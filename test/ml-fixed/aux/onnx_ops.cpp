// Validates the graph-prover op handlers that no available ONNX file exercises,
// by building synthetic graphs in C++ (the same OnnxGraph the parser produces)
// and proving them:
//   (A) ResNet-style block:  Conv -> BatchNormalization -> Relu -> Add(residual)
//                            -> GlobalAveragePool -> Flatten -> Gemm   (exact-validated)
//   (B) transformer ops:     MatMul -> Add(bias) -> LayerNormalization -> Gelu -> Softmax
//                            (MatMul+Add exact-validated; LN/Gelu/Softmax dispatch + prove)
//
//   test_ml-fixed_onnx_ops 1 12345  /  2 12345

#include <iostream>
#include <vector>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-fixed/graph_infer.h"

using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = 4, P = 12;

static OnnxNode nd(string op, vector<string> in, vector<string> out) { OnnxNode n; n.op_type = op; n.name = op; n.input = in; n.output = out; return n; }
static void ai(OnnxNode &n, string k, int64_t v) { OnnxAttr a; a.name = k; a.ints = {v}; a.i = v; n.attr[k] = a; }
static void aiv(OnnxNode &n, string k, vector<int64_t> v) { OnnxAttr a; a.name = k; a.ints = v; n.attr[k] = a; }
static void af(OnnxNode &n, string k, float v) { OnnxAttr a; a.name = k; a.f = v; n.attr[k] = a; }

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i) ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "----- graph-prover op handlers (synthetic ResNet block + transformer ops) -----" << endl << endl;

  block seed = makeBlock(0, 1234); PRG prg(&seed);                         // deterministic weights on both parties
  auto rf = [&](int n, float s) { vector<float> v(n); for (auto &x : v) { uint64_t r; prg.random_data(&r, 8); x = s * (2.f * (float)(r >> 40) / (float)(1 << 24) - 1.f); } return v; };
  auto mk = [](vector<int64_t> dims, vector<float> d) { OnnxTensor t; t.dtype = 1; t.dims = dims; t.data = std::move(d); return t; };

  // ---------- graph A: ResNet-style block ----------
  const int C = 4, Hh = 6, Wd = 6;
  OnnxGraph A; A.inputs = {"x"}; A.outputs = {"outA"}; A.input_shapes["x"] = {1, C, Hh, Wd};
  A.initializer["cw"] = mk({C, C, 3, 3}, rf(C * C * 9, 0.15f));  A.initializer["cb"] = mk({C}, rf(C, 0.05f));
  A.initializer["bs"] = mk({C}, rf(C, 0.2f));                    A.initializer["bb"] = mk({C}, rf(C, 0.1f));
  A.initializer["bm"] = mk({C}, rf(C, 0.1f));
  { vector<float> var(C); for (auto &v : var) { uint64_t r; prg.random_data(&r, 8); v = 0.5f + (float)(r >> 40) / (float)(1 << 24); } A.initializer["bv"] = mk({C}, var); }  // var > 0
  A.initializer["fw"] = mk({3, C}, rf(3 * C, 0.2f));             A.initializer["fb"] = mk({3}, rf(3, 0.05f));
  { OnnxNode c = nd("Conv", {"x", "cw", "cb"}, {"c"}); ai(c, "strides", 1); ai(c, "pads", 1); A.nodes.push_back(c); }
  { OnnxNode b = nd("BatchNormalization", {"c", "bs", "bb", "bm", "bv"}, {"bn"}); af(b, "epsilon", 1e-5f); A.nodes.push_back(b); }
  A.nodes.push_back(nd("Relu", {"bn"}, {"r"}));
  A.nodes.push_back(nd("Add", {"r", "x"}, {"a"}));               // residual (same shape)
  A.nodes.push_back(nd("GlobalAveragePool", {"a"}, {"g"}));
  { OnnxNode f = nd("Flatten", {"g"}, {"fl"}); ai(f, "axis", 1); A.nodes.push_back(f); }
  { OnnxNode gm = nd("Gemm", {"fl", "fw", "fb"}, {"outA"}); ai(gm, "transB", 1); A.nodes.push_back(gm); }

  // ---------- graph B: transformer ops ----------
  const int S = 2, D = 4;
  OnnxGraph B; B.inputs = {"y"}; B.outputs = {"outB"}; B.input_shapes["y"] = {1, S, D};
  B.initializer["mw"] = mk({D, D}, rf(D * D, 0.2f));             // MatMul weight [in,out]
  B.initializer["mb"] = mk({D}, rf(D, 0.05f));                   // bias (per-feature)
  B.initializer["lg"] = mk({D}, rf(D, 0.1f));                    // LN gamma ~1
  for (auto &v : B.initializer["lg"].data) v += 1.0f;
  B.initializer["lb"] = mk({D}, rf(D, 0.05f));                   // LN beta
  B.nodes.push_back(nd("MatMul", {"y", "mw"}, {"m"}));
  B.nodes.push_back(nd("Add", {"m", "mb"}, {"ab"}));            // per-feature bias broadcast
  B.nodes.push_back(nd("LayerNormalization", {"ab", "lg", "lb"}, {"ln"}));
  B.nodes.push_back(nd("Gelu", {"ln"}, {"gl"}));
  B.nodes.push_back(nd("Softmax", {"gl"}, {"outB"}));

  // ---------- graph C: self-attention glue (Gather/Split/Transpose/two-act MatMul/Mul) ----------
  auto mki = [](vector<int64_t> dims, vector<int64_t> idata) { OnnxTensor t; t.dtype = 7; t.dims = dims; t.i64 = std::move(idata); return t; };
  const int Sa = 2, Da = 4, Va = 5;
  auto build_attn = [&](bool full) {
    OnnxGraph G; G.inputs = {"px"}; G.outputs = {full ? "ctx" : "sc"}; G.input_shapes["px"] = {1, Sa, Da};
    G.initializer["wte"] = mk({Va, Da}, rf(Va * Da, 0.2f));
    G.initializer["ids"] = mki({Sa}, {1, 3});                          // token ids -> embedding rows
    G.initializer["Wqkv"] = mk({Da, 3 * Da}, rf(Da * 3 * Da, 0.15f));  // fused QKV projection
    G.initializer["Wo"] = mk({Da, Da}, rf(Da * Da, 0.15f));
    G.initializer["scale"] = mk({1}, {(float)(1.0 / sqrt((double)Da))});
    G.nodes.push_back(nd("Gather", {"wte", "ids"}, {"emb"}));          // embeddings
    G.nodes.push_back(nd("Add", {"emb", "px"}, {"h"}));                // + positional (the input)
    G.nodes.push_back(nd("MatMul", {"h", "Wqkv"}, {"qkv"}));
    G.nodes.push_back(nd("Split", {"qkv"}, {"Q", "K", "V"}));          // fused QKV -> 3
    { OnnxNode t = nd("Transpose", {"K"}, {"Kt"}); aiv(t, "perm", {1, 0}); G.nodes.push_back(t); }
    G.nodes.push_back(nd("MatMul", {"Q", "Kt"}, {"qk"}));              // Q . K^T  (two-activation)
    G.nodes.push_back(nd("Mul", {"qk", "scale"}, {"sc"}));             // / sqrt(d)
    if (full) { G.nodes.push_back(nd("Softmax", {"sc"}, {"attn"}));
      G.nodes.push_back(nd("MatMul", {"attn", "V"}, {"av"}));          // attn . V  (two-activation)
      G.nodes.push_back(nd("MatMul", {"av", "Wo"}, {"ctx"})); }        // output projection
    return G;
  };
  OnnxGraph Cg = build_attn(false), Dg = build_attn(true);

  // ---------- graph E: MULTI-HEAD scores (Reshape->Transpose->batched MatMul) ----------
  const int Sh = 2, Hd = 2, Dh = 2, Dm = Hd * Dh;   // 2 heads
  OnnxGraph E; E.inputs = {"hx"}; E.outputs = {"msc"}; E.input_shapes["hx"] = {1, Sh, Dm};
  E.initializer["Wqkv2"] = mk({Dm, 3 * Dm}, rf(Dm * 3 * Dm, 0.15f));
  E.initializer["rs"] = mki({4}, {1, Sh, Hd, Dh});   // reshape [S,D] -> [b,S,H,Dh]
  E.nodes.push_back(nd("MatMul", {"hx", "Wqkv2"}, {"qkv2"}));
  E.nodes.push_back(nd("Split", {"qkv2"}, {"Q2", "K2", "V2"}));
  E.nodes.push_back(nd("Reshape", {"Q2", "rs"}, {"Qr"}));
  { OnnxNode t = nd("Transpose", {"Qr"}, {"Qh"}); aiv(t, "perm", {1, 0, 2}); E.nodes.push_back(t); }   // [S,H,Dh]->[H,S,Dh]
  E.nodes.push_back(nd("Reshape", {"K2", "rs"}, {"Kr"}));
  { OnnxNode t = nd("Transpose", {"Kr"}, {"Kht"}); aiv(t, "perm", {1, 2, 0}); E.nodes.push_back(t); }   // [S,H,Dh]->[H,Dh,S]
  E.nodes.push_back(nd("MatMul", {"Qh", "Kht"}, {"msc"}));   // [H,S,Dh]@[H,Dh,S] -> [H,S,S] (batched)

  // ---------- run ----------
  const uint64_t EXP_T = 1u << 16, GELU_T = 1u << 14, SIG_T = 1u << 18;
  const int64_t GELU_OFF = 1 << 13, SIG_OFF = 1 << 17;
  auto sig_tab_build = [](uint64_t T, int p, int64_t off) { vector<uint64_t> t(T); for (uint64_t i = 0; i < T; ++i) { double x = (double)((int64_t)i - off) / (1 << p); t[i] = (uint64_t)llround(1.0 / (1.0 + exp(-x)) * (1 << p)); } return t; };

  setup_zk_arith(ios, threads, party);
  FixedInfer qi(party, P, ios[0], 20);
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, 16);
  FixedTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table(EXP_T, P));
  FixedTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table(GELU_T, P, GELU_OFF));
  FixedTable sigt(SIG_T, party, ios[0]->io); sigt.Setup(sig_tab_build(SIG_T, P, SIG_OFF));
  FixedGraphInfer gi(&qi, &cmp, party, P);
  gi.divr = &divr; gi.rcmp = &rcmp; gi.exp_tab = &expt; gi.gelu_tab = &gtab; gi.gelu_off = GELU_OFF; gi.sig_tab = &sigt; gi.sig_off = SIG_OFF;

  int fails = 0;
  auto run_graph = [&](OnnxGraph &g, const char *name, bool exact) {
    int n_in = 1; for (size_t i = 1; i < g.input_shapes[g.inputs[0]].size(); ++i) n_in *= (int)g.input_shapes[g.inputs[0]][i];
    vector<int64_t> xr(n_in, 0);
    if (party == ALICE) { for (auto &v : xr) { uint64_t t; prg.random_data(&t, 8); v = (int64_t)(t % 600) - 300; } }
    vector<IntFp> cx; qi.commit_vec(cx, xr);
    FixedGraphInfer g2(&qi, &cmp, party, P); g2.divr = &divr; g2.rcmp = &rcmp; g2.exp_tab = &expt; g2.gelu_tab = &gtab; g2.gelu_off = GELU_OFF; g2.sig_tab = &sigt; g2.sig_off = SIG_OFF;
    vector<int64_t> oref; vector<IntFp> out = g2.run(g, cx, xr, oref);
    int lf = 0;
    for (size_t j = 0; j < out.size(); ++j) { int64_t got = FixedInfer::decode_signed(out[j].reveal()); if (exact && party == ALICE && !oref.empty() && got != oref[j]) ++lf; }
    if (party == ALICE) { cout << "  [" << name << "] ops="; for (auto &op : g2.ops_run) cout << op << " "; cout << (exact ? (lf ? "  EXACT-MISMATCH(" + to_string(lf) + ")" : "  exact-OK") : "  dispatched+proven") << endl; }
    fails += lf;
  };
  run_graph(A, "ResNet block", true);
  run_graph(B, "transformer ops", false);
  run_graph(Cg, "attention glue (scaled scores)", true);   // Gather/Split/Transpose/two-act MatMul/Mul exact
  run_graph(Dg, "full self-attention", false);            // + Softmax + attn.V + proj
  run_graph(E, "multi-head scores (batched)", true);      // Reshape/Transpose/batched MatMul exact

  rcmp.Teardown(); expt.Teardown(); gtab.Teardown(); sigt.Teardown();
  finalize_zk_arith();
  if (party == ALICE) { if (fails) error("handler mismatch\n"); cout << "pass: all op handlers proved (ResNet block exact-validated; transformer ops dispatched + proven)" << endl; }
  else cout << "verifier: proof complete" << endl;
  // proof finalized + verified; skip emp-zk's crash-prone global teardown
  cout.flush(); fflush(stdout); _exit(0);
}
