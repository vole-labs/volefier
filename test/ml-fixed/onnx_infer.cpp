// Graph-driven ONNX inference proof (ml-fixed). Reads a real ONNX model file,
// extracts its operator graph + weights, and composes a ZK proof of its
// inference by dispatching each operator to a fixed-point ZKP component. Aborts
// if the model uses an unsupported operator.
//
//   test_ml-fixed_onnx_infer 1 12345 [model.onnx]
//   test_ml-fixed_onnx_infer 2 12345 [model.onnx]

#include <iostream>
#include <vector>
#include <unistd.h>   // _exit
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-fixed/graph_infer.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4, P = 12;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  const char *model = argc > 3 ? argv[3] : "references/deepprove/deep-prove-master/zkml/assets/model.onnx";
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE")) ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
  cout << endl << "----- graph-driven ONNX inference proof (ml-fixed) : " << model << " -----" << endl << endl;

  OnnxGraph g = load_onnx(model);
  cout << g.nodes.size() << " raw operators" << endl;

  // set a concrete shape for every real (non-weight) input so the passes can run
  for (auto &nm : g.inputs) { if (g.initializer.count(nm)) continue; auto &s = g.input_shapes[nm];
    if (s.empty()) s = {1, 8}; for (size_t i = 0; i < s.size(); ++i) if (s[i] <= 0) s[i] = (i == 0 ? 1 : 8); }
  // graph-optimization passes: constant-fold static plumbing, then fuse the
  // decomposed LayerNorm/GELU subgraphs back into the ops we handle.
  FoldStats fs; g = fold_constants(g, fs);
  FuseStats us; g = fuse_patterns(g, us);
  cout << "after fold+fuse: " << g.nodes.size() << " operators  (folded " << fs.folded
       << ", fused " << us.layernorm << " LayerNorm + " << us.gelu << " GELU)" << endl;

  // pre-flight: list every operator with no ZKP component
  { std::map<std::string,int> bad; for (auto &n : g.nodes) if (!FixedGraphInfer::supported(n.op_type)) bad[n.op_type] += 1;
    if (!bad.empty()) { int tot = 0; cout << "CANNOT PROVE -- unsupported ops remain:"; for (auto &kv : bad) { cout << " " << kv.first << "x" << kv.second; tot += kv.second; } cout << "  (" << tot << " nodes)" << endl;
      delete ios[0]; if(ios[1]) delete ios[1]; return 2; } }
  cout << "all operators supported -> composing proof" << endl;

  // total input elements from the declared input shape (drop batch); fall back
  // to the first Gemm's in-dim if the shape is missing/dynamic.
  int in_dim = 0;
  { auto it = g.input_shapes.find(g.inputs[0]);
    if (it != g.input_shapes.end() && !it->second.empty()) { in_dim = 1; for (size_t i = (it->second.size() > 1 ? 1 : 0); i < it->second.size(); ++i) in_dim *= (int)it->second[i]; } }
  if (in_dim <= 0) for (auto &n : g.nodes) if (n.op_type == "Gemm") { const auto &W = g.initializer.at(n.input[1]); int tb = n.attr.count("transB") ? (int)n.attr.at("transB").i : 0; in_dim = (int)(tb ? W.dims[1] : W.dims[0]); break; }

  // sigmoid lookup table (cnn-covid uses Sigmoid); exp/gelu for transformer ops
  const uint64_t EXP_T = 1u << 16, GELU_T = 1u << 14, SIG_T = 1u << 20;
  const int64_t GELU_OFF = 1 << 13, SIG_OFF = 1 << 19;   // sigmoid domain +-128.0
  auto build_sigmoid_table = [](uint64_t T, int p, int64_t off) { std::vector<uint64_t> t(T); for (uint64_t i = 0; i < T; ++i) { double x = (double)((int64_t)i - off) / (1 << p); t[i] = (uint64_t)llround(1.0 / (1.0 + exp(-x)) * (1 << p)); } return t; };

  BenchE2E<BoolIO> _bench(party, ios, ios[1] ? 2 : 1);
  setup_zk_arith(ios, threads, party, getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL, ios[1]);
  FixedInfer qi(party, P, ios[0], 24);
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, 16);
  FixedTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table(EXP_T, P));
  FixedTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table(GELU_T, P, GELU_OFF));
  FixedTable sigt(SIG_T, party, ios[0]->io); sigt.Setup(build_sigmoid_table(SIG_T, P, SIG_OFF));
  FixedGraphInfer gi(&qi, &cmp, party, P);
  gi.divr = &divr; gi.rcmp = &rcmp;
  gi.exp_tab = &expt; gi.gelu_tab = &gtab; gi.gelu_off = GELU_OFF; gi.sig_tab = &sigt; gi.sig_off = SIG_OFF;

  // commit a random input vector (scale 2^P)
  vector<int64_t> xr(in_dim, 0);
  if (party == ALICE) { PRG prg; for (auto &v : xr) { uint64_t t; prg.random_data(&t, 8); v = (int64_t)(t % 200) - 100; } }   // small (no real input image)
  if (party == ALICE) { const char *ip = getenv("INPUT_F32"); if (ip) { FILE *f = fopen(ip, "rb"); vector<float> fb(in_dim); size_t rd = fread(fb.data(), 4, in_dim, f); fclose(f); for (int i = 0; i < in_dim && i < (int)rd; ++i) xr[i] = llround((double)fb[i] * (1 << P)); cout << "loaded real input (" << rd << " vals) from " << ip << endl; } }
  vector<IntFp> cx; qi.commit_vec(cx, xr);

  vector<int64_t> out_ref;
  vector<IntFp> out = gi.run(g, cx, xr, out_ref);

  int fails = 0;
  for (size_t j = 0; j < out.size(); ++j) {
    int64_t got = FixedInfer::decode_signed(out[j].reveal());
    if (party == ALICE && got != out_ref[j]) { if (++fails <= 5) cout << "  out[" << j << "] " << got << " vs ref " << out_ref[j] << endl; }
  }

  rcmp.Teardown(); expt.Teardown(); gtab.Teardown(); sigt.Teardown();
  divr.destroy(); qi.destroy();   // BEFORE finalize: range-check batches must be in the proof
  finalize_zk_arith();
  _bench.report("onnx-infer (ml-fixed)");
  if (party == ALICE) {
    cout << "dispatched: "; for (auto &op : gi.ops_run) cout << op << " "; cout << endl;
    if (fails) error("graph inference mismatch\n");
    { int am = 0; for (size_t j = 1; j < out_ref.size(); ++j) if (out_ref[j] > out_ref[am]) am = (int)j; cout << "ARGMAX (predicted class) = " << am << endl; }
    { const char *od = getenv("OREF_DUMP"); if (od) { FILE *f = fopen(od, "wb"); fwrite(out_ref.data(), 8, out_ref.size(), f); fclose(f); cout << "dumped out_ref to " << od << endl; } }
    cout << "pass: proved ONNX model inference (" << g.nodes.size() << " ops, input dim " << in_dim
         << ", output dim " << out.size() << ") matches plaintext reference" << endl;
  } else cout << "verifier: proof complete" << endl;
  // The proof is finalized and verified on both sides at this point. emp-zk's
  // global VOLE/IO state segfaults if torn down via the normal destructor path
  // (deleting `ios` / static dtors) after finalize, so flush and hard-exit
  // instead of running cleanup — the OS reclaims everything.
  cout.flush(); fflush(stdout);
  _exit(0);
}
