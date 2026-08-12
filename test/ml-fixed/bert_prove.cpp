// Fixed-point (ml-fixed) proof of a real BERT-base forward pass.
// Uniform 2^-P fixed-point throughout (the fp32 model; int8 weights are NOT used -- this is the
// fixed-point counterpart of test/ml/quant_bert_prove.cpp).  Graph: encoder + pooler(Tanh) + 2-class
// classifier.  Exercises the new ml-fixed Tanh (pooler) and Div (attention score / sqrt(d_head)) handlers.
//   build: test_ml-fixed_bert_prove ;  Env: P (precision), IDS, SEQ, MODEL.
#include <iostream>
#include <vector>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-fixed/graph_infer.h"
#include "test/bench_e2e.h"
using namespace emp; using namespace volefier; using namespace std;
int port, party; const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4; const int P = getenv("P") ? atoi(getenv("P")) : 12;
int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port); setvbuf(stdout, NULL, _IONBF, 0);
  int seq = getenv("SEQ") ? atoi(getenv("SEQ")) : 16;   // must match the fixed-seq export
  const char *mp = getenv("MODEL") ? getenv("MODEL") : "model/bert_base_fp32_basic.onnx";
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE")) ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
  OnnxGraph g = load_onnx(mp);
  auto inj = [&](const char *nm, vector<int64_t> iv) { OnnxTensor t; t.name = nm; t.dtype = 7; t.dims = {1, (int64_t)iv.size()}; t.i64 = iv; g.initializer[nm] = t; g.input_shapes[nm] = {1, (int64_t)iv.size()}; };
  vector<int64_t> ids;
  if (getenv("IDS")) { string s = getenv("IDS"), cur; for (char c : s + ",") { if (c == ',') { if (!cur.empty()) ids.push_back(atoll(cur.c_str())); cur.clear(); } else cur += c; } seq = (int)ids.size(); }
  else for (int i = 0; i < seq; ++i) ids.push_back(101 + i * 7);   // arbitrary ids (random-init classifier -> path test)
  inj("input_ids", ids);
  FoldStats fs; g = fold_constants(g, fs); FuseStats us; g = fuse_patterns(g, us);
  printf("[%d] folded+fused: %zu nodes (seq=%d)\n", party, g.nodes.size(), seq);
  { map<string, int> bad; for (auto &n : g.nodes) if (!FixedGraphInfer::supported(n.op_type)) bad[n.op_type]++;
    if (!bad.empty()) { printf("UNSUPPORTED:"); for (auto &kv : bad) printf(" %s", kv.first.c_str()); printf("\n"); delete ios[0]; if (ios[1]) delete ios[1]; return 2; } }
  BenchE2E<BoolIO> _bench(party, ios, ios[1] ? 2 : 1);
  setup_zk_arith(ios, threads, party, getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL, ios[1]);
  {
    FixedInfer qi(party, P, ios[0], 32, threads);
    RangeCheckLU rclu(party, ios[0]->io, 16); CompareLU cmp(&rclu); Div divr(party, ios[0]->io, 16);
    // tables keep their REAL domain as P scales: exp [0,512) (score_bits=P+9), gelu & tanh +-64 (tanh saturates within +-4).
    const std::size_t SB = (std::size_t)(P + 9); const uint64_t ET = 1ull << SB;
    const uint64_t GT = 1u << (P + 7); const int64_t GOFF = 1LL << (P + 6);
    const uint64_t TT = 1u << (P + 7); const int64_t TOFF = 1LL << (P + 6);
    const bool lse = getenv("LSE") != nullptr;   // DeepProve LogSumExp softmax (no max_value/division)
    FixedTable expt(ET, party, ios[0]->io); expt.Setup(build_exp_table(ET, P, !lse));  // LSE needs the non-clamped table
    FixedTable gtab(GT, party, ios[0]->io); gtab.Setup(build_gelu_table(GT, P, GOFF));
    FixedTable ttab(TT, party, ios[0]->io); ttab.Setup(build_tanh_table(TT, P, TOFF));   // <-- new Tanh pooler gadget
    FixedGraphInfer gi(&qi, &cmp, party, P); gi.divr = &divr; gi.rcmp = &rclu; gi.exp_tab = &expt; gi.gelu_tab = &gtab; gi.gelu_off = GOFF;
    gi.tanh_tab = &ttab; gi.tanh_off = TOFF; gi.score_bits = SB; gi.lse_softmax = lse;
    vector<int64_t> xr(seq, 0); vector<IntFp> cx; qi.commit_vec(cx, xr); vector<int64_t> oref;
    printf("[%d] running %zu ops...\n", party, g.nodes.size());
    vector<IntFp> out = gi.run(g, cx, xr, oref);
    printf("[%d] dispatched %zu ops, %zu class logits\n", party, gi.ops_run.size(), out.size());
    int cls = -1; if (!getenv("NO_ARGMAX")) { IntFp idx = cmp.argmax(out, 24); cls = (int)idx.reveal(); } else for (auto &v : out) v.reveal();
    rclu.Teardown(); expt.Teardown(); gtab.Teardown(); ttab.Teardown(); divr.destroy(); qi.destroy(); finalize_zk_arith();   // divr/qi BEFORE finalize: range-check batches must be in the proof
    _bench.report("bert-base (ml-fixed)");
    if (party == ALICE) { printf("PROVED bert-base (fp32, ml-fixed): %zu ops, %zu class logits, argmax class=%d\n", gi.ops_run.size(), out.size(), cls);
      printf("pass: ml-fixed proved real BERT-base forward incl. Tanh pooler (verifier accepted)\n"); }
    else printf("verifier: bert fixed-point proof complete\n");
  }
  cout.flush(); fflush(stdout); _exit(0);
}
