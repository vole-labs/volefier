// Graph-driven ONNX inference proof on the ml-float (Mystique) backend: same
// ONNX front-end (parse/fold/fuse), but linear layers truncate via field
// range-check, ReLU/MaxPool run in the Boolean domain, and BatchNorm/Sigmoid/
// Gelu use IEEE-754 float circuits. Needs the dual bool+arith setup and a
// writable ./data/ for the Boolean COT.
//
//   test_ml-float_onnx_infer 1 12345 [model.onnx]   /   2 12345 [model.onnx]

#include <iostream>
#include <map>
#include <vector>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-float/float_graph_infer.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace volefier;
using namespace std;
typedef BoolIO ZIO;  // new emp: BoolIO is not a template

int port, party;
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
const int S = 12;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  const char *model = argc > 3 ? argv[3] : "references/deepprove/deep-prove-master/zkml/assets/model.onnx";
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";  // BOB(verifier) connects to ALICE(prover); set PEER_IP=<prover IP> on BOB
  ZIO *ios[threads];
  for (int i = 0; i < threads; ++i) ios[i] = new ZIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + i), party == ALICE);
  cout << endl << "----- ml-float graph-driven ONNX proof : " << model << " -----" << endl << endl;

  OnnxGraph g = load_onnx(model);
  for (auto &nm : g.inputs) { if (g.initializer.count(nm)) continue; auto &s = g.input_shapes[nm]; if (s.empty()) s = {1, 8}; for (size_t i = 0; i < s.size(); ++i) if (s[i] <= 0) s[i] = (i == 0 ? 1 : 8); }
  FoldStats fs; g = fold_constants(g, fs); FuseStats us; g = fuse_patterns(g, us);
  cout << g.nodes.size() << " ops after fold+fuse (folded " << fs.folded << ", fused " << us.layernorm << " LN + " << us.gelu << " GELU)" << endl;
  { map<string, int> bad; for (auto &n : g.nodes) if (!FloatGraphInfer<ZIO>::supported(n.op_type)) bad[n.op_type]++;
    if (!bad.empty()) { cout << "CANNOT PROVE:"; for (auto &kv : bad) cout << " " << kv.first << "x" << kv.second; cout << endl; for (int i = 0; i < threads; ++i) delete ios[i]; return 2; } }
  cout << "all operators supported -> composing ml-float proof" << endl;

  int in_dim = 0; { auto it = g.input_shapes.find(g.inputs[0]); if (it != g.input_shapes.end() && !it->second.empty()) { in_dim = 1; for (size_t i = (it->second.size() > 1 ? 1 : 0); i < it->second.size(); ++i) in_dim *= (int)it->second[i]; } }

  BenchE2E<ZIO> _bench(party, ios, threads);
  // Background COT: run Ferret on a distinct socket (ios[1]) in a producer thread (bool analogue
  // of arith's BG_VOLE), overlapping COT generation with the serial gate stream.  n_threads sizes
  // the engine pool, cot_threads the correlation producers -- split to avoid oversubscription.
  const int cot_threads = getenv("COT_THREADS") ? atoi(getenv("COT_THREADS")) : threads;
  ZKBoolSession bsess(ios[0], party, /*expected_cots=*/0, threads, /*cot_io=*/ios[1], cot_threads);
  setup_zk_arith(ios[0], party, bsess, threads);                      // arith + bool<->arith conversion
  IPProof<ZIO> ipp(party, ios[0], volefier::get_ostriple(party));
  FloatGraphInfer<ZIO> net(&ipp, &bsess, party, S, threads);

  vector<int64_t> xr(in_dim, 0);
  if (party == ALICE) { PRG prg; for (auto &v : xr) { uint64_t t; prg.random_data(&t, 8); v = (int64_t)(t % 200) - 100; } }
  if (party == ALICE) { const char *ip = getenv("INPUT_F32"); if (ip) { FILE *f = fopen(ip, "rb"); vector<float> fb(in_dim); size_t rd = fread(fb.data(), 4, in_dim, f); fclose(f); for (int i = 0; i < in_dim && i < (int)rd; ++i) xr[i] = llround((double)fb[i] * (1 << S)); cout << "loaded real input (" << rd << " vals) from " << ip << endl; } }
  vector<IntFp> cx = net.commit_vec(xr);

  auto t0 = clock_start();
  vector<int64_t> oref; vector<IntFp> out = net.run(g, cx, xr, oref);
  int64_t maxerr = 0;
  for (size_t j = 0; j < out.size(); ++j) { int64_t got = FloatGraphInfer<ZIO>::decode(out[j].reveal()); if (party == ALICE && !oref.empty()) maxerr = max<int64_t>(maxerr, llabs(got - oref[j])); }
  double secs = time_from(t0) / 1e6;
  uint64_t nand = bsess.num_and();   // total AND gates emitted (read before finalize deletes the engine)

  finalize_zk_arith();     // flush arith + conversion (uses bsess)
  bsess.finalize();        // bool closing checks; aborts via error() on a cheating prover
  bool cheated = false;
  _bench.report("onnx-infer (ml-float)");
  if (party == ALICE) {
    cout << "dispatched: "; for (auto &op : net.ops_run) cout << op << " "; cout << endl;
    if (cheated) error("bool ZK rejected\n");
    cout << "edaBits=" << net.n_eda << " float-ops=" << net.n_flt << " and-gates=" << nand << "  proving " << secs << " s  max-abs-err " << maxerr << " (scale 2^" << S << ")" << endl;
    { int am = 0; for (size_t j = 1; j < oref.size(); ++j) if (oref[j] > oref[am]) am = (int)j; cout << "ARGMAX (predicted class) = " << am << endl; }
    { const char *od = getenv("OREF_DUMP"); if (od) { FILE *f = fopen(od, "wb"); fwrite(oref.data(), 8, oref.size(), f); fclose(f); cout << "dumped oref to " << od << endl; } }
    if (maxerr > 8192) error("ml-float graph inference error too large\n");   // float32-vs-double ULPs scale with input magnitude
    cout << "pass: ml-float proved ONNX inference (" << g.nodes.size() << " ops, in " << in_dim << " -> out " << out.size() << ")" << endl;
  } else cout << "verifier: ml-float proof complete" << endl;
  cout.flush(); fflush(stdout);
  _exit(0);
}
