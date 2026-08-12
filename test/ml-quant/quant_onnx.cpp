// Graph-driven QDQ-int8 ONNX inference proof on the ml-quant (per-tensor) backend.
//   test_ml-quant_quant_onnx 1 12345 model/resnet50-v1-7-qdq.onnx
#include <iostream>
#include <vector>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx.h"
#include "volefier/ml-quant/quant_graph_infer.h"
#include "test/bench_e2e.h"
using namespace emp; using namespace volefier; using namespace std;
int port, party; const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  setvbuf(stdout, NULL, _IONBF, 0); std::cout.setf(std::ios::unitbuf);
  const char *model = argc > 3 ? argv[3] : "model/resnet50-v1-7-qdq.onnx";
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE")) ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
  cout << "\n----- ml-quant QDQ ONNX proof : " << model << " -----\n\n";
  OnnxGraph g = load_onnx(model);
  for (auto &nm : g.inputs) { if (g.initializer.count(nm)) continue; auto &s = g.input_shapes[nm]; for (size_t i=0;i<s.size();++i) if (s[i]<=0) s[i]=(i==0?1:224); }
  { map<string,int> bad; for (auto &n : g.nodes) if (!QuantGraphInfer::supported(n.op_type)) bad[n.op_type]++;
    if (!bad.empty()) { cout << "CANNOT PROVE:"; for (auto &kv : bad) cout << " " << kv.first << "x" << kv.second; cout << "\n"; delete ios[0]; if(ios[1]) delete ios[1]; return 2; } }
  cout << g.nodes.size() << " ops, all supported -> ml-quant proof\n";
  double in_scale = g.initializer.count("data_scale") ? g.initializer.at("data_scale").data[0] : 1.0;
  int in_zp = g.initializer.count("data_zero_point") && !g.initializer.at("data_zero_point").i64.empty() ? (int)g.initializer.at("data_zero_point").i64[0] : 0;
  int in_dim = 1; for (size_t i = 1; i < g.input_shapes[g.inputs[0]].size(); ++i) in_dim *= (int)g.input_shapes[g.inputs[0]][i];

  BenchE2E<BoolIO> _bench(party, ios, ios[1] ? 2 : 1);
  setup_zk_arith(ios, threads, party, getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL, ios[1]);
  {
    QuantInfer qi(party, 16, ios[0], 24, 1, 4);
    RangeCheckLU rclu(party, ios[0]->io, 16); CompareLU cmp(&rclu); Div divr(party, ios[0]->io, 16);
    QuantGraphInfer gi(&qi, &cmp, &divr, party);
    // QuantizeLinear [0,255] saturation as ONE LogUp lookup (vs 2 comparisons/element) -- the QDQ requant bottleneck.
    QuantTable clampt(1u<<16, party, ios[0]->io); clampt.Setup(build_clamp_table(1u<<16, 1<<15, 0, 255));
    if (getenv("USE_CLAMP_LUT")) { gi.clamp_tab = &clampt; gi.clamp_off = 1<<15; }   // opt-in: pre-clamp values must fit +-2^15. Works on WELL-CALIBRATED nets (imagenet-calib ResNet-50: QuantizeLinear 18.3->3.1s, total -40%, comm -55%, bit-exact); only the SYNTHETIC-calib default resnet50-v1-7-qdq.onnx overflows 2^15 (bad scales).
    vector<int64_t> xr(in_dim, 0);
    if (party == ALICE) { PRG prg; for (auto &v : xr) { uint64_t t; prg.random_data(&t, 8); v = (int64_t)(t % 256); } }
    if (party == ALICE) { const char *ip = getenv("INPUT_F32"); if (ip) { FILE *f = fopen(ip, "rb"); vector<float> fb(in_dim); size_t rd = fread(fb.data(), 4, in_dim, f); fclose(f); for (int i = 0; i < in_dim && i < (int)rd; ++i) { long q = lround((double)fb[i] / in_scale) + in_zp; xr[i] = q < 0 ? 0 : (q > 255 ? 255 : q); } cout << "loaded real input (" << rd << " vals) from " << ip << "\n"; } }
    vector<IntFp> cx; qi.commit_vec(cx, xr);
    auto t0 = clock_start();
    vector<int64_t> oref; vector<IntFp> out = gi.run(g, cx, xr, in_scale, in_zp, oref);
    int64_t maxerr = 0; for (size_t j = 0; j < out.size(); ++j) { int64_t got = QuantInfer::decode_signed(out[j].reveal()); if (party == ALICE && !oref.empty()) maxerr = max(maxerr, (int64_t)llabs(got - oref[j])); }  // cast: int64_t==long != long long on Linux
    double secs = time_from(t0) / 1e6;
    clampt.Teardown(); rclu.Teardown(); divr.destroy(); qi.destroy(); finalize_zk_arith();   // divr/qi BEFORE finalize: range-check batches must be in the proof
    _bench.report("onnx (ml-quant)");
    if (party == ALICE) { cout << "dispatched " << gi.ops_run.size() << " ops, proving " << secs << " s, max-abs-err " << maxerr << "\n";
      if (maxerr != 0) cout << "MISMATCH (integer arithmetic must be exact)\n"; else cout << "pass: ml-quant proved QDQ-int8 ONNX inference (out dim " << out.size() << ") EXACT\n";
      { int am = 0; for (size_t j = 1; j < oref.size(); ++j) if (oref[j] > oref[am]) am = (int)j; cout << "ARGMAX (predicted class) = " << am << "\n"; }
      { const char *od = getenv("OREF_DUMP"); if (od) { FILE *f = fopen(od, "wb"); fwrite(oref.data(), 8, oref.size(), f); fclose(f); cout << "dumped oref (" << oref.size() << " int64) to " << od << "\n"; } } }
    else cout << "verifier: ml-quant proof complete\n";
  }
  delete ios[0]; if(ios[1]) delete ios[1];
  return 0;
}
