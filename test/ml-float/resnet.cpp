// High-precision ResNet inference proof: fixed-point conv/fc in the field +
// float BatchNorm/ReLU/pool/residual in IEEE-754 Boolean circuits, bridged by
// emp-zk edaBit + Mystique Fix<->Float at each boundary.
//
//   test_ml-float_resnet PARTY PORT [HW base threads b1 b2 ...]
//     HW      input height=width (3 x HW x HW), default 8
//     base    stem channels, default 4
//     threads VOLE/COT threads, default 1
//     b1 b2.. basic blocks per stage (channels double + spatial halves per stage
//             after the first), default "1 1"   e.g. 2 2 2 2 ~ ResNet-18 structure

#include <iostream>
#include <vector>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/ml-float/float_resnet.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int S = 16;
typedef BoolIO<NetIO> ZIO;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  int HW = argc > 3 ? atoi(argv[3]) : 8;
  int base = argc > 4 ? atoi(argv[4]) : 4;
  int threads = argc > 5 ? atoi(argv[5]) : 1;
  bool resnet50 = (argc > 6 && string(argv[6]) == "50");   // pass "50" -> full bottleneck ResNet-50
  vector<int> blocks;
  for (int i = 6; i < argc; ++i) blocks.push_back(atoi(argv[i]));
  if (blocks.empty()) blocks = {1, 1};

  ZIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new ZIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "----- high-precision float ResNet : 3x" << HW << "x" << HW << ", base=" << base
       << ", blocks={"; for (int b : blocks) cout << b << ","; cout << "}, threads=" << threads << " -----" << endl << endl;

  BenchE2E<ZIO> _bench(party, ios, threads);
  setup_zk_bool<ZIO>(ios, threads, party);
  setup_zk_arith<ZIO>(ios, threads, party, /*enable_conversion=*/true);
  IPProof<ZIO> ipp(party, ios[0], volefier::get_ostriple<ZIO>(party));
  PRG prg;
  FloatResNet<ZIO> net(&ipp, party, S, &prg, threads);

  FloatResNet<ZIO>::St inp; inp.C = 3; inp.H = HW; inp.W = HW;
  inp.X.assign(3, vector<vector<Float>>(HW, vector<Float>(HW)));
  if (party == ALICE) inp.Xr.assign(3, vector<vector<double>>(HW, vector<double>(HW)));
  for (int c = 0; c < 3; ++c) for (int h = 0; h < HW; ++h) for (int w = 0; w < HW; ++w) {
    double v = net.rnd(-1.0, 1.0);
    inp.X[c][h][w] = Float((float)v, ALICE);
    if (party == ALICE) inp.Xr[c][h][w] = v;
  }

  auto t0 = clock_start();
  auto out = resnet50 ? net.forward_resnet50(inp, base, 10) : net.forward(inp, base, 10, blocks);
  ipp.batch_check();
  ios[0]->flush();
  double secs = time_from(t0) / 1e6;

  double max_abs = 0, max_rel = 0;
  for (size_t j = 0; j < out.X.size(); ++j) {
    double g = out.X[j].reveal<double>(PUBLIC);
    if (party == ALICE) {
      double err = fabs(g - out.Xr[j]), rel = err / (fabs(out.Xr[j]) + 1e-6);
      max_abs = std::max(max_abs, err); max_rel = std::max(max_rel, rel);
    }
  }

  finalize_zk_arith<ZIO>();
  bool cheated = finalize_zk_bool<ZIO>();
  _bench.report("resnet (ml-float)");
  if (party == ALICE) {
    cout << "Fix<->Float conversions: " << net.n_conv_acts
         << "   proving: " << secs << " s   (" << (secs * 1e6 / net.n_conv_acts) << " us / conversion)" << endl;
    cout << "committed-vs-double max abs err " << max_abs << ", max rel err " << max_rel
         << "  (float32 circuit vs double ref; the PROOF is exact)" << endl;
    if (max_abs > 0.5) error("gross divergence -- likely a bug, not float accumulation\n");
    cout << "pass high-precision float ResNet (fixed-point conv + float BN/ReLU/pool/residual)" << endl;
  } else {
    cout << "verifier check: " << (cheated ? "REJECT" : "OK") << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
