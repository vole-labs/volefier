// FIELD-NATIVE high-precision ResNet (Mystique architecture): activations in the
// fixed-point field; conv/fc/residual in field; ReLU as an integer comparison;
// float ONLY for the truncation rescale and BatchNorm sqrt. Compare timing vs
// the float-native ml-float/resnet.cpp on the same config.
//
//   test_ml-float_field_resnet PARTY PORT [HW base threads b1 b2 .. | 50]

#include <iostream>
#include <vector>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/ml-float/field_resnet.h"
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
  bool resnet50 = (argc > 6 && string(argv[6]) == "50");
  vector<int> blocks; for (int i = 6; i < argc; ++i) blocks.push_back(atoi(argv[i]));
  if (blocks.empty()) blocks = {1, 1};

  ZIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new ZIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "----- FIELD-NATIVE float ResNet : 3x" << HW << "x" << HW << ", base=" << base
       << (resnet50 ? ", ResNet-50" : "") << ", threads=" << threads << " -----" << endl << endl;

  BenchE2E<ZIO> _bench(party, ios, threads);
  setup_zk_bool<ZIO>(ios, threads, party);
  setup_zk_arith<ZIO>(ios, threads, party, /*enable_conversion=*/true);
  IPProof<ZIO> ipp(party, ios[0], volefier::get_ostriple<ZIO>(party));
  PRG prg;
  FieldResNet<ZIO> net(&ipp, party, S, &prg, threads);

  // input committed as fixed-point IntFp (scale 2^S)
  FieldResNet<ZIO>::St inp; inp.C = 3; inp.H = HW; inp.W = HW;
  inp.X.assign(3, vector<vector<IntFp>>(HW, vector<IntFp>(HW)));
  if (party == ALICE) inp.Xr.assign(3, vector<vector<double>>(HW, vector<double>(HW)));
  for (int c = 0; c < 3; ++c) for (int h = 0; h < HW; ++h) for (int w = 0; w < HW; ++w) {
    double v = net.rnd(-1.0, 1.0);
    inp.X[c][h][w] = net.commit_q((int64_t)llround(v * (double)(1LL << S)));
    if (party == ALICE) inp.Xr[c][h][w] = v;
  }

  auto t0 = clock_start();
  auto out = resnet50 ? net.forward_resnet50(inp, base, 10) : net.forward(inp, base, 10, blocks);
  ipp.batch_check();
  ios[0]->flush();
  double secs = time_from(t0) / 1e6;

  double max_abs = 0;
  for (size_t j = 0; j < out.X.size(); ++j) {
    int64_t g = FieldResNet<ZIO>::dec(out.X[j].reveal());
    if (party == ALICE) max_abs = std::max(max_abs, fabs((double)g / (1LL << S) - out.Xr[j]));
  }

  auto tf = clock_start();
  finalize_zk_arith<ZIO>();
  bool cheated = finalize_zk_bool<ZIO>();
  _bench.report("field-resnet (ml-float)");
  double t_final = time_from(tf) / 1e6;
  if (party == ALICE) {
    double tm = net.t_matmul/1e6, te = net.t_eda/1e6, tfl = net.t_float/1e6, tfm = net.t_fieldmul/1e6, tc = net.t_commit/1e6;
    double other = secs - (tm + te + tfl + tfm + tc);
    cout << "proving " << secs << " s,  finalize " << t_final << " s   (max abs err " << max_abs << ")" << endl;
    cout << "---- microbenchmark breakdown (forward proving) ----" << endl;
    printf("  field matmul (conv+fc)     : %7.2f s  (%4.1f%%)\n", tm,  100*tm/secs);
    printf("  edaBit A2B/B2A (%9ld) : %7.2f s  (%4.1f%%)\n", net.n_eda, te, 100*te/secs);
    printf("  float circuits (%9ld) : %7.2f s  (%4.1f%%)\n", net.n_flt, tfl, 100*tfl/secs);
    printf("  field multiplies (BN)      : %7.2f s  (%4.1f%%)\n", tfm, 100*tfm/secs);
    printf("  weight/param commit        : %7.2f s  (%4.1f%%)\n", tc,  100*tc/secs);
    printf("  other (reshape/batch_check): %7.2f s  (%4.1f%%)\n", other, 100*other/secs);
    if (max_abs > 0.5) error("gross divergence -- likely a bug\n");
    cout << "pass FIELD-NATIVE float ResNet (field conv/fc/relu/add, float only for truncate + BN-sqrt)" << endl;
  } else {
    cout << "verifier check: " << (cheated ? "REJECT" : "OK") << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
