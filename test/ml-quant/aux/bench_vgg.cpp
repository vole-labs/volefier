// Per-operator microbenchmark of a VGG proof, for an apples-to-apples
// comparison with VerfCNN (which proves conv+ReLU+maxpool but NOT the FC
// layers). Reports time + communication for: weight commit (preprocessing),
// conv, ReLU, maxpool, FC, and finalize/teardown -- so we can compare both the
// VerfCNN-proven subset (conv+relu+pool) and the full forward (incl. FC).
//
//   bin/test_ml-quant_bench_vgg <party> <port> [depth base in_hw classes mat_threads]
//   default VerfCNN config: VGG-16, base 64, 32x32, 10 classes, 1 thread.

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "test/utility.h"
#include "volefier/ml-quant/quant_vgg.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  int depth      = (argc >= 4) ? atoi(argv[3]) : 16;
  int base       = (argc >= 5) ? atoi(argv[4]) : 64;
  int in_hw      = (argc >= 6) ? atoi(argv[5]) : 32;
  int classes    = (argc >= 7) ? atoi(argv[6]) : 10;
  int mat_thr    = (argc >= 8) ? atoi(argv[7]) : 1;   // VerfCNN is single-thread
  const int fc_dim = base * 8;

  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  auto comm = [&]() { uint64_t c = 0; for (int i = 0; i < threads; ++i) c += io_counter(ios[i]->io); return c; };

  PRG prg;
  setup_zk_arith(ios, threads, party);
  QuantInfer qi(party, P, ios[0], /*out_bits=*/20, /*wc=*/1, /*mat_threads=*/mat_thr);
  RangeCheckLU rclu(party, ios[0]->io, 16);
  CompareLU cmp(&rclu);
  Div divr(party, ios[0]->io, /*range_bits=*/24);

  VGGCfg cfg = vgg_cfg(depth, base, fc_dim, classes, in_hw, /*bn=*/false);
  if (party == ALICE) cout << "config: " << cfg.name << " base=" << base << " " << in_hw << "x" << in_hw
                           << " classes=" << classes << " mat_threads=" << mat_thr << endl;

  auto rndv = [&](int64_t lo, int64_t hi) -> int64_t { if (party != ALICE) return 0; uint64_t r; prg.random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); };
  St input; input.C = 3; input.H = in_hw; input.W = in_hw;
  input.X.assign(3, vector<vector<IntFp>>(in_hw, vector<IntFp>(in_hw)));
  if (party == ALICE) input.Xr.assign(3, vector<vector<int64_t>>(in_hw, vector<int64_t>(in_hw)));
  for (int c = 0; c < 3; ++c) for (int h = 0; h < in_hw; ++h) for (int w = 0; w < in_hw; ++w) {
    int64_t v = rndv(-127, 127); input.X[c][h][w] = qi.commit(v); if (party == ALICE) input.Xr[c][h][w] = v;
  }

  ResNet net(&qi, &cmp, &divr, &prg, party, /*act_bits=*/24);
  net.commf = comm;
  VecSt logits = vgg_forward(net, cfg, input);
  for (int j = 0; j < classes; ++j) (void)logits.X[j].reveal();   // force the output

  // finalize + LogUp teardowns (the batched verification, like VerfCNN's batched ReLU proof)
  uint64_t fb0 = comm(); auto ft0 = clock_start();
  rclu.Teardown(); divr.destroy(); qi.destroy();
  finalize_zk_arith();
  double fin_us = time_from(ft0); uint64_t fin_b = comm() - fb0;

  if (party == ALICE) {
    auto &C = net.ph_commit; auto &Cv = net.ph_conv; auto &R = net.ph_relu; auto &Po = net.ph_pool; auto &Fc = net.ph_fc;
    double prove_us = Cv.us + R.us + Po.us + Fc.us + fin_us;
    uint64_t prove_b = Cv.bytes + R.bytes + Po.bytes + Fc.bytes + fin_b;
    double verf_us = Cv.us + R.us + Po.us + fin_us;      // VerfCNN-comparable: conv+relu+pool
    uint64_t verf_b = Cv.bytes + R.bytes + Po.bytes + fin_b;
    auto row = [](const char *n, double us, uint64_t b){ char s[160]; snprintf(s,sizeof s,"  %-26s %8.3f s   %8.1f MB", n, us/1e6, b/1e6); cout<<s<<endl; };
    cout << "\n  phase                          time         comm" << endl;
    cout << "  ----------------------------------------------------------" << endl;
    row("weight commit (preproc)", C.us, C.bytes);
    cout << "  -- per-inference proving --" << endl;
    row("conv", Cv.us, Cv.bytes);
    row("relu", R.us, R.bytes);
    row("maxpool", Po.us, Po.bytes);
    row("fc (classifier)", Fc.us, Fc.bytes);
    row("finalize + teardown", fin_us, fin_b);
    cout << "  ----------------------------------------------------------" << endl;
    row("PROVING total (incl. FC)", prove_us, prove_b);
    row("conv+relu+pool (=VerfCNN proves)", verf_us, verf_b);
    cout << "\n  -- VerfCNN (VGG-16 CIFAR-10, 1 thread): prover 12.6 s, proof 196 KB, FC NOT proven --" << endl;
    cout << "  peak RSS: " << getPeakRSS() / 1e6 << " MB" << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
