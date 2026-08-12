// End-to-end VGG-{11,13,16,19} proof (optionally with BatchNorm). Runs ONE
// variant (depth from argv) at reduced width/resolution with random committed
// weights; ALICE compares revealed logits to a parallel plaintext forward and
// the proof must verify.
//
//   test_ml_quant_vgg 1 12345 16 [bn]   # prover, vgg-16
//   test_ml_quant_vgg 2 12345 16 [bn]   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "test/utility.h"
#include "volefier/ml-fixed/fixed_vgg.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  int depth = (argc >= 4) ? atoi(argv[3]) : 16;
  const int base    = (argc >= 5) ? atoi(argv[4]) : 4;
  const int in_hw   = (argc >= 6) ? atoi(argv[5]) : 16;
  const int classes = (argc >= 7) ? atoi(argv[6]) : 4;
  bool bn = (argc >= 8 && string(argv[7]) == "bn");
  const int fc_dim = base * 8;

  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);

  PRG prg;
  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);
  FixedInfer qi(party, P, ios[0], /*out_bits=*/20, /*mat_threads=*/threads);
  RangeCheckLU rclu(party, ios[0]->io, 16);
  CompareLU cmp(&rclu);
  Div divr(party, ios[0]->io, /*range_bits=*/24);

  VGGCfg cfg = vgg_cfg(depth, base, fc_dim, classes, in_hw, bn);

  auto rndv = [&](int64_t lo, int64_t hi) -> int64_t { if (party != ALICE) return 0; uint64_t r; prg.random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); };
  St input; input.C = 3; input.H = in_hw; input.W = in_hw;
  input.X.assign(3, vector<vector<IntFp>>(in_hw, vector<IntFp>(in_hw)));
  if (party == ALICE) input.Xr.assign(3, vector<vector<int64_t>>(in_hw, vector<int64_t>(in_hw)));
  for (int c = 0; c < 3; ++c) for (int h = 0; h < in_hw; ++h) for (int w = 0; w < in_hw; ++w) {
    int64_t v = rndv(-127, 127);
    input.X[c][h][w] = qi.commit(v);
    if (party == ALICE) input.Xr[c][h][w] = v;
  }

  ResNet net(&qi, &cmp, &divr, &prg, party, /*act_bits=*/24);
  auto t0 = clock_start();
  VecSt logits = vgg_forward(net, cfg, input);
  double fwd_us = time_from(t0);

  size_t fails = 0;
  vector<int64_t> got(classes);
  for (int j = 0; j < classes; ++j) {
    got[j] = FixedInfer::decode_signed(logits.X[j].reveal());
    if (party == ALICE && got[j] != logits.Xr[j]) { if (fails < 5) cout << "logit[" << j << "] " << got[j] << " vs " << logits.Xr[j] << endl; fails++; }
  }

  rclu.Teardown();
  divr.destroy();
  qi.destroy();
  finalize_zk_arith();
  _bench.report("vgg (ml-fixed)");
  if (party == ALICE) {
    if (fails) { cout << cfg.name << ": " << fails << " logit mismatches" << endl; error("vgg failed\n"); }
    int64_t am = 0; for (int j = 1; j < classes; ++j) if (got[j] > got[am]) am = j;
    cout << "pass " << cfg.name << "  (forward " << fwd_us / 1e6 << " s, logits[";
    for (int j = 0; j < classes; ++j) cout << got[j] << (j + 1 < classes ? "," : "");
    cout << "], argmax=" << am << ", peak RSS " << getPeakRSS() / 1e6 << " MB)" << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
