// bench_transformer.cpp - benchmark VOLE-ZK proving time for a STACK of
// quantized transformer blocks (the GPT-2 "body"), and compare against prior
// reported numbers for the same circuit shape.
//
// Matches a standard benchmarked circuit: L transformer blocks (no token
// embedding / LM head, like their main_demo_llm), each block =
//   LN1 -> MHA(causal) -> residual -> LN2 -> FFN(GeLU) -> residual.
// Weights are small random integers (a timing harness, not a correctness test);
// magnitudes are kept in range so the activation/range lookups never abort.
//
//   Usage:
//     bin/test_ml-quant_bench_transformer <party> <port> [L d heads seq ffn] [IP]
//   Defaults to a standard config: L=12 d=768 heads=12 seq=30 ffn=3072.
//   For an in-session run pass a smaller config, e.g.  ... 1 12345 4 128 4 8 512
//
//   Output ends with one parseable line:
//     [RESULT] L=.. d=.. heads=.. seq=.. ffn=.. setup_us=.. blocks_us=.. teardown_us=.. total_us=.. bytes=..

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "test/utility.h"
#include "volefier/ml-quant/quant_transformer.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;

static PRG *gprg;
static int party_g;
static int64_t rnd(int64_t lo, int64_t hi) {
  if (party_g != ALICE) return 0;
  uint64_t r; gprg->random_data(&r, sizeof(r));
  return lo + (int64_t)(r % (uint64_t)(hi - lo + 1));
}

int main(int argc, char **argv) {
  if (argc < 3) { cout << "usage: " << argv[0] << " PARTY PORT [L d heads seq ffn] [IP]\n"; return -1; }
  parse_party_and_port(argv, &party, &port);
  party_g = party;

  // config: a standard 12-layer GPT-2 body by default
  size_t L = 12, d = 768, heads = 12, seq = 30, ffn = 3072;
  if (argc >= 8) { L = atoi(argv[3]); d = atoi(argv[4]); heads = atoi(argv[5]); seq = atoi(argv[6]); ffn = atoi(argv[7]); }
  const size_t dh = d / heads;
  const int mat_threads = threads;   // IP matmul uses the same thread count as setup
  // ALICE listens (server), BOB connects to ALICE's IP (default localhost).
  const char *bob_ip = (argc == 9) ? argv[8] : "127.0.0.1";

  // derived table / range sizes
  const uint64_t QINV = (uint64_t)llround((double)(1 << P) / sqrt((double)dh));
  // LayerNorm normalizes to ~z-score scale, max |.| ~ sqrt(d-1) * 2^P, so the
  // requant output clamp width must cover that.
  size_t zbits = (size_t)ceil(log2(ceil(sqrt((double)d))));
  size_t out_bits = P + zbits + 2;
  const uint64_t EXP_T = 1u << 16, GELU_T = 1u << 16;
  const int64_t  GELU_OFF = 1 << 15;

  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : bob_ip, port + i), party == ALICE);

  PRG prg; gprg = &prg;

  auto comm = [&]() { uint64_t c = 0; for (int i = 0; i < threads; ++i) c += io_counter(ios[i]->io); return c; };
  auto rvec = [&](size_t n, int64_t lo, int64_t hi) { vector<int64_t> v(n); for (auto &e : v) e = rnd(lo, hi); return v; };
  auto rmat = [&](size_t r, size_t c, int64_t lo, int64_t hi) { vector<vector<int64_t>> m(r); for (auto &row : m) row = rvec(c, lo, hi); return m; };

  if (party == ALICE)
    cout << "config: L=" << L << " d=" << d << " heads=" << heads << " seq=" << seq
         << " ffn=" << ffn << " (out_bits=" << out_bits << ")" << endl;

  setup_zk_arith(ios, threads, party);
  uint64_t c0 = comm();
  auto t_setup0 = clock_start();

  QuantInfer qi(party, P, ios[0], out_bits, /*wc=*/1, mat_threads);
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, /*range_bits=*/ P + (int)ceil(log2((double)(seq + 1))) + 2);
  TScale ts;
  QuantTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table_s(EXP_T, P, ts.score));
  QuantTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table_s(GELU_T, P, GELU_OFF, ts.hid));
  QuantSoftmax softmax(&qi, &cmp, &expt, &divr, P, /*score_bits=*/16);
  QuantActivation gelu(&gtab, GELU_OFF);
  QuantLayerNorm ln1(&qi, &rcmp, &divr, P, d), ln2(&qi, &rcmp, &divr, P, d);
  QuantAttention attn(&qi, &softmax, P, d, heads, ts);
  QuantTransformerBlock block(&qi, &ln1, &ln2, &attn, &gelu, d, ffn, ts);
  double setup_us = time_from(t_setup0);

  // residual stream (small, keeps activations / LayerNorm variance in range)
  Mat H; qi.commit_mat(H, rmat(seq, d, -100, 100));

  // Weight commitment (all L blocks) is ONE-TIME preprocessing -- committed once
  // and reused across every inference input. Only the input-dependent "advice"
  // commitment counts per inference, so we measure it
  // separately and exclude it from the per-inference proving figure too.
  double commit_us = 0, fwd_us = 0;
  uint64_t commit_b = 0, fwd_b = 0;
  for (size_t l = 0; l < L; ++l) {
    vector<IntFp> cg1, cb1n, cg2, cb2n, cbq, cbk, cbv, cbo, cb1, cb2;
    Mat cWq, cWk, cWv, cWo, cW1, cW2;
    { uint64_t b0 = comm(); auto tc = clock_start();   // --- preprocessing: commit weights ---
      qi.commit_vec(cg1, rvec(d, (1 << P) - 50, (1 << P) + 50));
      qi.commit_vec(cb1n, rvec(d, -20, 20));
      qi.commit_vec(cg2, rvec(d, (1 << P) - 50, (1 << P) + 50));
      qi.commit_vec(cb2n, rvec(d, -20, 20));
      qi.commit_mat(cWq, rmat(d, d, -2, 2)); qi.commit_mat(cWk, rmat(d, d, -2, 2));
      qi.commit_mat(cWv, rmat(d, d, -2, 2)); qi.commit_mat(cWo, rmat(d, d, -2, 2));
      qi.commit_vec(cbq, rvec(d, -20, 20)); qi.commit_vec(cbk, rvec(d, -20, 20));
      qi.commit_vec(cbv, rvec(d, -20, 20)); qi.commit_vec(cbo, rvec(d, -20, 20));
      qi.commit_mat(cW1, rmat(ffn, d, -2, 2)); qi.commit_mat(cW2, rmat(d, ffn, -2, 2));
      qi.commit_vec(cb1, rvec(ffn, -20, 20)); qi.commit_vec(cb2, rvec(d, -20, 20));
      commit_us += time_from(tc); commit_b += comm() - b0;
    }
    { uint64_t b0 = comm(); auto tf = clock_start();   // --- per-inference proving ---
      Mat out;
      block.forward(out, H, cg1, cb1n, cg2, cb2n, cWq, cbq, cWk, cbk, cWv, cbv, cWo, cbo, cW1, cb1, cW2, cb2);
      H = out;  // thread the residual stream into the next block
      fwd_us += time_from(tf); fwd_b += comm() - b0;
    }
    if (party == ALICE) cout << "  block " << (l + 1) << "/" << L << " done" << endl;
  }
  double blocks_us = fwd_us;

  auto t_td0 = clock_start();
  rcmp.Teardown();
  divr.destroy();
  expt.Teardown();
  gtab.Teardown();
  qi.destroy();
  double teardown_us = time_from(t_td0);

  finalize_zk_arith();
  uint64_t bytes = comm() - c0;                          // everything after setup
  // per-inference proving = total - one-time weight commitment
  double prove_us = setup_us + fwd_us + teardown_us;
  uint64_t prove_b = bytes - commit_b;

  if (party == ALICE) {
    cout << "\n[RESULT] L=" << L << " d=" << d << " heads=" << heads << " seq=" << seq
         << " ffn=" << ffn
         << " weightcommit_us=" << (uint64_t)commit_us << " weightcommit_bytes=" << commit_b
         << " prove_us=" << (uint64_t)prove_us << " prove_bytes=" << prove_b << endl;
    cout << "  weight commit (one-time preprocessing) : " << commit_us / 1e6 << " s, " << commit_b / 1e6 << " MB" << endl;
    cout << "  per-inference PROVING (excl. weights)  : " << prove_us / 1e6 << " s, " << prove_b / 1e6 << " MB" << endl;
    cout << "    of which per block (avg)             : " << blocks_us / 1e6 / L << " s" << endl;
    cout << "  full total (incl. weights)             : " << (prove_us + commit_us) / 1e6 << " s, " << bytes / 1e6 << " MB" << endl;
    cout << "\n  (this prover: VOLE-ZK, interactive/designated-verifier, " << threads << " threads)" << endl;
    cout << "  peak RSS: " << getPeakRSS() / 1e6 << " MB" << endl;
  }

  for (int i = 0; i < threads; ++i) { auto *_io = ios[i]->io; delete ios[i]; delete _io; }
  return 0;
}
