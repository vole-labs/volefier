// bench_full.cpp - end-to-end benchmark: the FULL GPT-2
// forward graph -- token embedding -> L transformer blocks -> LM head -> argmax
// certification. (Contrast bench_transformer, which is the body only and matches
// the body-only benchmarked circuit.)
//
//   bin/test_ml-quant_bench_full <party> <port> [L d heads seq ffn vocab] [bob_ip]
//   default: L=12 d=768 heads=12 seq=30 ffn=3072 vocab=8192
//   (real GPT-2 vocab is 50257; reduce for an in-session run, extrapolate.)
//
// Weight/table commitment (block weights + embedding table + LM-head We) is
// one-time preprocessing and reported separately from per-inference proving.

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
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
const int P = 12;

static PRG *gprg;
static int party_g;
static int64_t rnd(int64_t lo, int64_t hi) {
  if (party_g != ALICE) return 0;
  uint64_t r; gprg->random_data(&r, sizeof(r));
  return lo + (int64_t)(r % (uint64_t)(hi - lo + 1));
}

int main(int argc, char **argv) {
  if (argc < 3) { cout << "usage: " << argv[0] << " PARTY PORT [L d heads seq ffn vocab] [bob_ip]\n"; return -1; }
  parse_party_and_port(argv, &party, &port);
  party_g = party;
  size_t L = 12, d = 768, heads = 12, seq = 30, ffn = 3072, vocab = 8192;
  if (argc >= 9) { L = atoi(argv[3]); d = atoi(argv[4]); heads = atoi(argv[5]); seq = atoi(argv[6]); ffn = atoi(argv[7]); vocab = atoi(argv[8]); }
  const int mat_threads = threads;
  const char *bob_ip = (argc >= 10) ? argv[9] : "127.0.0.1";

  size_t out_bits = P + (size_t)ceil(log2(ceil(sqrt((double)d)))) + 2;
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
    cout << "config: L=" << L << " d=" << d << " heads=" << heads << " seq=" << seq << " ffn=" << ffn << " vocab=" << vocab << endl;

  setup_zk_arith(ios, threads, party);
  uint64_t c0 = comm();
  double commit_us = 0, fwd_us = 0; uint64_t commit_b = 0, fwd_b = 0;
  auto P0 = [&]{ return clock_start(); };
  auto Pre = [&](decltype(clock_start()) t, uint64_t b0){ commit_us += time_from(t); commit_b += comm() - b0; };
  auto Inf = [&](decltype(clock_start()) t, uint64_t b0){ fwd_us += time_from(t); fwd_b += comm() - b0; };

  QuantInfer qi(party, P, ios[0], out_bits, /*wc=*/1, mat_threads);
  RangeCheckLU rclu(party, ios[0]->io, /*table_bits=*/16);   // shared: softmax max + LM-head argmax
  CompareLU cmplu(&rclu);
  Div divr(party, ios[0]->io, P + (int)ceil(log2((double)(seq + 1))) + 2);
  TScale ts;
  const bool LSE = getenv("FUSED_NL") ? atoi(getenv("FUSED_NL")) != 0 : true;  // LSE softmax default; FUSED_NL=0 -> round_div
  QuantTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table_s(EXP_T, P, ts.score, /*clamp=*/!LSE));
  QuantTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table_s(GELU_T, P, GELU_OFF, ts.hid));
  QuantSoftmax softmax(&qi, &cmplu, &expt, &divr, P, /*score_bits=*/16, /*lse=*/LSE, /*in_scale=*/ts.score);
  QuantActivation gelu(&gtab, GELU_OFF);
  QuantLayerNorm ln1(&qi, &rclu, &divr, P, d), ln2(&qi, &rclu, &divr, P, d);
  QuantAttention attn(&qi, &softmax, P, d, heads, ts);
  QuantTransformerBlock block(&qi, &ln1, &ln2, &attn, &gelu, d, ffn, ts);
  QuantEmbedding embed(vocab, d, party, ios[0]->io);
  QuantLMHead lm(&qi, &cmplu, d, vocab, /*logit_bits=*/20);

  // ---- preprocessing: commit embedding table (vocab x d) ----
  { auto t = P0(); uint64_t b0 = comm();
    vector<uint64_t> emb_raw(vocab * d);
    if (party == ALICE) for (size_t i = 0; i < vocab * d; ++i) { int64_t v = rnd(-2000, 2000); emb_raw[i] = (v >= 0) ? (uint64_t)v : (uint64_t)(v + (int64_t)PR); }
    embed.Setup(emb_raw);
    Pre(t, b0);
  }

  // ---- per-inference: embedding lookup (token ids + positional) ----
  Mat H;
  { auto t = P0(); uint64_t b0 = comm();
    Mat pos; qi.commit_mat(pos, rmat(seq, d, -300, 300));
    vector<IntFp> ids(seq); for (size_t i = 0; i < seq; ++i) ids[i] = qi.commit(rnd(0, vocab - 1));
    embed.embed(H, ids, pos);
    Inf(t, b0);
  }
  if (party == ALICE) cout << "  embedding done" << endl;

  // ---- L transformer blocks (weights = preprocessing, forward = inference) ----
  for (size_t l = 0; l < L; ++l) {
    vector<IntFp> cg1, cb1n, cg2, cb2n, cbq, cbk, cbv, cbo, cb1, cb2;
    Mat cWq, cWk, cWv, cWo, cW1, cW2;
    { auto t = P0(); uint64_t b0 = comm();
      qi.commit_vec(cg1, rvec(d, (1 << P) - 50, (1 << P) + 50)); qi.commit_vec(cb1n, rvec(d, -20, 20));
      qi.commit_vec(cg2, rvec(d, (1 << P) - 50, (1 << P) + 50)); qi.commit_vec(cb2n, rvec(d, -20, 20));
      qi.commit_mat(cWq, rmat(d, d, -2, 2)); qi.commit_mat(cWk, rmat(d, d, -2, 2));
      qi.commit_mat(cWv, rmat(d, d, -2, 2)); qi.commit_mat(cWo, rmat(d, d, -2, 2));
      qi.commit_vec(cbq, rvec(d, -20, 20)); qi.commit_vec(cbk, rvec(d, -20, 20));
      qi.commit_vec(cbv, rvec(d, -20, 20)); qi.commit_vec(cbo, rvec(d, -20, 20));
      qi.commit_mat(cW1, rmat(ffn, d, -2, 2)); qi.commit_mat(cW2, rmat(d, ffn, -2, 2));
      qi.commit_vec(cb1, rvec(ffn, -20, 20)); qi.commit_vec(cb2, rvec(d, -20, 20));
      Pre(t, b0); }
    { auto t = P0(); uint64_t b0 = comm();
      Mat out; block.forward(out, H, cg1, cb1n, cg2, cb2n, cWq, cbq, cWk, cbk, cWv, cbv, cWo, cbo, cW1, cb1, cW2, cb2);
      H = out; Inf(t, b0); }
    if (party == ALICE) cout << "  block " << (l + 1) << "/" << L << " done" << endl;
  }

  // ---- LM head: commit We (preprocessing) ; forward + argmax certify (inference) ----
  // Time the hidden->vocab MATMUL and the ARGMAX certify SEPARATELY.
  double head_us = 0, argmax_us = 0; uint64_t head_b = 0, argmax_b = 0;
  Mat cWe; vector<IntFp> cbe;
  { auto t = P0(); uint64_t b0 = comm();
    qi.commit_mat(cWe, rmat(vocab, d, -2, 2)); qi.commit_vec(cbe, rvec(vocab, -50, 50));
    Pre(t, b0); }
  Mat logits;
  { auto t = P0(); uint64_t b0 = comm();             // LM head matmul (threaded IP)
    lm.forward(logits, H, cWe, cbe);
    head_us = time_from(t); head_b = comm() - b0; fwd_us += head_us; fwd_b += head_b; }
  { auto t = P0(); uint64_t b0 = comm();             // argmax certification
    vector<IntFp> tok(seq);                          // claimed tokens = true argmax (ALICE)
    if (party == ALICE)
      for (size_t i = 0; i < seq; ++i) {
        uint64_t best = 0; int64_t bv = -(1LL << 60);
        for (size_t v = 0; v < vocab; ++v) { int64_t lv = QuantInfer::decode_signed((uint64_t)((uint64_t)(logits[i][v].value))); if (lv > bv) { bv = lv; best = v; } }
        tok[i] = qi.commit((int64_t)best);
      }
    else for (size_t i = 0; i < seq; ++i) tok[i] = qi.commit(0);
    lm.certify(logits, tok);
    argmax_us = time_from(t); argmax_b = comm() - b0; fwd_us += argmax_us; fwd_b += argmax_b; }
  if (party == ALICE) cout << "  LM head matmul: " << head_us / 1e6 << " s / " << head_b / 1e6
                           << " MB ;  argmax: " << argmax_us / 1e6 << " s / " << argmax_b / 1e6 << " MB" << endl;

  auto t_td = clock_start();
  rclu.Teardown(); divr.destroy(); expt.Teardown(); gtab.Teardown(); embed.Teardown(); qi.destroy();
  double teardown_us = time_from(t_td);
  finalize_zk_arith();
  uint64_t bytes = comm() - c0;
  double prove_us = fwd_us + teardown_us;
  uint64_t prove_b = bytes - commit_b;

  if (party == ALICE) {
    cout << "\n[RESULT-FULL] L=" << L << " d=" << d << " seq=" << seq << " vocab=" << vocab
         << " commit_us=" << (uint64_t)commit_us << " commit_bytes=" << commit_b
         << " prove_us=" << (uint64_t)prove_us << " prove_bytes=" << prove_b << endl;
    cout << "  weight+table commit (preprocessing) : " << commit_us / 1e6 << " s, " << commit_b / 1e6 << " MB" << endl;
    cout << "  per-inference PROVING (excl. above) : " << prove_us / 1e6 << " s, " << prove_b / 1e6 << " MB" << endl;
    cout << "  full total                          : " << (prove_us + commit_us) / 1e6 << " s, " << bytes / 1e6 << " MB" << endl;
    cout << "\n  peak RSS: " << getPeakRSS() / 1e6 << " MB" << endl;
  }
  for (int i = 0; i < threads; ++i) { auto *_io = ios[i]->io; delete ios[i]; delete _io; }
  return 0;
}
