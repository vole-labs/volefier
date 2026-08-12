// bench_micro.cpp - per-operator microbenchmark of ONE transformer block at
// GPT-2 dims, to locate the performance bottleneck. Inlines the block forward
// (LN, projections, attention QK/softmax/AV, output proj, FFN, GeLU, residuals)
// with a timer + byte counter around each phase, plus the LogUp teardown.
//
//   bin/test_ml-quant_bench_micro <party> <port> [d heads seq ffn] [IP]
//   default: d=768 heads=12 seq=30 ffn=3072  (one block)

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
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
  if (argc < 3) { cout << "usage: " << argv[0] << " PARTY PORT [d heads seq ffn] [IP]\n"; return -1; }
  parse_party_and_port(argv, &party, &port);
  party_g = party;
  size_t d = 768, heads = 12, seq = 30, ffn = 3072;
  if (argc >= 7) { d = atoi(argv[3]); heads = atoi(argv[4]); seq = atoi(argv[5]); ffn = atoi(argv[6]); }
  const size_t dh = d / heads;
  const char *bob_ip = (argc == 8) ? argv[7] : "127.0.0.1";

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

  struct Phase { string name; double us; uint64_t bytes; };
  vector<Phase> ph;
  auto run = [&](const string &name, const std::function<void()> &fn) {
    uint64_t c0 = comm(); auto t0 = clock_start();
    fn();
    ph.push_back({name, time_from(t0), comm() - c0});
  };

  if (party == ALICE) cout << "config: d=" << d << " heads=" << heads << " seq=" << seq << " ffn=" << ffn << endl;
  setup_zk_arith(ios, threads, party);

  QuantInfer qi(party, P, ios[0], out_bits, 1, threads);   // mat_threads = threads (thread the matmuls)
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, P + (int)ceil(log2((double)(seq + 1))) + 2);
  TScale ts;                                                // per-tensor activation scales (match bench_full)
  const bool LSE = getenv("FUSED_NL") ? atoi(getenv("FUSED_NL")) != 0 : true;  // LSE softmax default; FUSED_NL=0 -> round_div
  const bool FV = getenv("FREIVALDS") && atoi(getenv("FREIVALDS")) != 0;        // Freivalds matmul path
  if (party == ALICE) cout << "matmul path: " << (FV ? "Freivalds" : "per-entry") << endl;
  QuantTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table_s(EXP_T, P, ts.score, /*clamp=*/!LSE));
  QuantTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table_s(GELU_T, P, GELU_OFF, ts.hid));
  QuantSoftmax softmax(&qi, &cmp, &expt, &divr, P, /*score_bits=*/16, /*lse=*/LSE, /*in_scale=*/ts.score);
  QuantActivation gelu(&gtab, GELU_OFF);
  QuantLayerNorm ln1(&qi, &rcmp, &divr, P, d), ln2(&qi, &rcmp, &divr, P, d);
  QuantAttention attn(&qi, &softmax, P, d, heads, ts);      // source the per-tensor QK/AV multipliers + S_a anchor

  // committed inputs + weights. In VOLE-ZK every committed witness value is sent
  // (~1 field element each), so this is the witness-commitment communication.
  Mat H, Wq, Wk, Wv, Wo, W1, W2;
  vector<IntFp> g1, b1n, g2, b2n, bq, bk, bv, bo, b1, b2;
  run("commit input + weights", [&]{
    qi.commit_mat(H, rmat(seq, d, -100, 100));
    qi.commit_vec(g1, rvec(d, (1 << P) - 50, (1 << P) + 50)); qi.commit_vec(b1n, rvec(d, -20, 20));
    qi.commit_vec(g2, rvec(d, (1 << P) - 50, (1 << P) + 50)); qi.commit_vec(b2n, rvec(d, -20, 20));
    qi.commit_mat(Wq, rmat(d, d, -2, 2)); qi.commit_mat(Wk, rmat(d, d, -2, 2));
    qi.commit_mat(Wv, rmat(d, d, -2, 2)); qi.commit_mat(Wo, rmat(d, d, -2, 2));
    qi.commit_vec(bq, rvec(d, -20, 20)); qi.commit_vec(bk, rvec(d, -20, 20));
    qi.commit_vec(bv, rvec(d, -20, 20)); qi.commit_vec(bo, rvec(d, -20, 20));
    qi.commit_mat(W1, rmat(ffn, d, -2, 2)); qi.commit_mat(W2, rmat(d, ffn, -2, 2));
    qi.commit_vec(b1, rvec(ffn, -20, 20)); qi.commit_vec(b2, rvec(d, -20, 20));
  });

  Mat normed(seq), Q(seq), K(seq), V(seq), ctx(seq, vector<IntFp>(d)), attn_out(seq), a(seq), n2(seq), hid(seq), go(seq), ff(seq), out(seq);
  vector<vector<vector<IntFp>>> scaled(seq, vector<vector<IntFp>>(heads)), probs(seq, vector<vector<IntFp>>(heads));

  run("LN1", [&]{ for (size_t i = 0; i < seq; ++i) ln1.forward(normed[i], H[i], g1, b1n); });
  run("proj Q/K/V (matmul d->d)", [&]{ qi.dense_layer_q_mm(Q, Wq, bq, normed, attn.S_a, ts.w, ts.q); qi.dense_layer_q_mm(K, Wk, bk, normed, attn.S_a, ts.w, ts.k); qi.dense_layer_q_mm(V, Wv, bv, normed, attn.S_a, ts.w, ts.v); });
  // QK^T via the batched/threaded matmul_attn (full square, like the real graph prover) instead of
  // per-score dot_ip.  Repack Q,K from [seq][d] (head-concatenated) into the per-head flat layouts.
  run("attn Q.K^T (matmul_attn)", [&]{
    vector<IntFp> Aq((size_t)heads * seq * dh), Bk((size_t)heads * dh * seq);
    for (size_t g = 0; g < heads; ++g) for (size_t m = 0; m < seq; ++m) for (size_t k = 0; k < dh; ++k)
      Aq[((g * seq) + m) * dh + k] = Q[m][g * dh + k];                   // A = Q  [heads, seq, dh]
    for (size_t g = 0; g < heads; ++g) for (size_t k = 0; k < dh; ++k) for (size_t j = 0; j < seq; ++j)
      Bk[((g * dh) + k) * seq + j] = K[j][g * dh + k];                   // B = K^T [heads, dh, seq]
    vector<IntFp> qk;
    if (FV) qi.ipp->matmul_attn_freivalds(qk, Aq.data(), Bk.data(), (int)heads, (int)seq, (int)seq, (int)dh, true, threads);
    else    qi.ipp->matmul_attn(qk, Aq.data(), Bk.data(), (int)heads, (int)seq, (int)seq, (int)dh, true, threads);
    for (size_t i = 0; i < seq; ++i) for (size_t hh = 0; hh < heads; ++hh) {
      scaled[i][hh].resize(seq);                                         // full row (graph-prover form)
      for (size_t j = 0; j < seq; ++j)
        scaled[i][hh][j] = qi.requant_cm_one(qk[(hh * seq + i) * seq + j], attn.cQK, attn.mQK);
    } });
  // softmax dispatches internally: round_div (default) or LogSumExp (FUSED_NL), same as bench_full.
  run("softmax", [&]{ for (size_t i = 0; i < seq; ++i) for (size_t hh = 0; hh < heads; ++hh) probs[i][hh] = softmax.forward(scaled[i][hh]); });
  // A.V via matmul_attn (probs[heads,seq,seq] x V[heads,seq,dh] -> ctx[heads,seq,dh]).
  run("attn A.V (matmul_attn)", [&]{
    vector<IntFp> Ap((size_t)heads * seq * seq), Bv((size_t)heads * seq * dh);
    for (size_t g = 0; g < heads; ++g) for (size_t m = 0; m < seq; ++m) for (size_t j = 0; j < seq; ++j)
      Ap[((g * seq) + m) * seq + j] = probs[m][g][j];                   // A = probs [heads, seq, seq]
    for (size_t g = 0; g < heads; ++g) for (size_t j = 0; j < seq; ++j) for (size_t k = 0; k < dh; ++k)
      Bv[((g * seq) + j) * dh + k] = V[j][g * dh + k];                  // B = V [heads, seq, dh]
    vector<IntFp> av;
    if (FV) qi.ipp->matmul_attn_freivalds(av, Ap.data(), Bv.data(), (int)heads, (int)seq, (int)dh, (int)seq, true, threads);
    else    qi.ipp->matmul_attn(av, Ap.data(), Bv.data(), (int)heads, (int)seq, (int)dh, (int)seq, true, threads);
    for (size_t i = 0; i < seq; ++i) for (size_t hh = 0; hh < heads; ++hh) for (size_t k = 0; k < dh; ++k)
      ctx[i][hh * dh + k] = qi.requant_cm_one(av[(hh * seq + i) * dh + k], attn.cAV, attn.mAV);
  });
  run("proj O (matmul d->d)", [&]{ qi.dense_layer_q_mm(attn_out, Wo, bo, ctx, ts.ctx, ts.w, ts.res); });
  run("residual 1", [&]{ for (size_t i = 0; i < seq; ++i) qi.add(a[i], H[i], attn_out[i]); });
  run("LN2", [&]{ for (size_t i = 0; i < seq; ++i) ln2.forward(n2[i], a[i], g2, b2n); });
  run("FFN W1 (matmul d->ffn)", [&]{ qi.dense_layer_q_mm(hid, W1, b1, n2, attn.S_a, ts.w, ts.hid); });
  run("GeLU", [&]{ for (size_t i = 0; i < seq; ++i) gelu.forward(go[i], hid[i]); });
  run("FFN W2 (matmul ffn->d)", [&]{ qi.dense_layer_q_mm(ff, W2, b2, go, attn.S_a, ts.w, ts.res); });
  run("residual 2", [&]{ for (size_t i = 0; i < seq; ++i) qi.add(out[i], a[i], ff[i]); });
  size_t acc_rc_rem = qi.rc_rem->witnesses.size();   // requant remainder accesses
  run("teardown: cmp range", [&]{ rcmp.Teardown(); });
  run("teardown: div range (chunk)", [&]{ divr.destroy(); });
  run("teardown: exp table (2^16)", [&]{ expt.Teardown(); });
  run("teardown: gelu table (2^16)", [&]{ gtab.Teardown(); });
  run("teardown: qi (rc_rem/div/msb)", [&]{ qi.destroy(); });
  run("finalize (batched mult check)", [&]{ finalize_zk_arith(); });

  if (party == ALICE) {
    double tot = 0; uint64_t tb = 0;
    for (auto &p : ph) { tot += p.us; tb += p.bytes; }
    sort(ph.begin(), ph.end(), [](const Phase &x, const Phase &y){ return x.us > y.us; });
    cout << "\n  phase                         time(s)   %time     MB    %bytes" << endl;
    cout << "  ------------------------------------------------------------------" << endl;
    for (auto &p : ph) {
      char buf[256];
      snprintf(buf, sizeof(buf), "  %-28s %7.3f  %5.1f%%  %7.1f  %5.1f%%",
               p.name.c_str(), p.us / 1e6, 100.0 * p.us / tot, p.bytes / 1e6, 100.0 * p.bytes / tb);
      cout << buf << endl;
    }
    char buf[128]; snprintf(buf, sizeof(buf), "  %-28s %7.3f          %7.1f", "TOTAL (1 block)", tot / 1e6, tb / 1e6);
    cout << "  ------------------------------------------------------------------\n" << buf << endl;
    cout << "\n  rc_rem accesses (rem+clamp limbs): " << acc_rc_rem << endl;
  }

  for (int i = 0; i < threads; ++i) { auto *_io = ios[i]->io; delete ios[i]; delete _io; }
  return 0;
}
