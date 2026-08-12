// bench_micro.cpp (ml-fixed) - per-operator microbenchmark of ONE transformer block
// at GPT-2 dims, on the UNIFORM fixed-point backend (FixedInfer + Fixed{Softmax,
// LayerNorm,Activation}).  Inlines the block forward (LN, projections, attention
// QK/softmax/AV, output proj, FFN, GeLU, residuals) with a timer + byte counter
// around each phase, plus the LogUp teardown.  This is the ml-fixed twin of
// test/ml-quant/bench_micro.cpp.
//
//   bin/test_ml-fixed_bench_micro <party> <port> [d heads seq ffn] [IP]
//   default: d=768 heads=12 seq=30 ffn=3072  (one block);  ZK_THREADS env -> mat_threads
//
// NOTE: like the ml-quant micro, the attention here is the hand-built CAUSAL TRIANGLE
// (j <= i, no mask) -- it does NOT model the full-square + causal-mask cost of the real
// graph-driven gpt2_prove path.

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-fixed/fixed_transformer.h"

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

  const uint64_t QINV = (uint64_t)llround((double)(1 << P) / sqrt((double)dh));
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

  if (party == ALICE) cout << "config: [FIXED-POINT] d=" << d << " heads=" << heads << " seq=" << seq << " ffn=" << ffn << endl;
  setup_zk_arith(ios, threads, party);

  FixedInfer qi(party, P, ios[0], out_bits, threads);   // mat_threads = threads
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, P + (int)ceil(log2((double)(seq + 1))) + 2);
  FixedTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table(EXP_T, P));
  FixedTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table(GELU_T, P, GELU_OFF));
  FixedSoftmax softmax(&qi, &cmp, &expt, &divr, P, /*score_bits=*/16);
  FixedActivation gelu(&gtab, GELU_OFF);
  FixedLayerNorm ln1(&qi, &rcmp, &divr, P, d), ln2(&qi, &rcmp, &divr, P, d);

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
  run("proj Q/K/V (matmul d->d)", [&]{ for (size_t i = 0; i < seq; ++i) { qi.dense_layer(Q[i], Wq, bq, normed[i]); qi.dense_layer(K[i], Wk, bk, normed[i]); qi.dense_layer(V[i], Wv, bv, normed[i]); } });
  run("attn Q.K^T (dot_ip dh)", [&]{
    for (size_t i = 0; i < seq; ++i) for (size_t hh = 0; hh < heads; ++hh) {
      size_t off = hh * dh;
      vector<IntFp> qh(Q[i].begin() + off, Q[i].begin() + off + dh), raw(i + 1);
      for (size_t j = 0; j <= i; ++j) { vector<IntFp> kh(K[j].begin() + off, K[j].begin() + off + dh); raw[j] = qi.requant_one(qi.dot_ip(qh, kh)); }
      qi.requant_fp(scaled[i][hh], raw, QINV);
    } });
  run("softmax", [&]{ for (size_t i = 0; i < seq; ++i) for (size_t hh = 0; hh < heads; ++hh) probs[i][hh] = softmax.forward(scaled[i][hh]); });
  run("attn A.V (dot_ip)", [&]{
    for (size_t i = 0; i < seq; ++i) for (size_t hh = 0; hh < heads; ++hh) {
      size_t off = hh * dh;
      for (size_t k = 0; k < dh; ++k) { vector<IntFp> vk(i + 1); for (size_t j = 0; j <= i; ++j) vk[j] = V[j][off + k]; ctx[i][off + k] = qi.requant_one(qi.dot_ip(probs[i][hh], vk)); }
    } });
  run("proj O (matmul d->d)", [&]{ for (size_t i = 0; i < seq; ++i) qi.dense_layer(attn_out[i], Wo, bo, ctx[i]); });
  run("residual 1", [&]{ for (size_t i = 0; i < seq; ++i) qi.add(a[i], H[i], attn_out[i]); });
  run("LN2", [&]{ for (size_t i = 0; i < seq; ++i) ln2.forward(n2[i], a[i], g2, b2n); });
  run("FFN W1 (matmul d->ffn)", [&]{ for (size_t i = 0; i < seq; ++i) qi.dense_layer(hid[i], W1, b1, n2[i]); });
  run("GeLU", [&]{ for (size_t i = 0; i < seq; ++i) gelu.forward(go[i], hid[i]); });
  run("FFN W2 (matmul ffn->d)", [&]{ for (size_t i = 0; i < seq; ++i) qi.dense_layer(ff[i], W2, b2, go[i]); });
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
