// End-to-end ZK proof of transformer building blocks: token embedding,
// multi-head causal self-attention, and a full GPT-2 style block. ALICE checks
// every output against an exact integer reference assembled from the same ops.
//
//   test_ml_transformer_block 1 12345   # prover
//   test_ml_transformer_block 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-fixed/fixed_transformer.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;

const int P = 12;
const size_t S = 3, D = 4, H = 2, HD = D / H, FFN = 8, VOCAB = 10;
const uint64_t EXP_T = 1u << 16, GELU_T = 1u << 14;
const int64_t GELU_OFF = 1 << 13;
const uint64_t QINV = 2896;   // round(2^12 / sqrt(2))
const int64_t  WC = 1;        // FIXED-POINT: weights at uniform 2^-P (wc == 1)

typedef vector<int64_t> Vec;
typedef vector<vector<int64_t>> RMat;

static vector<uint64_t> EXP_REF, GELU_REF;

// ---- reference ops (exact integer semantics of the gadgets) ----
static int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b, r = a % b; if (r != 0 && ((r < 0) != (b < 0))) q--; return q; }
static int64_t rdiv(int64_t N, int64_t D) { return (N >= 0) ? (N + D / 2) / D : -(((-N) + D / 2) / D); }
static int64_t ars(int64_t x, int p) { return x >> p; }
static int64_t rcm(int64_t acc, int64_t c, int M) {   // round(acc*c / 2^M), half up
  return (acc * c + ((int64_t)1 << (M - 1))) >> M;
}
static uint64_t enc(int64_t v) { return v >= 0 ? (uint64_t)v : (uint64_t)(v + (int64_t)PR); }

static Vec r_linear(const RMat &W, const Vec &b, const Vec &x, bool relu) {
  Vec y(W.size());
  for (size_t m = 0; m < W.size(); ++m) {
    int64_t a = 0; for (size_t k = 0; k < x.size(); ++k) a += W[m][k] * x[k];
    int64_t v = rcm(a, WC, P) + b[m];
    y[m] = (relu && v < 0) ? 0 : v;
  }
  return y;
}
static Vec r_softmax(const Vec &s) {
  int64_t m = s[0]; for (auto v : s) m = max(m, v);
  int64_t sum = 0; Vec e(s.size());
  for (size_t i = 0; i < s.size(); ++i) { e[i] = (int64_t)EXP_REF[m - s[i]]; sum += e[i]; }
  Vec o(s.size()); for (size_t i = 0; i < s.size(); ++i) o[i] = rdiv(e[i] * (1ll << P), sum);
  return o;
}
static Vec r_layernorm(const Vec &x, const Vec &g, const Vec &b) {
  size_t d = x.size(); int64_t sx = 0; for (auto v : x) sx += v;
  int64_t mu = rdiv(sx, d), ssq = 0; Vec c(d);
  for (size_t i = 0; i < d; ++i) { c[i] = x[i] - mu; ssq += c[i] * c[i]; }
  int64_t vq = rdiv(ssq, d);
  int64_t dinv = llround((double)(1ull << (2 * P)) / sqrt((double)vq));
  Vec o(d);
  for (size_t i = 0; i < d; ++i) { int64_t y = rcm(c[i] * dinv, 1, P); o[i] = rcm(y * g[i], 1, P) + b[i]; }
  return o;
}
static RMat r_attention(const RMat &Hs,
                        const RMat &Wq, const Vec &bq, const RMat &Wk, const Vec &bk,
                        const RMat &Wv, const Vec &bv, const RMat &Wo, const Vec &bo) {
  size_t s = Hs.size();
  RMat Q(s), K(s), V(s), ctx(s, Vec(D));
  for (size_t i = 0; i < s; ++i) { Q[i] = r_linear(Wq, bq, Hs[i], false); K[i] = r_linear(Wk, bk, Hs[i], false); V[i] = r_linear(Wv, bv, Hs[i], false); }
  for (size_t i = 0; i < s; ++i)
    for (size_t hh = 0; hh < H; ++hh) {
      size_t off = hh * HD; Vec scaled(i + 1);
      for (size_t j = 0; j <= i; ++j) {
        int64_t acc = 0; for (size_t k = 0; k < HD; ++k) acc += Q[i][off + k] * K[j][off + k];
        scaled[j] = rcm(acc, (int64_t)QINV, 2 * P);
      }
      Vec probs = r_softmax(scaled);
      for (size_t k = 0; k < HD; ++k) {           // A.V deferred: no truncate (scale 2^-2P)
        int64_t acc = 0; for (size_t j = 0; j <= i; ++j) acc += probs[j] * V[j][off + k];
        ctx[i][off + k] = acc;
      }
    }
  RMat out(s);                                    // output proj truncates 2P, then +bias
  for (size_t i = 0; i < s; ++i) {
    out[i] = Vec(D);
    for (size_t j = 0; j < D; ++j) {
      int64_t acc = 0; for (size_t k = 0; k < D; ++k) acc += Wo[j][k] * ctx[i][k];
      out[i][j] = rcm(acc, WC, 2 * P) + bo[j];
    }
  }
  return out;
}

// ---- random committed helpers (meaningful on ALICE) ----
static PRG *gprg;
static int64_t rnd(int64_t lo, int64_t hi) { if (party != ALICE) return 0; uint64_t r; gprg->random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); }
static Vec rvec(size_t n, int64_t lo, int64_t hi) { Vec v(n); for (auto &e : v) e = rnd(lo, hi); return v; }
static RMat rmat(size_t r, size_t c, int64_t lo, int64_t hi) { RMat m(r); for (auto &row : m) row = rvec(c, lo, hi); return m; }

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ ZK proof of transformer block ------------" << endl << endl;

  PRG prg; gprg = &prg;
  EXP_REF = build_exp_table(EXP_T, P);
  GELU_REF = build_gelu_table(GELU_T, P, GELU_OFF);

  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);
  FixedInfer qi(party, P, ios[0], 18);
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, /*range_bits=*/16);
  FixedTable expt(EXP_T, party, ios[0]->io); expt.Setup(EXP_REF);
  FixedTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(GELU_REF);
  FixedSoftmax softmax(&qi, &cmp, &expt, &divr, P, /*score_bits=*/16);
  FixedActivation gelu(&gtab, GELU_OFF);
  FixedLayerNorm ln1(&qi, &rcmp, &divr, P, D), ln2(&qi, &rcmp, &divr, P, D);
  FixedAttention attn(&qi, &softmax, P, D, H, QINV);

  auto commit_mat = [&](const RMat &m) { Mat c; qi.commit_mat(c, m); return c; };

  // ===== 1. Embedding =====
  {
    RMat emb = rmat(VOCAB, D, -2000, 2000);     // token-embedding table
    RMat pos = rmat(S, D, -500, 500);
    Vec ids(S); for (size_t i = 0; i < S; ++i) ids[i] = rnd(0, VOCAB - 1);

    vector<uint64_t> emb_raw(VOCAB * D);
    for (size_t r = 0; r < VOCAB; ++r) for (size_t c = 0; c < D; ++c) emb_raw[r * D + c] = enc(emb[r][c]);

    FixedEmbedding embed(VOCAB, D, party, ios[0]->io);
    embed.Setup(party == ALICE ? emb_raw : vector<uint64_t>(VOCAB * D, 0));

    Mat cpos = commit_mat(pos);
    vector<IntFp> cids(S); for (size_t i = 0; i < S; ++i) cids[i] = qi.commit(ids[i]);

    Mat Hc; embed.embed(Hc, cids, cpos);
    embed.Teardown();

    for (size_t i = 0; i < S; ++i)
      for (size_t k = 0; k < D; ++k) {
        int64_t got = FixedInfer::decode_signed(Hc[i][k].reveal());
        if (party == ALICE && got != emb[ids[i]][k] + pos[i][k]) error("embedding mismatch\n");
      }
    if (party == ALICE) cout << "pass Embedding" << endl;
  }

  // ===== 2. Attention =====
  {
    RMat Hs = rmat(S, D, -2500, 2500);
    RMat Wq = rmat(D, D, -200, 200), Wk = rmat(D, D, -200, 200), Wv = rmat(D, D, -200, 200), Wo = rmat(D, D, -200, 200);
    Vec bq = rvec(D, -100, 100), bk = rvec(D, -100, 100), bv = rvec(D, -100, 100), bo = rvec(D, -100, 100);

    Mat cH = commit_mat(Hs), cWq = commit_mat(Wq), cWk = commit_mat(Wk), cWv = commit_mat(Wv), cWo = commit_mat(Wo);
    vector<IntFp> cbq, cbk, cbv, cbo;
    qi.commit_vec(cbq, bq); qi.commit_vec(cbk, bk); qi.commit_vec(cbv, bv); qi.commit_vec(cbo, bo);

    Mat ao; attn.forward(ao, cH, cWq, cbq, cWk, cbk, cWv, cbv, cWo, cbo);
    RMat ref = r_attention(Hs, Wq, bq, Wk, bk, Wv, bv, Wo, bo);
    for (size_t i = 0; i < S; ++i)
      for (size_t k = 0; k < D; ++k) {
        int64_t got = FixedInfer::decode_signed(ao[i][k].reveal());
        if (party == ALICE && got != ref[i][k]) { cout << "attn[" << i << "][" << k << "] " << got << " vs " << ref[i][k] << endl; error("attention mismatch\n"); }
      }
    if (party == ALICE) cout << "pass Attention" << endl;
  }

  // ===== 3. Full transformer block =====
  {
    RMat x = rmat(S, D, -2500, 2500);
    Vec g1 = rvec(D, (1 << P) - 200, (1 << P) + 200), b1n = rvec(D, -100, 100);
    Vec g2 = rvec(D, (1 << P) - 200, (1 << P) + 200), b2n = rvec(D, -100, 100);
    RMat Wq = rmat(D, D, -200, 200), Wk = rmat(D, D, -200, 200), Wv = rmat(D, D, -200, 200), Wo = rmat(D, D, -200, 200);
    Vec bq = rvec(D, -100, 100), bk = rvec(D, -100, 100), bv = rvec(D, -100, 100), bo = rvec(D, -100, 100);
    RMat W1 = rmat(FFN, D, -200, 200), W2 = rmat(D, FFN, -200, 200);
    Vec b1 = rvec(FFN, -100, 100), b2 = rvec(D, -100, 100);

    Mat cx = commit_mat(x), cWq = commit_mat(Wq), cWk = commit_mat(Wk), cWv = commit_mat(Wv), cWo = commit_mat(Wo), cW1 = commit_mat(W1), cW2 = commit_mat(W2);
    vector<IntFp> cg1, cb1n, cg2, cb2n, cbq, cbk, cbv, cbo, cb1, cb2;
    qi.commit_vec(cg1, g1); qi.commit_vec(cb1n, b1n); qi.commit_vec(cg2, g2); qi.commit_vec(cb2n, b2n);
    qi.commit_vec(cbq, bq); qi.commit_vec(cbk, bk); qi.commit_vec(cbv, bv); qi.commit_vec(cbo, bo);
    qi.commit_vec(cb1, b1); qi.commit_vec(cb2, b2);

    FixedTransformerBlock block(&qi, &ln1, &ln2, &attn, &gelu, D, FFN);
    Mat out;
    block.forward(out, cx, cg1, cb1n, cg2, cb2n, cWq, cbq, cWk, cbk, cWv, cbv, cWo, cbo, cW1, cb1, cW2, cb2);

    // reference
    RMat normed(S), a(S), ref(S);
    if (party == ALICE) {
      for (size_t i = 0; i < S; ++i) normed[i] = r_layernorm(x[i], g1, b1n);
      RMat ao = r_attention(normed, Wq, bq, Wk, bk, Wv, bv, Wo, bo);
      for (size_t i = 0; i < S; ++i) { a[i] = Vec(D); for (size_t k = 0; k < D; ++k) a[i][k] = x[i][k] + ao[i][k]; }
      for (size_t i = 0; i < S; ++i) {
        Vec n2 = r_layernorm(a[i], g2, b2n);
        Vec hid = r_linear(W1, b1, n2, false);
        Vec go(FFN); for (size_t m = 0; m < FFN; ++m) go[m] = FixedInfer::decode_signed(GELU_REF[hid[m] + GELU_OFF]);
        Vec ff = r_linear(W2, b2, go, false);
        ref[i] = Vec(D); for (size_t k = 0; k < D; ++k) ref[i][k] = a[i][k] + ff[k];
      }
    }
    for (size_t i = 0; i < S; ++i)
      for (size_t k = 0; k < D; ++k) {
        int64_t got = FixedInfer::decode_signed(out[i][k].reveal());
        if (party == ALICE && got != ref[i][k]) { cout << "block[" << i << "][" << k << "] " << got << " vs " << ref[i][k] << endl; error("block mismatch\n"); }
      }
    if (party == ALICE) cout << "pass Transformer block" << endl;
  }

  rcmp.Teardown();

  divr.destroy();
  expt.Teardown();
  gtab.Teardown();
  qi.destroy();
  finalize_zk_arith();
  _bench.report("transformer-block (ml-fixed)");
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
