// Test: ml-fixed SINGLE GPT-2 transformer-block TRAIN STEP, bound to the REAL ZK forward.
// The forward is now a PROVEN FixedTransformerBlock::forward() inside the ZK session
// (LN1 -> Attn -> residual -> LN2 -> FFN -> residual); the backward gadgets consume the
// committed intermediates straight from the forward caches (no fresh commits), so the
// train step is one continuous proof.  Loss head (last token): logits = We@out[last]+be,
// softmax cross-entropy -> LMHeadBackward -> dL/dout[last]; block backward + SGD follow.
// Block grads checked vs a composed-analytic float reference built from the REVEALED ZK
// intermediates (so only backward quantization contributes, tol 0.06); loss head exact; SGD exact.
//
// Every intermediate the backward needs is read straight from the forward caches,
// including normed (LN1 output = attention input, for dLdWq/k/v) and geluout
// (post-GELU, for dLdW2) -- no reconstruction.
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <chrono>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/div.h"
#include "volefier/ml-fixed/fixed_transformer.h"
#include "volefier/ml-fixed/fixed_backward.h"
using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
const int P = getenv("P") ? atoi(getenv("P")) : 7;
static const int s = 3, d = 4, h = 2, dh = 2, ffn = 8, vocab = 8;

int main(int argc, char **argv)
{
  parse_party_and_port(argv, &party, &port);
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE"))
    ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
  const int64_t _ev = getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL;
  const int _vt = getenv("VOLE_THREADS") ? atoi(getenv("VOLE_THREADS")) : -1;
  setup_zk_arith(ios, threads, party, _ev, ios[1], _vt);
  auto t0 = chrono::steady_clock::now();

  bool ok = true;
  {
    FixedInfer qi(party, P, ios[0], 32, threads);
    const double SCALE = static_cast<double>(1u << P);
    const uint64_t qinv = (uint64_t)llround((1.0 / std::sqrt((double)dh)) * SCALE);
    const double asc = (double)qinv / SCALE;
    const uint64_t lr_q = 16;
    const std::size_t true_token = 3;
    const std::size_t SB = 13; // softmax score bits (covers attention + LM-head logits)
    const uint64_t ET = 1u << SB;
    const uint64_t GT = 1u << (P + 7);
    const int64_t GOFF = 1LL << (P + 6);

    // ---- ZK forward machinery (mirrors gpt2_prove.cpp) ----
    RangeCheckLU rclu(party, ios[0]->io, 16);
    CompareLU cmp(&rclu);
    Div divr_ln(party, ios[0]->io, 24); // LN forward (mean/var) + LayerNormBackward means
    Div divr_sm(party, ios[0]->io, 16); // softmax normalization
    FixedTable expt(ET, party, ios[0]->io);
    expt.Setup(build_exp_table(ET, P, true));
    FixedTable gtab(GT, party, ios[0]->io);
    gtab.Setup(build_gelu_table(GT, P, GOFF));
    FixedSoftmax sm(&qi, &cmp, &expt, &divr_sm, P, SB);
    FixedActivation gelu_act(&gtab, GOFF);
    FixedLayerNorm ln1(&qi, &rclu, &divr_ln, P, d), ln2(&qi, &rclu, &divr_ln, P, d);
    FixedAttention attn(&qi, &sm, P, d, h, qinv);
    FixedTransformerBlock block(&qi, &ln1, &ln2, &attn, &gelu_act, d, ffn);
    // backward gadgets
    GELUBackward gelu_bwd(&qi, GT, GOFF);
    FFNBackward fnb(&qi, &gelu_bwd);
    LayerNormBackward lnb(&qi, &divr_ln);
    AttentionBackward atb(&qi, nullptr, h, dh, qinv, P);
    LMHeadBackward lmb(&qi);
    SoftmaxBackward sb(&qi);
    SGDOptimizer sgd(&qi, lr_q);

    // ---- random params + commit ----
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]()
    { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi)
    { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };
    auto q1 = [&](double v)
    { return (int64_t)llround(v * SCALE); };
    auto qd = [&](double v)
    { return (double)q1(v) / SCALE; }; // committed value (as real)
    double x[s][d], Wq[d][d], Wk[d][d], Wv[d][d], Wo[d][d], bq[d], bk[d], bv[d], bo[d];
    double g1[d], b1n[d], g2[d], b2n[d], W1[ffn][d], b1[ffn], W2[d][ffn], b2[d], We[vocab][d], be[vocab];
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k)
        x[i][k] = ur(-1, 1);
    for (int a = 0; a < d; ++a)
    {
      bq[a] = ur(-0.3, 0.3);
      bk[a] = ur(-0.3, 0.3);
      bv[a] = ur(-0.3, 0.3);
      bo[a] = ur(-0.3, 0.3);
      g1[a] = ur(0.7, 1.3);
      b1n[a] = ur(-0.2, 0.2);
      g2[a] = ur(0.7, 1.3);
      b2n[a] = ur(-0.2, 0.2);
      b2[a] = ur(-0.2, 0.2);
    }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b)
      {
        Wq[a][b] = ur(-0.5, 0.5);
        Wk[a][b] = ur(-0.5, 0.5);
        Wv[a][b] = ur(-0.5, 0.5);
        Wo[a][b] = ur(-0.5, 0.5);
      }
    for (int a = 0; a < ffn; ++a)
    {
      b1[a] = ur(-0.2, 0.2);
      for (int b = 0; b < d; ++b)
        W1[a][b] = ur(-0.5, 0.5);
    }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < ffn; ++b)
        W2[a][b] = ur(-0.5, 0.5);
    for (int v = 0; v < vocab; ++v)
    {
      be[v] = ur(-0.3, 0.3);
      for (int k = 0; k < d; ++k)
        We[v][k] = ur(-0.5, 0.5);
    }

    auto cMat = [&](int rows, int cols, function<double(int, int)> f)
    { Mat c(rows, vector<IntFp>(cols)); for (int r = 0; r < rows; ++r) { vector<int64_t> row(cols); for (int cc = 0; cc < cols; ++cc) row[cc] = q1(f(r, cc)); qi.commit_vec(c[r], row); } return c; };
    auto cVec = [&](int n_, function<double(int)> f)
    { vector<int64_t> row(n_); for (int r = 0; r < n_; ++r) row[r] = q1(f(r)); vector<IntFp> c; qi.commit_vec(c, row); return c; };
    Mat cx = cMat(s, d, [&](int i, int k)
                  { return x[i][k]; });
    vector<IntFp> cg1 = cVec(d, [&](int k)
                             { return g1[k]; }),
                  cb1n = cVec(d, [&](int k)
                              { return b1n[k]; });
    vector<IntFp> cg2 = cVec(d, [&](int k)
                             { return g2[k]; }),
                  cb2n = cVec(d, [&](int k)
                              { return b2n[k]; });
    Mat cWq = cMat(d, d, [&](int a, int b)
                   { return Wq[a][b]; }),
        cWk = cMat(d, d, [&](int a, int b)
                   { return Wk[a][b]; });
    Mat cWv = cMat(d, d, [&](int a, int b)
                   { return Wv[a][b]; }),
        cWo = cMat(d, d, [&](int a, int b)
                   { return Wo[a][b]; });
    vector<IntFp> cbq = cVec(d, [&](int k)
                             { return bq[k]; }),
                  cbk = cVec(d, [&](int k)
                             { return bk[k]; }),
                  cbv = cVec(d, [&](int k)
                             { return bv[k]; }),
                  cbo = cVec(d, [&](int k)
                             { return bo[k]; });
    Mat cW1 = cMat(ffn, d, [&](int a, int b)
                   { return W1[a][b]; }),
        cW2 = cMat(d, ffn, [&](int a, int b)
                   { return W2[a][b]; });
    vector<IntFp> cb1f = cVec(ffn, [&](int k)
                              { return b1[k]; }),
                  cb2 = cVec(d, [&](int k)
                             { return b2[k]; });
    Mat cWe = cMat(vocab, d, [&](int v, int k)
                   { return We[v][k]; });
    vector<IntFp> cbe = cVec(vocab, [&](int v)
                             { return be[v]; });

    // ---- PROVEN forward ----
    ln1.save_intermediates = true;
    ln2.save_intermediates = true;
    attn.save_intermediates = true;
    block.save_intermediates = true;
    ln1.clear_cache();
    ln2.clear_cache();
    attn.clear_cache();
    block.clear_cache();
    Mat cout;
    block.forward(cout, cx, cg1, cb1n, cg2, cb2n, cWq, cbq, cWk, cbk, cWv, cbv, cWo, cbo, cW1, cb1f, cW2, cb2);
    printf("[%d] proven forward done\n", party);

    // ---- pull committed intermediates from caches ----
    const int last = s - 1;
    Mat &c1 = ln1.cache_mat["c"], &y1 = ln1.cache_mat["y"], &c2 = ln2.cache_mat["c"];
    vector<IntFp> &dinv1 = ln1.cache_scalar_vec["dinv"], &dinv2 = ln2.cache_scalar_vec["dinv"];
    Mat &Q = attn.cache["Q"], &K = attn.cache["K"], &V = attn.cache["V"], &ctx = attn.cache["ctx"], &probsF = attn.cache["probs"];
    Mat &hidden = block.cache["hidden"], &n2 = block.cache["normed2"];
    // reorganize ragged probs [i*h+hh] -> [i][hh]
    vector<vector<vector<IntFp>>> probs(s, vector<vector<IntFp>>(h));
    for (int i = 0; i < s; ++i)
      for (int hh = 0; hh < h; ++hh)
        probs[i][hh] = probsF[i * h + hh];
    // normed (attention input) and geluout (post-GELU) now come straight from the block cache
    Mat &normed = block.cache["normed"], &geluout = block.cache["geluout"];

    // ---- LOSS HEAD ----
    vector<IntFp> logits;
    qi.dense_layer(logits, cWe, cbe, cout[last]);
    vector<IntFp> p = sm.forward(logits);
    vector<IntFp> dLdlogits;
    sb.cross_entropy(dLdlogits, p, true_token);
    vector<IntFp> dLdHlast, dLdWe, dLdbe;
    lmb.forward(dLdHlast, dLdWe, dLdbe, dLdlogits, cout[last], cWe, (uint64_t)vocab, (uint64_t)d);
    const IntFp zero(0ULL, PUBLIC);
    Mat cdO(s, vector<IntFp>(d, zero));
    for (int k = 0; k < d; ++k)
      cdO[last][k] = dLdHlast[k];

    // ---- BLOCK BACKWARD ----
    Mat dLdn2;
    vector<IntFp> dLdW1, dLdb1, dLdW2, dLdb2;
    fnb.forward(dLdn2, dLdW1, dLdb1, dLdW2, dLdb2, cdO, n2, hidden, geluout, cW1, cW2, (uint64_t)d, (uint64_t)ffn);
    Mat dLda(s, vector<IntFp>(d, zero));
    vector<IntFp> dLdg2(d, zero), dLdb2n(d, zero);
    for (int i = 0; i < s; ++i)
    {
      vector<IntFp> dxi, dgi, dbi;
      lnb.forward(dxi, dgi, dbi, dLdn2[i], c2[i], cg2, dinv2[i], (uint64_t)d);
      for (int k = 0; k < d; ++k)
      {
        dLda[i][k] = cdO[i][k] + dxi[k];
        dLdg2[k] = dLdg2[k] + dgi[k];
        dLdb2n[k] = dLdb2n[k] + dbi[k];
      }
    }
    Mat dLdH, dLdWq, dLdWk, dLdWv, dLdQ, dLdK, dLdV;
    vector<IntFp> dLdWq_f, dLdWk_f, dLdWv_f, dLdWo_f;
    atb.forward_full(dLdH, dLdWq, dLdWk, dLdWv, dLdWq_f, dLdWk_f, dLdWv_f, dLdQ, dLdK, dLdV, dLdWo_f, dLda, Q, K, V, probs, ctx, cWo, cWq, cWk, cWv, normed, (uint64_t)d);
    Mat dLdx(s, vector<IntFp>(d, zero));
    vector<IntFp> dLdg1(d, zero), dLdb1n(d, zero);
    for (int i = 0; i < s; ++i)
    {
      vector<IntFp> dxi, dgi, dbi;
      lnb.forward(dxi, dgi, dbi, dLdH[i], c1[i], cg1, dinv1[i], (uint64_t)d);
      for (int k = 0; k < d; ++k)
      {
        dLdx[i][k] = dLda[i][k] + dxi[k];
        dLdg1[k] = dLdg1[k] + dgi[k];
        dLdb1n[k] = dLdb1n[k] + dbi[k];
      }
    }

    // ---- SGD (Wq + We + be) ----
    Mat Wq_new;
    sgd.forward_mat(Wq_new, cWq, dLdWq);
    Mat dLdWe_mat(vocab, vector<IntFp>(d));
    for (int v = 0; v < vocab; ++v)
      for (int k = 0; k < d; ++k)
        dLdWe_mat[v][k] = dLdWe[v * d + k];
    Mat We_new;
    sgd.forward_mat(We_new, cWe, dLdWe_mat);
    vector<IntFp> be_new;
    sgd.forward(be_new, cbe, dLdbe);
    printf("[%d] loss head + block backward + SGD done\n", party);

    // ================= REVEAL (both parties, symmetric) =================
    auto rv = [&](IntFp &z)
    { return FixedInfer::decode_signed(z.reveal()); };
    double nrmd[s][d], Qr[s][d], Kr[s][d], Vr[s][d], ctxr[s][d], c1r[s][d], c2r[s][d], hidr[s][ffn], goutr[s][ffn], n2r[s][d];
    double d1r[s], d2r[s], outLastr[d];
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k)
      {
        nrmd[i][k] = rv(normed[i][k]) / SCALE;
        Qr[i][k] = rv(Q[i][k]) / SCALE;
        Kr[i][k] = rv(K[i][k]) / SCALE;
        Vr[i][k] = rv(V[i][k]) / SCALE;
        ctxr[i][k] = rv(ctx[i][k]) / (SCALE * SCALE);
        c1r[i][k] = rv(c1[i][k]) / SCALE;
        c2r[i][k] = rv(c2[i][k]) / SCALE;
        n2r[i][k] = rv(n2[i][k]) / SCALE;
      }
    for (int i = 0; i < s; ++i)
      for (int b = 0; b < ffn; ++b)
      {
        hidr[i][b] = rv(hidden[i][b]) / SCALE;
        goutr[i][b] = rv(geluout[i][b]) / SCALE;
      }
    for (int i = 0; i < s; ++i)
    {
      d1r[i] = rv(dinv1[i]) / SCALE;
      d2r[i] = rv(dinv2[i]) / SCALE;
    }
    for (int k = 0; k < d; ++k)
      outLastr[k] = rv(cout[last][k]) / SCALE;
    vector<vector<vector<double>>> pr(s, vector<vector<double>>(h));
    for (int i = 0; i < s; ++i)
      for (int hh = 0; hh < h; ++hh)
      {
        pr[i][hh].resize(i + 1);
        for (int j = 0; j <= i; ++j)
          pr[i][hh][j] = rv(probs[i][hh][j]) / SCALE;
      }
    vector<int64_t> logits_q(vocab), p_q(vocab), dl_q(vocab), dHlast_q(d), dWe_q(vocab * d), dbe_q(vocab);
    for (int v = 0; v < vocab; ++v)
    {
      logits_q[v] = rv(logits[v]);
      p_q[v] = rv(p[v]);
      dl_q[v] = rv(dLdlogits[v]);
      dbe_q[v] = rv(dLdbe[v]);
    }
    for (int k = 0; k < d; ++k)
      dHlast_q[k] = rv(dLdHlast[k]);
    for (int v = 0; v < vocab; ++v)
      for (int k = 0; k < d; ++k)
        dWe_q[v * d + k] = rv(dLdWe[v * d + k]);
    vector<vector<int64_t>> dx_q(s, vector<int64_t>(d));
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k)
        dx_q[i][k] = rv(dLdx[i][k]);
    vector<int64_t> dWq_q(d * d), dWo_q(d * d), dW1_q(ffn * d), dW2_q(d * ffn), dg1_q(d), dg2_q(d);
    for (int t = 0; t < d * d; ++t)
    {
      dWq_q[t] = rv(dLdWq_f[t]);
      dWo_q[t] = rv(dLdWo_f[t]);
    }
    for (int t = 0; t < ffn * d; ++t)
      dW1_q[t] = rv(dLdW1[t]);
    for (int t = 0; t < d * ffn; ++t)
      dW2_q[t] = rv(dLdW2[t]);
    for (int k = 0; k < d; ++k)
    {
      dg1_q[k] = rv(dLdg1[k]);
      dg2_q[k] = rv(dLdg2[k]);
    }
    vector<int64_t> Wqn_q(d * d), Wen_q(vocab * d), ben_q(vocab);
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b)
        Wqn_q[a * d + b] = rv(Wq_new[a][b]);
    for (int v = 0; v < vocab; ++v)
    {
      for (int k = 0; k < d; ++k)
        Wen_q[v * d + k] = rv(We_new[v][k]);
      ben_q[v] = rv(be_new[v]);
    }

    // ================= CHECK (ALICE) =================
    if (party == ALICE)
    {
      auto rha = [&](int64_t N) -> int64_t
      { int64_t hf = 1LL << (P - 1), Dn = 1LL << P; return (N >= 0) ? (N + hf) / Dn : -(((-N) + hf) / Dn); };
      // --- loss head: softmax sanity + exact ---
      double mx = -1e30, sum = 0, pf[vocab];
      for (int v = 0; v < vocab; ++v)
        if (logits_q[v] / SCALE > mx)
          mx = logits_q[v] / SCALE;
      for (int v = 0; v < vocab; ++v)
      {
        pf[v] = std::exp(logits_q[v] / SCALE - mx);
        sum += pf[v];
      }
      for (int v = 0; v < vocab; ++v)
        pf[v] /= sum;
      for (int v = 0; v < vocab; ++v)
      {
        if (std::fabs(p_q[v] / SCALE - pf[v]) > 0.03)
        {
          printf("[%d] p[%d] %.4f vs %.4f\n", party, v, p_q[v] / SCALE, pf[v]);
          ok = false;
        }
        int64_t ce = p_q[v] - ((std::size_t)v == true_token ? (1LL << P) : 0);
        if (dl_q[v] != ce || dbe_q[v] != dl_q[v])
        {
          printf("[%d] dLdlogits/dbe[%d] MISMATCH\n", party, v);
          ok = false;
        }
      }
      for (int k = 0; k < d; ++k)
      {
        int64_t acc = 0;
        for (int v = 0; v < vocab; ++v)
          acc += q1(We[v][k]) * dl_q[v];
        if (dHlast_q[k] != rha(acc))
        {
          printf("[%d] dLdHlast[%d] MISMATCH\n", party, k);
          ok = false;
        }
      }
      for (int v = 0; v < vocab; ++v)
        for (int k = 0; k < d; ++k)
          if (dWe_q[v * d + k] != rha(dl_q[v] * (int64_t)llround(outLastr[k] * SCALE)))
          {
            printf("[%d] dLdWe[%d][%d] MISMATCH\n", party, v, k);
            ok = false;
          }

      // --- composed-analytic block reference from REVEALED ZK intermediates ---
      double dO[s][d] = {};
      for (int k = 0; k < d; ++k)
        dO[last][k] = dHlast_q[k] / SCALE;
      vector<uint64_t> ggt = build_gelu_grad_table(GT, P, GOFF);
      auto gg = [&](double hq)
      { int64_t xq = llround(hq * SCALE); return FixedInfer::decode_signed(ggt[(size_t)(xq + GOFF)]) / SCALE; };
      double gN2[s][d] = {}, gHid[s][ffn], dGr[ffn], gW1r[ffn][d] = {}, gW2r[d][ffn] = {};
      for (int i = 0; i < s; ++i)
      {
        for (int p2 = 0; p2 < ffn; ++p2)
        {
          double g = 0;
          for (int k = 0; k < d; ++k)
            g += qd(W2[k][p2]) * dO[i][k];
          dGr[p2] = g;
        }
        for (int p2 = 0; p2 < ffn; ++p2)
          gHid[i][p2] = dGr[p2] * gg(hidr[i][p2]);
        for (int b = 0; b < d; ++b)
        {
          double g = 0;
          for (int p2 = 0; p2 < ffn; ++p2)
            g += qd(W1[p2][b]) * gHid[i][p2];
          gN2[i][b] = g;
        }
        for (int p2 = 0; p2 < ffn; ++p2)
          for (int b = 0; b < d; ++b)
            gW1r[p2][b] += gHid[i][p2] * n2r[i][b];
        for (int k = 0; k < d; ++k)
          for (int p2 = 0; p2 < ffn; ++p2)
            gW2r[k][p2] += dO[i][k] * goutr[i][p2];
      }
      auto ln_bwd = [&](const double *dLdy, const double *cc, double dinv, const double *g, double *dLdin, double *dg_acc)
      {
        double xh[d], gg2[d];
        for (int k = 0; k < d; ++k)
        {
          xh[k] = cc[k] * dinv;
          gg2[k] = dLdy[k] * g[k];
        }
        double e1 = 0, e2 = 0;
        for (int k = 0; k < d; ++k)
        {
          e1 += gg2[k] * xh[k];
          e2 += gg2[k];
        }
        e1 /= d;
        e2 /= d;
        for (int k = 0; k < d; ++k)
        {
          dLdin[k] = dinv * (gg2[k] - e2 - xh[k] * e1);
          dg_acc[k] += dLdy[k] * xh[k];
        }
      };
      double g2v[d];
      for (int k = 0; k < d; ++k)
        g2v[k] = qd(g2[k]);
      double g1v[d];
      for (int k = 0; k < d; ++k)
        g1v[k] = qd(g1[k]);
      double gDa[s][d], gG2[d] = {};
      for (int i = 0; i < s; ++i)
        ln_bwd(gN2[i], c2r[i], d2r[i], g2v, gDa[i], gG2);
      double gA[s][d];
      for (int i = 0; i < s; ++i)
        for (int k = 0; k < d; ++k)
          gA[i][k] = dO[i][k] + gDa[i][k];
      double gQ[s][d] = {}, gK[s][d] = {}, gV[s][d] = {}, gctx[s][d];
      for (int i = 0; i < s; ++i)
        for (int b = 0; b < d; ++b)
        {
          double g = 0;
          for (int aa = 0; aa < d; ++aa)
            g += qd(Wo[aa][b]) * gA[i][aa];
          gctx[i][b] = g;
        }
      for (int i = 0; i < s; ++i)
        for (int hh = 0; hh < h; ++hh)
        {
          int off = hh * dh, L = i + 1;
          double gp[s];
          for (int j = 0; j < L; ++j)
          {
            double g = 0;
            for (int k = 0; k < dh; ++k)
              g += gctx[i][off + k] * Vr[j][off + k];
            gp[j] = g;
          }
          for (int j = 0; j < L; ++j)
            for (int k = 0; k < dh; ++k)
              gV[j][off + k] += pr[i][hh][j] * gctx[i][off + k];
          double sd = 0;
          for (int j = 0; j < L; ++j)
            sd += pr[i][hh][j] * gp[j];
          double gd[s];
          for (int j = 0; j < L; ++j)
            gd[j] = pr[i][hh][j] * (gp[j] - sd) * asc;
          for (int k = 0; k < dh; ++k)
          {
            double g = 0;
            for (int j = 0; j < L; ++j)
              g += gd[j] * Kr[j][off + k];
            gQ[i][off + k] += g;
          }
          for (int j = 0; j < L; ++j)
            for (int k = 0; k < dh; ++k)
              gK[j][off + k] += gd[j] * Qr[i][off + k];
        }
      double gNr[s][d], gWqr[d][d] = {}, gWor[d][d] = {};
      for (int i = 0; i < s; ++i)
        for (int b = 0; b < d; ++b)
        {
          double g = 0;
          for (int aa = 0; aa < d; ++aa)
            g += qd(Wq[aa][b]) * gQ[i][aa] + qd(Wk[aa][b]) * gK[i][aa] + qd(Wv[aa][b]) * gV[i][aa];
          gNr[i][b] = g;
        }
      for (int i = 0; i < s; ++i)
      {
        for (int aa = 0; aa < d; ++aa)
          for (int b = 0; b < d; ++b)
            gWqr[aa][b] += gQ[i][aa] * nrmd[i][b];
        for (int aa = 0; aa < d; ++aa)
          for (int b = 0; b < d; ++b)
            gWor[aa][b] += gA[i][aa] * ctxr[i][b];
      }
      double gDx[s][d], gG1[d] = {};
      for (int i = 0; i < s; ++i)
        ln_bwd(gNr[i], c1r[i], d1r[i], g1v, gDx[i], gG1);
      double gX[s][d];
      for (int i = 0; i < s; ++i)
        for (int k = 0; k < d; ++k)
          gX[i][k] = gA[i][k] + gDx[i][k];

      const double tol = 0.06;
      auto ckf = [&](const char *nm, int i, int j, int64_t got, double ref)
      { if (std::fabs(got / SCALE - ref) > tol) { printf("[%d] %s[%d][%d]: gadget %.4f vs float %.4f (diff %.4f)\n", party, nm, i, j, got / SCALE, ref, std::fabs(got / SCALE - ref)); ok = false; } };
      for (int i = 0; i < s; ++i)
        for (int k = 0; k < d; ++k)
          ckf("dLdx", i, k, dx_q[i][k], gX[i][k]);
      for (int a2 = 0; a2 < d; ++a2)
        for (int b = 0; b < d; ++b)
        {
          ckf("dLdWq", a2, b, dWq_q[a2 * d + b], gWqr[a2][b]);
          ckf("dLdWo", a2, b, dWo_q[a2 * d + b], gWor[a2][b]);
        }
      for (int a2 = 0; a2 < ffn; ++a2)
        for (int b = 0; b < d; ++b)
          ckf("dLdW1", a2, b, dW1_q[a2 * d + b], gW1r[a2][b]);
      for (int a2 = 0; a2 < d; ++a2)
        for (int b = 0; b < ffn; ++b)
          ckf("dLdW2", a2, b, dW2_q[a2 * ffn + b], gW2r[a2][b]);
      for (int k = 0; k < d; ++k)
      {
        ckf("dLdg1", 0, k, dg1_q[k], gG1[k]);
        ckf("dLdg2", 0, k, dg2_q[k], gG2[k]);
      }

      // --- SGD exact ---
      for (int a2 = 0; a2 < d; ++a2)
        for (int b = 0; b < d; ++b)
          if (Wqn_q[a2 * d + b] != q1(Wq[a2][b]) - rha((int64_t)lr_q * dWq_q[a2 * d + b]))
          {
            printf("[%d] Wq_new[%d][%d] MISMATCH\n", party, a2, b);
            ok = false;
          }
      for (int v = 0; v < vocab; ++v)
      {
        for (int k = 0; k < d; ++k)
          if (Wen_q[v * d + k] != q1(We[v][k]) - rha((int64_t)lr_q * dWe_q[v * d + k]))
          {
            printf("[%d] We_new[%d][%d] MISMATCH\n", party, v, k);
            ok = false;
          }
        if (ben_q[v] != q1(be[v]) - rha((int64_t)lr_q * dbe_q[v]))
        {
          printf("[%d] be_new[%d] MISMATCH\n", party, v);
          ok = false;
        }
      }
      printf("[%d] TRAIN STEP checked: proven forward + loss head exact + block float(<%.3f) + SGD exact\n", party, tol);
    }

    gelu_bwd.Teardown();
    gtab.Teardown();
    expt.Teardown();
    rclu.Teardown();
    divr_sm.destroy();
    divr_ln.destroy();
    qi.destroy();
  }
  finalize_zk_arith();
  auto dt = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

  if (party == ALICE)
  {
    printf("[timing] ZK session %.2fs\n", dt);
    printf("%s\n", ok ? "pass block_train_step (ml-fixed)" : "FAIL block_train_step (ml-fixed)");
  }
  else
    printf("verifier: ml-fixed block_train_step proof complete\n");
  cout.flush();
  fflush(stdout);
  _exit(0);
}
