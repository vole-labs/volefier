// Test: ml-fixed SINGLE-BLOCK LoRA FINE-TUNING train step, bound to the REAL ZK forward.
// Copy of block_train_step.cpp adapted for LoRA: the Q and V projections use EFFECTIVE
// weights  W*_eff = W*_base + (alpha/Rk) * B * A  (committed as ALICE witnesses, so the
// proof covers the effective weight, not the low-rank decomposition).  The forward is a
// proven FixedTransformerBlock::forward(); forward_full yields dL/dW_Q_eff, dL/dW_V_eff;
// those are decomposed into adapter grads via matmul_backward, and ONLY the adapters
// (A_A,B_A,A_V,B_V) are SGD-updated -- the base model is frozen.
//
// NOTE: per spec the decomposition matmul_backward carries NO (alpha/Rk) factor, i.e. it
// computes dL/dB = dL/dW_eff @ A^T (and dL/dA = B^T @ dL/dW_eff); the reference mirrors
// that exactly, so both the exact and float checks validate the gadget as specified.
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
static const int s = 3, d = 4, h = 2, dh = 2, ffn = 8, vocab = 8, Rk = 2;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE")) ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
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
    const std::size_t SB = 13;
    const uint64_t ET = 1u << SB;
    const uint64_t GT = 1u << (P + 7);
    const int64_t GOFF = 1LL << (P + 6);
    const double alpha = 1.0;
    const uint64_t scale_q = (uint64_t)llround((alpha / (double)Rk) * SCALE); // (alpha/Rk)@2^-P = 64

    // ---- ZK forward machinery ----
    RangeCheckLU rclu(party, ios[0]->io, 16);
    CompareLU cmp(&rclu);
    Div divr_ln(party, ios[0]->io, 24), divr_sm(party, ios[0]->io, 16);
    FixedTable expt(ET, party, ios[0]->io); expt.Setup(build_exp_table(ET, P, true));
    FixedTable gtab(GT, party, ios[0]->io); gtab.Setup(build_gelu_table(GT, P, GOFF));
    FixedSoftmax sm(&qi, &cmp, &expt, &divr_sm, P, SB);
    FixedActivation gelu_act(&gtab, GOFF);
    FixedLayerNorm ln1(&qi, &rclu, &divr_ln, P, d), ln2(&qi, &rclu, &divr_ln, P, d);
    FixedAttention attn(&qi, &sm, P, d, h, qinv);
    FixedTransformerBlock block(&qi, &ln1, &ln2, &attn, &gelu_act, d, ffn);
    GELUBackward gelu_bwd(&qi, GT, GOFF);
    FFNBackward fnb(&qi, &gelu_bwd);
    LayerNormBackward lnb(&qi, &divr_ln);
    AttentionBackward atb(&qi, nullptr, h, dh, qinv, P);
    LMHeadBackward lmb(&qi);
    SoftmaxBackward sb(&qi);
    SGDOptimizer sgd(&qi, lr_q);
    volefier::IPProof<BoolIO> &ipp = *qi.ipp;

    // ---- random params ----
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };
    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };
    auto qd = [&](double v) { return (double)q1(v) / SCALE; };
    auto rha = [&](int64_t N, int M) -> int64_t { int64_t hf = 1LL << (M - 1), Dn = 1LL << M; return (N >= 0) ? (N + hf) / Dn : -(((-N) + hf) / Dn); };
    double x[s][d], Wq[d][d], Wk[d][d], Wv[d][d], Wo[d][d], bq[d], bk[d], bv[d], bo[d];
    double g1[d], b1n[d], g2[d], b2n[d], W1[ffn][d], b1[ffn], W2[d][ffn], b2[d], We[vocab][d], be[vocab];
    double A_A[Rk][d], B_A[d][Rk], A_V[Rk][d], B_V[d][Rk]; // LoRA adapters (ALICE private)
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) x[i][k] = ur(-1, 1);
    for (int a = 0; a < d; ++a) { bq[a] = ur(-0.3, 0.3); bk[a] = ur(-0.3, 0.3); bv[a] = ur(-0.3, 0.3); bo[a] = ur(-0.3, 0.3); g1[a] = ur(0.7, 1.3); b1n[a] = ur(-0.2, 0.2); g2[a] = ur(0.7, 1.3); b2n[a] = ur(-0.2, 0.2); b2[a] = ur(-0.2, 0.2); }
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { Wq[a][b] = ur(-0.5, 0.5); Wk[a][b] = ur(-0.5, 0.5); Wv[a][b] = ur(-0.5, 0.5); Wo[a][b] = ur(-0.5, 0.5); }
    for (int a = 0; a < ffn; ++a) { b1[a] = ur(-0.2, 0.2); for (int b = 0; b < d; ++b) W1[a][b] = ur(-0.5, 0.5); }
    for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) W2[a][b] = ur(-0.5, 0.5);
    for (int v = 0; v < vocab; ++v) { be[v] = ur(-0.3, 0.3); for (int k = 0; k < d; ++k) We[v][k] = ur(-0.5, 0.5); }
    for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { A_A[k][j] = ur(-0.5, 0.5); A_V[k][j] = ur(-0.5, 0.5); }
    for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { B_A[i][k] = ur(-0.5, 0.5); B_V[i][k] = ur(-0.5, 0.5); }

    // ---- effective weights  W_eff = W_base + requant_2P(scale_q * (B @ A))  (plaintext) ----
    auto eff = [&](double Wbase[d][d], double A[Rk][d], double B[d][Rk], int64_t out[d][d]) {
      for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
        int64_t ba = 0; for (int k = 0; k < Rk; ++k) ba += q1(B[i][k]) * q1(A[k][j]); // 2^-2P
        out[i][j] = q1(Wbase[i][j]) + rha((int64_t)scale_q * ba, 2 * P);             // 2^-P
      }
    };
    int64_t WQe_q[d][d], WVe_q[d][d];
    eff(Wq, A_A, B_A, WQe_q); eff(Wv, A_V, B_V, WVe_q);

    // ---- commit ----
    auto cMat = [&](int rows, int cols, function<double(int, int)> f) { Mat c(rows, vector<IntFp>(cols)); for (int rr = 0; rr < rows; ++rr) { vector<int64_t> row(cols); for (int cc = 0; cc < cols; ++cc) row[cc] = q1(f(rr, cc)); qi.commit_vec(c[rr], row); } return c; };
    auto cMatI = [&](int rows, int cols, function<int64_t(int, int)> f) { Mat c(rows, vector<IntFp>(cols)); for (int rr = 0; rr < rows; ++rr) { vector<int64_t> row(cols); for (int cc = 0; cc < cols; ++cc) row[cc] = f(rr, cc); qi.commit_vec(c[rr], row); } return c; };
    auto cVec = [&](int n_, function<double(int)> f) { vector<int64_t> row(n_); for (int rr = 0; rr < n_; ++rr) row[rr] = q1(f(rr)); vector<IntFp> c; qi.commit_vec(c, row); return c; };
    Mat cx = cMat(s, d, [&](int i, int k) { return x[i][k]; });
    vector<IntFp> cg1 = cVec(d, [&](int k) { return g1[k]; }), cb1n = cVec(d, [&](int k) { return b1n[k]; });
    vector<IntFp> cg2 = cVec(d, [&](int k) { return g2[k]; }), cb2n = cVec(d, [&](int k) { return b2n[k]; });
    Mat cWQe = cMatI(d, d, [&](int a, int b) { return WQe_q[a][b]; }); // effective Q weight (committed witness)
    Mat cWVe = cMatI(d, d, [&](int a, int b) { return WVe_q[a][b]; }); // effective V weight
    Mat cWk = cMat(d, d, [&](int a, int b) { return Wk[a][b]; }), cWo = cMat(d, d, [&](int a, int b) { return Wo[a][b]; });
    vector<IntFp> cbq = cVec(d, [&](int k) { return bq[k]; }), cbk = cVec(d, [&](int k) { return bk[k]; }), cbv = cVec(d, [&](int k) { return bv[k]; }), cbo = cVec(d, [&](int k) { return bo[k]; });
    Mat cW1 = cMat(ffn, d, [&](int a, int b) { return W1[a][b]; }), cW2 = cMat(d, ffn, [&](int a, int b) { return W2[a][b]; });
    vector<IntFp> cb1f = cVec(ffn, [&](int k) { return b1[k]; }), cb2 = cVec(d, [&](int k) { return b2[k]; });
    Mat cWe = cMat(vocab, d, [&](int v, int k) { return We[v][k]; });
    vector<IntFp> cbe = cVec(vocab, [&](int v) { return be[v]; });
    // LoRA adapters (private witnesses)
    Mat cA_A = cMat(Rk, d, [&](int k, int j) { return A_A[k][j]; }), cB_A = cMat(d, Rk, [&](int i, int k) { return B_A[i][k]; });
    Mat cA_V = cMat(Rk, d, [&](int k, int j) { return A_V[k][j]; }), cB_V = cMat(d, Rk, [&](int i, int k) { return B_V[i][k]; });

    // ---- PROVEN forward (uses EFFECTIVE Q/V weights) ----
    ln1.save_intermediates = ln2.save_intermediates = attn.save_intermediates = block.save_intermediates = true;
    ln1.clear_cache(); ln2.clear_cache(); attn.clear_cache(); block.clear_cache();
    Mat cout;
    block.forward(cout, cx, cg1, cb1n, cg2, cb2n, cWQe, cbq, cWk, cbk, cWVe, cbv, cWo, cbo, cW1, cb1f, cW2, cb2);
    printf("[%d] proven forward done\n", party);

    const int last = s - 1;
    Mat &c1 = ln1.cache_mat["c"], &c2 = ln2.cache_mat["c"];
    vector<IntFp> &dinv1 = ln1.cache_scalar_vec["dinv"], &dinv2 = ln2.cache_scalar_vec["dinv"];
    Mat &Q = attn.cache["Q"], &K = attn.cache["K"], &V = attn.cache["V"], &ctx = attn.cache["ctx"], &probsF = attn.cache["probs"];
    Mat &hidden = block.cache["hidden"], &n2 = block.cache["normed2"];
    Mat &normed = block.cache["normed"], &geluout = block.cache["geluout"];
    vector<vector<vector<IntFp>>> probs(s, vector<vector<IntFp>>(h));
    for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) probs[i][hh] = probsF[i * h + hh];

    // ---- LOSS HEAD ----
    vector<IntFp> logits; qi.dense_layer(logits, cWe, cbe, cout[last]);
    vector<IntFp> p = sm.forward(logits);
    vector<IntFp> dLdlogits; sb.cross_entropy(dLdlogits, p, true_token);
    vector<IntFp> dLdHlast, dLdWe, dLdbe;
    lmb.forward(dLdHlast, dLdWe, dLdbe, dLdlogits, cout[last], cWe, (uint64_t)vocab, (uint64_t)d);
    const IntFp zero(0ULL, PUBLIC);
    Mat cdO(s, vector<IntFp>(d, zero));
    for (int k = 0; k < d; ++k) cdO[last][k] = dLdHlast[k];

    // ---- BLOCK BACKWARD (forward_full uses the EFFECTIVE weights) ----
    Mat dLdn2; vector<IntFp> dLdW1, dLdb1, dLdW2, dLdb2;
    fnb.forward(dLdn2, dLdW1, dLdb1, dLdW2, dLdb2, cdO, n2, hidden, geluout, cW1, cW2, (uint64_t)d, (uint64_t)ffn);
    Mat dLda(s, vector<IntFp>(d, zero)); vector<IntFp> dLdg2(d, zero), dLdb2n(d, zero);
    for (int i = 0; i < s; ++i) { vector<IntFp> dxi, dgi, dbi; lnb.forward(dxi, dgi, dbi, dLdn2[i], c2[i], cg2, dinv2[i], (uint64_t)d); for (int k = 0; k < d; ++k) { dLda[i][k] = cdO[i][k] + dxi[k]; dLdg2[k] = dLdg2[k] + dgi[k]; dLdb2n[k] = dLdb2n[k] + dbi[k]; } }
    Mat dLdH, dLdWq, dLdWk, dLdWv, dLdQ, dLdK, dLdV; vector<IntFp> dLdWq_f, dLdWk_f, dLdWv_f, dLdWo_f;
    atb.forward_full(dLdH, dLdWq, dLdWk, dLdWv, dLdWq_f, dLdWk_f, dLdWv_f, dLdQ, dLdK, dLdV, dLdWo_f, dLda, Q, K, V, probs, ctx, cWo, cWQe, cWk, cWVe, normed, (uint64_t)d);
    Mat dLdx(s, vector<IntFp>(d, zero)); vector<IntFp> dLdg1(d, zero), dLdb1n(d, zero);
    for (int i = 0; i < s; ++i) { vector<IntFp> dxi, dgi, dbi; lnb.forward(dxi, dgi, dbi, dLdH[i], c1[i], cg1, dinv1[i], (uint64_t)d); for (int k = 0; k < d; ++k) { dLdx[i][k] = dLda[i][k] + dxi[k]; dLdg1[k] = dLdg1[k] + dgi[k]; dLdb1n[k] = dLdb1n[k] + dbi[k]; } }

    // ---- LoRA gradient decomposition:  dL/dW_eff -> adapter grads ----
    auto flatMat = [&](const Mat &M, int rows, int cols) { vector<IntFp> f(rows * cols); for (int a = 0; a < rows; ++a) for (int b = 0; b < cols; ++b) f[a * cols + b] = M[a][b]; return f; };
    vector<IntFp> A_A_flat = flatMat(cA_A, Rk, d), B_A_flat = flatMat(cB_A, d, Rk);
    vector<IntFp> A_V_flat = flatMat(cA_V, Rk, d), B_V_flat = flatMat(cB_V, d, Rk);
    vector<IntFp> dLdB_A, dLdA_A, dLdB_V, dLdA_V, dLdB_A_p, dLdA_A_p, dLdB_V_p, dLdA_V_p;
    matmul_backward(ipp, dLdB_A, dLdA_A, B_A_flat, A_A_flat, dLdWq_f, (int)d, (int)Rk, (int)d);
    qi.requant_cm(dLdB_A_p, dLdB_A, 1, P);
    qi.requant_cm(dLdA_A_p, dLdA_A, 1, P);
    matmul_backward(ipp, dLdB_V, dLdA_V, B_V_flat, A_V_flat, dLdWv_f, (int)d, (int)Rk, (int)d);
    qi.requant_cm(dLdB_V_p, dLdB_V, 1, P);
    qi.requant_cm(dLdA_V_p, dLdA_V, 1, P);

    // ---- SGD on ADAPTERS ONLY (base model frozen) ----
    auto toMat = [&](const vector<IntFp> &f, int rows, int cols) { Mat M(rows, vector<IntFp>(cols)); for (int a = 0; a < rows; ++a) for (int b = 0; b < cols; ++b) M[a][b] = f[a * cols + b]; return M; };
    Mat A_A_new, B_A_new, A_V_new, B_V_new;
    sgd.forward_mat(A_A_new, cA_A, toMat(dLdA_A_p, Rk, d));
    sgd.forward_mat(B_A_new, cB_A, toMat(dLdB_A_p, d, Rk));
    sgd.forward_mat(A_V_new, cA_V, toMat(dLdA_V_p, Rk, d));
    sgd.forward_mat(B_V_new, cB_V, toMat(dLdB_V_p, d, Rk));
    printf("[%d] loss head + block backward + LoRA decomp + adapter SGD done\n", party);

    // ================= REVEAL =================
    auto rv = [&](IntFp &z) { return FixedInfer::decode_signed(z.reveal()); };
    double nrmd[s][d], Qr[s][d], Kr[s][d], Vr[s][d], ctxr[s][d], c1r[s][d], c2r[s][d], hidr[s][ffn], goutr[s][ffn], n2r[s][d];
    double d1r[s], d2r[s], outLastr[d];
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) { nrmd[i][k] = rv(normed[i][k]) / SCALE; Qr[i][k] = rv(Q[i][k]) / SCALE; Kr[i][k] = rv(K[i][k]) / SCALE; Vr[i][k] = rv(V[i][k]) / SCALE; ctxr[i][k] = rv(ctx[i][k]) / (SCALE * SCALE); c1r[i][k] = rv(c1[i][k]) / SCALE; c2r[i][k] = rv(c2[i][k]) / SCALE; n2r[i][k] = rv(n2[i][k]) / SCALE; }
    for (int i = 0; i < s; ++i) for (int b = 0; b < ffn; ++b) { hidr[i][b] = rv(hidden[i][b]) / SCALE; goutr[i][b] = rv(geluout[i][b]) / SCALE; }
    for (int i = 0; i < s; ++i) { d1r[i] = rv(dinv1[i]) / SCALE; d2r[i] = rv(dinv2[i]) / SCALE; }
    for (int k = 0; k < d; ++k) outLastr[k] = rv(cout[last][k]) / SCALE;
    vector<vector<vector<double>>> pr(s, vector<vector<double>>(h));
    for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) { pr[i][hh].resize(i + 1); for (int j = 0; j <= i; ++j) pr[i][hh][j] = rv(probs[i][hh][j]) / SCALE; }
    vector<int64_t> logits_q(vocab), p_q(vocab), dl_q(vocab), dHlast_q(d), dWe_q(vocab * d), dbe_q(vocab);
    for (int v = 0; v < vocab; ++v) { logits_q[v] = rv(logits[v]); p_q[v] = rv(p[v]); dl_q[v] = rv(dLdlogits[v]); dbe_q[v] = rv(dLdbe[v]); }
    for (int k = 0; k < d; ++k) dHlast_q[k] = rv(dLdHlast[k]);
    for (int v = 0; v < vocab; ++v) for (int k = 0; k < d; ++k) dWe_q[v * d + k] = rv(dLdWe[v * d + k]);
    vector<vector<int64_t>> dx_q(s, vector<int64_t>(d));
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) dx_q[i][k] = rv(dLdx[i][k]);
    vector<int64_t> dWq_q(d * d), dWv_q(d * d), dWo_q(d * d), dW1_q(ffn * d), dW2_q(d * ffn), dg1_q(d), dg2_q(d);
    for (int t = 0; t < d * d; ++t) { dWq_q[t] = rv(dLdWq_f[t]); dWv_q[t] = rv(dLdWv_f[t]); dWo_q[t] = rv(dLdWo_f[t]); }
    for (int t = 0; t < ffn * d; ++t) dW1_q[t] = rv(dLdW1[t]);
    for (int t = 0; t < d * ffn; ++t) dW2_q[t] = rv(dLdW2[t]);
    for (int k = 0; k < d; ++k) { dg1_q[k] = rv(dLdg1[k]); dg2_q[k] = rv(dLdg2[k]); }
    // LoRA adapter grads + updated adapters
    vector<int64_t> dAA_q(Rk * d), dBA_q(d * Rk), dAV_q(Rk * d), dBV_q(d * Rk);
    vector<int64_t> AAn_q(Rk * d), BAn_q(d * Rk), AVn_q(Rk * d), BVn_q(d * Rk);
    for (int t = 0; t < Rk * d; ++t) { dAA_q[t] = rv(dLdA_A_p[t]); dAV_q[t] = rv(dLdA_V_p[t]); }
    for (int t = 0; t < d * Rk; ++t) { dBA_q[t] = rv(dLdB_A_p[t]); dBV_q[t] = rv(dLdB_V_p[t]); }
    for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { AAn_q[k * d + j] = rv(A_A_new[k][j]); AVn_q[k * d + j] = rv(A_V_new[k][j]); }
    for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { BAn_q[i * Rk + k] = rv(B_A_new[i][k]); BVn_q[i * Rk + k] = rv(B_V_new[i][k]); }

    // ================= CHECK (ALICE) =================
    if (party == ALICE) {
      // --- loss head exact + softmax sanity ---
      double mx = -1e30, sum = 0, pf[vocab];
      for (int v = 0; v < vocab; ++v) if (logits_q[v] / SCALE > mx) mx = logits_q[v] / SCALE;
      for (int v = 0; v < vocab; ++v) { pf[v] = std::exp(logits_q[v] / SCALE - mx); sum += pf[v]; }
      for (int v = 0; v < vocab; ++v) pf[v] /= sum;
      for (int v = 0; v < vocab; ++v) { if (std::fabs(p_q[v] / SCALE - pf[v]) > 0.03) { printf("[%d] p[%d] MISMATCH\n", party, v); ok = false; } int64_t ce = p_q[v] - ((std::size_t)v == true_token ? (1LL << P) : 0); if (dl_q[v] != ce || dbe_q[v] != dl_q[v]) { printf("[%d] dLdlogits/dbe[%d] MISMATCH\n", party, v); ok = false; } }
      for (int k = 0; k < d; ++k) { int64_t acc = 0; for (int v = 0; v < vocab; ++v) acc += q1(We[v][k]) * dl_q[v]; if (dHlast_q[k] != rha(acc, P)) { printf("[%d] dLdHlast[%d] MISMATCH\n", party, k); ok = false; } }
      for (int v = 0; v < vocab; ++v) for (int k = 0; k < d; ++k) if (dWe_q[v * d + k] != rha(dl_q[v] * (int64_t)llround(outLastr[k] * SCALE), P)) { printf("[%d] dLdWe[%d][%d] MISMATCH\n", party, v, k); ok = false; }

      // --- composed-analytic block reference (uses EFFECTIVE Q/V weights) ---
      auto WQe = [&](int a, int b) { return WQe_q[a][b] / SCALE; };
      auto WVe = [&](int a, int b) { return WVe_q[a][b] / SCALE; };
      double dO[s][d] = {}; for (int k = 0; k < d; ++k) dO[last][k] = dHlast_q[k] / SCALE;
      vector<uint64_t> ggt = build_gelu_grad_table(GT, P, GOFF);
      auto gg = [&](double hq) { int64_t xq = llround(hq * SCALE); return FixedInfer::decode_signed(ggt[(size_t)(xq + GOFF)]) / SCALE; };
      double gN2[s][d] = {}, gHid[s][ffn], dGr[ffn], gW1r[ffn][d] = {}, gW2r[d][ffn] = {};
      for (int i = 0; i < s; ++i) { for (int p2 = 0; p2 < ffn; ++p2) { double g = 0; for (int k = 0; k < d; ++k) g += qd(W2[k][p2]) * dO[i][k]; dGr[p2] = g; } for (int p2 = 0; p2 < ffn; ++p2) gHid[i][p2] = dGr[p2] * gg(hidr[i][p2]); for (int b = 0; b < d; ++b) { double g = 0; for (int p2 = 0; p2 < ffn; ++p2) g += qd(W1[p2][b]) * gHid[i][p2]; gN2[i][b] = g; } for (int p2 = 0; p2 < ffn; ++p2) for (int b = 0; b < d; ++b) gW1r[p2][b] += gHid[i][p2] * n2r[i][b]; for (int k = 0; k < d; ++k) for (int p2 = 0; p2 < ffn; ++p2) gW2r[k][p2] += dO[i][k] * goutr[i][p2]; }
      auto ln_bwd = [&](const double *dLdy, const double *cc, double dinv, const double *g, double *dLdin, double *dg_acc) { double xh[d], gg2[d]; for (int k = 0; k < d; ++k) { xh[k] = cc[k] * dinv; gg2[k] = dLdy[k] * g[k]; } double e1 = 0, e2 = 0; for (int k = 0; k < d; ++k) { e1 += gg2[k] * xh[k]; e2 += gg2[k]; } e1 /= d; e2 /= d; for (int k = 0; k < d; ++k) { dLdin[k] = dinv * (gg2[k] - e2 - xh[k] * e1); dg_acc[k] += dLdy[k] * xh[k]; } };
      double g2v[d], g1v[d]; for (int k = 0; k < d; ++k) { g2v[k] = qd(g2[k]); g1v[k] = qd(g1[k]); }
      double gDa[s][d], gG2[d] = {}; for (int i = 0; i < s; ++i) ln_bwd(gN2[i], c2r[i], d2r[i], g2v, gDa[i], gG2);
      double gA[s][d]; for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) gA[i][k] = dO[i][k] + gDa[i][k];
      double gQ[s][d] = {}, gK[s][d] = {}, gV[s][d] = {}, gctx[s][d];
      for (int i = 0; i < s; ++i) for (int b = 0; b < d; ++b) { double g = 0; for (int aa = 0; aa < d; ++aa) g += qd(Wo[aa][b]) * gA[i][aa]; gctx[i][b] = g; }
      for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) { int off = hh * dh, L = i + 1; double gp[s]; for (int j = 0; j < L; ++j) { double g = 0; for (int k = 0; k < dh; ++k) g += gctx[i][off + k] * Vr[j][off + k]; gp[j] = g; } for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gV[j][off + k] += pr[i][hh][j] * gctx[i][off + k]; double sd = 0; for (int j = 0; j < L; ++j) sd += pr[i][hh][j] * gp[j]; double gd[s]; for (int j = 0; j < L; ++j) gd[j] = pr[i][hh][j] * (gp[j] - sd) * asc; for (int k = 0; k < dh; ++k) { double g = 0; for (int j = 0; j < L; ++j) g += gd[j] * Kr[j][off + k]; gQ[i][off + k] += g; } for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gK[j][off + k] += gd[j] * Qr[i][off + k]; }
      double gNr[s][d], gWqr[d][d] = {}, gWvr[d][d] = {}, gWor[d][d] = {};
      for (int i = 0; i < s; ++i) for (int b = 0; b < d; ++b) { double g = 0; for (int aa = 0; aa < d; ++aa) g += WQe(aa, b) * gQ[i][aa] + qd(Wk[aa][b]) * gK[i][aa] + WVe(aa, b) * gV[i][aa]; gNr[i][b] = g; }
      for (int i = 0; i < s; ++i) { for (int aa = 0; aa < d; ++aa) for (int b = 0; b < d; ++b) { gWqr[aa][b] += gQ[i][aa] * nrmd[i][b]; gWvr[aa][b] += gV[i][aa] * nrmd[i][b]; gWor[aa][b] += gA[i][aa] * ctxr[i][b]; } }
      double gDx[s][d], gG1[d] = {}; for (int i = 0; i < s; ++i) ln_bwd(gNr[i], c1r[i], d1r[i], g1v, gDx[i], gG1);
      double gX[s][d]; for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) gX[i][k] = gA[i][k] + gDx[i][k];

      const double tolB = 0.06;
      auto ckf = [&](const char *nm, int i, int j, int64_t got, double ref, double tol) { if (std::fabs(got / SCALE - ref) > tol) { printf("[%d] %s[%d][%d]: gadget %.4f vs float %.4f (diff %.4f)\n", party, nm, i, j, got / SCALE, ref, std::fabs(got / SCALE - ref)); ok = false; } };
      for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) ckf("dLdx", i, k, dx_q[i][k], gX[i][k], tolB);
      for (int a2 = 0; a2 < d; ++a2) for (int b = 0; b < d; ++b) { ckf("dLdWq_eff", a2, b, dWq_q[a2 * d + b], gWqr[a2][b], tolB); ckf("dLdWv_eff", a2, b, dWv_q[a2 * d + b], gWvr[a2][b], tolB); ckf("dLdWo", a2, b, dWo_q[a2 * d + b], gWor[a2][b], tolB); }
      for (int a2 = 0; a2 < ffn; ++a2) for (int b = 0; b < d; ++b) ckf("dLdW1", a2, b, dW1_q[a2 * d + b], gW1r[a2][b], tolB);
      for (int a2 = 0; a2 < d; ++a2) for (int b = 0; b < ffn; ++b) ckf("dLdW2", a2, b, dW2_q[a2 * ffn + b], gW2r[a2][b], tolB);
      for (int k = 0; k < d; ++k) { ckf("dLdg1", 0, k, dg1_q[k], gG1[k], tolB); ckf("dLdg2", 0, k, dg2_q[k], gG2[k], tolB); }

      // --- LoRA decomposition: EXACT replay (mirrors gadget: no alpha/Rk factor) + float ---
      const double tolA = 2.0 / SCALE;
      // dLdB[i][k] = requant(sum_j dLdWeff[i][j]*A[k][j]) ; dLdA[k][j] = requant(sum_i B[i][k]*dLdWeff[i][j])
      auto chk_decomp = [&](const char *nm, const vector<int64_t> &dWeff_q, double A[Rk][d], double B[d][Rk],
                            const vector<int64_t> &dB_p, const vector<int64_t> &dA_p) {
        for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { int64_t acc = 0; for (int j = 0; j < d; ++j) acc += dWeff_q[i * d + j] * q1(A[k][j]); int64_t ex = rha(acc, P); if (dB_p[i * Rk + k] != ex) { printf("[%d] dLdB_%s[%d][%d] MISMATCH got %ld exp %ld\n", party, nm, i, k, (long)dB_p[i * Rk + k], (long)ex); ok = false; } double fl = 0; for (int j = 0; j < d; ++j) fl += (dWeff_q[i * d + j] / SCALE) * qd(A[k][j]); if (std::fabs(dB_p[i * Rk + k] / SCALE - fl) > tolA) { printf("[%d] dLdB_%s[%d][%d] float diff\n", party, nm, i, k); ok = false; } }
        for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { int64_t acc = 0; for (int i = 0; i < d; ++i) acc += q1(B[i][k]) * dWeff_q[i * d + j]; int64_t ex = rha(acc, P); if (dA_p[k * d + j] != ex) { printf("[%d] dLdA_%s[%d][%d] MISMATCH got %ld exp %ld\n", party, nm, k, j, (long)dA_p[k * d + j], (long)ex); ok = false; } double fl = 0; for (int i = 0; i < d; ++i) fl += qd(B[i][k]) * (dWeff_q[i * d + j] / SCALE); if (std::fabs(dA_p[k * d + j] / SCALE - fl) > tolA) { printf("[%d] dLdA_%s[%d][%d] float diff\n", party, nm, k, j); ok = false; } }
      };
      chk_decomp("A", dWq_q, A_A, B_A, dBA_q, dAA_q);
      chk_decomp("V", dWv_q, A_V, B_V, dBV_q, dAV_q);

      // --- adapter SGD: exact + float ---
      auto chk_sgd_A = [&](const char *nm, double A[Rk][d], const vector<int64_t> &dA_p, const vector<int64_t> &An) {
        for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { int64_t ex = q1(A[k][j]) - rha((int64_t)lr_q * dA_p[k * d + j], P); if (An[k * d + j] != ex) { printf("[%d] %s_new[%d][%d] SGD MISMATCH\n", party, nm, k, j); ok = false; } double fl = qd(A[k][j]) - ((double)lr_q / SCALE) * (dA_p[k * d + j] / SCALE); if (std::fabs(An[k * d + j] / SCALE - fl) > tolA) { printf("[%d] %s_new[%d][%d] float diff\n", party, nm, k, j); ok = false; } }
      };
      auto chk_sgd_B = [&](const char *nm, double B[d][Rk], const vector<int64_t> &dB_p, const vector<int64_t> &Bn) {
        for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { int64_t ex = q1(B[i][k]) - rha((int64_t)lr_q * dB_p[i * Rk + k], P); if (Bn[i * Rk + k] != ex) { printf("[%d] %s_new[%d][%d] SGD MISMATCH\n", party, nm, i, k); ok = false; } double fl = qd(B[i][k]) - ((double)lr_q / SCALE) * (dB_p[i * Rk + k] / SCALE); if (std::fabs(Bn[i * Rk + k] / SCALE - fl) > tolA) { printf("[%d] %s_new[%d][%d] float diff\n", party, nm, i, k); ok = false; } }
      };
      chk_sgd_A("A_A", A_A, dAA_q, AAn_q); chk_sgd_B("B_A", B_A, dBA_q, BAn_q);
      chk_sgd_A("A_V", A_V, dAV_q, AVn_q); chk_sgd_B("B_V", B_V, dBV_q, BVn_q);

      printf("[%d] LoRA TRAIN STEP checked: proven forward + loss head + block float(<%.3f) + adapter decomp/SGD exact + float(<%.4f)\n", party, tolB, tolA);
    }

    gelu_bwd.Teardown(); gtab.Teardown(); expt.Teardown(); rclu.Teardown(); divr_sm.destroy(); divr_ln.destroy(); qi.destroy();
  }
  finalize_zk_arith();
  auto dt = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

  if (party == ALICE) { printf("[timing] ZK session %.2fs\n", dt); printf("%s\n", ok ? "pass lora_train_step (ml-fixed)" : "FAIL lora_train_step (ml-fixed)"); }
  else printf("verifier: ml-fixed lora_train_step proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
