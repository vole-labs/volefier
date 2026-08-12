// Test: ml-fixed MULTI-BLOCK LoRA fine-tuning train step, one continuous ZK proof.
// num_blocks stacked FixedTransformerBlocks (each Q/V projection LoRA-adapted via an
// effective weight committed as an ALICE witness), a proven forward chain H_0=x ->
// block[0] -> ... -> block[B-1] -> LM head, softmax cross-entropy loss, then a reverse
// backward chain (dLdH flows block b -> block b-1) with LoRA gradient decomposition and
// SGD on the adapters ONLY (base model frozen).
//
// Validation (tight): each block's backward is checked against a composed-analytic
// reference driven by the block's REVEALED upstream + REVEALED forward intermediates
// (0.06), including the chain link (reference dLdx vs the gadget's next-block upstream);
// adapter decomposition is exact from the revealed dL/dW_eff, adapter SGD exact + float
// (<2/SCALE); loss-head dLdlogits exact.  ONE setup_zk_arith / finalize; shared ZK
// machinery across blocks; per-block GELUBackward owns its own table.
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <cstring>
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
static const int NB = 12, s = 512, d = 768, h = 12, dh = 64, ffn = 3072, vocab = 50257, Rk = 8;

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
    const uint64_t scale_q = (uint64_t)llround((alpha / (double)Rk) * SCALE); // 64

    // ---- SHARED ZK machinery (instantiated ONCE) ----
    RangeCheckLU rclu(party, ios[0]->io, 16);
    CompareLU cmp(&rclu);
    Div divr_ln(party, ios[0]->io, 24), divr_sm(party, ios[0]->io, 16);
    FixedTable expt(ET, party, ios[0]->io); expt.Setup(build_exp_table(ET, P, true));
    FixedTable gtab(GT, party, ios[0]->io); gtab.Setup(build_gelu_table(GT, P, GOFF));
    FixedSoftmax sm(&qi, &cmp, &expt, &divr_sm, P, SB);
    FixedActivation gelu_act(&gtab, GOFF);
    LayerNormBackward lnb(&qi, &divr_ln);
    AttentionBackward atb(&qi, nullptr, h, dh, qinv, P);
    LMHeadBackward lmb(&qi);
    SoftmaxBackward sbk(&qi);
    SGDOptimizer sgd(&qi, lr_q);
    volefier::IPProof<BoolIO> &ipp = *qi.ipp;
    // per-block objects (no default ctor -> heap)
    FixedLayerNorm *ln1[NB], *ln2[NB];
    FixedAttention *attn[NB];
    FixedTransformerBlock *block[NB];
    GELUBackward *gelu_bwd[NB]; // each owns its own GELU' table
    for (int b = 0; b < NB; ++b) {
      ln1[b] = new FixedLayerNorm(&qi, &rclu, &divr_ln, P, d);
      ln2[b] = new FixedLayerNorm(&qi, &rclu, &divr_ln, P, d);
      attn[b] = new FixedAttention(&qi, &sm, P, d, h, qinv);
      block[b] = new FixedTransformerBlock(&qi, ln1[b], ln2[b], attn[b], &gelu_act, d, ffn);
      gelu_bwd[b] = new GELUBackward(&qi, GT, GOFF);
    }

    // ---- random params ----
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };
    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };
    auto qd = [&](double v) { return (double)q1(v) / SCALE; };
    auto rha = [&](int64_t N, int M) -> int64_t { int64_t hf = 1LL << (M - 1), Dn = 1LL << M; return (N >= 0) ? (N + hf) / Dn : -(((-N) + hf) / Dn); };
    vector<vector<double>> x(s, vector<double>(d));
    vector<vector<vector<double>>> Wq(NB, vector<vector<double>>(d, vector<double>(d))), Wk(NB, vector<vector<double>>(d, vector<double>(d))), Wv(NB, vector<vector<double>>(d, vector<double>(d))), Wo(NB, vector<vector<double>>(d, vector<double>(d)));
    vector<vector<double>> bq(NB, vector<double>(d)), bk(NB, vector<double>(d)), bv(NB, vector<double>(d)), bo(NB, vector<double>(d));
    vector<vector<double>> g1(NB, vector<double>(d)), b1n(NB, vector<double>(d)), g2(NB, vector<double>(d)), b2n(NB, vector<double>(d)), b2(NB, vector<double>(d));
    vector<vector<double>> b1(NB, vector<double>(ffn));
    vector<vector<vector<double>>> W1(NB, vector<vector<double>>(ffn, vector<double>(d))), W2(NB, vector<vector<double>>(d, vector<double>(ffn)));
    vector<vector<vector<double>>> A_A(NB, vector<vector<double>>(Rk, vector<double>(d))), B_A(NB, vector<vector<double>>(d, vector<double>(Rk))), A_V(NB, vector<vector<double>>(Rk, vector<double>(d))), B_V(NB, vector<vector<double>>(d, vector<double>(Rk)));
    vector<vector<double>> We(vocab, vector<double>(d));
    vector<double> be(vocab);
    vector<vector<vector<int64_t>>> WQe_q(NB, vector<vector<int64_t>>(d, vector<int64_t>(d))), WVe_q(NB, vector<vector<int64_t>>(d, vector<int64_t>(d)));
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) x[i][k] = ur(-1, 1);
    for (int b = 0; b < NB; ++b) {
      for (int a = 0; a < d; ++a) { bq[b][a] = ur(-0.3, 0.3); bk[b][a] = ur(-0.3, 0.3); bv[b][a] = ur(-0.3, 0.3); bo[b][a] = ur(-0.3, 0.3); g1[b][a] = 1.0; b1n[b][a] = 0.0; g2[b][a] = 1.0; b2n[b][a] = 0.0; b2[b][a] = ur(-0.2, 0.2); }
      for (int a = 0; a < d; ++a) for (int c = 0; c < d; ++c) { Wq[b][a][c] = ur(-0.02, 0.02); Wk[b][a][c] = ur(-0.02, 0.02); Wv[b][a][c] = ur(-0.02, 0.02); Wo[b][a][c] = ur(-0.02, 0.02); }
      for (int a = 0; a < ffn; ++a) { b1[b][a] = ur(-0.2, 0.2); for (int c = 0; c < d; ++c) W1[b][a][c] = ur(-0.02, 0.02); }
      for (int a = 0; a < d; ++a) for (int c = 0; c < ffn; ++c) W2[b][a][c] = ur(-0.02, 0.02);
      for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { A_A[b][k][j] = ur(-0.01, 0.01); A_V[b][k][j] = ur(-0.01, 0.01); }
      for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { B_A[b][i][k] = 0.0; B_V[b][i][k] = 0.0; }
    }
    for (int v = 0; v < vocab; ++v) { be[v] = ur(-0.3, 0.3); for (int k = 0; k < d; ++k) We[v][k] = ur(-0.02, 0.02); }
    // effective weights (plaintext): W_eff = W_base + requant_2P(scale_q * (B @ A))
    for (int b = 0; b < NB; ++b)
      for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
        int64_t baA = 0, baV = 0;
        for (int k = 0; k < Rk; ++k) { baA += q1(B_A[b][i][k]) * q1(A_A[b][k][j]); baV += q1(B_V[b][i][k]) * q1(A_V[b][k][j]); }
        WQe_q[b][i][j] = q1(Wq[b][i][j]) + rha((int64_t)scale_q * baA, 2 * P);
        WVe_q[b][i][j] = q1(Wv[b][i][j]) + rha((int64_t)scale_q * baV, 2 * P);
      }

    // ---- commit ----
    auto cMat = [&](int rows, int cols, function<double(int, int)> f) { Mat c(rows, vector<IntFp>(cols)); for (int rr = 0; rr < rows; ++rr) { vector<int64_t> row(cols); for (int cc = 0; cc < cols; ++cc) row[cc] = q1(f(rr, cc)); qi.commit_vec(c[rr], row); } return c; };
    auto cMatI = [&](int rows, int cols, function<int64_t(int, int)> f) { Mat c(rows, vector<IntFp>(cols)); for (int rr = 0; rr < rows; ++rr) { vector<int64_t> row(cols); for (int cc = 0; cc < cols; ++cc) row[cc] = f(rr, cc); qi.commit_vec(c[rr], row); } return c; };
    auto cVec = [&](int n_, function<double(int)> f) { vector<int64_t> row(n_); for (int rr = 0; rr < n_; ++rr) row[rr] = q1(f(rr)); vector<IntFp> c; qi.commit_vec(c, row); return c; };
    Mat cx = cMat(s, d, [&](int i, int k) { return x[i][k]; });
    Mat cWQe[NB], cWVe[NB], cWk[NB], cWo[NB], cW1[NB], cW2[NB], cA_A[NB], cB_A[NB], cA_V[NB], cB_V[NB];
    vector<IntFp> cbq[NB], cbk[NB], cbv[NB], cbo[NB], cb1f[NB], cb2[NB], cg1[NB], cb1n[NB], cg2[NB], cb2n[NB];
    for (int b = 0; b < NB; ++b) {
      cWQe[b] = cMatI(d, d, [&](int a, int c) { return WQe_q[b][a][c]; });
      cWVe[b] = cMatI(d, d, [&](int a, int c) { return WVe_q[b][a][c]; });
      cWk[b] = cMat(d, d, [&](int a, int c) { return Wk[b][a][c]; });
      cWo[b] = cMat(d, d, [&](int a, int c) { return Wo[b][a][c]; });
      cW1[b] = cMat(ffn, d, [&](int a, int c) { return W1[b][a][c]; });
      cW2[b] = cMat(d, ffn, [&](int a, int c) { return W2[b][a][c]; });
      cbq[b] = cVec(d, [&](int k) { return bq[b][k]; }); cbk[b] = cVec(d, [&](int k) { return bk[b][k]; });
      cbv[b] = cVec(d, [&](int k) { return bv[b][k]; }); cbo[b] = cVec(d, [&](int k) { return bo[b][k]; });
      cb1f[b] = cVec(ffn, [&](int k) { return b1[b][k]; }); cb2[b] = cVec(d, [&](int k) { return b2[b][k]; });
      cg1[b] = cVec(d, [&](int k) { return g1[b][k]; }); cb1n[b] = cVec(d, [&](int k) { return b1n[b][k]; });
      cg2[b] = cVec(d, [&](int k) { return g2[b][k]; }); cb2n[b] = cVec(d, [&](int k) { return b2n[b][k]; });
      cA_A[b] = cMat(Rk, d, [&](int k, int j) { return A_A[b][k][j]; }); cB_A[b] = cMat(d, Rk, [&](int i, int k) { return B_A[b][i][k]; });
      cA_V[b] = cMat(Rk, d, [&](int k, int j) { return A_V[b][k][j]; }); cB_V[b] = cMat(d, Rk, [&](int i, int k) { return B_V[b][i][k]; });
    }
    Mat cWe = cMat(vocab, d, [&](int v, int k) { return We[v][k]; });
    vector<IntFp> cbe = cVec(vocab, [&](int v) { return be[v]; });

    // ================= PROVEN FORWARD (chain of blocks) =================
    Mat H = cx;
    for (int b = 0; b < NB; ++b) {
      ln1[b]->save_intermediates = ln2[b]->save_intermediates = attn[b]->save_intermediates = block[b]->save_intermediates = true;
      ln1[b]->clear_cache(); ln2[b]->clear_cache(); attn[b]->clear_cache(); block[b]->clear_cache();
      Mat Ho;
      block[b]->forward(Ho, H, cg1[b], cb1n[b], cg2[b], cb2n[b], cWQe[b], cbq[b], cWk[b], cbk[b], cWVe[b], cbv[b], cWo[b], cbo[b], cW1[b], cb1f[b], cW2[b], cb2[b]);
      H = Ho;
    }
    const int last = s - 1;
    // LM head + loss
    vector<IntFp> logits; qi.dense_layer(logits, cWe, cbe, H[last]);
    vector<IntFp> p = sm.forward(logits);
    vector<IntFp> dLdlogits; sbk.cross_entropy(dLdlogits, p, true_token);
    vector<IntFp> dLdHlast, dLdWe, dLdbe;
    lmb.forward(dLdHlast, dLdWe, dLdbe, dLdlogits, H[last], cWe, (uint64_t)vocab, (uint64_t)d);
    printf("[%d] proven forward (%d blocks) + loss head done\n", party, NB);

    // ================= BACKWARD CHAIN (reverse) =================
    const IntFp zero(0ULL, PUBLIC);
    Mat dLdH(s, vector<IntFp>(d, zero));
    for (int k = 0; k < d; ++k) dLdH[last][k] = dLdHlast[k];
    Mat up_snap[NB];                    // gadget upstream INTO each block (for the tight reference)
    vector<IntFp> dWq_fB[NB], dWv_fB[NB], dAAp[NB], dBAp[NB], dAVp[NB], dBVp[NB];
    auto flatMat = [&](const Mat &M, int rows, int cols) { vector<IntFp> f(rows * cols); for (int a = 0; a < rows; ++a) for (int c = 0; c < cols; ++c) f[a * cols + c] = M[a][c]; return f; };
    for (int b = NB - 1; b >= 0; --b) {
      up_snap[b] = dLdH; // snapshot upstream into block b
      Mat &c1 = ln1[b]->cache_mat["c"], &c2 = ln2[b]->cache_mat["c"];
      vector<IntFp> &dinv1 = ln1[b]->cache_scalar_vec["dinv"], &dinv2 = ln2[b]->cache_scalar_vec["dinv"];
      Mat &Q = attn[b]->cache["Q"], &K = attn[b]->cache["K"], &V = attn[b]->cache["V"], &ctx = attn[b]->cache["ctx"], &probsF = attn[b]->cache["probs"];
      Mat &hidden = block[b]->cache["hidden"], &n2 = block[b]->cache["normed2"], &normed = block[b]->cache["normed"], &geluout = block[b]->cache["geluout"];
      vector<vector<vector<IntFp>>> probs(s, vector<vector<IntFp>>(h));
      for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) probs[i][hh] = probsF[i * h + hh];

      // FFN backward (existing gadget; owns per-block GELUBackward)
      FFNBackward fnb(&qi, gelu_bwd[b]);
      Mat dLdn2; vector<IntFp> dW1, db1, dW2, db2;
      fnb.forward(dLdn2, dW1, db1, dW2, db2, dLdH, n2, hidden, geluout, cW1[b], cW2[b], (uint64_t)d, (uint64_t)ffn);
      // LN2 backward + residual
      Mat dLda(s, vector<IntFp>(d, zero)); vector<IntFp> dg2(d, zero), db2n(d, zero);
      for (int i = 0; i < s; ++i) { vector<IntFp> dxi, dgi, dbi; lnb.forward(dxi, dgi, dbi, dLdn2[i], c2[i], cg2[b], dinv2[i], (uint64_t)d); for (int k = 0; k < d; ++k) { dLda[i][k] = dLdH[i][k] + dxi[k]; dg2[k] = dg2[k] + dgi[k]; db2n[k] = db2n[k] + dbi[k]; } }
      // attention backward (effective weights) -> dL/dnormed, dL/dW_Q_eff, dL/dW_V_eff
      Mat dLdnormed, dWq, dWk, dWv, dQ, dK, dV; vector<IntFp> dWq_f, dWk_f, dWv_f, dWo_f;
      atb.forward_full(dLdnormed, dWq, dWk, dWv, dWq_f, dWk_f, dWv_f, dQ, dK, dV, dWo_f, dLda, Q, K, V, probs, ctx, cWo[b], cWQe[b], cWk[b], cWVe[b], normed, (uint64_t)d);
      dWq_fB[b] = dWq_f; dWv_fB[b] = dWv_f;
      // LoRA gradient decomposition (no alpha/Rk factor, per spec)
      vector<IntFp> A_A_flat = flatMat(cA_A[b], Rk, d), B_A_flat = flatMat(cB_A[b], d, Rk);
      vector<IntFp> A_V_flat = flatMat(cA_V[b], Rk, d), B_V_flat = flatMat(cB_V[b], d, Rk);
      vector<IntFp> dBA, dAA, dBV, dAV;
      matmul_backward(ipp, dBA, dAA, B_A_flat, A_A_flat, dWq_f, (int)d, (int)Rk, (int)d);
      qi.requant_cm(dBAp[b], dBA, 1, P); qi.requant_cm(dAAp[b], dAA, 1, P);
      matmul_backward(ipp, dBV, dAV, B_V_flat, A_V_flat, dWv_f, (int)d, (int)Rk, (int)d);
      qi.requant_cm(dBVp[b], dBV, 1, P); qi.requant_cm(dAVp[b], dAV, 1, P);
      // LN1 backward + residual -> dLdx = upstream for block b-1
      Mat dLdx(s, vector<IntFp>(d, zero)); vector<IntFp> dg1(d, zero), db1n(d, zero);
      for (int i = 0; i < s; ++i) { vector<IntFp> dxi, dgi, dbi; lnb.forward(dxi, dgi, dbi, dLdnormed[i], c1[i], cg1[b], dinv1[i], (uint64_t)d); for (int k = 0; k < d; ++k) { dLdx[i][k] = dLda[i][k] + dxi[k]; dg1[k] = dg1[k] + dgi[k]; db1n[k] = db1n[k] + dbi[k]; } }
      dLdH = dLdx;
    }
    Mat final_dLdH = dLdH; // block 0's dLdx (grad w.r.t. model input)

    // ================= SGD on ADAPTERS ONLY (all blocks) =================
    auto toMat = [&](const vector<IntFp> &f, int rows, int cols) { Mat M(rows, vector<IntFp>(cols)); for (int a = 0; a < rows; ++a) for (int c = 0; c < cols; ++c) M[a][c] = f[a * cols + c]; return M; };
    Mat AAn[NB], BAn[NB], AVn[NB], BVn[NB];
    for (int b = 0; b < NB; ++b) {
      sgd.forward_mat(AAn[b], cA_A[b], toMat(dAAp[b], Rk, d));
      sgd.forward_mat(BAn[b], cB_A[b], toMat(dBAp[b], d, Rk));
      sgd.forward_mat(AVn[b], cA_V[b], toMat(dAVp[b], Rk, d));
      sgd.forward_mat(BVn[b], cB_V[b], toMat(dBVp[b], d, Rk));
    }
    printf("[%d] backward chain + LoRA decomp + adapter SGD done\n", party);

    // ================= REVEAL =================
    auto rv = [&](IntFp &z) { return FixedInfer::decode_signed(z.reveal()); };
    // per-block forward intermediates + upstream (for the tight reference)
    static double nrmd[NB][s][d], Qr[NB][s][d], Kr[NB][s][d], Vr[NB][s][d], ctxr[NB][s][d], c1r[NB][s][d], c2r[NB][s][d], n2r[NB][s][d];
    static double hidr[NB][s][ffn], goutr[NB][s][ffn], d1r[NB][s], d2r[NB][s], upq[NB][s][d], finq[s][d], prb[NB][s][h][s];
    for (int b = 0; b < NB; ++b) {
      Mat &c1 = ln1[b]->cache_mat["c"], &c2 = ln2[b]->cache_mat["c"];
      vector<IntFp> &dinv1 = ln1[b]->cache_scalar_vec["dinv"], &dinv2 = ln2[b]->cache_scalar_vec["dinv"];
      Mat &Q = attn[b]->cache["Q"], &K = attn[b]->cache["K"], &V = attn[b]->cache["V"], &ctx = attn[b]->cache["ctx"], &probsF = attn[b]->cache["probs"];
      Mat &normed = block[b]->cache["normed"], &n2 = block[b]->cache["normed2"], &hidden = block[b]->cache["hidden"], &geluout = block[b]->cache["geluout"];
      for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) { nrmd[b][i][k] = rv(normed[i][k]) / SCALE; Qr[b][i][k] = rv(Q[i][k]) / SCALE; Kr[b][i][k] = rv(K[i][k]) / SCALE; Vr[b][i][k] = rv(V[i][k]) / SCALE; ctxr[b][i][k] = rv(ctx[i][k]) / (SCALE * SCALE); c1r[b][i][k] = rv(c1[i][k]) / SCALE; c2r[b][i][k] = rv(c2[i][k]) / SCALE; n2r[b][i][k] = rv(n2[i][k]) / SCALE; upq[b][i][k] = rv(up_snap[b][i][k]) / SCALE; }
      for (int i = 0; i < s; ++i) for (int j = 0; j < ffn; ++j) { hidr[b][i][j] = rv(hidden[i][j]) / SCALE; goutr[b][i][j] = rv(geluout[i][j]) / SCALE; }
      for (int i = 0; i < s; ++i) { d1r[b][i] = rv(dinv1[i]) / SCALE; d2r[b][i] = rv(dinv2[i]) / SCALE; }
      for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) for (int j = 0; j <= i; ++j) prb[b][i][hh][j] = rv(probsF[i * h + hh][j]) / SCALE;
    }
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) finq[i][k] = rv(final_dLdH[i][k]) / SCALE;
    // effective-weight grads + adapter grads/updates
    static int64_t dWqf_q[NB][d * d], dWvf_q[NB][d * d];
    static int64_t dAAp_q[NB][Rk * d], dBAp_q[NB][d * Rk], dAVp_q[NB][Rk * d], dBVp_q[NB][d * Rk];
    static int64_t AAn_q[NB][Rk * d], BAn_q[NB][d * Rk], AVn_q[NB][Rk * d], BVn_q[NB][d * Rk];
    for (int b = 0; b < NB; ++b) {
      for (int t = 0; t < d * d; ++t) { dWqf_q[b][t] = rv(dWq_fB[b][t]); dWvf_q[b][t] = rv(dWv_fB[b][t]); }
      for (int t = 0; t < Rk * d; ++t) { dAAp_q[b][t] = rv(dAAp[b][t]); dAVp_q[b][t] = rv(dAVp[b][t]); }
      for (int t = 0; t < d * Rk; ++t) { dBAp_q[b][t] = rv(dBAp[b][t]); dBVp_q[b][t] = rv(dBVp[b][t]); }
      for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { AAn_q[b][k * d + j] = rv(AAn[b][k][j]); AVn_q[b][k * d + j] = rv(AVn[b][k][j]); }
      for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { BAn_q[b][i * Rk + k] = rv(BAn[b][i][k]); BVn_q[b][i * Rk + k] = rv(BVn[b][i][k]); }
    }
    // loss head
    int64_t logits_q[vocab], p_q[vocab], dl_q[vocab];
    for (int v = 0; v < vocab; ++v) { logits_q[v] = rv(logits[v]); p_q[v] = rv(p[v]); dl_q[v] = rv(dLdlogits[v]); }

    // ================= CHECK (ALICE) =================
    if (party == ALICE) {
      // --- loss head: softmax sanity + dLdlogits exact ---
      double mx = -1e30, sum = 0, pf[vocab];
      for (int v = 0; v < vocab; ++v) if (logits_q[v] / SCALE > mx) mx = logits_q[v] / SCALE;
      for (int v = 0; v < vocab; ++v) { pf[v] = std::exp(logits_q[v] / SCALE - mx); sum += pf[v]; }
      for (int v = 0; v < vocab; ++v) pf[v] /= sum;
      for (int v = 0; v < vocab; ++v) { if (std::fabs(p_q[v] / SCALE - pf[v]) > 0.03) { printf("[%d] p[%d] MISMATCH\n", party, v); ok = false; } int64_t ce = p_q[v] - ((std::size_t)v == true_token ? (1LL << P) : 0); if (dl_q[v] != ce) { printf("[%d] dLdlogits[%d] MISMATCH\n", party, v); ok = false; } }

      vector<uint64_t> ggt = build_gelu_grad_table(GT, P, GOFF);
      auto gg = [&](double hq) { int64_t xq = llround(hq * SCALE); return FixedInfer::decode_signed(ggt[(size_t)(xq + GOFF)]) / SCALE; };
      auto ln_bwd = [&](const double *dLdy, const double *cc, double dinv, const double *g, double *dLdin, double *dg) { double xh[d], gg2[d]; for (int k = 0; k < d; ++k) { xh[k] = cc[k] * dinv; gg2[k] = dLdy[k] * g[k]; } double e1 = 0, e2 = 0; for (int k = 0; k < d; ++k) { e1 += gg2[k] * xh[k]; e2 += gg2[k]; } e1 /= d; e2 /= d; for (int k = 0; k < d; ++k) { dLdin[k] = dinv * (gg2[k] - e2 - xh[k] * e1); dg[k] += dLdy[k] * xh[k]; } };
      // per-block reference backward using this block's REVEALED upstream + intermediates
      auto blk_ref = [&](int b, double dO[s][d], double gWqr[d][d], double gWvr[d][d], double gX[s][d]) {
        static double gN2[s][d], gHid[s][ffn]; double dGr[ffn]; memset(gN2, 0, sizeof(gN2));
        for (int i = 0; i < s; ++i) { for (int p2 = 0; p2 < ffn; ++p2) { double g = 0; for (int k = 0; k < d; ++k) g += qd(W2[b][k][p2]) * dO[i][k]; dGr[p2] = g; } for (int p2 = 0; p2 < ffn; ++p2) gHid[i][p2] = dGr[p2] * gg(hidr[b][i][p2]); for (int c = 0; c < d; ++c) { double g = 0; for (int p2 = 0; p2 < ffn; ++p2) g += qd(W1[b][p2][c]) * gHid[i][p2]; gN2[i][c] = g; } }
        double g2v[d], g1v[d]; for (int k = 0; k < d; ++k) { g2v[k] = qd(g2[b][k]); g1v[k] = qd(g1[b][k]); }
        static double gDa[s][d]; double junk[d] = {}; for (int i = 0; i < s; ++i) ln_bwd(gN2[i], c2r[b][i], d2r[b][i], g2v, gDa[i], junk);
        static double gA[s][d]; for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) gA[i][k] = dO[i][k] + gDa[i][k];
        static double gQ[s][d], gK[s][d], gV[s][d], gctx[s][d]; memset(gQ, 0, sizeof(gQ)); memset(gK, 0, sizeof(gK)); memset(gV, 0, sizeof(gV));
        for (int i = 0; i < s; ++i) for (int c = 0; c < d; ++c) { double g = 0; for (int aa = 0; aa < d; ++aa) g += qd(Wo[b][aa][c]) * gA[i][aa]; gctx[i][c] = g; }
        for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) { int off = hh * dh, L = i + 1; double gp[s]; for (int j = 0; j < L; ++j) { double g = 0; for (int k = 0; k < dh; ++k) g += gctx[i][off + k] * Vr[b][j][off + k]; gp[j] = g; } for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gV[j][off + k] += prb[b][i][hh][j] * gctx[i][off + k]; double sd = 0; for (int j = 0; j < L; ++j) sd += prb[b][i][hh][j] * gp[j]; double gd[s]; for (int j = 0; j < L; ++j) gd[j] = prb[b][i][hh][j] * (gp[j] - sd) * asc; for (int k = 0; k < dh; ++k) { double g = 0; for (int j = 0; j < L; ++j) g += gd[j] * Kr[b][j][off + k]; gQ[i][off + k] += g; } for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gK[j][off + k] += gd[j] * Qr[b][i][off + k]; }
        static double gNr[s][d];
        for (int i = 0; i < s; ++i) for (int c = 0; c < d; ++c) { double g = 0; for (int aa = 0; aa < d; ++aa) g += (WQe_q[b][aa][c] / SCALE) * gQ[i][aa] + qd(Wk[b][aa][c]) * gK[i][aa] + (WVe_q[b][aa][c] / SCALE) * gV[i][aa]; gNr[i][c] = g; }
        for (int i = 0; i < s; ++i) for (int aa = 0; aa < d; ++aa) for (int c = 0; c < d; ++c) { gWqr[aa][c] += gQ[i][aa] * nrmd[b][i][c]; gWvr[aa][c] += gV[i][aa] * nrmd[b][i][c]; }
        static double gDx[s][d]; for (int i = 0; i < s; ++i) ln_bwd(gNr[i], c1r[b][i], d1r[b][i], g1v, gDx[i], junk);
        for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) gX[i][k] = gA[i][k] + gDx[i][k];
      };

      const double tolB = 0.06, tolA = 2.0 / SCALE;
      auto near = [&](int64_t got, double ref, double tol) { return std::fabs(got / SCALE - ref) <= tol; };
      for (int b = NB - 1; b >= 0; --b) {
        static double gWqr[d][d], gWvr[d][d], gX[s][d]; memset(gWqr, 0, sizeof(gWqr)); memset(gWvr, 0, sizeof(gWvr));
        static double dO[s][d]; for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) dO[i][k] = upq[b][i][k];
        blk_ref(b, dO, gWqr, gWvr, gX);
        // block backward: effective-weight grads vs reference
        for (int a = 0; a < d; ++a) for (int c = 0; c < d; ++c) { if (!near(dWqf_q[b][a * d + c], gWqr[a][c], tolB)) { printf("[%d] blk%d dLdWQeff[%d][%d] %.4f vs %.4f\n", party, b, a, c, dWqf_q[b][a * d + c] / SCALE, gWqr[a][c]); ok = false; } if (!near(dWvf_q[b][a * d + c], gWvr[a][c], tolB)) { printf("[%d] blk%d dLdWVeff[%d][%d] %.4f vs %.4f\n", party, b, a, c, dWvf_q[b][a * d + c] / SCALE, gWvr[a][c]); ok = false; } }
        // chain link: reference dLdx (gX) == gadget's next-block upstream
        for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) { double tgt = (b > 0) ? upq[b - 1][i][k] : finq[i][k]; if (!near((int64_t)llround(gX[i][k] * SCALE), tgt, tolB)) { /* compare gadget-vs-ref both reals */ if (std::fabs(gX[i][k] - tgt) > tolB) { printf("[%d] blk%d chain dLdx[%d][%d] ref %.4f vs gadget %.4f\n", party, b, i, k, gX[i][k], tgt); ok = false; } } }

        // LoRA decomposition: EXACT replay from revealed dL/dW_eff (mirrors gadget) + float
        auto chk = [&](const char *nm, int64_t dWeff[d * d], const vector<vector<double>> &A, const vector<vector<double>> &B, int64_t dBp[], int64_t dAp[], int64_t An[], int64_t Bn[]) {
          for (int i = 0; i < d; ++i) for (int k = 0; k < Rk; ++k) { int64_t acc = 0; for (int j = 0; j < d; ++j) acc += dWeff[i * d + j] * q1(A[k][j]); int64_t ex = rha(acc, P); if (dBp[i * Rk + k] != ex) { printf("[%d] blk%d dLdB_%s[%d][%d] MISMATCH\n", party, b, nm, i, k); ok = false; } int64_t sgdx = q1(B[i][k]) - rha((int64_t)lr_q * dBp[i * Rk + k], P); if (Bn[i * Rk + k] != sgdx) { printf("[%d] blk%d B_%s_new[%d][%d] SGD MISMATCH\n", party, b, nm, i, k); ok = false; } double fl = qd(B[i][k]) - ((double)lr_q / SCALE) * (dBp[i * Rk + k] / SCALE); if (std::fabs(Bn[i * Rk + k] / SCALE - fl) > tolA) { printf("[%d] blk%d B_%s_new[%d][%d] float\n", party, b, nm, i, k); ok = false; } }
          for (int k = 0; k < Rk; ++k) for (int j = 0; j < d; ++j) { int64_t acc = 0; for (int i = 0; i < d; ++i) acc += q1(B[i][k]) * dWeff[i * d + j]; int64_t ex = rha(acc, P); if (dAp[k * d + j] != ex) { printf("[%d] blk%d dLdA_%s[%d][%d] MISMATCH\n", party, b, nm, k, j); ok = false; } int64_t sgdx = q1(A[k][j]) - rha((int64_t)lr_q * dAp[k * d + j], P); if (An[k * d + j] != sgdx) { printf("[%d] blk%d A_%s_new[%d][%d] SGD MISMATCH\n", party, b, nm, k, j); ok = false; } double fl = qd(A[k][j]) - ((double)lr_q / SCALE) * (dAp[k * d + j] / SCALE); if (std::fabs(An[k * d + j] / SCALE - fl) > tolA) { printf("[%d] blk%d A_%s_new[%d][%d] float\n", party, b, nm, k, j); ok = false; } }
        };
        chk("A", dWqf_q[b], A_A[b], B_A[b], dBAp_q[b], dAAp_q[b], AAn_q[b], BAn_q[b]);
        chk("V", dWvf_q[b], A_V[b], B_V[b], dBVp_q[b], dAVp_q[b], AVn_q[b], BVn_q[b]);
      }
      printf("[%d] %d-BLOCK LoRA TRAIN STEP checked: loss head exact + per-block chain float(<%.3f) + adapter decomp/SGD exact + float(<%.4f)\n", party, NB, tolB, tolA);
    }

    for (int b = 0; b < NB; ++b) gelu_bwd[b]->Teardown();
    gtab.Teardown(); expt.Teardown(); rclu.Teardown(); divr_sm.destroy(); divr_ln.destroy(); qi.destroy();
  }
  finalize_zk_arith();
  auto dt = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

  if (party == ALICE) { printf("[timing] ZK session %.2fs\n", dt); printf("%s\n", ok ? "pass multiblock_lora_train_step (ml-fixed)" : "FAIL multiblock_lora_train_step (ml-fixed)"); }
  else printf("verifier: ml-fixed multiblock_lora_train_step proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
