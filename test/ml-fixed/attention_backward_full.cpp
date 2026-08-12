// Test: ml-fixed attention backward THROUGH the Q/K/V projections
// (AttentionBackward::forward_full).  Same inner attention backward as
// attention_backward.cpp, but forward_full also backprops Q[i]=Wq@H_normed[i]
// etc., producing the TRUE dL/dH = requant_P(Wq^T dQ + Wk^T dK + Wv^T dV) and the
// projection weight grads dL/dWq,dWk,dWv.  All outputs checked EXACTLY against a
// fixed-point replay and cross-checked against a float reference within tolerance.
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-fixed/fixed_backward.h"
using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
const int P = getenv("P") ? atoi(getenv("P")) : 7;
static const int s = 4, d = 8, h = 2, dh = 4;

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

  bool ok = true;
  {
    FixedInfer qi(party, P, ios[0], 32, threads);
    const double SCALE = static_cast<double>(1u << P);
    const uint64_t qinv = (uint64_t)llround((1.0 / std::sqrt((double)dh)) * SCALE);

    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };

    double Qr[s][d], Kr[s][d], Vr[s][d], dOr[s][d], ctxr[s][d], Wor[d][d];
    double Wqr[d][d], Wkr[d][d], Wvr[d][d], Hnr[s][d];
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k) {
        Qr[i][k] = ur(-1, 1); Kr[i][k] = ur(-1, 1); Vr[i][k] = ur(-1, 1);
        dOr[i][k] = ur(-1, 1); ctxr[i][k] = ur(-1, 1); Hnr[i][k] = ur(-1, 1);
      }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) { Wor[a][b] = ur(-0.5, 0.5); Wqr[a][b] = ur(-0.5, 0.5); Wkr[a][b] = ur(-0.5, 0.5); Wvr[a][b] = ur(-0.5, 0.5); }
    vector<vector<vector<double>>> pr(s, vector<vector<double>>(h));
    for (int i = 0; i < s; ++i)
      for (int hh = 0; hh < h; ++hh) {
        pr[i][hh].resize(i + 1);
        double sum = 0;
        for (int j = 0; j <= i; ++j) { pr[i][hh][j] = ur(0.05, 1.0); sum += pr[i][hh][j]; }
        for (int j = 0; j <= i; ++j) pr[i][hh][j] /= sum;
      }

    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };
    auto q2 = [&](double v) { return (int64_t)llround(v * SCALE * SCALE); };
    vector<vector<int64_t>> Qi(s, vector<int64_t>(d)), Ki(s, vector<int64_t>(d)), Vi(s, vector<int64_t>(d)),
        dOi(s, vector<int64_t>(d)), ctxi(s, vector<int64_t>(d)), Hni(s, vector<int64_t>(d));
    vector<vector<int64_t>> Wi(d, vector<int64_t>(d)), Wqi(d, vector<int64_t>(d)), Wki(d, vector<int64_t>(d)), Wvi(d, vector<int64_t>(d));
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k) {
        Qi[i][k] = q1(Qr[i][k]); Ki[i][k] = q1(Kr[i][k]); Vi[i][k] = q1(Vr[i][k]);
        dOi[i][k] = q1(dOr[i][k]); ctxi[i][k] = q2(ctxr[i][k]); Hni[i][k] = q1(Hnr[i][k]);
      }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) { Wi[a][b] = q1(Wor[a][b]); Wqi[a][b] = q1(Wqr[a][b]); Wki[a][b] = q1(Wkr[a][b]); Wvi[a][b] = q1(Wvr[a][b]); }
    vector<vector<vector<int64_t>>> pri(s, vector<vector<int64_t>>(h));
    for (int i = 0; i < s; ++i)
      for (int hh = 0; hh < h; ++hh) { pri[i][hh].resize(i + 1); for (int j = 0; j <= i; ++j) pri[i][hh][j] = q1(pr[i][hh][j]); }

    auto rha = [&](int64_t N, int M) -> int64_t { int64_t half = 1LL << (M - 1), Dn = 1LL << M; return (N >= 0) ? (N + half) / Dn : -(((-N) + half) / Dn); };

    // ===== EXACT replay: inner attention backward (identical to attention_backward.cpp) =====
    vector<vector<int64_t>> dQref(s, vector<int64_t>(d, 0)), dKref(s, vector<int64_t>(d, 0)), dVref(s, vector<int64_t>(d, 0));
    vector<int64_t> dWoref(d * d, 0);
    vector<vector<int64_t>> dCtxr(s, vector<int64_t>(d, 0)), dWoacc(d, vector<int64_t>(d, 0));
    for (int i = 0; i < s; ++i) {
      for (int b = 0; b < d; ++b) { int64_t acc = 0; for (int a = 0; a < d; ++a) acc += Wi[a][b] * dOi[i][a]; dCtxr[i][b] = acc; }
      for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) dWoacc[a][b] += dOi[i][a] * ctxi[i][b];
    }
    for (int i = 0; i < s; ++i) {
      int L = i + 1;
      for (int hh = 0; hh < h; ++hh) {
        int off = hh * dh;
        vector<int64_t> dCtx_h(dh); for (int k = 0; k < dh; ++k) dCtx_h[k] = dCtxr[i][off + k];
        vector<int64_t> dprobsP(L);
        for (int j = 0; j < L; ++j) { int64_t acc = 0; for (int k = 0; k < dh; ++k) acc += dCtx_h[k] * Vi[j][off + k]; dprobsP[j] = rha(acc, 2 * P); }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) dVref[j][off + k] += rha(pri[i][hh][j] * dCtx_h[k], 2 * P);
        int64_t dot2p = 0; for (int j = 0; j < L; ++j) dot2p += pri[i][hh][j] * dprobsP[j];
        int64_t dotP = rha(dot2p, P);
        vector<int64_t> dScores(L); for (int j = 0; j < L; ++j) dScores[j] = rha(pri[i][hh][j] * (dprobsP[j] - dotP), P);
        vector<int64_t> dSc(L); for (int j = 0; j < L; ++j) dSc[j] = rha(dScores[j] * (int64_t)qinv, P);
        for (int k = 0; k < dh; ++k) { int64_t acc = 0; for (int j = 0; j < L; ++j) acc += dSc[j] * Ki[j][off + k]; dQref[i][off + k] += rha(acc, P); }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) dKref[j][off + k] += rha(Qi[i][off + k] * dSc[j], P);
      }
    }
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) dWoref[a * d + b] = rha(dWoacc[a][b], 2 * P);

    // ===== EXACT replay: projection backward (uses the inner dQref/dKref/dVref) =====
    // dWq_acc[a][b] += dQ[i][a]*Hn[i][b] (2^-2P);  dH[i][b] = sum_a Wq[a][b]*dQ[i][a] (2^-2P)
    vector<vector<int64_t>> dHtrue(s, vector<int64_t>(d, 0)), dWqacc(d, vector<int64_t>(d, 0)), dWkacc(d, vector<int64_t>(d, 0)), dWvacc(d, vector<int64_t>(d, 0));
    for (int i = 0; i < s; ++i) {
      for (int b = 0; b < d; ++b) {
        int64_t hq = 0, hk = 0, hv = 0;
        for (int a = 0; a < d; ++a) { hq += Wqi[a][b] * dQref[i][a]; hk += Wki[a][b] * dKref[i][a]; hv += Wvi[a][b] * dVref[i][a]; }
        dHtrue[i][b] = rha(hq + hk + hv, P);
      }
      for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { dWqacc[a][b] += dQref[i][a] * Hni[i][b]; dWkacc[a][b] += dKref[i][a] * Hni[i][b]; dWvacc[a][b] += dVref[i][a] * Hni[i][b]; }
    }
    vector<int64_t> dWqref(d * d), dWkref(d * d), dWvref(d * d);
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { dWqref[a * d + b] = rha(dWqacc[a][b], P); dWkref[a * d + b] = rha(dWkacc[a][b], P); dWvref[a * d + b] = rha(dWvacc[a][b], P); }

    // ===== FLOAT reference =====
    double gQ[s][d] = {}, gK[s][d] = {}, gV[s][d] = {};
    double gctx[s][d];
    for (int i = 0; i < s; ++i) for (int b = 0; b < d; ++b) { double a = 0; for (int aa = 0; aa < d; ++aa) a += Wor[aa][b] * dOr[i][aa]; gctx[i][b] = a; }
    const double dsc = (double)qinv / SCALE;
    for (int i = 0; i < s; ++i) {
      int L = i + 1;
      for (int hh = 0; hh < h; ++hh) {
        int off = hh * dh;
        vector<double> gp(L);
        for (int j = 0; j < L; ++j) { double a = 0; for (int k = 0; k < dh; ++k) a += gctx[i][off + k] * Vr[j][off + k]; gp[j] = a; }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gV[j][off + k] += pr[i][hh][j] * gctx[i][off + k];
        double sdot = 0; for (int j = 0; j < L; ++j) sdot += pr[i][hh][j] * gp[j];
        vector<double> gd(L); for (int j = 0; j < L; ++j) gd[j] = pr[i][hh][j] * (gp[j] - sdot) * dsc;
        for (int k = 0; k < dh; ++k) { double a = 0; for (int j = 0; j < L; ++j) a += gd[j] * Kr[j][off + k]; gQ[i][off + k] += a; }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gK[j][off + k] += gd[j] * Qr[i][off + k];
      }
    }
    double gHf[s][d] = {}, gWqf[d][d] = {}, gWkf[d][d] = {}, gWvf[d][d] = {};
    for (int i = 0; i < s; ++i)
      for (int b = 0; b < d; ++b) { double a = 0; for (int aa = 0; aa < d; ++aa) a += Wqr[aa][b] * gQ[i][aa] + Wkr[aa][b] * gK[i][aa] + Wvr[aa][b] * gV[i][aa]; gHf[i][b] = a; }
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { double vq = 0, vk = 0, vv = 0; for (int i = 0; i < s; ++i) { vq += gQ[i][a] * Hnr[i][b]; vk += gK[i][a] * Hnr[i][b]; vv += gV[i][a] * Hnr[i][b]; } gWqf[a][b] = vq; gWkf[a][b] = vk; gWvf[a][b] = vv; }

    // ===== commit =====
    auto commitMat = [&](const vector<vector<int64_t>> &m) { Mat c(m.size()); for (size_t i = 0; i < m.size(); ++i) qi.commit_vec(c[i], m[i]); return c; };
    Mat cQ = commitMat(Qi), cK = commitMat(Ki), cV = commitMat(Vi), cdO = commitMat(dOi), cctx = commitMat(ctxi);
    Mat cWo = commitMat(Wi), cWq = commitMat(Wqi), cWk = commitMat(Wki), cWv = commitMat(Wvi), cHn = commitMat(Hni);
    vector<vector<vector<IntFp>>> cprobs(s, vector<vector<IntFp>>(h));
    for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) qi.commit_vec(cprobs[i][hh], pri[i][hh]);
    printf("[%d] committed Q,K,V,probs,ctx,Wo,Wq,Wk,Wv,H_normed,dLdOut\n", party);

    // ===== run forward_full =====
    AttentionBackward ab(&qi, nullptr, h, dh, qinv, P);
    Mat dLdH, dLdWq, dLdWk, dLdWv, dLdQ, dLdK, dLdV;
    vector<IntFp> dLdWq_f, dLdWk_f, dLdWv_f, dLdWo_f;
    ab.forward_full(dLdH, dLdWq, dLdWk, dLdWv, dLdWq_f, dLdWk_f, dLdWv_f, dLdQ, dLdK, dLdV, dLdWo_f,
                    cdO, cQ, cK, cV, cprobs, cctx, cWo, cWq, cWk, cWv, cHn, (uint64_t)d);
    printf("[%d] forward_full done\n", party);

    // ===== reveal + check =====
    auto check_mat = [&](const char *nm, Mat &G, const vector<vector<int64_t>> &Rref, const double Gf[s][d]) {
      for (int i = 0; i < s; ++i)
        for (int k = 0; k < d; ++k) {
          int64_t got = FixedInfer::decode_signed(G[i][k].reveal());
          if (party == ALICE) {
            if (got != Rref[i][k]) { printf("[%d] %s[%d][%d]: got %ld exp %ld MISMATCH\n", party, nm, i, k, (long)got, (long)Rref[i][k]); ok = false; }
            if (std::fabs(got / SCALE - Gf[i][k]) > 0.05) { printf("[%d] %s[%d][%d]: float diff %.4f\n", party, nm, i, k, std::fabs(got / SCALE - Gf[i][k])); ok = false; }
          }
        }
    };
    auto check_flat = [&](const char *nm, vector<IntFp> &G, const vector<int64_t> &Rref, const double Gf[d][d]) {
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
          int64_t got = FixedInfer::decode_signed(G[a * d + b].reveal());
          if (party == ALICE) {
            if (got != Rref[a * d + b]) { printf("[%d] %s[%d][%d]: got %ld exp %ld MISMATCH\n", party, nm, a, b, (long)got, (long)Rref[a * d + b]); ok = false; }
            if (std::fabs(got / SCALE - Gf[a][b]) > 0.05) { printf("[%d] %s[%d][%d]: float diff %.4f\n", party, nm, a, b, std::fabs(got / SCALE - Gf[a][b])); ok = false; }
          }
        }
    };
    // inner grads unchanged from forward()
    check_mat("dLdQ", dLdQ, dQref, gQ);
    check_mat("dLdK", dLdK, dKref, gK);
    check_mat("dLdV", dLdV, dVref, gV);
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { int64_t got = FixedInfer::decode_signed(dLdWo_f[a * d + b].reveal()); if (party == ALICE && got != dWoref[a * d + b]) { printf("[%d] dLdWo[%d][%d] MISMATCH got %ld exp %ld\n", party, a, b, (long)got, (long)dWoref[a * d + b]); ok = false; } }
    // NEW: true dLdH through projections, and dLdWq/k/v
    check_mat("dLdH", dLdH, dHtrue, gHf);
    check_flat("dLdWq", dLdWq_f, dWqref, gWqf);
    check_flat("dLdWk", dLdWk_f, dWkref, gWkf);
    check_flat("dLdWv", dLdWv_f, dWvref, gWvf);
    // Mat variant of dLdWq must equal its flat
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { int64_t m = FixedInfer::decode_signed(dLdWq[a][b].reveal()), f = FixedInfer::decode_signed(dLdWq_f[a * d + b].reveal()); if (party == ALICE && m != f) { printf("[%d] dLdWq Mat!=flat at [%d][%d]\n", party, a, b); ok = false; } }
    if (party == ALICE) printf("[%d] all forward_full gradients checked (exact + float within 0.05)\n", party);

    qi.destroy();
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass attention_backward_full (ml-fixed)" : "FAIL attention_backward_full (ml-fixed)");
  else printf("verifier: ml-fixed attention_backward_full proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
