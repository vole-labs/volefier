// Test: ml-fixed multi-head causal self-attention backward (AttentionBackward).
// Retained forward tensors Q,K,V (2^-P), per (token,head) softmax probs (2^-P),
// ctx (2^-2P), Wo, and upstream dL/dOut (2^-P) are committed directly; the gadget
// produces dL/dH, dL/dQ, dL/dK, dL/dV (2^-P) and dL/dWo (flat, 2^-P).
// Outputs are checked EXACTLY against a fixed-point replay (same round-half-up
// requant + causal mask) and cross-checked against a float attention-backward
// reference within tolerance.  Setup mirrors test/ml-fixed/gelu_backward.cpp.
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

// dimensions
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
    const uint64_t qinv = (uint64_t)llround((1.0 / std::sqrt((double)dh)) * SCALE); // dh=4 -> 64

    // -------- deterministic plaintext generation (identical on both parties) --------
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };

    // reals
    double Qr[s][d], Kr[s][d], Vr[s][d], dOr[s][d], ctxr[s][d], Wor[d][d];
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k) {
        Qr[i][k] = ur(-1, 1); Kr[i][k] = ur(-1, 1); Vr[i][k] = ur(-1, 1);
        dOr[i][k] = ur(-1, 1); ctxr[i][k] = ur(-1, 1);
      }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) Wor[a][b] = ur(-0.5, 0.5);
    // probs[i][hh] length i+1, normalized (softmax-like)
    vector<vector<vector<double>>> pr(s, vector<vector<double>>(h));
    for (int i = 0; i < s; ++i)
      for (int hh = 0; hh < h; ++hh) {
        pr[i][hh].resize(i + 1);
        double sum = 0;
        for (int j = 0; j <= i; ++j) { pr[i][hh][j] = ur(0.05, 1.0); sum += pr[i][hh][j]; }
        for (int j = 0; j <= i; ++j) pr[i][hh][j] /= sum;
      }

    // quantized ints (Q,K,V,dOut,Wo @ 2^-P ; ctx @ 2^-2P ; probs @ 2^-P)
    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };
    auto q2 = [&](double v) { return (int64_t)llround(v * SCALE * SCALE); };
    vector<vector<int64_t>> Qi(s, vector<int64_t>(d)), Ki(s, vector<int64_t>(d)),
        Vi(s, vector<int64_t>(d)), dOi(s, vector<int64_t>(d)), ctxi(s, vector<int64_t>(d));
    vector<vector<int64_t>> Wi(d, vector<int64_t>(d));
    for (int i = 0; i < s; ++i)
      for (int k = 0; k < d; ++k) {
        Qi[i][k] = q1(Qr[i][k]); Ki[i][k] = q1(Kr[i][k]); Vi[i][k] = q1(Vr[i][k]);
        dOi[i][k] = q1(dOr[i][k]); ctxi[i][k] = q2(ctxr[i][k]);
      }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) Wi[a][b] = q1(Wor[a][b]);
    vector<vector<vector<int64_t>>> pri(s, vector<vector<int64_t>>(h));
    for (int i = 0; i < s; ++i)
      for (int hh = 0; hh < h; ++hh) {
        pri[i][hh].resize(i + 1);
        for (int j = 0; j <= i; ++j) pri[i][hh][j] = q1(pr[i][hh][j]);
      }

    // ============ EXACT fixed-point reference (mirror of the gadget) ============
    auto rhu = [&](int64_t N, int M) -> int64_t { return (N + (1LL << (M - 1))) >> M; }; // round-half-up by 2^M
    vector<vector<int64_t>> dQref(s, vector<int64_t>(d, 0)), dKref(s, vector<int64_t>(d, 0)),
        dVref(s, vector<int64_t>(d, 0)), dHref(s, vector<int64_t>(d, 0));
    vector<int64_t> dWoref(d * d, 0);
    // STEP 1
    vector<vector<int64_t>> dCtxr(s, vector<int64_t>(d, 0)); // 2^-2P
    vector<vector<int64_t>> dWoacc(d, vector<int64_t>(d, 0)); // 2^-3P
    for (int i = 0; i < s; ++i) {
      for (int b = 0; b < d; ++b) { int64_t acc = 0; for (int a = 0; a < d; ++a) acc += Wi[a][b] * dOi[i][a]; dCtxr[i][b] = acc; }
      for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) dWoacc[a][b] += dOi[i][a] * ctxi[i][b];
    }
    // STEP 2
    for (int i = 0; i < s; ++i) {
      int L = i + 1;
      for (int hh = 0; hh < h; ++hh) {
        int off = hh * dh;
        // 2a
        vector<int64_t> dCtx_h(dh); for (int k = 0; k < dh; ++k) dCtx_h[k] = dCtxr[i][off + k];
        vector<int64_t> dprobsP(L), dprobs_h(L);
        for (int j = 0; j < L; ++j) { int64_t acc = 0; for (int k = 0; k < dh; ++k) acc += dCtx_h[k] * Vi[j][off + k]; dprobs_h[j] = acc; dprobsP[j] = rhu(acc, 2 * P); }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) dVref[j][off + k] += rhu(pri[i][hh][j] * dCtx_h[k], 2 * P);
        // 2b softmax general
        int64_t dot2p = 0; for (int j = 0; j < L; ++j) dot2p += pri[i][hh][j] * dprobsP[j];
        int64_t dotP = rhu(dot2p, P);
        vector<int64_t> dScores(L);
        for (int j = 0; j < L; ++j) dScores[j] = rhu(pri[i][hh][j] * (dprobsP[j] - dotP), P);
        // 2c undo 1/sqrt(dh)
        vector<int64_t> dSc(L); for (int j = 0; j < L; ++j) dSc[j] = rhu(dScores[j] * (int64_t)qinv, P);
        // 2d Q@Kt
        for (int k = 0; k < dh; ++k) { int64_t acc = 0; for (int j = 0; j < L; ++j) acc += dSc[j] * Ki[j][off + k]; dQref[i][off + k] += rhu(acc, P); }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) dKref[j][off + k] += rhu(Qi[i][off + k] * dSc[j], P);
      }
    }
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) dWoref[a * d + b] = rhu(dWoacc[a][b], 2 * P);
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) dHref[i][k] = dQref[i][k] + dKref[i][k] + dVref[i][k];

    // ============ FLOAT reference (true attention backward, sanity cross-check) ============
    double gQ[s][d] = {}, gK[s][d] = {}, gV[s][d] = {}, gWo[d][d] = {};
    double gctx[s][d];
    for (int i = 0; i < s; ++i) for (int b = 0; b < d; ++b) { double a = 0; for (int aa = 0; aa < d; ++aa) a += Wor[aa][b] * dOr[i][aa]; gctx[i][b] = a; }
    const double dsc = (double)qinv / SCALE; // the quantized 1/sqrt(dh) actually applied
    for (int i = 0; i < s; ++i) {
      int L = i + 1;
      for (int hh = 0; hh < h; ++hh) {
        int off = hh * dh;
        vector<double> gp(L);
        for (int j = 0; j < L; ++j) { double a = 0; for (int k = 0; k < dh; ++k) a += gctx[i][off + k] * Vr[j][off + k]; gp[j] = a; }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gV[j][off + k] += pr[i][hh][j] * gctx[i][off + k];
        double sdot = 0; for (int j = 0; j < L; ++j) sdot += pr[i][hh][j] * gp[j];
        vector<double> gd(L);
        for (int j = 0; j < L; ++j) gd[j] = pr[i][hh][j] * (gp[j] - sdot) * dsc;
        for (int k = 0; k < dh; ++k) { double a = 0; for (int j = 0; j < L; ++j) a += gd[j] * Kr[j][off + k]; gQ[i][off + k] += a; }
        for (int j = 0; j < L; ++j) for (int k = 0; k < dh; ++k) gK[j][off + k] += gd[j] * Qr[i][off + k];
      }
    }
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) { double v = 0; for (int i = 0; i < s; ++i) v += dOr[i][a] * ctxr[i][b]; gWo[a][b] = v; }

    // -------- commit retained forward tensors + upstream gradient --------
    auto commitMat = [&](const vector<vector<int64_t>> &m) { Mat c(m.size()); for (size_t i = 0; i < m.size(); ++i) qi.commit_vec(c[i], m[i]); return c; };
    Mat cQ = commitMat(Qi), cK = commitMat(Ki), cV = commitMat(Vi), cdO = commitMat(dOi), cctx = commitMat(ctxi), cWo = commitMat(Wi);
    vector<vector<vector<IntFp>>> cprobs(s, vector<vector<IntFp>>(h));
    for (int i = 0; i < s; ++i) for (int hh = 0; hh < h; ++hh) qi.commit_vec(cprobs[i][hh], pri[i][hh]);
    printf("[%d] committed Q,K,V,probs,ctx,Wo,dLdOut\n", party);

    // -------- run attention backward --------
    AttentionBackward ab(&qi, nullptr, h, dh, qinv, P); // sm unused (SoftmaxBackward internal)
    Mat dLdH, dLdQ, dLdK, dLdV;
    vector<IntFp> dLdWo;
    ab.forward(dLdH, dLdQ, dLdK, dLdV, dLdWo, cdO, cQ, cK, cV, cprobs, cctx, cWo, (uint64_t)d);
    printf("[%d] attention backward done\n", party);

    // -------- reveal + check --------
    auto check_mat = [&](const char *nm, Mat &G, const vector<vector<int64_t>> &Rref, const double Gf[s][d]) {
      for (int i = 0; i < s; ++i)
        for (int k = 0; k < d; ++k) {
          int64_t got = FixedInfer::decode_signed(G[i][k].reveal());
          if (party == ALICE) {
            if (got != Rref[i][k]) { printf("[%d] %s[%d][%d]: got %ld exp %ld MISMATCH\n", party, nm, i, k, (long)got, (long)Rref[i][k]); ok = false; }
            double fdiff = std::fabs(got / SCALE - Gf[i][k]);
            if (fdiff > 0.05) { printf("[%d] %s[%d][%d]: float diff %.4f (got %.4f vs %.4f)\n", party, nm, i, k, fdiff, got / SCALE, Gf[i][k]); ok = false; }
          }
        }
    };
    check_mat("dLdQ", dLdQ, dQref, gQ);
    check_mat("dLdK", dLdK, dKref, gK);
    check_mat("dLdV", dLdV, dVref, gV);
    // dLdH = dLdQ+dLdK+dLdV
    { double gH[s][d]; for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) gH[i][k] = gQ[i][k] + gK[i][k] + gV[i][k]; check_mat("dLdH", dLdH, dHref, gH); }
    // dLdWo (flat)
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        int64_t got = FixedInfer::decode_signed(dLdWo[a * d + b].reveal());
        if (party == ALICE) {
          if (got != dWoref[a * d + b]) { printf("[%d] dLdWo[%d][%d]: got %ld exp %ld MISMATCH\n", party, a, b, (long)got, (long)dWoref[a * d + b]); ok = false; }
          double fdiff = std::fabs(got / SCALE - gWo[a][b]);
          if (fdiff > 0.05) { printf("[%d] dLdWo[%d][%d]: float diff %.4f\n", party, a, b, fdiff); ok = false; }
        }
      }
    if (party == ALICE) printf("[%d] all gradients checked (exact fixed-point + float within 0.05)\n", party);

    qi.destroy(); // flush LogUp range-check batches BEFORE finalize_zk_arith
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass attention_backward (ml-fixed)" : "FAIL attention_backward (ml-fixed)");
  else printf("verifier: ml-fixed attention_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
