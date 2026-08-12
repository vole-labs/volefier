// Test: ml-fixed FFN (GPT-2 MLP) backward (FFNBackward).
// Forward per token:  hidden = requant(W1 @ n2)+b1 ; geluout = GELU(hidden) ;
//                     ff = requant(W2 @ geluout)+b2.
// Given upstream dL/dff and retained n2, hidden, geluout, produce dL/dn2 and the
// parameter grads dL/dW1, dL/db1, dL/dW2, dL/db2.  All checked EXACTLY against a
// fixed-point replay (same round-half-away requant + GELU' table) and against a
// float reference within tolerance.  Setup mirrors attention_backward.cpp.
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
static const int s = 4, d = 8, ffn = 16;

// tanh-approx GELU (matches build_gelu_table / build_gelu_grad_table)
static double gelu_f(double x) {
  double k = std::sqrt(2.0 / M_PI);
  return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
}

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
    const uint64_t GT = 1u << (P + 7);
    const int64_t GOFF = 1LL << (P + 6);

    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };
    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };

    // reals
    double dffr[s][d], n2r[s][d], hidr[s][ffn], W1r[ffn][d], W2r[d][ffn];
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) { dffr[i][k] = ur(-1, 1); n2r[i][k] = ur(-1, 1); }
    for (int i = 0; i < s; ++i) for (int b = 0; b < ffn; ++b) hidr[i][b] = ur(-2, 2);
    for (int a = 0; a < ffn; ++a) for (int b = 0; b < d; ++b) W1r[a][b] = ur(-0.5, 0.5);
    for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) W2r[a][b] = ur(-0.5, 0.5);

    // quantized ints (all 2^-P); geluout = GELU(hidden)
    vector<vector<int64_t>> dffi(s, vector<int64_t>(d)), n2i(s, vector<int64_t>(d));
    vector<vector<int64_t>> hidi(s, vector<int64_t>(ffn)), gouti(s, vector<int64_t>(ffn));
    vector<vector<int64_t>> W1i(ffn, vector<int64_t>(d)), W2i(d, vector<int64_t>(ffn));
    for (int i = 0; i < s; ++i) for (int k = 0; k < d; ++k) { dffi[i][k] = q1(dffr[i][k]); n2i[i][k] = q1(n2r[i][k]); }
    for (int i = 0; i < s; ++i) for (int b = 0; b < ffn; ++b) { hidi[i][b] = q1(hidr[i][b]); gouti[i][b] = q1(gelu_f((double)hidi[i][b] / SCALE)); }
    for (int a = 0; a < ffn; ++a) for (int b = 0; b < d; ++b) W1i[a][b] = q1(W1r[a][b]);
    for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) W2i[a][b] = q1(W2r[a][b]);

    // GELU' table (same as GELUBackward uses) for the exact replay
    vector<uint64_t> ggt = build_gelu_grad_table(GT, P, GOFF);
    auto gg = [&](int64_t xq) { return FixedInfer::decode_signed(ggt[(size_t)(xq + GOFF)]); };
    auto rha = [&](int64_t N, int M) -> int64_t { int64_t half = 1LL << (M - 1), Dn = 1LL << M; return (N >= 0) ? (N + half) / Dn : -(((-N) + half) / Dn); };

    // ===== EXACT fixed-point replay =====
    vector<vector<int64_t>> dN2ref(s, vector<int64_t>(d, 0)), dHid(s, vector<int64_t>(ffn, 0));
    vector<vector<int64_t>> dW2acc(d, vector<int64_t>(ffn, 0)), dW1acc(ffn, vector<int64_t>(d, 0));
    vector<int64_t> db2ref(d, 0), db1ref(ffn, 0);
    for (int i = 0; i < s; ++i) {
      // W2 backward: dGelu[b] = sum_a W2[a][b]*dff[i][a]
      vector<int64_t> dGeluP(ffn);
      for (int b = 0; b < ffn; ++b) { int64_t acc = 0; for (int a = 0; a < d; ++a) acc += W2i[a][b] * dffi[i][a]; dGeluP[b] = rha(acc, P); }
      for (int j = 0; j < d; ++j) db2ref[j] += dffi[i][j];
      for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) dW2acc[a][b] += dffi[i][a] * gouti[i][b];
      // GELU backward: dHidden[b] = rha(dGeluP[b] * GELU'(hidden[b]), P)
      for (int b = 0; b < ffn; ++b) dHid[i][b] = rha(dGeluP[b] * gg(hidi[i][b]), P);
      for (int j = 0; j < ffn; ++j) db1ref[j] += dHid[i][j];
      // W1 backward: dN2[b] = sum_a W1[a][b]*dHidden[a]
      for (int b = 0; b < d; ++b) { int64_t acc = 0; for (int a = 0; a < ffn; ++a) acc += W1i[a][b] * dHid[i][a]; dN2ref[i][b] = rha(acc, P); }
      for (int a = 0; a < ffn; ++a) for (int b = 0; b < d; ++b) dW1acc[a][b] += dHid[i][a] * n2i[i][b];
    }
    vector<int64_t> dW2ref(d * ffn), dW1ref(ffn * d);
    for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) dW2ref[a * ffn + b] = rha(dW2acc[a][b], P);
    for (int a = 0; a < ffn; ++a) for (int b = 0; b < d; ++b) dW1ref[a * d + b] = rha(dW1acc[a][b], P);

    // ===== FLOAT reference (from committed ints; quantized GELU') =====
    double gN2[s][d] = {}, gW2[d][ffn] = {}, gW1[ffn][d] = {}, gb2[d] = {}, gb1[ffn] = {};
    for (int i = 0; i < s; ++i) {
      double dGr[ffn], dHr[ffn];
      for (int b = 0; b < ffn; ++b) { double a = 0; for (int aa = 0; aa < d; ++aa) a += (W2i[aa][b] / SCALE) * (dffi[i][aa] / SCALE); dGr[b] = a; }
      for (int b = 0; b < ffn; ++b) dHr[b] = dGr[b] * (gg(hidi[i][b]) / SCALE);
      for (int j = 0; j < d; ++j) gb2[j] += dffi[i][j] / SCALE;
      for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) gW2[a][b] += (dffi[i][a] / SCALE) * (gouti[i][b] / SCALE);
      for (int j = 0; j < ffn; ++j) gb1[j] += dHr[j];
      for (int b = 0; b < d; ++b) { double a = 0; for (int aa = 0; aa < ffn; ++aa) a += (W1i[aa][b] / SCALE) * dHr[aa]; gN2[i][b] = a; }
      for (int a = 0; a < ffn; ++a) for (int b = 0; b < d; ++b) gW1[a][b] += dHr[a] * (n2i[i][b] / SCALE);
    }

    // ===== commit =====
    auto commitMat = [&](const vector<vector<int64_t>> &m) { Mat c(m.size()); for (size_t i = 0; i < m.size(); ++i) qi.commit_vec(c[i], m[i]); return c; };
    Mat cdff = commitMat(dffi), cn2 = commitMat(n2i), chid = commitMat(hidi), cgout = commitMat(gouti), cW1 = commitMat(W1i), cW2 = commitMat(W2i);
    printf("[%d] committed dLdff, n2, hidden, geluout, W1, W2\n", party);

    // ===== run FFN backward =====
    GELUBackward gback(&qi, GT, GOFF); // owns GELU' table
    FFNBackward fb(&qi, &gback);
    Mat dLdn2;
    vector<IntFp> dLdW1, dLdb1, dLdW2, dLdb2;
    fb.forward(dLdn2, dLdW1, dLdb1, dLdW2, dLdb2, cdff, cn2, chid, cgout, cW1, cW2, (uint64_t)d, (uint64_t)ffn);
    printf("[%d] ffn backward done\n", party);

    // ===== reveal + check =====
    auto ck = [&](const char *nm, int i, int j, int64_t got, int64_t ref, double fl) {
      if (party != ALICE) return;
      if (got != ref) { printf("[%d] %s[%d][%d]: got %ld exp %ld MISMATCH\n", party, nm, i, j, (long)got, (long)ref); ok = false; }
      if (std::fabs(got / SCALE - fl) > 0.05) { printf("[%d] %s[%d][%d]: float diff %.4f\n", party, nm, i, j, std::fabs(got / SCALE - fl)); ok = false; }
    };
    for (int i = 0; i < s; ++i) for (int b = 0; b < d; ++b) ck("dLdn2", i, b, FixedInfer::decode_signed(dLdn2[i][b].reveal()), dN2ref[i][b], gN2[i][b]);
    for (int a = 0; a < ffn; ++a) for (int b = 0; b < d; ++b) ck("dLdW1", a, b, FixedInfer::decode_signed(dLdW1[a * d + b].reveal()), dW1ref[a * d + b], gW1[a][b]);
    for (int a = 0; a < d; ++a) for (int b = 0; b < ffn; ++b) ck("dLdW2", a, b, FixedInfer::decode_signed(dLdW2[a * ffn + b].reveal()), dW2ref[a * ffn + b], gW2[a][b]);
    for (int j = 0; j < ffn; ++j) ck("dLdb1", 0, j, FixedInfer::decode_signed(dLdb1[j].reveal()), db1ref[j], gb1[j]);
    for (int j = 0; j < d; ++j) ck("dLdb2", 0, j, FixedInfer::decode_signed(dLdb2[j].reveal()), db2ref[j], gb2[j]);
    if (party == ALICE) printf("[%d] all FFN gradients checked (exact + float within 0.05)\n", party);

    gback.Teardown(); // GELU' table LogUp BEFORE finalize
    qi.destroy();
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass ffn_backward (ml-fixed)" : "FAIL ffn_backward (ml-fixed)");
  else printf("verifier: ml-fixed ffn_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
