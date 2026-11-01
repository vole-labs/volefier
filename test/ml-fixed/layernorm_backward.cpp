// Test: ml-fixed LayerNorm backward gadget.
// Forward proved  y_i = x_hat_i = (x_i - mu)*dinv,  out_i = gamma_i*y_i + beta_i,
// dinv = 1/sigma (committed sqrt-free witness).  Backward (retaining c = x-mu,
// gamma, dinv, plus upstream dL/dy) produces:
//   dL/dgamma_i = dL/dy_i * x_hat_i
//   dL/dbeta_i  = dL/dy_i
//   dL/dx_i     = dinv*( g_i - mean(g) - x_hat_i*mean(g (.) x_hat) ),  g = dL/dy (.) gamma
// c, gamma, dinv, dL/dy are committed as IntFp; the revealed gradients are checked
// EXACTLY against a fixed-point replay (same round-half-up requant + Div rounding)
// and cross-checked (within tolerance) against the float LayerNorm-backward reference.
// Setup mirrors test/ml-fixed/gelu_backward.cpp / rmsnorm_backward.cpp.
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/div.h"
#include "volefier/ml-fixed/fixed_backward.h"
using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
const int P = getenv("P") ? atoi(getenv("P")) : 7;

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
    Div divr(party, ios[0]->io, 24); // range for the two means' two-sided terms (~2*n*2^P)
    const double SCALE = static_cast<double>(1u << P);

    const int N = 8;
    const uint64_t n = N;
    double x_f[N]     = {1.5, -2.0, 0.5, 3.0, -1.0, 0.25, -0.75, 2.0};
    double gamma_f[N] = {1.0,  0.8, 1.2, 0.5,  1.1, 0.9,   1.0,  0.7};
    double beta_f[N]  = {0.1, -0.2, 0.0, 0.3, -0.1, 0.2,  -0.05, 0.15}; // generated, unused in backward
    double dLdy_f[N]  = {0.5, -1.0, 0.25, 2.0, -0.5, 1.0,  -0.3,  0.8};
    (void)beta_f;

    // -------- quantize inputs to scale 2^-P --------
    vector<int64_t> x_int(N), gamma_int(N), dLdy_int(N);
    for (int i = 0; i < N; ++i) {
      x_int[i] = llround(x_f[i] * SCALE);
      gamma_int[i] = llround(gamma_f[i] * SCALE);
      dLdy_int[i] = llround(dLdy_f[i] * SCALE);
    }

    // rounding helpers: FixedInfer::requant_cm (round-half-up) and Div::round_div (half away)
    auto rhu = [&](int64_t Nn) -> int64_t { return (Nn + (1LL << (P - 1))) >> P; }; // requant by 2^P
    auto rdiv = [&](int64_t Nn, int64_t Dn) -> int64_t { // round_div by Dn (half away)
      int64_t h = Dn / 2;
      return (Nn >= 0) ? (Nn + h) / Dn : -(((-Nn) + h) / Dn);
    };

    // -------- plaintext forward (same formulas as FixedLayerNorm::forward) --------
    int64_t sumx = 0; for (int i = 0; i < N; ++i) sumx += x_int[i];
    int64_t mu = rdiv(sumx, (int64_t)n);
    vector<int64_t> c_int(N); for (int i = 0; i < N; ++i) c_int[i] = x_int[i] - mu;
    int64_t ssq = 0; for (int i = 0; i < N; ++i) ssq += c_int[i] * c_int[i];
    int64_t var_q = rdiv(ssq, (int64_t)n);
    double sq = std::sqrt((double)var_q);
    int64_t dinv_int = llround((double)(1LL << (2 * P)) / (sq > 0 ? sq : 1.0));

    // -------- EXACT fixed-point reference (mirror of the gadget) --------
    vector<int64_t> y_ref(N), dgamma_ref(N), dbeta_ref(N), g_ref(N), dx_ref(N);
    for (int i = 0; i < N; ++i) y_ref[i] = rhu(c_int[i] * dinv_int);            // x_hat
    for (int i = 0; i < N; ++i) dgamma_ref[i] = rhu(dLdy_int[i] * y_ref[i]);
    for (int i = 0; i < N; ++i) dbeta_ref[i] = dLdy_int[i];                     // free
    for (int i = 0; i < N; ++i) g_ref[i] = rhu(dLdy_int[i] * gamma_int[i]);
    int64_t gy = 0; for (int i = 0; i < N; ++i) gy += g_ref[i] * y_ref[i];
    int64_t dot1 = rdiv(gy, (int64_t)n << P);                                   // <g,y>/n
    int64_t sum_g = 0; for (int i = 0; i < N; ++i) sum_g += g_ref[i];
    int64_t dot2 = rdiv(sum_g, (int64_t)n);                                     // mean(g)
    for (int i = 0; i < N; ++i) {
      int64_t yp = rhu(y_ref[i] * dot1);
      int64_t term = g_ref[i] - dot2 - yp;
      dx_ref[i] = rhu(dinv_int * term);
    }

    // -------- float reference (true LayerNorm backward, sanity cross-check) --------
    double dinv_r = dinv_int / SCALE;
    double y_r[N], g_r[N];
    for (int i = 0; i < N; ++i) y_r[i] = (c_int[i] / SCALE) * dinv_r;
    for (int i = 0; i < N; ++i) g_r[i] = (dLdy_int[i] / SCALE) * (gamma_int[i] / SCALE);
    double d1 = 0, d2 = 0; for (int i = 0; i < N; ++i) { d1 += g_r[i] * y_r[i]; d2 += g_r[i]; }
    d1 /= n; d2 /= n;
    double dx_f[N], dgamma_f[N];
    for (int i = 0; i < N; ++i) {
      dx_f[i] = dinv_r * (g_r[i] - d2 - y_r[i] * d1);
      dgamma_f[i] = (dLdy_int[i] / SCALE) * y_r[i];
    }

    // -------- commit retained forward outputs + upstream gradient --------
    vector<IntFp> c_c, gamma_c, dLdy_c;
    qi.commit_vec(c_c, c_int);
    qi.commit_vec(gamma_c, gamma_int);
    qi.commit_vec(dLdy_c, dLdy_int);
    IntFp dinv_c = qi.commit(dinv_int);
    printf("[%d] committed c, gamma, dinv, dL/dy\n", party);

    // -------- run LayerNorm backward --------
    LayerNormBackward lb(&qi, &divr);
    vector<IntFp> dLdx, dLdgamma, dLdbeta;
    lb.forward(dLdx, dLdgamma, dLdbeta, dLdy_c, c_c, gamma_c, dinv_c, n);
    printf("[%d] layernorm backward done: dLdx=%zu dLdgamma=%zu dLdbeta=%zu\n",
           party, dLdx.size(), dLdgamma.size(), dLdbeta.size());

    // -------- reveal + check --------
    const double tol = 2.0 / SCALE;
    for (int i = 0; i < N; ++i) {
      int64_t gx = FixedInfer::decode_signed(dLdx[i].reveal());
      int64_t gg = FixedInfer::decode_signed(dLdgamma[i].reveal());
      int64_t gb = FixedInfer::decode_signed(dLdbeta[i].reveal());
      if (party == ALICE) {
        bool exx = (gx == dx_ref[i]), exg = (gg == dgamma_ref[i]), exb = (gb == dbeta_ref[i]);
        double fx = gx / SCALE, fg = gg / SCALE;
        printf("[%d] dLdx[%d]: got %ld exp %ld %s (float %.4f gadget %.4f)\n",
               party, i, (long)gx, (long)dx_ref[i], exx ? "OK" : "MISMATCH", dx_f[i], fx);
        printf("[%d] dLdgamma[%d]: got %ld exp %ld %s ; dLdbeta[%d]: got %ld exp %ld %s\n",
               party, i, (long)gg, (long)dgamma_ref[i], exg ? "OK" : "MISMATCH",
               i, (long)gb, (long)dbeta_ref[i], exb ? "OK" : "MISMATCH");
        if (!exx || !exg || !exb) ok = false;
        if (std::fabs(fx - dx_f[i]) > tol) { printf("[%d] dLdx[%d] float diff %.4f > %.4f\n", party, i, std::fabs(fx - dx_f[i]), tol); ok = false; }
        if (std::fabs(fg - dgamma_f[i]) > tol) { printf("[%d] dLdgamma[%d] float diff %.4f > %.4f\n", party, i, std::fabs(fg - dgamma_f[i]), tol); ok = false; }
      }
    }

    divr.destroy(); // flush Div's LogUp BEFORE finalize_zk_arith
    qi.destroy();   // flush FixedInfer's LogUp range-check batches
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass layernorm_backward (ml-fixed)" : "FAIL layernorm_backward (ml-fixed)");
  else printf("verifier: ml-fixed layernorm_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
