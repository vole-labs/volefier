// Test: ml-fixed RMSNorm backward gadget.
// Forward proved  y_i = gamma_i * x_hat_i,  x_hat_i = x_i * dinv,  dinv = 1/RMS.
// Backward (retaining x_hat, gamma, dinv, plus upstream dL/dy) produces:
//   dL/dgamma_i = dL/dy_i * x_hat_i
//   dL/dx_i     = (dinv/d)*( d*gamma_i*dL/dy_i - x_hat_i * sum_j gamma_j dL/dy_j x_hat_j )
// x_hat, gamma, dinv, dL/dy are committed as IntFp; the revealed committed gradients are
// checked EXACTLY against a fixed-point replay (same round-half-away requant + Div rounding)
// and cross-checked (within tolerance) against the float RMSNorm-backward reference.
// Setup mirrors test/ml-fixed/gelu_backward.cpp (adds a Div gadget for the mean).
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
    Div divr(party, ios[0]->io, 24); // range for the mean's two-sided terms (~2*d*2^P)
    const double SCALE = static_cast<double>(1u << P);

    // Feature vector x, scale gamma, upstream gradient dL/dy (real).
    const int N = 6;
    const uint64_t d = N;
    double x_f[N]     = {1.5, -2.0, 0.5, 3.0, -1.0, 0.25};
    double gamma_f[N] = {1.0,  0.8, 1.2, 0.5,  1.1, 0.9};
    double dLdy_f[N]  = {0.5, -1.0, 0.25, 2.0, -0.5, 1.0};

    // Forward RMS witnesses: RMS = sqrt(mean(x^2)); x_hat = x/RMS; dinv = 1/RMS.
    double ms = 0.0; for (int i = 0; i < N; ++i) ms += x_f[i] * x_f[i]; ms /= N;
    double rms = std::sqrt(ms), dinv_r = 1.0 / rms;
    double xhat_f[N]; for (int i = 0; i < N; ++i) xhat_f[i] = x_f[i] * dinv_r;

    // Quantize everything to scale 2^-P (these are the committed values).
    vector<int64_t> xh_int(N), gamma_int(N), dLdy_int(N);
    for (int i = 0; i < N; ++i) {
      xh_int[i]    = llround(xhat_f[i] * SCALE);
      gamma_int[i] = llround(gamma_f[i] * SCALE);
      dLdy_int[i]  = llround(dLdy_f[i] * SCALE);
    }
    int64_t dinv_int = llround(dinv_r * SCALE);

    // ---- Fixed-point reference replay (exact mirror of the gadget) ----
    auto rha = [&](int64_t Nn) -> int64_t { // requant_cm round-half-away by 2^P
      int64_t h = 1LL << (P - 1), Dn = 1LL << P;
      return (Nn >= 0) ? (Nn + h) / Dn : -(((-Nn) + h) / Dn);
    };
    const int64_t Dbig = (int64_t)d << P; // d * 2^P, the mean divisor
    auto rdiv = [&](int64_t Nn) -> int64_t { // Div::round_div round-half-away by Dbig
      int64_t h = Dbig / 2;
      return (Nn >= 0) ? (Nn + h) / Dbig : -(((-Nn) + h) / Dbig);
    };
    vector<int64_t> dgamma_ref(N), g_ref(N), dx_ref(N);
    for (int i = 0; i < N; ++i) dgamma_ref[i] = rha(dLdy_int[i] * xh_int[i]);
    for (int i = 0; i < N; ++i) g_ref[i] = rha(gamma_int[i] * dLdy_int[i]);
    int64_t dot2p = 0; for (int j = 0; j < N; ++j) dot2p += g_ref[j] * xh_int[j];
    int64_t mean = rdiv(dot2p);
    for (int i = 0; i < N; ++i) { int64_t Q = rha(xh_int[i] * mean); dx_ref[i] = rha(dinv_int * (g_ref[i] - Q)); }

    // ---- Float reference (sanity cross-check within tolerance) ----
    double dotf = 0.0; for (int j = 0; j < N; ++j) dotf += gamma_f[j] * dLdy_f[j] * xhat_f[j];
    double dx_f[N];
    for (int i = 0; i < N; ++i)
      dx_f[i] = dinv_r * (gamma_f[i] * dLdy_f[i] - xhat_f[i] * dotf / (double)d);

    // Commit retained forward outputs + upstream gradient.
    vector<IntFp> c_xh, c_gamma, c_dLdy;
    qi.commit_vec(c_xh, xh_int);
    qi.commit_vec(c_gamma, gamma_int);
    qi.commit_vec(c_dLdy, dLdy_int);
    IntFp c_dinv = qi.commit(dinv_int);
    printf("[%d] committed x_hat, gamma, dinv, dL/dy\n", party);

    // Run RMSNorm backward.
    RMSNormBackward rb(&qi, &divr);
    vector<IntFp> dLdx, dLdgamma;
    rb.forward(dLdx, dLdgamma, c_dLdy, c_xh, c_gamma, c_dinv, d);
    printf("[%d] rmsnorm backward done, dLdx=%zu dLdgamma=%zu\n", party, dLdx.size(), dLdgamma.size());

    // Reveal committed gradients; check exact vs fixed-point replay, sanity vs float.
    for (int i = 0; i < N; ++i) {
      int64_t gx = FixedInfer::decode_signed(dLdx[i].reveal());
      int64_t gg = FixedInfer::decode_signed(dLdgamma[i].reveal());
      if (party == ALICE) {
        bool exx = (gx == dx_ref[i]), exg = (gg == dgamma_ref[i]);
        double fx = gx / SCALE;
        printf("[%d] dLdx[%d]: got %ld exp %ld %s (float %.4f gadget %.4f)\n",
               party, i, (long)gx, (long)dx_ref[i], exx ? "OK" : "MISMATCH", dx_f[i], fx);
        printf("[%d] dLdgamma[%d]: got %ld exp %ld %s\n",
               party, i, (long)gg, (long)dgamma_ref[i], exg ? "OK" : "MISMATCH");
        if (!exx || !exg) ok = false;
        if (std::fabs(fx - dx_f[i]) > 0.05) { printf("[%d] dLdx[%d] float mismatch!\n", party, i); ok = false; }
      }
    }

    divr.destroy(); // flush Div's LogUp BEFORE finalize_zk_arith
    qi.destroy();   // flush FixedInfer's LogUp range-check batches
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass rmsnorm_backward (ml-fixed)" : "FAIL rmsnorm_backward (ml-fixed)");
  else printf("verifier: ml-fixed rmsnorm_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
