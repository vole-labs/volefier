// Test: ml-fixed SGD parameter update (SGDOptimizer).
// SGD:  W_new = W_old - lr * dW.  lr is a public hyperparameter pre-quantized to
// lr_q = round(lr * 2^P); lr_q*dW is at 2^-2P and requantized to 2^-P, then a free
// subtraction gives W_new.  Committed W_old and dW; the revealed W_new is checked
// EXACTLY against a fixed-point replay (same round-half-up requant) and cross-
// checked against a float reference.  Also exercises the forward_mat() variant.
// Setup mirrors test/ml-fixed/gelu_backward.cpp.
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

    // lr = 0.01 at P=7  ->  lr_q = round(0.01 * 128) = round(1.28) = 1.
    // lr_q = 1 is the genuine quantized learning rate here, not a placeholder.
    const double lr = 0.01;
    const uint64_t lr_q = (uint64_t)llround(lr * SCALE); // = 1

    const int N = 8;
    double W_f[N]  = {1.5, -2.0, 0.5, 3.0, -1.0, 0.25, -0.75, 2.0};
    double dW_f[N] = {40.0, -25.0, 60.0, -128.0, 32.0, 100.0, -64.0, 12.0}; // large so lr_q=1 moves W

    vector<int64_t> W_int(N), dW_int(N);
    for (int i = 0; i < N; ++i) { W_int[i] = llround(W_f[i] * SCALE); dW_int[i] = llround(dW_f[i] * SCALE); }

    // round-half-up requant by 2^P (mirror FixedInfer::requant_cm)
    auto rhu = [&](int64_t Nn) -> int64_t { return (Nn + (1LL << (P - 1))) >> P; };

    // -------- reference: W_new_i = W_old_i - requant(lr_q * dW_i) --------
    vector<int64_t> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = W_int[i] - rhu((int64_t)lr_q * dW_int[i]);
    // Float reference uses the EFFECTIVE quantized learning rate lr_q/2^P (what the
    // gadget actually applies), not the ideal lr: at P=7, lr=0.01 rounds to lr_q=1 =
    // 0.0078125, so comparing to the ideal 0.01 would measure lr-quantization error,
    // not gadget correctness.  (Same reasoning as using qinv/2^P in attention_backward.)
    const double lr_eff = (double)lr_q / SCALE;
    double ref_f[N];
    for (int i = 0; i < N; ++i) ref_f[i] = W_f[i] - lr_eff * dW_f[i];

    if (party == ALICE) printf("[%d] lr=%.4f lr_q=%llu\n", party, lr, (unsigned long long)lr_q);

    // -------- commit W_old, dW --------
    vector<IntFp> W_c, dW_c;
    qi.commit_vec(W_c, W_int);
    qi.commit_vec(dW_c, dW_int);
    printf("[%d] committed W_old, dW\n", party);

    // -------- run SGD (vector) --------
    SGDOptimizer sgd(&qi, lr_q);
    vector<IntFp> W_new;
    sgd.forward(W_new, W_c, dW_c);
    printf("[%d] sgd update done, size=%zu\n", party, W_new.size());

    const double tol = 1.0 / SCALE;
    for (int i = 0; i < N; ++i) {
      int64_t got = FixedInfer::decode_signed(W_new[i].reveal());
      if (party == ALICE) {
        bool ex = (got == ref[i]);
        double fv = got / SCALE;
        printf("[%d] W_new[%d]: got %ld exp %ld %s (float %.4f gadget %.4f)\n",
               party, i, (long)got, (long)ref[i], ex ? "OK" : "MISMATCH", ref_f[i], fv);
        if (!ex) ok = false;
        if (std::fabs(fv - ref_f[i]) > tol) { printf("[%d] W_new[%d] float diff %.4f > %.4f\n", party, i, std::fabs(fv - ref_f[i]), tol); ok = false; }
      }
    }

    // -------- run SGD (matrix variant, 2x4 = same 8 elements, row-major) --------
    const int R = 2, C = 4;
    Mat Wm(R), dWm(R);
    for (int r = 0; r < R; ++r) {
      vector<int64_t> wr(C), dwr(C);
      for (int c = 0; c < C; ++c) { wr[c] = W_int[r * C + c]; dwr[c] = dW_int[r * C + c]; }
      qi.commit_vec(Wm[r], wr);
      qi.commit_vec(dWm[r], dwr);
    }
    Mat Wm_new;
    sgd.forward_mat(Wm_new, Wm, dWm);
    printf("[%d] sgd forward_mat done: %zux%zu\n", party, Wm_new.size(), Wm_new.empty() ? 0 : Wm_new[0].size());
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) {
        int64_t got = FixedInfer::decode_signed(Wm_new[r][c].reveal());
        if (party == ALICE && got != ref[r * C + c]) {
          printf("[%d] Wm_new[%d][%d]: got %ld exp %ld MISMATCH\n", party, r, c, (long)got, (long)ref[r * C + c]);
          ok = false;
        }
      }
    if (party == ALICE) printf("[%d] forward_mat matches vector update\n", party);

    qi.destroy(); // flush LogUp range-check batches BEFORE finalize_zk_arith
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass sgd_optimizer (ml-fixed)" : "FAIL sgd_optimizer (ml-fixed)");
  else printf("verifier: ml-fixed sgd_optimizer proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
