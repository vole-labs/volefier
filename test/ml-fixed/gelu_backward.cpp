// Test: ml-fixed GELU backward gadget (elementwise chain rule).
// Forward proved  y = GELU(x)  as a table lookup; backward is  dL/dx = dL/dy * GELU'(x).
// x and dL/dy are committed as IntFp (retained forward-pass activations + upstream
// gradient), GELUBackward::forward() looks up GELU'(x_i) and proves the committed
// multiply + requant, and the revealed committed dL/dx is checked against a plaintext
// reference built from the SAME gradient table (so the match is exact).
// Setup mirrors test/ml-fixed/softmax_backward.cpp.
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
    // GELU table domain matches gpt2_prove: x in [-64, 64) at scale 2^-P.
    const uint64_t GT = 1u << (P + 7);
    const int64_t GOFF = 1LL << (P + 6);

    // Inputs (real) and upstream gradient (real).
    const int N = 5;
    double x_f[N]    = {-2.0, -0.5, 0.0, 0.5, 2.0};
    double dLdy_f[N] = { 1.0, -0.5, 2.0, 0.25, -1.5};

    vector<int64_t> x_int(N), dLdy_int(N);
    for (int i = 0; i < N; ++i) { x_int[i] = llround(x_f[i] * SCALE); dLdy_int[i] = llround(dLdy_f[i] * SCALE); }

    // Plaintext reference from the SAME gradient table + same round-half-away requant.
    vector<uint64_t> grad_tab = build_gelu_grad_table(GT, P, GOFF);
    auto requant_rha = [&](int64_t Nv) -> int64_t { // matches FixedInfer::requant_cm_one
      int64_t half = 1LL << (P - 1), D = 1LL << P;
      return (Nv >= 0) ? (Nv + half) / D : -(((-Nv) + half) / D);
    };
    vector<int64_t> ref(N);
    for (int i = 0; i < N; ++i) {
      int64_t gp_q = FixedInfer::decode_signed(grad_tab[(std::size_t)(x_int[i] + GOFF)]);
      ref[i] = requant_rha(dLdy_int[i] * gp_q);
    }

    if (party == ALICE) {
      printf("[%d] x_int=[", party); for (int i = 0; i < N; ++i) printf("%ld%s", (long)x_int[i], i + 1 < N ? "," : "]\n");
      printf("[%d] ref  =[", party); for (int i = 0; i < N; ++i) printf("%ld%s", (long)ref[i], i + 1 < N ? "," : "]\n");
    }

    // Commit x and dL/dy (retained forward activations + upstream gradient).
    vector<IntFp> cx, cdLdy;
    qi.commit_vec(cx, x_int);
    qi.commit_vec(cdLdy, dLdy_int);
    printf("[%d] committed x and dL/dy\n", party);

    // Run GELU backward.
    GELUBackward gb(&qi, GT, GOFF);
    vector<IntFp> dLdx;
    gb.forward(dLdx, cdLdy, cx);
    printf("[%d] gelu backward done, size=%zu\n", party, dLdx.size());

    // Reveal committed dL/dx and check against plaintext reference.
    for (int i = 0; i < N; ++i) {
      int64_t got = FixedInfer::decode_signed(dLdx[i].reveal());
      if (party == ALICE) {
        printf("[%d] dLdx[%d]: got %ld expected %ld %s\n", party, i, (long)got, (long)ref[i],
               got == ref[i] ? "OK" : "MISMATCH");
        if (got != ref[i]) ok = false;
      }
    }

    gb.Teardown(); // flush gradient-table LogUp BEFORE finalize_zk_arith
    qi.destroy();  // flush LogUp range-check batches BEFORE finalize_zk_arith
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass gelu_backward (ml-fixed)" : "FAIL gelu_backward (ml-fixed)");
  else printf("verifier: ml-fixed gelu_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
