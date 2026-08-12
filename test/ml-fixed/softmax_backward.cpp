// Test: ml-fixed softmax backward gadget (cross-entropy special case).
// Forward:  p = softmax(scores),  scores = [3, 1, 0]  ->  p ~ [0.844, 0.114, 0.042]
// Backward: dL/dscores[i] = p[i] - 1{i==true_token}   (softmax+CE fused case)
// The softmax outputs are committed as IntFp (simulating retained forward-pass
// activations), cross_entropy() runs the backward, and the revealed committed
// gradients are checked against a plaintext reference.  Setup mirrors
// test/ml-fixed/gpt2_prove.cpp (BoolIO + setup_zk_arith + FixedInfer).
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

    // Plaintext softmax forward (probabilities at scale 2^-P), true_token = 0.
    const int N = 3;
    const std::size_t true_token = 0;
    double scores_f[N] = {3.0, 1.0, 0.0};
    double max_s = 3.0, sum = 0.0, p_f[N];
    for (int i = 0; i < N; ++i) { p_f[i] = std::exp(scores_f[i] - max_s); sum += p_f[i]; }
    for (int i = 0; i < N; ++i) p_f[i] /= sum;

    vector<int64_t> p_int(N);
    for (int i = 0; i < N; ++i) p_int[i] = llround(p_f[i] * SCALE);

    // Reference: dL/dscores[i] = p[i] - 1{i==true_token}   (one-hot "1" = 2^P).
    vector<int64_t> ref(N);
    for (int i = 0; i < N; ++i)
      ref[i] = p_int[i] - ((std::size_t)i == true_token ? (int64_t)(1u << P) : 0);

    if (party == ALICE)
      printf("[%d] p_int=[%ld,%ld,%ld] ref=[%ld,%ld,%ld]\n", party,
             (long)p_int[0], (long)p_int[1], (long)p_int[2],
             (long)ref[0], (long)ref[1], (long)ref[2]);

    // Commit softmax outputs as IntFp (retained forward-pass activations).
    vector<IntFp> p_committed;
    qi.commit_vec(p_committed, p_int);
    printf("[%d] committed softmax outputs\n", party);

    // Run softmax backward (cross-entropy fused case).
    SoftmaxBackward sb(&qi);
    vector<IntFp> dLdscores;
    sb.cross_entropy(dLdscores, p_committed, true_token);
    printf("[%d] softmax backward done, size=%zu\n", party, dLdscores.size());

    // Reveal committed gradients and check against plaintext reference.
    for (int i = 0; i < N; ++i) {
      int64_t got = FixedInfer::decode_signed(dLdscores[i].reveal());
      if (party == ALICE) {
        printf("[%d] dLdscores[%d]: got %ld expected %ld %s\n", party, i,
               (long)got, (long)ref[i], got == ref[i] ? "OK" : "MISMATCH");
        if (got != ref[i]) ok = false;
      }
    }

    qi.destroy(); // flush LogUp range-check batches BEFORE finalize_zk_arith
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass softmax_backward (ml-fixed)" : "FAIL softmax_backward (ml-fixed)");
  else printf("verifier: ml-fixed softmax_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
