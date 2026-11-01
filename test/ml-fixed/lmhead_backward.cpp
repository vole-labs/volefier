// Test: ml-fixed loss + LM-head backward (the output side of a train step).
// Forward:  logits = requant(We @ H) + be   (We: vocab x d, H: d, last token).
// Loss:     p = softmax(logits) ; L = cross-entropy(p, true_token).
// Backward: dL/dlogits = p - onehot(true_token)   [FixedSoftmax + SoftmaxBackward]
//           dL/dWe = outer(dL/dlogits, H) ; dL/dH = We^T @ dL/dlogits ; dL/dbe = dL/dlogits
//                                                                        [LMHeadBackward]
// Chains the EXISTING FixedSoftmax (vocab softmax) + SoftmaxBackward::cross_entropy
// with the new LMHeadBackward.  Outputs checked EXACTLY against a fixed-point replay
// and against a float reference; softmax output cross-checked vs float softmax.
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
static const int vocab = 8, d = 8;

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
    const std::size_t score_bits = 12;      // (max - logit) width; exp table = 2^score_bits
    const uint64_t ET = 1u << score_bits;   // exp table size
    const std::size_t true_token = 3;

    // softmax machinery (same as gpt2_prove)
    RangeCheckLU rclu(party, ios[0]->io, 16);
    CompareLU cmp(&rclu);
    Div divr(party, ios[0]->io, 16);
    FixedTable expt(ET, party, ios[0]->io);
    expt.Setup(build_exp_table(ET, P, /*clamp_pos=*/true));
    FixedSoftmax sm(&qi, &cmp, &expt, &divr, P, score_bits);

    // ---- random H, We, be ----
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };
    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };
    double Hr[d], Wer[vocab][d], ber[vocab];
    for (int k = 0; k < d; ++k) Hr[k] = ur(-1, 1);
    for (int v = 0; v < vocab; ++v) { ber[v] = ur(-0.3, 0.3); for (int k = 0; k < d; ++k) Wer[v][k] = ur(-0.5, 0.5); }
    vector<int64_t> Hq(d); for (int k = 0; k < d; ++k) Hq[k] = q1(Hr[k]);
    vector<vector<int64_t>> Weq(vocab, vector<int64_t>(d)); for (int v = 0; v < vocab; ++v) for (int k = 0; k < d; ++k) Weq[v][k] = q1(Wer[v][k]);
    vector<int64_t> beq(vocab); for (int v = 0; v < vocab; ++v) beq[v] = q1(ber[v]);

    // ---- commit ----
    vector<IntFp> cH; qi.commit_vec(cH, Hq);
    Mat cWe(vocab); for (int v = 0; v < vocab; ++v) qi.commit_vec(cWe[v], Weq[v]);
    vector<IntFp> cbe; qi.commit_vec(cbe, beq);
    printf("[%d] committed H, We, be\n", party);

    // ---- LM-head forward: logits = requant(We @ H) + be ----
    vector<IntFp> logits;
    qi.dense_layer(logits, cWe, cbe, cH);

    // ---- loss: softmax + cross-entropy grad ----
    vector<IntFp> p = sm.forward(logits);                 // vocab softmax, 2^-P
    SoftmaxBackward sb(&qi);
    vector<IntFp> dLdlogits;
    sb.cross_entropy(dLdlogits, p, true_token);            // p - onehot, 2^-P

    // ---- LM-head backward ----
    LMHeadBackward lmb(&qi);
    vector<IntFp> dLdH, dLdWe, dLdbe;
    lmb.forward(dLdH, dLdWe, dLdbe, dLdlogits, cH, cWe, (uint64_t)vocab, (uint64_t)d);
    printf("[%d] softmax + cross-entropy + LM-head backward done\n", party);

    // ---- reveal ----
    vector<int64_t> logits_q(vocab), p_q(vocab), dl_q(vocab), dH_q(d), dbe_q(vocab);
    vector<int64_t> dWe_q(vocab * d);
    for (int v = 0; v < vocab; ++v) { logits_q[v] = FixedInfer::decode_signed(logits[v].reveal()); p_q[v] = FixedInfer::decode_signed(p[v].reveal()); dl_q[v] = FixedInfer::decode_signed(dLdlogits[v].reveal()); dbe_q[v] = FixedInfer::decode_signed(dLdbe[v].reveal()); }
    for (int k = 0; k < d; ++k) dH_q[k] = FixedInfer::decode_signed(dLdH[k].reveal());
    for (int v = 0; v < vocab; ++v) for (int k = 0; k < d; ++k) dWe_q[v * d + k] = FixedInfer::decode_signed(dLdWe[v * d + k].reveal());

    if (party == ALICE) {
      auto rhu = [&](int64_t N) -> int64_t { return (N + (1LL << (P - 1))) >> P; };   // round-half-up: FixedInfer::requant_cm
      // float softmax over the (revealed) committed logits, for a sanity cross-check of p
      double lr[vocab], mx = -1e30, sum = 0, pf[vocab];
      for (int v = 0; v < vocab; ++v) { lr[v] = logits_q[v] / SCALE; if (lr[v] > mx) mx = lr[v]; }
      for (int v = 0; v < vocab; ++v) { pf[v] = std::exp(lr[v] - mx); sum += pf[v]; }
      for (int v = 0; v < vocab; ++v) pf[v] /= sum;
      for (int v = 0; v < vocab; ++v) if (std::fabs(p_q[v] / SCALE - pf[v]) > 0.03) { printf("[%d] p[%d]: gadget %.4f vs float %.4f\n", party, v, p_q[v] / SCALE, pf[v]); ok = false; }

      // EXACT: dL/dlogits = p - onehot*2^P
      for (int v = 0; v < vocab; ++v) { int64_t ref = p_q[v] - ((std::size_t)v == true_token ? (1LL << P) : 0); if (dl_q[v] != ref) { printf("[%d] dLdlogits[%d]: got %ld exp %ld MISMATCH\n", party, v, (long)dl_q[v], (long)ref); ok = false; } if (dbe_q[v] != dl_q[v]) { printf("[%d] dLdbe[%d] != dLdlogits\n", party, v); ok = false; } }
      // EXACT: dH[k] = rhu(sum_v We_q[v][k]*dl_q[v]) ; float cross-check
      for (int k = 0; k < d; ++k) { int64_t acc = 0; for (int v = 0; v < vocab; ++v) acc += Weq[v][k] * dl_q[v]; int64_t ref = rhu(acc); if (dH_q[k] != ref) { printf("[%d] dLdH[%d]: got %ld exp %ld MISMATCH\n", party, k, (long)dH_q[k], (long)ref); ok = false; } double fl = 0; for (int v = 0; v < vocab; ++v) fl += (Weq[v][k] / SCALE) * (dl_q[v] / SCALE); if (std::fabs(dH_q[k] / SCALE - fl) > 0.05) { printf("[%d] dLdH[%d] float diff %.4f\n", party, k, std::fabs(dH_q[k] / SCALE - fl)); ok = false; } }
      // EXACT: dWe[v][k] = rhu(dl_q[v]*H_q[k]) ; float cross-check
      for (int v = 0; v < vocab; ++v) for (int k = 0; k < d; ++k) { int64_t ref = rhu(dl_q[v] * Hq[k]); if (dWe_q[v * d + k] != ref) { printf("[%d] dLdWe[%d][%d]: got %ld exp %ld MISMATCH\n", party, v, k, (long)dWe_q[v * d + k], (long)ref); ok = false; } double fl = (dl_q[v] / SCALE) * (Hq[k] / SCALE); if (std::fabs(dWe_q[v * d + k] / SCALE - fl) > 0.05) { printf("[%d] dLdWe[%d][%d] float diff %.4f\n", party, v, k, std::fabs(dWe_q[v * d + k] / SCALE - fl)); ok = false; } }
      printf("[%d] LM-head backward checked (softmax sanity + exact dLdlogits/dH/dWe/dbe + float)\n", party);
    }

    expt.Teardown(); rclu.Teardown(); divr.destroy(); qi.destroy();
  }
  finalize_zk_arith();

  if (party == ALICE) printf("%s\n", ok ? "pass lmhead_backward (ml-fixed)" : "FAIL lmhead_backward (ml-fixed)");
  else printf("verifier: ml-fixed lmhead_backward proof complete\n");
  cout.flush(); fflush(stdout);
  _exit(0);
}
