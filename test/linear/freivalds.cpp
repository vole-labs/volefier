// Validate the Freivalds matmul proof (volefier/linear/ip_matmul.h: matmul_mm_freivalds)
// against the per-entry IP proof (matmul_mm): same committed inputs, both must reveal the
// plaintext product kernel . cols^T, and the verifier must ACCEPT both. Then a soundness
// check: a tampered committed C must make the Freivalds batch_check REJECT.
//
//   test_linear_freivalds 1 12345   # prover
//   test_linear_freivalds 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/linear/ip_matmul.h"

using namespace emp;
using namespace std;
using volefier::IPProof;
using volefier::get_ostriple;

int port, party;
const int threads = 4;

static IntFp commit_s(int64_t v, int party) {
  if (party == ALICE) return IntFp((v >= 0) ? (uint64_t)v : (uint64_t)(v + (int64_t)PR), ALICE);
  return IntFp((uint64_t)0, ALICE);
}
static int64_t decode_s(uint64_t v) { return (v > (PR >> 1)) ? (int64_t)v - (int64_t)PR : (int64_t)v; }

void run(BoolIO *ios[threads], int party) {
  PRG prg;
  setup_zk_arith(ios, threads, party);
  IPProof<BoolIO> ip(party, ios[0], get_ostriple(party));

  const size_t Cout = getenv("COUT") ? atol(getenv("COUT")) : 96,    // out[Cout][Pn] = kernel[Cout][M].cols[Pn][M]^T
               Pn   = getenv("PN")   ? atol(getenv("PN"))   : 40,
               M    = getenv("MM")   ? atol(getenv("MM"))   : 256;
  vector<vector<int64_t>> K(Cout, vector<int64_t>(M)), X(Pn, vector<int64_t>(M));
  if (party == ALICE) {
    for (auto &r : K) for (auto &w : r) { uint64_t t; prg.random_data(&t, 8); w = (int64_t)(t % 9) - 4; }
    for (auto &r : X) for (auto &w : r) { uint64_t t; prg.random_data(&t, 8); w = (int64_t)(t % 9) - 4; }
  }
  vector<vector<IntFp>> cK(Cout, vector<IntFp>(M)), cX(Pn, vector<IntFp>(M));
  for (size_t i = 0; i < Cout; ++i) for (size_t k = 0; k < M; ++k) cK[i][k] = commit_s(K[i][k], party);
  for (size_t p = 0; p < Pn; ++p) for (size_t k = 0; k < M; ++k) cX[p][k] = commit_s(X[p][k], party);

  // SOUNDNESS: commit a deliberately-wrong C[0]; the Freivalds check MUST reject (BOB aborts
  // inside matmul_mm_freivalds). Reaching finalize means the tampered output was accepted.
  if (getenv("SOUNDNESS")) {
    vector<vector<IntFp>> out_t;
    ip.matmul_mm_freivalds(out_t, cX, cK, threads, true, /*tamper_idx=*/0);
    finalize_zk_arith();
    cout << (party == BOB ? "TEST FAILED: tampered C was accepted (Freivalds check did not reject)"
                          : "(prover finished tampered proof; verifier must have aborted)") << endl;
    return;
  }

  // per-entry IP and Freivalds on the SAME committed inputs
  vector<vector<IntFp>> out_mm, out_fr;
  auto t0 = clock_start(); ip.matmul_mm(out_mm, cX, cK, threads);            double us_mm = time_from(t0);
  t0 = clock_start();      ip.matmul_mm_freivalds(out_fr, cX, cK, threads);  double us_fr = time_from(t0);

  size_t fails = 0;
  for (size_t co = 0; co < Cout; ++co)
    for (size_t p = 0; p < Pn; ++p) {
      int64_t g_mm = decode_s(out_mm[co][p].reveal()), g_fr = decode_s(out_fr[co][p].reveal());
      if (party == ALICE) {
        int64_t ref = 0; for (size_t k = 0; k < M; ++k) ref += K[co][k] * X[p][k];
        if (g_mm != ref || g_fr != ref) {
          if (fails < 5) cout << "  (" << co << "," << p << ") mm=" << g_mm << " fr=" << g_fr << " exp=" << ref << endl;
          fails++;
        }
      }
    }

  // ---- attention Freivalds: per-head A[G,M,K] . B[G,K,N] = C[G,M,N] ----
  const size_t G = 6, AM = 48, AN = 48, AK = 32;
  vector<int64_t> Af(G * AM * AK), Bf(G * AK * AN);
  if (party == ALICE) {
    for (auto &w : Af) { uint64_t t; prg.random_data(&t, 8); w = (int64_t)(t % 7) - 3; }
    for (auto &w : Bf) { uint64_t t; prg.random_data(&t, 8); w = (int64_t)(t % 7) - 3; }
  }
  vector<IntFp> cAf(G * AM * AK), cBf(G * AK * AN);
  for (size_t i = 0; i < cAf.size(); ++i) cAf[i] = commit_s(Af[i], party);
  for (size_t i = 0; i < cBf.size(); ++i) cBf[i] = commit_s(Bf[i], party);
  vector<IntFp> at_mm, at_fr;
  ip.matmul_attn(at_mm, cAf.data(), cBf.data(), G, AM, AN, AK, true, threads);
  ip.matmul_attn_freivalds(at_fr, cAf.data(), cBf.data(), G, AM, AN, AK, true, threads);
  size_t afails = 0;
  for (size_t g = 0; g < G; ++g) for (size_t m = 0; m < AM; ++m) for (size_t j = 0; j < AN; ++j) {
    size_t idx = (g * AM + m) * AN + j;
    int64_t gm = decode_s(at_mm[idx].reveal()), gf = decode_s(at_fr[idx].reveal());
    if (party == ALICE) {
      int64_t ref = 0; for (size_t k = 0; k < AK; ++k) ref += Af[(g * AM + m) * AK + k] * Bf[(g * AK + k) * AN + j];
      if (gm != ref || gf != ref) { if (afails < 5) cout << "  attn (" << g << "," << m << "," << j << ") mm=" << gm << " fr=" << gf << " exp=" << ref << endl; afails++; }
    }
  }

  finalize_zk_arith();
  if (party == ALICE) {
    if (fails) { cout << fails << " mismatches" << endl; error("Freivalds matmul FAILED\n"); }
    if (afails) { cout << afails << " attn mismatches" << endl; error("Freivalds attn FAILED\n"); }
    cout << "pass: Freivalds matmul agrees with per-entry IP and plaintext (" << Cout << "x" << Pn << "x" << M << ")" << endl;
    cout << "pass: Freivalds attention agrees with matmul_attn and plaintext (" << G << "x" << AM << "x" << AN << "x" << AK << ")" << endl;
    cout << "  matmul_mm (per-entry) : " << us_mm / 1e3 << " ms" << endl;
    cout << "  matmul_mm_freivalds   : " << us_fr / 1e3 << " ms   (" << us_mm / us_fr << "x)" << endl;
  }
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ Freivalds matmul validation ------------" << endl << endl;
  run(ios, party);
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
