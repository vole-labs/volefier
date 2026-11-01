// Test the sqrt-free 1/sigma proof (volefier/nonlinear/rsqrt.h):
//   dinv = round(2^2P / sqrt(var)), proven with bounded, wrap-free inequalities.
// ALICE checks each revealed dinv against the plaintext formula; BOB verifies the proof.
//
//   test_nonlinear_rsqrt 1 12345   # prover
//   test_nonlinear_rsqrt 2 12345   # verifier

#include <iostream>
#include <cmath>
#include <cstdint>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/rsqrt.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  setup_zk_arith(ios, threads, party);
  RangeCheckLU rc(party, ios[0]->io, 16);

  const int P = 12, VB = 2 * P + 20;
  PRG prg;
  const size_t TESTS = 3000;
  size_t fails = 0;
  for (size_t k = 0; k < TESTS; ++k) {
    uint64_t v = 1;
    if (party == ALICE) {   // log-uniform var over [1, 2^VB), plus the edge values
      uint64_t r; prg.random_data(&r, sizeof(r));
      if (k == 0) v = 1;
      else if (k == 1) v = (1ULL << VB) - 1;
      else { int b = 1 + (int)(r % VB); prg.random_data(&r, sizeof(r)); v = (r & ((1ULL << b) - 1)) | (1ULL << (b - 1)); }
    }
    IntFp var(party == ALICE ? v : 0ULL, ALICE);
    IntFp dinv = prove_rsqrt(&rc, party, var, P, VB);
    uint64_t got = dinv.reveal();
    if (party == ALICE) {
      uint64_t want = (uint64_t)llround((double)(1ULL << (2 * P)) / std::sqrt((double)v));
      if (got != want) { if (fails < 5) cout << "  var=" << v << " got " << got << " exp " << want << endl; fails++; }
    }
  }
  rc.Teardown();
  finalize_zk_arith();
  if (party == ALICE) {
    if (fails == 0) cout << "pass rsqrt (" << TESTS << " cases, var in [1, 2^" << VB << "))" << endl;
    else { cout << fails << " / " << TESTS << " mismatches" << endl; error("rsqrt test failed\n"); }
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
