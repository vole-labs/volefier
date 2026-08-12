// Test zkGPT-style rounded division (volefier/nonlinear/div.h):
//   q = round(N / D), proven with a two-sided range relation (no MSB).
// ALICE checks each q against a plaintext reference; BOB verifies the proof.
//
//   test_nonlinear_div 1 12345   # prover
//   test_nonlinear_div 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/div.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;

// reference: same round-half-away-from-zero as Div::round_div
static int64_t ref_round(int64_t N, int64_t D) {
  return (N >= 0) ? (N + D / 2) / D : -(((-N) + D / 2) / D);
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);

  cout << endl << "------------ ZK proof of zkGPT-style division ------------" << endl << endl;

  PRG prg;
  setup_zk_arith(ios, threads, party);

  // D in [1, 1000] -> 2*D <= 2000 < 2^12, so a single 12-bit range chunk covers the terms.
  Div dvd(party, ios[0]->io, /*range_bits=*/12, /*chunk_bits=*/12);

  const size_t TESTS = 2000;
  size_t fails = 0;
  for (size_t k = 0; k < TESTS; ++k) {
    int64_t Nv = 0, Dv = 1;
    if (party == ALICE) {
      uint64_t r; prg.random_data(&r, sizeof(r));
      Nv = (int64_t)(r % 200001) - 100000;   // N in [-100000, 100000]
      prg.random_data(&r, sizeof(r));
      Dv = (int64_t)(r % 1000) + 1;           // D in [1, 1000]
    }
    IntFp N = IntFp(party == ALICE ? (Nv >= 0 ? (uint64_t)Nv : (uint64_t)(Nv + (int64_t)PR)) : 0ULL, ALICE);
    IntFp D = IntFp(party == ALICE ? (uint64_t)Dv : 0ULL, ALICE);

    IntFp q = dvd.round_div(N, D);
    int64_t got = Div::decode_signed(q.reveal());
    if (party == ALICE && got != ref_round(Nv, Dv)) {
      if (fails < 5) cout << "  N=" << Nv << " D=" << Dv << " got " << got << " exp " << ref_round(Nv, Dv) << endl;
      fails++;
    }
  }

  dvd.destroy();
  finalize_zk_arith();

  if (party == ALICE) {
    if (fails == 0) cout << "pass division (" << TESTS << " cases, two-sided range, no MSB)" << endl;
    else { cout << fails << " / " << TESTS << " mismatches" << endl; error("division test failed\n"); }
  }

  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
