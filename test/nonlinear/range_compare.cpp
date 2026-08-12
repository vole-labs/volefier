// Test RangeCheckLU (lookup-table range check, #lookups = ceil(bits/table_bits))
// and CompareLU (comparison via range-check on the difference, DeepProve-style).
//
//   test_nonlinear_range_compare 1 12345   # prover
//   test_nonlinear_range_compare 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/comparison.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;

static PRG *gprg;
static int64_t rnd(int64_t lo, int64_t hi) { if (party != ALICE) return 0; uint64_t r; gprg->random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); }
static int64_t QuantInfer_decode(uint64_t v) { return (v > (PR >> 1)) ? (int64_t)v - (int64_t)PR : (int64_t)v; }

void run(BoolIO *ios[threads], int party) {
  PRG prg; gprg = &prg;
  setup_zk_arith(ios, threads, party);

  const size_t TB = 16;                       // 16-bit table
  RangeCheckLU rc(party, ios[0]->io, TB);
  CompareLU cmp(&rc);

  // #lookups tracks the operand width (host-side check, no proof)
  if (party == ALICE) {
    cout << "num_lookups: 8b=" << rc.num_lookups(8) << " 16b=" << rc.num_lookups(16)
         << " 17b=" << rc.num_lookups(17) << " 40b=" << rc.num_lookups(40) << " (table=" << TB << "b)" << endl;
  }

  // ---- range check: values in [0, 2^bits) pass ----
  for (size_t bits : {8u, 16u, 24u, 33u}) {
    int64_t v = rnd(0, (1LL << bits) - 1);
    rc.check(cmp.commit_signed(v), bits);
  }

  // ---- assert_le on ordered pairs (operand width ~17 bits) ----
  const size_t OB = 17;
  for (int t = 0; t < 50; ++t) {
    int64_t a = rnd(-60000, 60000), b = a + rnd(0, 5000);   // a <= b
    cmp.assert_le(cmp.commit_signed(a), cmp.commit_signed(b), OB);
  }

  // ---- is_le boolean: reveal and check ----
  size_t bad = 0;
  for (int t = 0; t < 80; ++t) {
    int64_t a = rnd(-60000, 60000), b = rnd(-60000, 60000);
    IntFp g = cmp.is_le(cmp.commit_signed(a), cmp.commit_signed(b), OB);
    int64_t got = QuantInfer_decode(g.reveal());
    if (party == ALICE && got != (a <= b ? 1 : 0)) { if (bad < 5) cout << "is_le(" << a << "," << b << ")=" << got << endl; bad++; }
  }

  // ---- argmax: reveal index, check against reference ----
  size_t abad = 0;
  for (int t = 0; t < 20; ++t) {
    const size_t n = 32;
    vector<int64_t> xs(n); for (auto &e : xs) e = rnd(-60000, 60000);
    vector<IntFp> cx(n); for (size_t i = 0; i < n; ++i) cx[i] = cmp.commit_signed(xs[i]);
    IntFp idx = cmp.argmax(cx, OB);
    int64_t got = QuantInfer_decode(idx.reveal());
    if (party == ALICE) {
      int64_t ref = max_element(xs.begin(), xs.end()) - xs.begin();
      if (got != ref) { if (abad < 5) cout << "argmax got " << got << " exp " << ref << endl; abad++; }
    }
  }

  rc.Teardown();
  finalize_zk_arith();

  if (party == ALICE) {
    if (bad || abad) { cout << bad << " is_le, " << abad << " argmax mismatches" << endl; error("range/compare failed\n"); }
    cout << "pass RangeCheckLU + CompareLU (range, assert_le, is_le, argmax)" << endl;
  }
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ lookup-table range check + comparison ------------" << endl << endl;
  run(ios, party);
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
