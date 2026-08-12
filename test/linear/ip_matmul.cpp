// Test the STANDALONE inner-product matmul (volefier/linear/ip_matmul.h), which
// reimplements emp-zk's QuickSilver IP proof without the global singleton.
// ALICE checks each output equals W[j].x; the IP proof's batch_check verifies
// the relation (BOB aborts via CheatRecord on a bad witness).
//
//   test_matmul_ip_matmul 1 12345   # prover
//   test_matmul_ip_matmul 2 12345   # verifier

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
using volefier::ip_linear;

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

  // n outputs, each an inner product of length m (a (n x m).(m) matvec).
  const size_t n = 8192, m = 768;
  vector<vector<int64_t>> W(n, vector<int64_t>(m));
  vector<int64_t> x(m);
  if (party == ALICE) {
    for (auto &row : W) for (auto &w : row) { uint64_t r; prg.random_data(&r, sizeof(r)); w = (int64_t)(r % 9) - 4; }
    for (auto &v : x) { uint64_t r; prg.random_data(&r, sizeof(r)); v = (int64_t)(r % 9) - 4; }
  }
  vector<vector<IntFp>> cW(n, vector<IntFp>(m));
  vector<IntFp> cx(m);
  for (size_t i = 0; i < m; ++i) cx[i] = commit_s(x[i], party);
  for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < m; ++i) cW[j][i] = commit_s(W[j][i], party);

  // prove the SAME matmul with 1 thread and with 4 threads; time each.
  vector<IntFp> out1, out4;
  auto t0 = clock_start(); ip.matmul(out1, cW, cx, 1); double us1 = time_from(t0);
  t0 = clock_start();      ip.matmul(out4, cW, cx, 4); double us4 = time_from(t0);

  size_t fails = 0;
  for (size_t j = 0; j < n; ++j) {
    int64_t g1 = decode_s(out1[j].reveal()), g4 = decode_s(out4[j].reveal());
    if (party == ALICE) {
      int64_t ref = 0; for (size_t i = 0; i < m; ++i) ref += W[j][i] * x[i];
      if (g1 != ref || g4 != ref) { if (fails < 5) cout << "row " << j << " got " << g1 << "/" << g4 << " exp " << ref << endl; fails++; }
    }
  }

  finalize_zk_arith();

  if (party == ALICE) {
    if (fails) { cout << fails << " mismatches" << endl; error("IP matmul failed\n"); }
    cout << "pass standalone IP matmul (" << n << " x " << m << " matvec, verified)" << endl;
    cout << "  1 thread : " << us1 / 1e6 << " s" << endl;
    cout << "  4 threads: " << us4 / 1e6 << " s   (speedup " << us1 / us4 << "x)" << endl;
  }
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ standalone IP matmul ------------" << endl << endl;
  run(ios, party);
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
