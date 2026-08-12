// Minimal end-to-end PROVER benchmark for the model-proof tests.
//
// Reports total wall-time + COMMUNICATION (the VOLE-ZK cost metric, which the
// per-op [TIMING]/[MATMUL_PROF] profilers do NOT capture) + peak RSS, gated
// behind BENCH=1 (or PHASES=1).  It is the 2-line drop-in equivalent of the
// time+MB accounting in bench_full.cpp.  Usage:
//
//   BenchE2E<BoolIO> _bench(party, ios, threads);  // just before setup_zk_arith
//   setup_zk_arith(...); ...prove...; finalize_zk_arith(...);
//   _bench.report("model-name");                          // just after finalize_zk_arith
//
// The measured window is setup_zk_arith -> finalize_zk_arith (i.e. the proving
// protocol itself), excluding model load / graph prep.  Figures are prover-side
// (ALICE); the byte counter is summed across all IO threads.
#ifndef TEST_BENCH_E2E_H
#define TEST_BENCH_E2E_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "test/utility.h"

template <class IO>
struct BenchE2E {
  int party;
  IO **ios;
  int n;
  bool on;
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;
  uint64_t b0;

  BenchE2E(int party_, IO **ios_, int n_)
      : party(party_), ios(ios_), n(n_),
        on(getenv("BENCH") || getenv("PHASES")) {
    t0 = emp::clock_start();
    b0 = bytes();
  }

  uint64_t bytes() const {
    uint64_t c = 0;
    for (int i = 0; i < n; ++i) c += io_counter(ios[i]->io);
    return c;
  }

  void report(const char *name) const {
    if (!on || party != emp::ALICE) return;
    double s = emp::time_from(t0) / 1e6;
    double mb = (double)(bytes() - b0) / 1e6;
    fprintf(stderr,
            "[BENCH] %-24s end-to-end %8.3f s  %9.2f MB comm  peakRSS %.0f MB\n",
            name, s, mb, getPeakRSS() / 1e6);
  }
};

#endif  // TEST_BENCH_E2E_H
