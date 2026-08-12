#ifndef EMP_ZKML_RLC_CHI_H
#define EMP_ZKML_RLC_CHI_H

#include "emp-zk/emp-zk.h"
#include <vector>
#include <cstring>
#include <algorithm>

// PRG-INDEPENDENT RLC coefficients, replacing the s^(i+1) power chains everywhere a
// random-linear-combination check needs per-element weights.
//
//   power chain:  a cheating prover survives iff the seed s is a root of the degree-n
//                 error polynomial  -> failure probability n/PR (~2^-36 at n = 32M).
//   independent:  each chi_i is a fresh uniform value expanded from the seed by AES-PRG
//                 -> the single linear combination vanishes with probability 1/PR (~2^-61),
//                 independent of n.
//
// Layout: fixed 4096-element blocks; block b is keyed on (seed ^ const, b), so the stream
// is deterministic given the seed, independent of thread count / segmentation (both parties
// MUST derive identical chi), and embarrassingly parallel (any worker can produce any block).
// The `mod` reduction of a 64-bit uniform to PR = 2^61-1 leaves the usual ~2^-3 bias per
// value, which only shifts the failure bound by that constant factor.

namespace volefier {

static constexpr int64_t kChiBlk = 4096;

inline void prg_chi_block(uint64_t seed, uint64_t blk, uint64_t *w /* kChiBlk */) {
  emp::block sb = emp::makeBlock(seed ^ 0x9e3779b97f4a7c15ULL, blk);
  emp::PRG prg(&sb);
  prg.random_data(w, (int)(kChiBlk * 8));
  for (int64_t j = 0; j < kChiBlk; ++j) w[j] = emp::mod(w[j]);
}

// chi for GLOBAL indices [lo, lo+n) of the seed's stream -- lets callers slice one logical
// stream (e.g. per-row segments of a rows*V argmax) without per-slice reseeding.
inline void prg_coeff_range(uint64_t *coeff, uint64_t seed, uint64_t lo, int64_t n) {
  if (n <= 0) return;
  std::vector<uint64_t> w((size_t)kChiBlk);
  const uint64_t hi = lo + (uint64_t)n;
  for (uint64_t b = lo / kChiBlk; b * kChiBlk < hi; ++b) {
    prg_chi_block(seed, b, w.data());
    const uint64_t base = b * (uint64_t)kChiBlk;
    const uint64_t s = std::max(base, lo), e = std::min(base + (uint64_t)kChiBlk, hi);
    std::memcpy(coeff + (s - lo), w.data() + (s - base), (size_t)(e - s) * 8);
  }
}

inline void prg_coeff_gen(uint64_t *coeff, uint64_t seed, int64_t n) {
  prg_coeff_range(coeff, seed, 0, n);
}

}  // namespace volefier

#endif
