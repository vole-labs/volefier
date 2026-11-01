#ifndef NONLINEAR_RSQRT_H
#define NONLINEAR_RSQRT_H

#include "emp-zk/emp-zk.h"
#include "volefier/nonlinear/range_check.h"   // RangeCheckLU
#include <algorithm>
#include <cmath>
#include <cstdint>
using namespace emp;

// Sqrt-free proof of the normalization factor  dinv = round(2^2P / sqrt(var))  (1/sigma at
// scale 2^-P) for a committed var in [1, 2^var_bits).  The prover commits dinv and an auxiliary
// s = round(sqrt(var * 2^2K)) (sigma with K extra fractional bits) and proves
//     0 <= s < 2^Sb,  0 <= dinv < 2^(2P+1),
//     |var * 2^2K - s^2|  <= s,                 (s is the rounded square root)
//     |2 (dinv * s - 2^(2P+K))| <= s + dinv + 4  (dinv is the rounded 2^(2P+K) / s)
// K is the largest value keeping s^2 < 2^60 and dinv * s < 2^59, so no product wraps the field
// and every relation is a true integer inequality.  The single check |dinv^2 var - 2^4P| <= TOL
// it replaces was satisfied by -dinv (negating the whole normalized row) and, through
// wraparound, by other large field elements.  Pins dinv to a relative ~1/(2s) + 1/(2 dinv).
inline IntFp prove_rsqrt(RangeCheckLU *rc, int party, const IntFp &var, int P, int var_bits) {
  const int Vh = (var_bits + 1) / 2, Db = 2 * P + 1;
  const int K = std::min(30 - Vh - 1, 59 - Db - Vh - 1);
  if (K < 0) error("prove_rsqrt: var_bits too wide for a wrap-free check\n");
  const int Sb = Vh + K + 1, Bm = std::max(Sb, Db) + 2;
  uint64_t sv = 0, dv = 0;
  if (party == ALICE) {
    const uint64_t v = static_cast<uint64_t>(VAL(var.value));
    if (v == 0 || v >> var_bits) error("prove_rsqrt: var outside [1, 2^var_bits)\n");
    dv = static_cast<uint64_t>(llround(static_cast<double>(1ULL << (2 * P)) / std::sqrt(static_cast<double>(v))));
    const uint64_t X = v << (2 * K);   // < 2^(var_bits + 2K) <= 2^60
    uint64_t r = static_cast<uint64_t>(sqrtl(static_cast<long double>(X)));
    while (r * r > X) --r;
    while ((r + 1) * (r + 1) <= X) ++r;   // r = floor(sqrt(X))
    sv = (X - r * r > r) ? r + 1 : r;     // round: X > r^2 + r  <=>  sqrt(X) > r + 1/2
  }
  IntFp s(sv, ALICE), dinv(dv, ALICE);
  rc->check(var, var_bits);
  rc->check(s, Sb);
  rc->check(dinv, Db);
  IntFp e = var * (1ULL << (2 * K)) + (s * s).negate();   // var * 2^2K - s^2
  rc->check(s + e, Sb + 1);
  rc->check(s + e.negate(), Sb + 1);
  IntFp m2 = (dinv * s + IntFp(1ULL << (2 * P + K), PUBLIC).negate()) * 2LL;
  IntFp w = s + dinv + IntFp(static_cast<uint64_t>(4), PUBLIC);
  rc->check(w + m2, Bm);
  rc->check(w + m2.negate(), Bm);
  return dinv;
}

#endif
