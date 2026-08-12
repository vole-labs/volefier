#ifndef NONLINEAR_DIV_H
#define NONLINEAR_DIV_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup.h"
#include <cstdint>
#include <vector>
using namespace emp;

// Div: zkGPT-style proof of rounded division  q = round(N / D)  with D > 0,
// using a TWO-SIDED RANGE RELATION and NO MSB / sign-bit gadget.
//
// zkGPT (neuralNetwork.cpp roundLayer / softmax_layer_2) proves a rounding by
// showing the rounding error is bounded on both sides. Rounding to nearest is
//   |N - q*D| <= D/2   <=>   -D <= 2N - 2qD <= D   <=>   both of:
//       t1 = D + 2N - 2qD  >= 0
//       t2 = D - 2N + 2qD  >= 0
// Note  t1 + t2 = 2D  is a field identity, independent of q. So if BOTH t1 and
// t2 are range-checked into [0, 2^B) with 2^(B+1) < PR, their integer sum is
// forced to equal 2D, which pins each into [0, 2D] -> both >= 0 -> the rounding
// is correct AND q is uniquely the honest round(N/D).
//
// Cost per division: ONE committed multiply (q*D) + TWO non-negative range
// checks (chunked LogUp). No squaring (so no field wraparound) and, crucially,
// no MSB sign extraction -- the whole point of replacing Div_LOGUP::divide_com,
// which costs 6 MSB calls.
//
// Works for signed N (q may be negative); D must be > 0 (the softmax / rescale
// denominator case). For D < 0 negate N and D first.
//
//   Usage:
//     Div div(party, io, /*range_bits=*/ B);   // B covers 2*D_max
//     div.range_nonneg(...) / IntFp q = div.round_div(N, D);
//     div.destroy();   // before finalize_zk_arith
class Div {
public:
  int party;
  IOChannel *io;
  std::size_t chunk_bits;   // LogUp range-table width (e.g. 12 -> table [0,2^12))
  std::size_t num_chunks;   // a range-checked term lies in [0, 2^(chunk_bits*num_chunks))
  LogUpRangeCheck *rc = nullptr;
  bool flag_teardown = false;
  bool owns_rc = true;      // false when `rc` is borrowed (shared with e.g. QuantInfer::rc_rem)

  // range_bits: an upper bound (in bits) on the largest term value (~ 2*D_max).
  // chunk_bits decomposition keeps the LogUp table small; the covered range is
  // rounded up to a multiple of chunk_bits and must stay below ~2^59.
  Div(int party, IOChannel *io, std::size_t range_bits, std::size_t chunk_bits = 12)
      : party(party), io(io), chunk_bits(chunk_bits) {
    num_chunks = (range_bits + chunk_bits - 1) / chunk_bits;
    rc = new LogUpRangeCheck(static_cast<uint64_t>(1) << chunk_bits, party, io);
    rc->Setup();
  }

  // SHARED-TABLE variant: borrow an existing [0,2^chunk_bits) range table (e.g. QuantInfer::rc_rem)
  // instead of allocating a second identical one -- both are plain membership tables, so ONE LogUp
  // instance backs both requant and division range checks (one fewer table to Setup + Prove).  The
  // owner Setup()s and Teardown()s `shared_rc`; Div only Access()es it and never frees it.
  // `chunk_bits` MUST equal the shared table's width (log2 of its size).
  Div(int party, IOChannel *io, std::size_t range_bits, std::size_t chunk_bits,
      LogUpRangeCheck *shared_rc)
      : party(party), io(io), chunk_bits(chunk_bits), owns_rc(false) {
    num_chunks = (range_bits + chunk_bits - 1) / chunk_bits;
    rc = shared_rc;   // borrowed: already Setup by its owner, torn down by its owner
  }

  ~Div() { if (!flag_teardown) destroy(); }

  void destroy() {
    if (flag_teardown) return;
    if (owns_rc) { rc->Teardown(); delete rc; }
    rc = nullptr;   // borrowed tables are torn down + freed by their owner
    flag_teardown = true;
  }

  static int64_t decode_signed(uint64_t v) {
    return (v > (PR >> 1)) ? static_cast<int64_t>(v) - static_cast<int64_t>(PR)
                           : static_cast<int64_t>(v);
  }

  // Prove a committed v lies in [0, 2^(chunk_bits*num_chunks)) by chunk
  // decomposition: each chunk is LogUp-range-checked into [0, 2^chunk_bits) and
  // the chunks are proven to reconstruct v.  v must be honestly non-negative.
  void range_nonneg(IntFp v) {
    std::vector<IntFp> chunk(num_chunks);
    if (party == ALICE) {
      uint64_t val = static_cast<uint64_t>(VAL(v.value));
      const uint64_t mask = (static_cast<uint64_t>(1) << chunk_bits) - 1;
      for (std::size_t i = 0; i < num_chunks; ++i)
        chunk[i] = IntFp((val >> (chunk_bits * i)) & mask, ALICE);
    } else {
      for (std::size_t i = 0; i < num_chunks; ++i) chunk[i] = IntFp(static_cast<uint64_t>(0), ALICE);
    }

    IntFp recon(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < num_chunks; ++i) {
      rc->Access(chunk[i]);                                       // chunk in [0, 2^chunk_bits)
      recon = recon + chunk[i] * (static_cast<int64_t>(1) << (chunk_bits * i));
    }
    (recon + v.negate()).reveal_zero();                           // reconstruction == v
  }

  // Prove and return  q = round(N / D),  D committed and > 0.
  IntFp round_div(IntFp N, IntFp D) {
    IntFp q;
    if (party == ALICE) {
      int64_t Nv = decode_signed(static_cast<uint64_t>(VAL(N.value)));
      int64_t Dv = decode_signed(static_cast<uint64_t>(VAL(D.value)));
      // round half away from zero (any q with |N - q*D| <= D/2 is accepted)
      int64_t qv = (Nv >= 0) ?  (Nv + Dv / 2) / Dv
                             : -(((-Nv) + Dv / 2) / Dv);
      uint64_t raw = (qv >= 0) ? static_cast<uint64_t>(qv)
                               : static_cast<uint64_t>(qv + static_cast<int64_t>(PR));
      q = IntFp(raw, ALICE);
    } else {
      q = IntFp(static_cast<uint64_t>(0), ALICE);
    }

    IntFp qD = q * D;                               // the only committed multiply
    IntFp t1 = D + N * 2LL + (qD * 2LL).negate();   // D + 2N - 2qD  >= 0
    IntFp t2 = D + (N * 2LL).negate() + qD * 2LL;   // D - 2N + 2qD  >= 0
    range_nonneg(t1);
    range_nonneg(t2);
    return q;
  }

  // PUBLIC divisor overload (LayerNorm mean/variance, avg-pool): D is a cleartext constant, so
  // q*D and D+... are mul_const/add-const (LOCAL) -- not the IntFp*IntFp mult-gate the committed-D
  // path takes.  Same rounding relation.
  IntFp round_div(IntFp N, uint64_t D) {
    IntFp q;
    if (party == ALICE) {
      int64_t Nv = decode_signed(static_cast<uint64_t>(VAL(N.value)));
      int64_t Dv = (int64_t)D;
      int64_t qv = (Nv >= 0) ?  (Nv + Dv / 2) / Dv
                             : -(((-Nv) + Dv / 2) / Dv);
      uint64_t raw = (qv >= 0) ? static_cast<uint64_t>(qv)
                               : static_cast<uint64_t>(qv + static_cast<int64_t>(PR));
      q = IntFp(raw, ALICE);
    } else {
      q = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    IntFp qD = q * D;                                          // mul_const (local)
    IntFp t1 = N * 2LL + (qD * 2LL).negate() + IntFp(D, PUBLIC);   // D + 2N - 2qD >= 0
    IntFp t2 = (N * 2LL).negate() + qD * 2LL + IntFp(D, PUBLIC);   // D - 2N + 2qD >= 0
    range_nonneg(t1);
    range_nonneg(t2);
    return q;
  }
};

#endif
