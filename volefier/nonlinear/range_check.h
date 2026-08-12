#ifndef NONLINEAR_RANGE_CHECK_H
#define NONLINEAR_RANGE_CHECK_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup.h"   // LogUpRangeCheck
#include "volefier/linear/ip_matmul.h"   // parallel_for + get_ostriple
#include <vector>
#include <cstdint>
using namespace emp;

// RangeCheckLU: a lookup-table range check with NO fixed public bound on the
// width of the checked value. ONE fixed-size table of 2^table_bits backs range
// checks of ANY bit length -- the caller passes the bound's bit length and the
// gadget decides how many lookups are needed:
//
//     #lookups(bit_length) = ceil(bit_length / table_bits)
//
// A bit_length-bit value v is split into that many table_bits-wide limbs, each
// looked up in the single table, and the limbs are proven to reconstruct v.
// This is DeepProve's "shift-check" approach (one wide lookup table, lookup
// count proportional to the operand width) -- as opposed to a fixed-width MSB
// bit-decomposition that pays for ~60 bits regardless of the actual operand.
class RangeCheckLU {
public:
  int party;
  IOChannel *io;
  std::size_t table_bits;       // width of the backing lookup table (e.g. 16)
  uint64_t table_size;          // 2^table_bits
  LogUpRangeCheck *table = nullptr;
  bool torn = false;

  RangeCheckLU(int party, IOChannel *io, std::size_t table_bits = 16)
      : party(party), io(io), table_bits(table_bits),
        table_size(static_cast<uint64_t>(1) << table_bits) {
    table = new LogUpRangeCheck(table_size, party, io);
    table->Setup();
  }
  ~RangeCheckLU() { Teardown(); }

  // Must run before finalize_zk_arith().
  void Teardown() {
    if (torn) return;
    table->Teardown();
    delete table; table = nullptr;
    torn = true;
  }

  // Number of table lookups a bit_length-bit value needs.
  std::size_t num_lookups(std::size_t bit_length) const {
    std::size_t n = (bit_length + table_bits - 1) / table_bits;
    return n == 0 ? 1 : n;
  }

  // Prove committed v lies in [0, 2^(num_lookups*table_bits)) -- a superset of
  // [0, 2^bit_length) (rounded up to a whole number of limbs, like DeepProve's
  // div_ceil). Sound for non-negativity as long as the true |v| < 2^bit_length
  // and 2^(num_lookups*table_bits) < PR (so a negative v would wrap to a huge
  // field element outside the limb range).
  void check(IntFp v, std::size_t bit_length) {
    const std::size_t nchunks = num_lookups(bit_length);
    // GUARD: sound only while the limb range 2^(nchunks*table_bits) stays BELOW the field
    // size -- a wrapped negative (~PR) must be UNrepresentable by the limbs.  Once
    // nchunks*table_bits >= log2(PR), EVERY field element decomposes and the check is
    // VACUOUS (e.g. ARGBITS=48 -> bit_length 49 -> 4x16 = 64 bits > 61).  Fail loudly.
    if (nchunks * table_bits >= 64 ||
        (static_cast<__uint128_t>(1) << (nchunks * table_bits)) >= static_cast<__uint128_t>(PR))
      error("RangeCheckLU::check: 2^(nchunks*table_bits) >= field size -> vacuous range check; "
            "reduce operand_bits (with a 16-bit table the max sound bit_length is 48, i.e. operand_bits <= 47)\n");
    if (nchunks <= 1) { table->Access(v); return; }   // v is the single limb
    const uint64_t mask = table_size - 1;
    std::vector<IntFp> chunk(nchunks);
    if (party == ALICE) {
      uint64_t val = static_cast<uint64_t>(VAL(v.value));
      for (std::size_t i = 0; i < nchunks; ++i)
        chunk[i] = IntFp((val >> (table_bits * i)) & mask, ALICE);
    } else {
      for (std::size_t i = 0; i < nchunks; ++i) chunk[i] = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    IntFp recon(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < nchunks; ++i) {
      table->Access(chunk[i]);                                   // limb in [0, 2^table_bits)
      recon = recon + chunk[i] * (static_cast<uint64_t>(1) << (table_bits * i));
    }
    (recon + v.negate()).reveal_zero();                          // limbs reconstruct v
  }

  // BATCHED range check of n values: identical constraints to n check() calls, but ALL
  // n*nchunks limbs are committed with ONE batch_feed (bulk VOLE + one send, vs n*nchunks
  // single-element feeds) and the n reconstruction identities close with ONE batched
  // zero-check.  Same bytes on the wire (the limb corrections are needed either way);
  // the saving is the per-call feed/check overhead at tens of millions of elements.
  void check_batch(const IntFp *v, std::size_t n, std::size_t bit_length) {
    if (n == 0) return;
    const std::size_t nchunks = num_lookups(bit_length);
    if (nchunks * table_bits >= 64 ||
        (static_cast<__uint128_t>(1) << (nchunks * table_bits)) >= static_cast<__uint128_t>(PR))
      error("RangeCheckLU::check_batch: 2^(nchunks*table_bits) >= field size -> vacuous range check; "
            "reduce operand_bits (with a 16-bit table the max sound bit_length is 48, i.e. operand_bits <= 47)\n");
    if (nchunks <= 1) {
      for (std::size_t i = 0; i < n; ++i) { IntFp t = v[i]; table->Access(t); }
      return;
    }
    const uint64_t mask = table_size - 1;
    const int NT = volefier::get_ostriple(party)->threads_;
    std::vector<uint64_t> craw(n * nchunks, 0);
    if (party == ALICE)
      volefier::parallel_for(n, NT, [&](std::size_t i) {         // disjoint limb extraction
        uint64_t val = static_cast<uint64_t>(VAL(v[i].value));
        for (std::size_t k = 0; k < nchunks; ++k)
          craw[i * nchunks + k] = (val >> (table_bits * k)) & mask;
      });
    std::vector<IntFp> ch(n * nchunks);
    batch_feed(ch.data(), craw.data(), (int64_t)(n * nchunks));
    std::vector<IntFp> zchk(n);
    volefier::parallel_for(n, NT, [&](std::size_t i) {           // disjoint recon lane math
      IntFp recon(static_cast<uint64_t>(0), PUBLIC);
      for (std::size_t k = 0; k < nchunks; ++k)
        recon = recon + ch[i * nchunks + k] * (static_cast<uint64_t>(1) << (table_bits * k));
      zchk[i] = recon + v[i].negate();
    });
    for (std::size_t i = 0; i < n; ++i)                          // SERIAL: shared witness buffer,
      for (std::size_t k = 0; k < nchunks; ++k)                  // same witness order as before
        table->Access(ch[i * nchunks + k]);
    batch_reveal_check_zero(zchk.data(), (int64_t)n);            // limbs reconstruct each v
  }
};

#endif
