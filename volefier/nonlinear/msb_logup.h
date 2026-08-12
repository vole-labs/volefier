#ifndef NONLINEAR_MSB_LOGUP_H
#define NONLINEAR_MSB_LOGUP_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup.h"
#include <cstdint>
using namespace emp;

// MSB_LOGUP: same MSB computation as MSB (msb.h), but uses a single
// LogUp range check in place of the ZKROM on the identity table [0, 4096).
//
// LogUp's range check has no notion of a per-teardown chunk size, so the
// ram_batch_block_sz parameter that MSB takes is dropped here.  ram_batch_sz
// is kept as an ignored parameter for API symmetry with MSB.

class MSB_LOGUP {
public:
  // public variables
  int party;
  LogUpRangeCheck *rc2To12 = nullptr;
  IOChannel *io = nullptr;
  bool flag_teardown;

  // fix n = 61, decide how the lower n-2=59 bits can be decomposed
  std::size_t divide_n[5] = {12, 12, 12, 12, 12};
  std::size_t num_section = 5;
  std::size_t fp_bit_sz = 61;

  // constructor
  MSB_LOGUP(int party, std::size_t /*ram_batch_sz*/, IOChannel *io)
      : party(party), io(io) {
    // Range check for [0, 4096).  The table is implicitly identity, so
    // we don't need to populate any values — LogUpRangeCheck handles it.
    rc2To12 = new LogUpRangeCheck(1LL<<12, party, io);
    rc2To12->Setup();
    flag_teardown = false;
  }

  // deconstructor
  ~MSB_LOGUP() {

  }

  void destroy() {
    rc2To12->Teardown();
    delete rc2To12;
    flag_teardown = true;
  }

  // input a committed x
  // output MSB(x)
  IntFp compute(IntFp x) {

    // 1. decompose x and commit to indices
    std::vector<uint64_t> indices(num_section);
    IntFp w;
    IntFp msb;

    // decompose x
    if(party == ALICE) {
      uint64_t val = static_cast<uint64_t>(x.value>>64);
      uint64_t msb_raw = val >> (fp_bit_sz - 1);

      uint64_t w_raw = (uint64_t)(bool)val;
      w = IntFp(w_raw, ALICE);

      uint64_t select[2];
      select[0] = val;
      select[1] = (uint64_t)p - val;
      val = select[msb_raw]; // abstract of input

      bool *bits = new bool[fp_bit_sz];
      for (std::size_t _i = 0; _i < (std::size_t)fp_bit_sz; ++_i) bits[_i] = (val >> _i) & 1; // was int_to_bool

      msb = IntFp(msb_raw, ALICE);
      std::size_t ptr = 0;
      for(std::size_t i = 0; i < num_section; ++i) {
        indices[i] = 0LL;
        for(std::size_t j = 0; j < divide_n[i]; ++j) {
          indices[i] += (((uint64_t)bits[ptr]) << j);
          ptr++;
        }
      }
      delete[] bits;
    } else {
      w = IntFp(false, ALICE);
      msb = IntFp(false, ALICE);
    }

    // indices
    std::vector<IntFp> com_indices(num_section);
    for(std::size_t i = 0; i < num_section; ++i) {
      com_indices[i] = IntFp(indices[i], ALICE);
    }

    // 2. lookup bits from tables (LogUp range-check version).
    // rc2To12->Access buffers the index and returns it unchanged (identity
    // lookup); the range-check proof is deferred to destroy().
    std::vector<IntFp> reads(num_section);
    for(std::size_t i = 0; i < num_section; ++i) {
      reads[i] = rc2To12->Access(com_indices[i]);
    }

    // 3. prove they are consistent
    IntFp check_z(reads[0]);
    check_z = check_z + (reads[1] * (1LL << 12));
    check_z = check_z + (reads[2] * (1LL << 24));
    check_z = check_z + (reads[3] * (1LL << 36));
    check_z = check_z + (reads[4] * (1LL << 48));
    check_z = check_z * (IntFp(1LL, PUBLIC) + (msb.negate() * 2LL));
    check_z = check_z + x.negate();

    // TODO: a better way for check_zero(check_z)
    check_z.reveal_zero();

    IntFp check_one_or_zero = msb * (IntFp(1LL, PUBLIC) + msb.negate());
    check_one_or_zero.reveal_zero();
    
    check_nonzero(x,w);

    IntFp soundness_check;
    soundness_check = msb * (IntFp(1LL, PUBLIC) + w.negate());
    soundness_check.reveal_zero();

    return msb;
  }

  // input a committed vector x
  // output MSB(x_i) for all x_i in x
  void compute(std::vector<IntFp> &y, 
      std::vector<IntFp> &x, 
      std::size_t num) {

    // 1. decompose x and commit to indices
    std::vector<uint64_t> indices(num * num_section);
    std::vector<IntFp> w(num);
    // decompose x
    if(party == ALICE) {
      bool *bits = new bool[fp_bit_sz];
      for(std::size_t k = 0; k < num; ++k) {
        uint64_t val = static_cast<uint64_t>(x[k].value>>64);
        uint64_t msb_raw = val >> (fp_bit_sz - 1);
        
        uint64_t w_raw = (uint64_t)(bool)val;
        w[k] = IntFp(w_raw, ALICE);

        uint64_t select[2];
        select[0] = val;
        select[1] = (uint64_t)p - val;
        val = select[msb_raw]; // abstract of input

        for (std::size_t _i = 0; _i < (std::size_t)fp_bit_sz; ++_i) bits[_i] = (val >> _i) & 1; // was int_to_bool

        y[k] = IntFp(msb_raw, ALICE);
        std::size_t ptr = 0;
        for(std::size_t i = 0; i < num_section; ++i) {
          indices[k*num_section+i] = 0LL;
          for(std::size_t j = 0; j < divide_n[i]; ++j) {
            indices[k*num_section+i] += (((uint64_t)bits[ptr]) << j);
            ptr++;
          }
        }
      }
      delete[] bits;
    } else {
      for(std::size_t k = 0; k < num; ++k){
        w[k] = IntFp(false, ALICE);
        y[k] = IntFp(false, ALICE);
      }
    }

    // indices
    std::vector<IntFp> com_indices(num*num_section);
    batch_feed(com_indices.data(), indices.data(), num*num_section);

    // 2. lookup bits from tables (LogUp range-check version).
    std::vector<IntFp> reads(num*num_section);
    std::size_t ptr = 0;
    for(std::size_t k = 0; k < num; ++k) {
      for(std::size_t i = 0; i < num_section; ++i) {
        reads[ptr] = rc2To12->Access(com_indices[ptr]);
        ptr++;
      }
    }

    // 3. prove they are consistent
    std::vector<IntFp> check_z(num);
    ptr = 0;
    for(std::size_t k = 0; k < num; ++k) {
      check_z[k] = reads[ptr];
      check_z[k] = check_z[k] + (reads[ptr+1] * (1LL << 12));
      check_z[k] = check_z[k] + (reads[ptr+2] * (1LL << 24));
      check_z[k] = check_z[k] + (reads[ptr+3] * (1LL << 36));
      check_z[k] = check_z[k] + (reads[ptr+4] * (1LL << 48));
      check_z[k] = check_z[k] * (IntFp(1LL, PUBLIC) + (y[k].negate() * 2LL));
      check_z[k] = check_z[k] + x[k].negate();
      ptr+=5;
    }

    // TODO: a better way for check_zero(check_z)
    batch_reveal_check_zero(check_z.data(), num);

    vector<IntFp> check_one_or_zero(num);
    for(std::size_t i = 0; i < num; ++i){
      check_one_or_zero[i] = y[i] * (IntFp(1LL, PUBLIC) + y[i].negate());
      
      check_nonzero(x[i],w[i]);
      
      IntFp soundness_check = y[i] * (IntFp(1LL, PUBLIC) + w[i].negate());
      soundness_check.reveal_zero();
    }
    batch_reveal_check_zero(check_one_or_zero.data(), num);
  }

  void check_nonzero(IntFp x, IntFp y){
    IntFp u;
    if (party == ALICE){
      uint64_t x_raw = static_cast<uint64_t>(x.value>>64);
      uint64_t u_raw = inv_mod(x_raw);
      u = IntFp(u_raw, ALICE);
    } else {
      u = IntFp(false, ALICE);
    }
    IntFp check_1 = y + u * x.negate();
    IntFp check_2 = (IntFp(1, PUBLIC) + y.negate()) * x ;
    check_1.reveal_zero();
    check_2.reveal_zero();
  }

  static inline uint64_t reduce_mod_p(__uint128_t x) {
      // Fold high bits: x = (low 61 bits) + (x >> 61)
      uint64_t lo = (uint64_t)(x & PR);
      uint64_t hi = (uint64_t)(x >> 61);
      __uint128_t t = ( (__uint128_t)lo + (__uint128_t)hi );

      // One more fold (because t may still exceed 2^61)
      lo = (uint64_t)(t & PR);
      hi = (uint64_t)(t >> 61);
      t = ( (__uint128_t)lo + (__uint128_t)hi );

      uint64_t r = (uint64_t)t;
      // Final normalization
      if (r >= PR) r -= PR;
      return r;
  }

  static inline uint64_t mul_mod(uint64_t a, uint64_t b) {
      // a,b assumed in [0, PR-1]
      __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
      return reduce_mod_p(prod);
  }

  static inline uint64_t pow_mod(uint64_t a, uint64_t e) {
      // a in [0, PR-1]
      uint64_t r = 1;
      while (e) {
          if (e & 1) r = mul_mod(r, a);
          a = mul_mod(a, a);
          e >>= 1;
      }
      return r;
  }

  // Fermat inverse: a^(PR-2) mod PR; define inv(0)=0
  static inline uint64_t inv_mod(uint64_t a) {
      a = mod(a);
      if (a == 0) return 0;
      return pow_mod(a, PR - 2);
  }

};

#endif
