#ifndef NONLINEAR_COMPARISON_H
#define NONLINEAR_COMPARISON_H

#include "emp-zk/emp-zk.h"
#include "volefier/nonlinear/range_check.h"
#include "volefier/rlc_chi.h"
#include "volefier/linear/ip_matmul.h"   // parallel_for + get_ostriple
#include "volefier/linear/hadamard_ip.h"   // Hadamard2IP: batched RLC-then-IP for bitness
#include <vector>
#include <cstdint>
using namespace emp;

// CompareLU: comparison via a RANGE CHECK on the difference (DeepProve-style,
// NO sign-bit / MSB decomposition):
//
//     a <= b   iff   (b - a) in [0, 2^operand_bits)
//
// where operand_bits bounds the operands so b-a cannot wrap the field. The
// number of lookups tracks operand_bits (RangeCheckLU::num_lookups), so a
// tight bound => few lookups -- exactly DeepProve's argmax cost model, instead
// of a fixed ~60-bit MSB per comparison.
class CompareLU {
public:
  int party;
  RangeCheckLU *rc;

  explicit CompareLU(RangeCheckLU *rc_) : party(rc_->party), rc(rc_) {}

  // Optional: when set, the bitness g(g-1)=0 / u(1-u)=0 checks are DEFERRED into the shared
  // Hadamard2IP batcher -- one RLC challenge for the whole batch across all comparison sites,
  // then one weighted inner product into fp_zkp_inner_prdt -- instead of a per-call seed exchange.
  volefier::Hadamard2IP *h2ip = nullptr;

  static IntFp pub(int64_t c) {
    uint64_t r = (c >= 0) ? (uint64_t)c : (uint64_t)(c + (int64_t)PR);
    return IntFp(r, PUBLIC);
  }
  IntFp commit_signed(int64_t v) const {
    if (party == ALICE) return IntFp((v >= 0) ? (uint64_t)v : (uint64_t)(v + (int64_t)PR), ALICE);
    return IntFp(static_cast<uint64_t>(0), ALICE);
  }
  static int64_t value_of(IntFp x) {            // ALICE-side plaintext
    uint64_t raw = static_cast<uint64_t>(VAL(x.value));
    return (raw > (PR >> 1)) ? (int64_t)raw - (int64_t)PR : (int64_t)raw;
  }

  // ASSERT a <= b. operand_bits = bit length of the operands, so b-a fits in
  // operand_bits+1 bits (extra bit for the sign of the subtraction).
  void assert_le(IntFp a, IntFp b, std::size_t operand_bits) {
    IntFp diff = b + a.negate();                // b - a
    rc->check(diff, operand_bits + 1);
  }
  // ASSERT a < b   <=>   b - a - 1 >= 0.
  void assert_lt(IntFp a, IntFp b, std::size_t operand_bits) {
    IntFp diff = b + a.negate() + pub(-1);      // b - a - 1
    rc->check(diff, operand_bits + 1);
  }

  // Committed boolean g = 1{a <= b}. Prover commits g and the gadget proves
  // g in {0,1} and the SELECTED branch is non-negative:
  //   selected = g ? (b-a) : (a-b-1)   must be in [0, 2^(operand_bits+1)).
  IntFp is_le(IntFp a, IntFp b, std::size_t operand_bits) {
    IntFp diff = b + a.negate();                          // b - a
    IntFp g = commit_signed(value_of(diff) >= 0 ? 1 : 0); // committed comparison bit
    (g * (g + pub(-1))).reveal_zero();                    // g(g-1) = 0  -> g in {0,1}
    // selected = g*(2*diff+1) + (-diff-1):  g=1 -> diff ; g=0 -> -diff-1
    IntFp two_diff_p1 = diff + diff + pub(1);
    IntFp selected = g * two_diff_p1 + (diff.negate() + pub(-1));
    rc->check(selected, operand_bits + 1);
    return g;
  }

  // BATCHED is_le: g[i] = 1{a[i] <= b[i]} for all i.  Replaces N per-call bitness g(g-1)=0
  // mult-gate + reveal_zero pairs with ONE random-linear-combination inner product:
  //   sum_i chi_i * g_i * (g_i - 1) = 0,   chi_i drawn from a POST-COMMITMENT PRG (independent,
  // eps ~ 1/PR), so a lying bit survives only if the RLC vanishes.  The `selected` non-negativity
  // range checks are batched via check_batch.  (Same technique as select_max_rows' selector bits.)
  void is_le_batch(std::vector<IntFp> &g, const std::vector<IntFp> &a, const std::vector<IntFp> &b,
                   std::size_t operand_bits) {
    const std::size_t N = a.size();
    g.resize(N);
    std::vector<IntFp> selv(N);
    for (std::size_t i = 0; i < N; ++i) {
      IntFp diff = b[i] + a[i].negate();
      g[i] = commit_signed(value_of(diff) >= 0 ? 1 : 0);
      IntFp two_diff_p1 = diff + diff + pub(1);
      selv[i] = g[i] * two_diff_p1 + (diff.negate() + pub(-1));  // g ? diff : -diff-1
    }
    // bitness g_i(g_i-1)=0: deferred into the shared Hadamard2IP collector (one RLC challenge for
    // the whole batch across all comparison sites) if set, else a local per-call RLC+IP fallback.
    std::vector<IntFp> gm1(N);
    for (std::size_t i = 0; i < N; ++i) gm1[i] = g[i] + pub(-1);   // g_i - 1
    if (h2ip) {
      h2ip->add_zero(g.data(), gm1.data(), N);                // deferred: g_i(g_i-1)=0, SHARED challenge
    } else {
      uint64_t seed = 0;
      IOChannel *io = rc->io;
      io->flush();
      if (party == ALICE) io->recv_data(&seed, sizeof(uint64_t));
      else { PRG prg; prg.random_data_unaligned(&seed, sizeof(uint64_t)); seed = mod(seed);
             io->send_data(&seed, sizeof(uint64_t)); io->flush(); }
      std::vector<uint64_t> chi(N);
      volefier::prg_coeff_gen(chi.data(), seed, (int64_t)N);
      std::vector<IntFp> xs(N);
      for (std::size_t i = 0; i < N; ++i) xs[i] = g[i] * chi[i];
      fp_zkp_inner_prdt(xs.data(), gm1.data(), 0, (int64_t)N); // sum chi_i g_i (g_i-1) = 0
    }
    rc->check_batch(selv.data(), N, operand_bits + 1);        // batched non-negativity of `selected`
  }

  // Core max gadget (DeepProve-style, no MSB). The prover commits a unit
  // selector u (1 at the max position) and the gadget proves:
  //   u_i in {0,1},  sum u_i = 1,  sel = sum u_i*x_i,  and  x_i <= sel for all i.
  // => sel is a real element AND an upper bound, i.e. the max; idx = sum u_i*i.
  // Each comparison is one RangeCheckLU (not an MSB).
  void select_max(const std::vector<IntFp> &x, std::size_t operand_bits, IntFp &sel, IntFp &idx) {
    const std::size_t n = x.size();
    std::size_t maxidx = 0;
    if (party == ALICE) {
      int64_t best = INT64_MIN;
      for (std::size_t i = 0; i < n; ++i) { int64_t v = value_of(x[i]); if (v > best) { best = v; maxidx = i; } }
    }
    std::vector<IntFp> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = commit_signed((party == ALICE && i == maxidx) ? 1 : 0);

    IntFp sum_u(static_cast<uint64_t>(0), PUBLIC);
    idx = IntFp(static_cast<uint64_t>(0), PUBLIC);
    sel = IntFp(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < n; ++i) {
      (u[i] * (u[i] + pub(-1))).reveal_zero();   // u_i in {0,1}
      sum_u = sum_u + u[i];
      idx   = idx + u[i] * (uint64_t)i;          // index = sum u_i * i
      sel   = sel + u[i] * x[i];                 // selected value = x[argmax]
    }
    (sum_u + pub(-1)).reveal_zero();             // exactly one selector set
    for (std::size_t i = 0; i < n; ++i) assert_le(x[i], sel, operand_bits);  // sel >= every x_i
  }

  // Committed argmax INDEX (certification primitive).
  IntFp argmax(const std::vector<IntFp> &x, std::size_t operand_bits) {
    IntFp sel, idx; select_max(x, operand_bits, sel, idx); return idx;
  }

  // BATCHED-ROWS select_max: same constraints as select_max for `rows` rows of V logits
  // (xf row-major), but with ZERO mult gates and bulk commits:
  //   - ONE batch_feed commits all rows' selector bits (vs rows*V single feeds), and one
  //     more commits the rows selected values.
  //   - ONE post-commitment challenge seed s (a single exchange for ALL rows); each bit is
  //     bound by its own INDEPENDENT weight chi_i (PRG-expanded, rlc_chi.h) in a per-row claim
  //         sum_i chi_i * u_i * (1 - u_i) = 0
  //     (per-element random weights drawn AFTER the commits -- the FpPolyProof/LogUp
  //     pattern; a lying bit survives only if s roots the error polynomial, eps ~ N/PR).
  //   - sel_r binds by the AGGREGATE claim  sum_i u_i*x_i - sel_r = 0  (that equality IS
  //     the constraint, so no per-element weights are needed).
  //   Both claim vectors are mul_const/affine transforms of committed values (local lane
  //   math) -- u_i(1-u_i) and u_i*x_i never materialize as mult gates.  The >= dominance
  //   range checks are unchanged (rc->check per element, batched in LogUp).
  void select_max_rows(const std::vector<IntFp> &xf, std::size_t rows, std::size_t V,
                       std::size_t operand_bits, std::vector<IntFp> &sel,
                       std::vector<IntFp> &idx) {
    const std::size_t N = rows * V;
    if (N == 0 || xf.size() < N) error("select_max_rows: bad dimensions\n");
    const int NT = volefier::get_ostriple(party)->threads_;   // lane math is local -> threaded
    sel.resize(rows);
    idx.resize(rows);
    // STREAMING (Phase 1): process rows in chunks so the O(rows*V) selector and (1-u) vectors are
    // resident per-chunk instead of all at once -- the argmax over a large vocab (e.g. Gemma's 262k)
    // was the RSS peak.  Soundness is unchanged: rows are independent, and every batched challenge
    // (h2ip flush, local seed, LogUp) is still drawn POST-commitment -- chunking only changes WHEN
    // columns are fed, not what is proven, so the committed witness and revealed indices are identical.
    std::size_t budget = (std::size_t)4 << 20;   // ~4M committed elements per chunk (bounded working set)
    if (const char *e = getenv("ARGMAX_CHUNK")) budget = (std::size_t)atoll(e);
    const std::size_t C = std::max<std::size_t>(1, budget / std::max<std::size_t>(1, V));
    std::vector<IntFp> xs(V + 1), ys(V + 1);     // per-row scratch, reused across all chunks
    std::vector<uint64_t> chi(V);
    for (std::size_t r0 = 0; r0 < rows; r0 += C) {
      const std::size_t rc_ = std::min(C, rows - r0);
      const std::size_t Nc = rc_ * V;
      // 1. ALICE scans this chunk's rows; ONE bulk commit of its selector bits + selected values.
      std::vector<uint64_t> uraw(Nc, 0), selraw(rc_, 0);
      if (party == ALICE)
        volefier::parallel_for(rc_, NT, [&](std::size_t rr) {   // rows independent
          std::size_t mi = 0; int64_t best = INT64_MIN;
          const IntFp *xr = &xf[(r0 + rr) * V];
          for (std::size_t i = 0; i < V; ++i) { int64_t v = value_of(xr[i]); if (v > best) { best = v; mi = i; } }
          uraw[rr * V + mi] = 1;
          selraw[rr] = static_cast<uint64_t>(VAL(xr[mi].value));
        });
      std::vector<IntFp> u(Nc);
      batch_feed(u.data(), uraw.data(), (int64_t)Nc);
      { std::vector<IntFp> selc(rc_); batch_feed(selc.data(), selraw.data(), (int64_t)rc_);
        for (std::size_t rr = 0; rr < rc_; ++rr) sel[r0 + rr] = selc[rr]; }
      // 2. Selector bitness u_i(1-u_i)=0.  h2ip -> defer into the shared collector; else one local seed.
      const bool local_seed = (h2ip == nullptr);
      uint64_t seed = 0;
      if (h2ip) {
        std::vector<IntFp> omu(Nc);                                   // 1 - u_i (affine, local)
        volefier::parallel_for(Nc, NT, [&](std::size_t k) { omu[k] = pub(1) + u[k].negate(); });
        h2ip->add_eq(u.data(), omu.data(), 0, Nc);                    // u_i(1-u_i)==0, shared challenge
      } else {
        IOChannel *io = rc->io;
        io->flush();
        if (party == ALICE) io->recv_data(&seed, sizeof(uint64_t));
        else {
          PRG prg; prg.random_data_unaligned(&seed, sizeof(uint64_t)); seed = mod(seed);
          io->send_data(&seed, sizeof(uint64_t)); io->flush();
        }
      }
      // 3. per row: bitness claim (local path), one-hot count, index, sel claim, dominance checks.
      for (std::size_t rr = 0; rr < rc_; ++rr) {
        const IntFp *ur = &u[rr * V];
        const IntFp *xr = &xf[(r0 + rr) * V];
        if (local_seed) {   // deferred to h2ip above otherwise
          volefier::prg_coeff_range(chi.data(), seed, (uint64_t)((r0 + rr) * V), (int64_t)V);  // independent chi
          volefier::parallel_for(V, NT, [&](std::size_t i) {   // disjoint lane math
            xs[i] = ur[i] * chi[i];                // chi_i * u_i   (mul_const: local)
            ys[i] = pub(1) + ur[i].negate();       // 1 - u_i       (affine: local)
          });
          fp_zkp_inner_prdt(xs.data(), ys.data(), 0, (int64_t)V);   // sum chi*u*(1-u) = 0
        }
        // one-hot count + index: per-worker IntFp partials, serial combine (lane adds are local)
        std::vector<IntFp> psum((size_t)NT, IntFp(static_cast<uint64_t>(0), PUBLIC));
        std::vector<IntFp> pix((size_t)NT, IntFp(static_cast<uint64_t>(0), PUBLIC));
        { const std::size_t per = (V + NT - 1) / (std::size_t)NT;
          volefier::parallel_for((std::size_t)NT, NT, [&](std::size_t t) {
            IntFp a(static_cast<uint64_t>(0), PUBLIC), b(static_cast<uint64_t>(0), PUBLIC);
            const std::size_t lo = t * per, hi = std::min(V, lo + per);
            for (std::size_t i = lo; i < hi; ++i) { a = a + ur[i]; b = b + ur[i] * (uint64_t)i; }
            psum[t] = a; pix[t] = b;
          }); }
        IntFp sum_u(static_cast<uint64_t>(0), PUBLIC);
        IntFp ix(static_cast<uint64_t>(0), PUBLIC);
        for (int t = 0; t < NT; ++t) { sum_u = sum_u + psum[(size_t)t]; ix = ix + pix[(size_t)t]; }
        (sum_u + pub(-1)).reveal_zero();         // exactly one selector set
        idx[r0 + rr] = ix;
        volefier::parallel_for(V, NT, [&](std::size_t i) { xs[i] = ur[i]; ys[i] = xr[i]; });
        xs[V] = sel[r0 + rr]; ys[V] = pub(-1);
        fp_zkp_inner_prdt(xs.data(), ys.data(), 0, (int64_t)(V + 1));  // sum u*x - sel = 0
        // BATCHED dominance: diffs = sel - x_i (affine, local), then ONE bulk limb feed +
        // one batched zero-check for the whole row (vs V single-feed assert_le calls).
        volefier::parallel_for(V, NT, [&](std::size_t i) { xs[i] = sel[r0 + rr] + xr[i].negate(); });
        rc->check_batch(xs.data(), V, operand_bits + 1);
      }
      // uraw / u / omu freed here -> peak selector memory is O(C*V), not O(rows*V)
    }
  }
  // Committed max VALUE (softmax max-subtraction; replaces Comp_LOGUP::find_max).
  IntFp max_value(const std::vector<IntFp> &x, std::size_t operand_bits) {
    IntFp sel, idx; select_max(x, operand_bits, sel, idx); return sel;
  }
};

#endif
