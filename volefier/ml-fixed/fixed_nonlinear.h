#ifndef ML_FIXED_NONLINEAR_H
#define ML_FIXED_NONLINEAR_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-fixed/fixed_infer.h"
#include "volefier/nonlinear/comparison.h" // CompareLU (max_value, replaces find_max)
#include "volefier/nonlinear/div.h"
#include "volefier/zk-ram/logup_keyvalue.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using namespace emp;

// Non-linear transformer operators built from emp-zk + LogUp, following
// DeepProve's "advice + lookup / sqrt-free" recipes:
//   - activations (exp, GeLU) are PUBLIC function tables looked up with LogUp;
//     the key-value binding simultaneously range-checks the index into [0, T).
//   - softmax = max-subtract (Comp) -> exp lookup -> sum -> committed divide.
//   - LayerNorm avoids a sqrt circuit: the prover commits d ~ 1/sigma and proves
//     the well-formedness  | d^2 * var - 2^4p | <= TOL  with two sign checks.

// ---------------------------------------------------------------------------
// FixedTable: a read-only public-function lookup.  Setup() loads T raw field
// values; lookup(idx) returns the committed table[idx] and (via the LogUp
// key-value binding at Teardown) forces idx in [0, T).
class FixedTable {
public:
  uint64_t T;
  int party;
  LogUpKeyValueLookup *tab = nullptr;

  FixedTable(uint64_t T_, int party_, IOChannel *io) : T(T_), party(party_) {
    tab = new LogUpKeyValueLookup(T_, party_, io);
  }
  ~FixedTable() {
    delete tab;
  }

  void Setup(const std::vector<uint64_t> &values) {
    tab->Setup(values);
  }
  IntFp lookup(IntFp idx) {
    return tab->Access(idx);
  }
  // batched: ONE feed for all returned values (DeepProve LSE softmax / activation batching).
  void lookup(std::vector<IntFp> &out, const std::vector<IntFp> &idx) {
    tab->Access(out, idx);
  }
  void Teardown() {
    tab->Teardown();
  }
};

// ---- host-side table builders (scale 2^-P) --------------------------------

// exp table for softmax:  expT[t] = round( exp(-t * 2^-P) * 2^P ), clamped >= 1
// so a softmax denominator is never zero (matches zkGPT's max(.,1)).  out_bits > P stores the
// output at the finer scale 2^-out_bits (the division softmax is scale-free in the table output),
// which keeps a long tail of tiny terms -- e.g. a 50k-token vocabulary -- from rounding away.
static inline std::vector<uint64_t> build_exp_table(uint64_t T, int P, bool clamp_pos = true,
                                                    int out_bits = -1) {
  std::vector<uint64_t> e(T);
  const double scale = static_cast<double>(1ULL << P);
  const double oscale = static_cast<double>(1ULL << (out_bits < 0 ? P : out_bits));
  for (uint64_t t = 0; t < T; ++t) {
    long long v = llround(std::exp(-static_cast<double>(t) / scale) * oscale);
    // clamp_pos=true (division path): >= 1 so the softmax denominator is never zero.
    // clamp_pos=false (LSE path): exact exp -> 0 for large t, so the row sums to exactly 2^P.
    e[t] = static_cast<uint64_t>(clamp_pos ? (v < 1 ? 1 : v) : (v < 0 ? 0 : v));
  }
  return e;
}

// GeLU table over inputs idx-OFFSET (scale 2^-P).  Output is signed and encoded
// into the field (negative -> +PR), so callers decode with decode_signed.
static inline std::vector<uint64_t> build_gelu_table(uint64_t T, int P, int64_t offset) {
  std::vector<uint64_t> g(T);
  const double scale = static_cast<double>(1ULL << P);
  const double k = std::sqrt(2.0 / M_PI);
  for (uint64_t i = 0; i < T; ++i) {
    double x = (static_cast<double>(static_cast<int64_t>(i) - offset)) / scale;
    double gelu = 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
    long long q = llround(gelu * scale);
    g[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return g;
}

// tanh table over inputs idx-OFFSET (scale 2^-P).  Output signed (BERT pooler), field-encoded.
static inline std::vector<uint64_t> build_tanh_table(uint64_t T, int P, int64_t offset) {
  std::vector<uint64_t> g(T);
  const double scale = static_cast<double>(1ULL << P);
  for (uint64_t i = 0; i < T; ++i) {
    double x = (static_cast<double>(static_cast<int64_t>(i) - offset)) / scale;
    long long q = llround(std::tanh(x) * scale);
    g[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return g;
}

// ReLU clamp table over inputs idx-OFFSET:  relu[idx] = max(0, idx - OFFSET).
// DeepProve treats ReLU as just another activation lookup (like GeLU), not a
// sign-bit multiply.  Outputs are non-negative, so no field encoding is needed.
static inline std::vector<uint64_t> build_relu_table(uint64_t T, int64_t offset) {
  std::vector<uint64_t> r(T);
  for (uint64_t i = 0; i < T; ++i) {
    int64_t x = static_cast<int64_t>(i) - offset;
    r[i] = static_cast<uint64_t>(x > 0 ? x : 0);
  }
  return r;
}

// ---------------------------------------------------------------------------
// Softmax over a committed logit vector (scale 2^-P), normalized to scale 2^-P
// (i.e. probabilities times 2^P).  expt must be Setup with build_exp_table.
class FixedSoftmax {
public:
  FixedInfer *qi;
  CompareLU *cmp; // lookup-table max (DeepProve-style, no MSB)
  FixedTable *expt;
  Div *divr; // zkGPT-style division (no MSB) for the normalization
  int P;
  std::size_t score_bits; // bit width of (max - logit); = log2(exp table size)
  bool lse = false;       // DeepProve LogSumExp path: no max_value, no division
  int out_bits = -1;      // LSE only: probabilities come out at 2^-out_bits (< 0 -> P); the exp
                          // table must be built with the same out_bits.  Each output's rounding
                          // adds 1/2 ulp to the normalization slack, so at long rows a finer
                          // output scale is what keeps the proven sum close to 1.
  double max_slack = 0;   // LSE only: if > 0, refuse a row whose normalization tolerance err/2^ob
                          // (the factor a prover may scale the row by) exceeds this

  // LSE normalization tolerance for a row of L, in output ulps: per-element rounding (<= L/2) +
  // shift quantization (~2^ob * in_scale) + slack.  Relative to the row sum it is err / 2^ob.
  uint64_t lse_err(std::size_t L) const {
    const double isc =
        (in_scale > 0) ? in_scale : 1.0 / static_cast<double>(static_cast<uint64_t>(1) << P);
    const int ob = (out_bits < 0) ? P : out_bits;
    return static_cast<uint64_t>(L) / 2 +
           static_cast<uint64_t>(std::ceil(std::pow(2.0, ob) * isc)) + 4;
  }
  double lse_slack(std::size_t L) const {
    return static_cast<double>(lse_err(L)) / std::pow(2.0, (out_bits < 0) ? P : out_bits);
  }
  double in_scale = -1.0; // exp-table input scale (default < 0 -> 2^-P)

  FixedSoftmax(FixedInfer *qi_, CompareLU *cmp_, FixedTable *expt_, Div *divr_, int P_,
               std::size_t score_bits_ = 16)
      : qi(qi_), cmp(cmp_), expt(expt_), divr(divr_), P(P_), score_bits(score_bits_) {}

  // logits: length L. returns probabilities at scale 2^-P.
  std::vector<IntFp> forward(std::vector<IntFp> &logits) {
    if (lse) return forward_lse_auto(logits);
    const std::size_t L = logits.size();
    IntFp m = cmp->max_value(logits, score_bits);

    std::vector<IntFp> e(L);
    IntFp sum(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < L; ++i) {
      IntFp t = m + logits[i].negate(); // t = max - logit >= 0, bound to [0,T)
      e[i] = expt->lookup(t);
      sum = sum + e[i];
    }

    std::vector<IntFp> out(L);
    const IntFp two_p(static_cast<uint64_t>(1) << P, PUBLIC);
    for (std::size_t i = 0; i < L; ++i)
      out[i] = divr->round_div(e[i] * two_p, sum); // round((e_i * 2^P) / sum), no MSB
    return out;
  }

  // DeepProve LogSumExp softmax: NO max_value, NO division.  softmax_i =
  // exp_table[(m - logit_i) + shift], with m committed as a plain upper-bound witness and
  // shift = round(log(sum_exp) * 2^P) committed too.  Soundness comes from the row summing
  // to ~2^P (which forces shift = logsumexp, so the output is invariant to the choice of m).
  // The exp table MUST be built non-clamped (build_exp_table(., ., false)).
  std::vector<IntFp> forward_lse_auto(std::vector<IntFp> &logits) {
    const std::size_t L = logits.size();
    const double isc =
        (in_scale > 0) ? in_scale : 1.0 / static_cast<double>(static_cast<uint64_t>(1) << P);
    const uint64_t err = lse_err(L);
    if (max_slack > 0 && lse_slack(L) > max_slack)
      throw std::runtime_error("LSE softmax: normalization slack " + std::to_string(lse_slack(L)) +
                               " for a row of " + std::to_string(L) + " exceeds max_slack " +
                               std::to_string(max_slack) + " (raise the exp table's out_bits)");
    std::size_t wfb = 0;
    for (uint64_t v = 2 * err; v; v >>= 1) ++wfb;
    std::vector<int64_t> rl;
    const int64_t *rlp = nullptr;
    if (qi->party == ALICE) {
      rl.resize(L);
      for (std::size_t j = 0; j < L; ++j)
        rl[j] = FixedInfer::decode_signed(static_cast<uint64_t>(VAL(logits[j].value)));
      rlp = rl.data();
    }
    return forward_lse(logits, rlp, cmp->rc, err, wfb, isc);
  }

  std::vector<IntFp> forward_lse(std::vector<IntFp> &logits, const int64_t *rlog, RangeCheckLU *rc,
                                 uint64_t err, std::size_t wf_bits, double isc) {
    const std::size_t L = logits.size();
    int64_t mref = INT64_MIN; // ALICE commits the row max as a plain upper-bound witness (one commit)
    if (qi->party == ALICE && rlog)
      for (std::size_t i = 0; i < L; ++i) if (rlog[i] > mref) mref = rlog[i];
    const int64_t mv = (qi->party == ALICE && rlog) ? mref : 0; // BOB commits 0
    IntFp m(mv >= 0 ? static_cast<uint64_t>(mv) : PR - static_cast<uint64_t>(-mv), ALICE);
    std::vector<IntFp> t(L);
    for (std::size_t i = 0; i < L; ++i) t[i] = m + logits[i].negate(); // t_i = m - logit_i >= 0
    int64_t shift_v = 0; // ALICE: shift = round(log(sum_exp) * 2^P)
    if (qi->party == ALICE && rlog) {
      double sum = 0;
      for (std::size_t i = 0; i < L; ++i)
        sum += std::exp(-static_cast<double>(mref - rlog[i]) * isc);
      shift_v = static_cast<int64_t>(llround(std::log(sum) / isc));
      if (shift_v < 0) shift_v = 0;
      // negative test only: a cheating prover offsets the shift, scaling the whole row by
      // exp(-k * isc); the normalization check must reject it once that exceeds the slack
      static const int64_t cheat = getenv("LSE_CHEAT_SHIFT") ? atoll(getenv("LSE_CHEAT_SHIFT")) : 0;
      shift_v += cheat;
    }
    IntFp shift(static_cast<uint64_t>(shift_v), ALICE); // committed LSE shift witness (>= 0)
    std::vector<IntFp> keys(L);
    for (std::size_t i = 0; i < L; ++i) keys[i] = t[i] + shift;
    std::vector<IntFp> out;
    expt->lookup(out, keys); // batched Access: ONE feed for all L returned values
    IntFp sumout(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < L; ++i) sumout = sumout + out[i];
    // normalization check (replaces the division): | sum_i out_i - 2^ob | <= err -> forces
    // shift = logsumexp.
    const int ob = (out_bits < 0) ? P : out_bits;
    IntFp delta = sumout + IntFp(static_cast<uint64_t>(1) << ob, PUBLIC).negate();
    rc->check(delta + IntFp(err, PUBLIC), wf_bits);          // delta + err >= 0
    rc->check(IntFp(err, PUBLIC) + delta.negate(), wf_bits); // err - delta >= 0
    return out;
  }
};

// ---------------------------------------------------------------------------
// Pointwise activation via table lookup (ReLU, GeLU, ...).  idx = x + offset
// must land in [0, T); the LogUp key-value binding range-checks it.  Load the
// table with build_relu_table / build_gelu_table.
class FixedActivation {
public:
  FixedTable *tbl;
  int64_t offset;

  FixedActivation(FixedTable *tbl_, int64_t offset_) : tbl(tbl_), offset(offset_) {}

  IntFp forward(IntFp x) {
    return tbl->lookup(x + IntFp(static_cast<uint64_t>(offset), PUBLIC));
  }

  void forward(std::vector<IntFp> &out, const std::vector<IntFp> &x) {
    out.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = forward(x[i]);
  }
};

// ---------------------------------------------------------------------------
// LayerNorm (DeepProve sqrt-free):
//   mu  = mean(x); c = x - mu; var = mean(c^2);
//   commit d ~ round(2^2P / sqrt(var_q)) (the 2^-P encoding of 1/sigma);
//   prove |d^2 * var_q - 2^4P| <= TOL;  y = trunc(c * d);  out = y*gamma + beta.
class FixedLayerNorm {
public:
  FixedInfer *qi;
  RangeCheckLU *rc; // well-formedness non-negativity checks (replaces MSB)
  Div *divr;        // zkGPT-style division (no MSB) for mean / variance
  int P;
  uint64_t d; // feature dimension
  uint64_t TOL;
  std::size_t wf_bits; // bit width of delta+TOL for the range check

  // Backward-pass intermediate cache (opt-in): retain the COMMITTED c, dinv, y that
  // LayerNormBackward consumes.  Same save_intermediates flag + map pattern used by the
  // FFN block; saved objects are the exact IntFp witnesses proven in forward(), not copies.
  bool save_intermediates = false;
  // Per-TOKEN cache (Mat == std::vector<std::vector<IntFp>>, spelled out because Mat is
  // defined in fixed_transformer.h which includes this header, not vice-versa).  Each
  // forward() call push_back's one token's row, so the block's LN loop over tokens
  // accumulates all rows instead of the last overwriting the rest.
  std::map<std::string, std::vector<std::vector<IntFp>>> cache_mat;   // [token][dim]: c, y
  std::map<std::string, std::vector<IntFp>> cache_scalar_vec;         // [token]: dinv
  void clear_cache() {
    cache_mat.clear();
    cache_scalar_vec.clear();
  }

  FixedLayerNorm(FixedInfer *qi_, RangeCheckLU *rc_, Div *divr_, int P_, uint64_t d_)
      : qi(qi_), rc(rc_), divr(divr_), P(P_), d(d_) {
    TOL = static_cast<uint64_t>(1)
          << (3 * P + 8); // bounds the 1/sqrt rounding error (widened: GPT-2 residual stream has
                          // large, un-normalized variance)
    wf_bits = 3 * P + 10; // delta+TOL in [0, 2*TOL = 2^(3P+9)] -> needs 3P+10 bits
  }

  void forward(std::vector<IntFp> &out, const std::vector<IntFp> &x,
               const std::vector<IntFp> &gamma, const std::vector<IntFp> &beta) {
    const std::size_t n = x.size();

    // mu = mean(x)
    IntFp sumx(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < n; ++i) sumx = sumx + x[i];
    IntFp mu = divr->round_div(sumx, (uint64_t)(d));

    // c = x - mu ;  ssq = sum c^2 ;  var_q = ssq / d   (scale 2^-2P)
    // QuickSilver degree-2 batching (CCS'21 sec 3.2): sum c_i^2 is one degree-2 polynomial, proven
    // with ONE deferred IPProof claim ([c...,-1].[c...,ssq]=0) -- ALICE commits just ssq, NO
    // per-square mult-gate commitments (was n IntFp*IntFp gates per row).
    std::vector<IntFp> c(n);
    for (std::size_t i = 0; i < n; ++i) c[i] = x[i] + mu.negate();
    IntFp ssq;
    if (qi->party == ALICE) {
      uint64_t s = 0;
      for (std::size_t i = 0; i < n; ++i) {
        uint64_t v = static_cast<uint64_t>(VAL(c[i].value));
        s = add_mod(s, mult_mod(v, v));
      }
      ssq = IntFp(s, ALICE);
    } else {
      ssq = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    {
      std::vector<IntFp> xx(n + 1), yy(n + 1);
      for (std::size_t i = 0; i < n; ++i) { xx[i] = c[i]; yy[i] = c[i]; }
      xx[n] = IntFp(PR - 1, PUBLIC); yy[n] = ssq;   // -1 , ssq : proves sum c_i^2 - ssq = 0
      qi->ipp->accumulate(xx.data(), yy.data(), 0, (int)(n + 1));
    }
    IntFp var_q = divr->round_div(ssq, (uint64_t)(d));
    var_q = var_q + IntFp(static_cast<uint64_t>(1), PUBLIC);
    // +1 epsilon (LayerNorm eps), matching QuantLayerNorm: a CONSTANT input gives var=0 -> without
    // eps dinv is 1/0 (garbage) and |dinv^2*var - 2^4P| = 2^4P > TOL is unsatisfiable. With +1,
    // var=0 -> dinv=2^2P and dinv^2*var=2^4P exactly (delta 0); negligible (2^-2P real) otherwise.

    // prover commits d ~ 1/sigma encoded at scale 2^-P
    IntFp dinv;
    if (qi->party == ALICE) {
      uint64_t vq = static_cast<uint64_t>(VAL(var_q.value));
      double sq = std::sqrt(static_cast<double>(vq));
      long long di = llround(static_cast<double>(1ULL << (2 * P)) / sq);
      dinv = IntFp(static_cast<uint64_t>(di), ALICE);
    } else {
      dinv = IntFp(static_cast<uint64_t>(0), ALICE);
    }

    // well-formedness: | dinv^2 * var_q - 2^4P | <= TOL
    IntFp lhs = (dinv * dinv) * var_q;
    IntFp target(static_cast<uint64_t>(1) << (4 * P), PUBLIC);
    IntFp delta = lhs + target.negate();
    // |delta| <= TOL via two non-negativity range checks (no MSB):
    rc->check(delta + IntFp(TOL, PUBLIC), wf_bits);          // delta + TOL >= 0
    rc->check(IntFp(TOL, PUBLIC) + delta.negate(), wf_bits); // TOL - delta >= 0

    // y = trunc(c * dinv)   (scale 2^-P normalized) ; out = y*gamma + beta
    std::vector<IntFp> prod(n);
    for (std::size_t i = 0; i < n; ++i) prod[i] = c[i] * dinv;
    std::vector<IntFp> y;
    qi->requant_cm(y, prod, 1, P); // normalize is activation-scale (c=1)
    if (save_intermediates) {
      // the EXACT committed witnesses from this proof (dinv's well-formedness was
      // range-checked above) -- never recomputed, so the backward stays bound to it.
      // push_back appends per-token; caller must clear_cache() between
      // forward passes to avoid stale token rows accumulating.
      cache_mat["c"].push_back(c);              // centered input x - mu, scale 2^-P
      cache_mat["y"].push_back(y);              // x_hat = requant(c*dinv), scale 2^-P
      cache_scalar_vec["dinv"].push_back(dinv); // 1/sigma witness, scale 2^-P
    }
    qi->affine(out, y, gamma, beta);
  }
};

#endif
