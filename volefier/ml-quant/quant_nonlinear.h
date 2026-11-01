#ifndef ML_QUANT_NONLINEAR_H
#define ML_QUANT_NONLINEAR_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-quant/quant_infer.h"
#include "volefier/nonlinear/comparison.h" // CompareLU (max_value, replaces find_max)
#include "volefier/nonlinear/div.h"
#include "volefier/nonlinear/rsqrt.h"     // prove_rsqrt (LayerNorm/RMSNorm 1/sigma)
#include "volefier/zk-ram/logup_keyvalue.h"
#include <cmath>
#include <cstdint>
#include <vector>
using namespace emp;

// Non-linear transformer operators built from emp-zk + LogUp, following
// DeepProve's "advice + lookup / sqrt-free" recipes:
//   - activations (exp, GeLU) are PUBLIC function tables looked up with LogUp;
//     the key-value binding simultaneously range-checks the index into [0, T).
//   - softmax = max-subtract (Comp) -> exp lookup -> sum -> committed divide.
//   - LayerNorm avoids a sqrt circuit: the prover commits d ~ 1/sigma and proves it
//     with bounded, wrap-free inequalities (prove_rsqrt).

// ---------------------------------------------------------------------------
// QuantTable: a read-only public-function lookup.  Setup() loads T raw field
// values; lookup(idx) returns the committed table[idx] and (via the LogUp
// key-value binding at Teardown) forces idx in [0, T).
class QuantTable {
public:
  uint64_t T;
  int party;
  LogUpKeyValueLookup *tab = nullptr;

  QuantTable(uint64_t T_, int party_, IOChannel *io) : T(T_), party(party_) {
    tab = new LogUpKeyValueLookup(T_, party_, io);
  }
  ~QuantTable() {
    delete tab;
  }

  void Setup(const std::vector<uint64_t> &values) {
    tab->Setup(values);
  }
  IntFp lookup(IntFp idx) {
    return tab->Access(idx);
  }
  void lookup(std::vector<IntFp> &out, const std::vector<IntFp> &idx) {   // batched: one feed for all values
    tab->Access(out, idx);
  }
  void Teardown() {
    tab->Teardown();
  }
};

// ---- host-side table builders (scale 2^-P) --------------------------------

// exp table for softmax:  expT[t] = round( exp(-t * 2^-P) * 2^P ).  clamp=true: >= 1 (zkGPT
// round_div needs a nonzero denominator).  clamp=false: exact (-> 0 for large t), used by the
// DeepProve LogSumExp path which has NO division -- so masked positions become exactly 0 (no
// max(.,1) leak), tightening the sum-to-one tolerance to the pure rounding bound.
static inline std::vector<uint64_t> build_exp_table(uint64_t T, int P, bool clamp = true) {
  std::vector<uint64_t> e(T);
  const double scale = static_cast<double>(1ULL << P);
  for (uint64_t t = 0; t < T; ++t) {
    long long v = llround(std::exp(-static_cast<double>(t) / scale) * scale);
    e[t] = static_cast<uint64_t>((clamp && v < 1) ? 1 : (v < 0 ? 0 : v));
  }
  return e;
}

// PER-TENSOR scale-aware exp table: index t is at scale `in_scale` (= S_score),
// so e[t] = round(2^P * exp(-t*in_scale)). Output K=2^P cancels in softmax's
// e_i/sum, so only the input scale matters.
// clamp=true (default): values >= 1 (zkGPT round_div needs a nonzero denominator).
// clamp=false: exact (-> 0 for large t), used by the LogSumExp path (no division) so
// tail/masked terms become exactly 0, tightening the sum-to-one tolerance.
static inline std::vector<uint64_t> build_exp_table_s(uint64_t T, int P, double in_scale,
                                                      bool clamp = true) {
  std::vector<uint64_t> e(T);
  const double K = static_cast<double>(1ULL << P);
  for (uint64_t t = 0; t < T; ++t) {
    long long v = llround(std::exp(-static_cast<double>(t) * in_scale) * K);
    e[t] = static_cast<uint64_t>((clamp && v < 1) ? 1 : (v < 0 ? 0 : v));
  }
  return e;
}

// PER-TENSOR scale-aware gelu table: input (idx-offset) at scale `in_scale`
// (= S_h), output at 2^-P:  g[idx] = round(2^P * gelu((idx-offset)*in_scale)).
static inline std::vector<uint64_t> build_gelu_table_s(uint64_t T, int P, int64_t offset,
                                                       double in_scale) {
  std::vector<uint64_t> g(T);
  const double K = static_cast<double>(1ULL << P), kk = std::sqrt(2.0 / M_PI);
  for (uint64_t i = 0; i < T; ++i) {
    double x = static_cast<double>(static_cast<int64_t>(i) - offset) * in_scale;
    double gelu = 0.5 * x * (1.0 + std::tanh(kk * (x + 0.044715 * x * x * x)));
    long long q = llround(gelu * K);
    g[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return g;
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

// DeepProve-style FUSED nonlinear lookup: input idx-offset is a PER-TENSOR int at scale s_in; the
// table output is the activation REQUANTIZED to the output per-tensor scale s_out and CLAMPED to
// [lo,hi] -- so ONE LogUp lookup does activation + requant + clamp, staying in the per-tensor scale
// (no 2^-P bus). g[idx] = clamp( round( gelu((idx-offset)*s_in) / s_out ), lo, hi ), signed ->
// field-encoded.
static inline std::vector<uint64_t> build_gelu_table_fused(uint64_t T, int64_t offset, double s_in,
                                                           double s_out, int64_t lo, int64_t hi) {
  std::vector<uint64_t> g(T);
  const double k = std::sqrt(2.0 / M_PI);
  for (uint64_t i = 0; i < T; ++i) {
    double x = static_cast<double>(static_cast<int64_t>(i) - offset) * s_in;
    double gv = 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
    long long q = llround(gv / s_out);
    if (q < lo) q = lo;
    if (q > hi) q = hi;
    g[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return g;
}

// DeepProve-style FUSED gelu lookup, ASYMMETRIC uint8 (BERT): input idx is the committed uint8 q in
// [0,T); dequant (q-zp_in)*s_in, gelu, requant round(gelu/s_out)+zp_out, clamp [lo,hi].  One LogUp
// lookup.
static inline std::vector<uint64_t> build_gelu_table_fused_asym(uint64_t T, double s_in,
                                                                double zp_in, double s_out,
                                                                double zp_out, int64_t lo,
                                                                int64_t hi) {
  std::vector<uint64_t> g(T);
  const double k = std::sqrt(2.0 / M_PI);
  for (uint64_t i = 0; i < T; ++i) {
    double x = (static_cast<double>(i) - zp_in) * s_in;
    double gv = 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
    long long q = llround(gv / s_out + zp_out);
    if (q < lo) q = lo;
    if (q > hi) q = hi;
    g[i] = static_cast<uint64_t>(q);
  }
  return g;
}

// DeepProve-style FUSED tanh lookup (BERT pooler), asymmetric uint8 aware: input idx is the
// committed uint8 q in [0,T); dequant (q-zp_in)*s_in, tanh, requant to the output
// (round(tanh/s_out)+zp_out), clamp. One LogUp lookup = dequant + tanh + requant + clamp; output
// uint8 (non-negative, no field encoding).
static inline std::vector<uint64_t> build_tanh_table_fused(uint64_t T, double s_in, double zp_in,
                                                           double s_out, double zp_out, int64_t lo,
                                                           int64_t hi) {
  std::vector<uint64_t> g(T);
  for (uint64_t i = 0; i < T; ++i) {
    double x = (static_cast<double>(i) - zp_in) * s_in;
    long long q = llround(std::tanh(x) / s_out + zp_out);
    if (q < lo) q = lo;
    if (q > hi) q = hi;
    g[i] = static_cast<uint64_t>(q);
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

// Saturating-clamp table: clamp[i] = min(hi, max(lo, i - offset)). ONE LogUp lookup
// (whose key-value binding also range-checks i-offset into [0,T)) replaces the TWO
// MSB-free comparisons + selects of QuantizeLinear's [lo,hi] saturation -- the
// DeepProve "activation/saturation as a lookup" optimization.
static inline std::vector<uint64_t> build_clamp_table(uint64_t T, int64_t offset, int64_t lo,
                                                      int64_t hi) {
  std::vector<uint64_t> t(T);
  for (uint64_t i = 0; i < T; ++i) {
    int64_t x = static_cast<int64_t>(i) - offset;
    t[i] = static_cast<uint64_t>(x < lo ? lo : (x > hi ? hi : x));
  }
  return t;
}

// ===========================================================================
// LLM operators for Llama/Gemma (beyond GPT-2): SiLU/SwiGLU, RoPE, GQA.
// All reuse ml-quant primitives -- no new soundness gadget.
// ===========================================================================

// SiLU / Swish activation table:  silu[idx] = round( silu((idx-offset)*2^-P) * 2^P ),
// silu(x) = x * sigmoid(x).  Signed output -> field-encoded (like GeLU).  Looked up
// with QuantActivation (one LogUp Access/elem).  SwiGLU = SiLU(gate) (.) up, both proj
// via the IP matmul; the element-wise (.) is a committed*committed mult + one requant.
static inline std::vector<uint64_t> build_silu_table(uint64_t T, int P, int64_t offset) {
  std::vector<uint64_t> g(T);
  const double scale = static_cast<double>(1ULL << P);
  for (uint64_t i = 0; i < T; ++i) {
    double x = static_cast<double>(static_cast<int64_t>(i) - offset) / scale;
    double v = x / (1.0 + std::exp(-x));
    long long q = llround(v * scale);
    g[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return g;
}

// Tanh activation via LogUp LUT (the BERT pooler:  pooled = tanh(dense([CLS]))).
//
// SCALE FORMAT -- DIFFERENT FROM FIXED-POINT, READ CAREFULLY:
//   * ml-fixed is UNIFORM 2^-P everywhere, so a tanh LUT there just hard-codes 2^-P in and out.
//   * ml-quant is PER-TENSOR c*2^e: each activation carries its OWN scale, NOT a global 2^-P. So
//     the INPUT scale must be BAKED INTO the table (it is the c*2^e "scale bridge", DeepProve/zkGPT
//     style) -- never assume the committed key is at 2^-P.
//   * tanh's OUTPUT is BOUNDED to (-1,1): at the gadget's 2^-P output convention the integer range
//     is only [-2^P, 2^P], and it SATURATES (tanh(x)->+-1 for |x|>~4). Unlike a matmul there is NO
//     requant afterwards -- the (-1,1) range already fits, so output stays at 2^-P. The caller must
//     still size (T, offset) so the committed input key stays inside [0,T) (clamp big pre-tanh
//     values first, exactly like the softmax mask clamp), since the lookup binds the key.
//   * signed output -> field-encoded (negatives as +PR); decode with QuantInfer::decode_signed.
//
//   build_tanh_table   : UNIFORM path  -- input (idx-offset) and output both at 2^-P.
//   build_tanh_table_s : PER-TENSOR path -- input (idx-offset) at the tensor's own scale `in_scale`
//                        (e.g. s_x = c*2^e from the dense layer's requant), output at 2^-P.
static inline std::vector<uint64_t> build_tanh_table(uint64_t T, int P, int64_t offset) {
  std::vector<uint64_t> t(T);
  const double scale = static_cast<double>(1ULL << P);
  for (uint64_t i = 0; i < T; ++i) {
    double x = static_cast<double>(static_cast<int64_t>(i) - offset) / scale; // input at 2^-P
    long long q = llround(std::tanh(x) * scale); // output at 2^-P, |tanh|<1 -> |q|<2^P
    t[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return t;
}
static inline std::vector<uint64_t> build_tanh_table_s(uint64_t T, int P, int64_t offset,
                                                       double in_scale) {
  std::vector<uint64_t> t(T);
  const double K = static_cast<double>(1ULL << P);
  for (uint64_t i = 0; i < T; ++i) {
    double x = static_cast<double>(static_cast<int64_t>(i) - offset) *
               in_scale;                     // input at PER-TENSOR in_scale (NOT 2^-P)
    long long q = llround(std::tanh(x) * K); // output at 2^-P
    t[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return t;
}

// field-encode a signed public integer as a PUBLIC IntFp constant.
static inline IntFp pub_signed(int64_t v) {
  return IntFp(v >= 0 ? static_cast<uint64_t>(v) : PR - static_cast<uint64_t>(-v), PUBLIC);
}

// RoPE: rotate dimension-pairs (2i,2i+1) of a committed [seq,H,dh] tensor by PUBLIC,
// position-dependent angles.  cos_q[s*half+i] / sin_q[s*half+i] are public ints at 2^-P
// (= round(cos(s*theta_i)*2^P)).  Each output is a linear combo of committed q with PUBLIC
// coefficients -> FREE (committed*public, no IP proof); one requant 2^-2P -> 2^-P at the end.
//   q'[2i]   = q[2i]*cos - q[2i+1]*sin
//   q'[2i+1] = q[2i]*sin + q[2i+1]*cos
static inline void apply_rope(QuantInfer *qi, std::vector<IntFp> &q, int seq, int H, int dh,
                              const std::vector<int64_t> &cos_q, const std::vector<int64_t> &sin_q,
                              int P) {
  const int half = dh / 2;
  std::vector<IntFp> prod(q.size()); // rotated, scale 2^-2P
  for (int s = 0; s < seq; ++s)
    for (int h = 0; h < H; ++h) {
      const int64_t base = (static_cast<int64_t>(s) * H + h) * dh;
      for (int i = 0; i < half; ++i) {
        const IntFp a = q[base + 2 * i], b = q[base + 2 * i + 1];
        const int64_t c = cos_q[static_cast<int64_t>(s) * half + i],
                      sn = sin_q[static_cast<int64_t>(s) * half + i];
        prod[base + 2 * i] = a * pub_signed(c) + b * pub_signed(-sn);
        prod[base + 2 * i + 1] = a * pub_signed(sn) + b * pub_signed(c);
      }
    }
  std::vector<IntFp> out;
  qi->requant_cm(out, prod, 1, static_cast<std::size_t>(P));
  q = std::move(out);
}

// GQA: expand committed K/V from [seq,H_kv,dh] to [seq,H_kv*rep,dh] by repeating each KV
// head `rep` times.  REUSES the same committed IntFps (no new commitment) -- pure reindexing.
// Query head h then pairs with expanded head h (== original KV head h/rep).
static inline std::vector<IntFp> repeat_kv(const std::vector<IntFp> &kv, int seq, int H_kv, int dh,
                                           int rep) {
  std::vector<IntFp> out(static_cast<size_t>(seq) * H_kv * rep * dh);
  for (int s = 0; s < seq; ++s)
    for (int hk = 0; hk < H_kv; ++hk)
      for (int r = 0; r < rep; ++r)
        for (int d = 0; d < dh; ++d)
          out[((static_cast<int64_t>(s) * H_kv * rep + hk * rep + r) * dh) + d] =
              kv[((static_cast<int64_t>(s) * H_kv + hk) * dh) + d];
  return out;
}

// ---------------------------------------------------------------------------
// Softmax over a committed logit vector (scale 2^-P), normalized to scale 2^-P
// (i.e. probabilities times 2^P).  expt must be Setup with build_exp_table.
class QuantSoftmax {
public:
  QuantInfer *qi;
  CompareLU *cmp; // lookup-table max (DeepProve-style, no MSB)
  QuantTable *expt;
  Div *divr; // zkGPT-style division (no MSB) for the normalization
  int P;
  std::size_t score_bits; // bit width of (max - logit); = log2(exp table size)
  bool lse;               // true: DeepProve LogSumExp (no division); false: zkGPT round_div
  double in_scale;        // exp-table input scale S_score (LSE shift/err); <0 -> 2^-P

  QuantSoftmax(QuantInfer *qi_, CompareLU *cmp_, QuantTable *expt_, Div *divr_, int P_,
               std::size_t score_bits_ = 16, bool lse_ = false, double in_scale_ = -1.0)
      : qi(qi_), cmp(cmp_), expt(expt_), divr(divr_), P(P_), score_bits(score_bits_), lse(lse_),
        in_scale(in_scale_) {}

  // Self-contained LSE: synthesize the plaintext score reference from the committed
  // logits (ALICE) and derive the sum-to-one tolerance, so callers just call forward().
  // The exp table must be Setup with clamp=false (so tail terms are exactly 0).
  std::vector<IntFp> forward_lse_auto(std::vector<IntFp> &logits) {
    const std::size_t L = logits.size();
    const double isc =
        (in_scale > 0) ? in_scale : 1.0 / static_cast<double>(static_cast<uint64_t>(1) << P);
    // err = per-elt rounding (<= L/2) + shift quantization (~2^P*in_scale) + slack.
    const uint64_t err = static_cast<uint64_t>(L) / 2 +
                         static_cast<uint64_t>(std::ceil(std::pow(2.0, P) * isc)) + 4;
    std::size_t wfb = 0;
    for (uint64_t v = 2 * err; v; v >>= 1) ++wfb;
    std::vector<int64_t> rl;
    const int64_t *rlp = nullptr;
    if (qi->party == ALICE) {
      rl.resize(L);
      for (std::size_t j = 0; j < L; ++j)
        rl[j] = QuantInfer::decode_signed(static_cast<uint64_t>(VAL(logits[j].value)));
      rlp = rl.data();
    }
    return forward_lse(logits, rlp, cmp->rc, err, wfb, isc);
  }

  // logits: length L. returns probabilities at scale 2^-P.
  // valid_len >= 0 (causal attention, driver sets it under CAUSAL_SM): the number of UNMASKED leading
  // keys in this row (keys [0,valid_len); the rest are the causal mask-fill).  On the SYM_BITS path the
  // masked keys overflow the exp domain hugely, and WHICH positions are masked is PUBLIC (causal), so:
  //  - masked i (>= valid_len): take the public exp_table[N-1] value (=1, clamp-true floor) DIRECTLY --
  //    no lookup, no is_le.
  //  - valid i (< valid_len): key t=m-x_i is in [0,N); the exp lookup already range-checks it on the
  //    VERIFIER (key >= N has no table row -> proof rejected), so the per-element is_le clamp is the
  //    REDUNDANT check -- drop it, look up t directly.  m is a bare committed upper bound (no select_max);
  //    the valid keys' in-range requirement forces m >= every valid x_i, which is all softmax needs.
  //  Byte-identical to the SM_CLAMP path (same m, e_i, out); ALICE self-checks the split (guard, not
  //  soundness) so a wrong causal assumption ERRORS instead of silently diverging.
  std::vector<IntFp> forward(std::vector<IntFp> &logits, int valid_len = -1) {
    if (lse) return forward_lse_auto(logits);
    const std::size_t L = logits.size();
    const bool causal = (valid_len >= 0 && (std::size_t)valid_len <= L);
    const bool cl = getenv("SM_CLAMP") != nullptr;
    const uint64_t Nm1 = (static_cast<uint64_t>(1) << score_bits) - 1;

    IntFp m;
    if (!causal) {
      m = cmp->max_value(logits, score_bits); // no public mask -> keep select_max
    } else {
      int64_t mref = INT64_MIN;
      if (qi->party == ALICE)
        for (int i = 0; i < valid_len; ++i) {
          int64_t v = QuantInfer::decode_signed(static_cast<uint64_t>(VAL(logits[i].value)));
          if (v > mref) mref = v;
        }
      m = qi->commit(mref);
      if (qi->party == ALICE)
        for (std::size_t i = 0; i < L; ++i) {
          int64_t key = mref - QuantInfer::decode_signed(static_cast<uint64_t>(VAL(logits[i].value)));
          if ((int)i < valid_len) {
            if (key < 0 || key > (int64_t)Nm1)
              error("causal softmax: valid key out of [0,N) -- mask assumption violated\n");
          } else {
            int64_t k = key < 0 ? 0 : (key > (int64_t)Nm1 ? (int64_t)Nm1 : key);
            double ev = std::round(std::exp(-(double)k / (double)(1 << P)) * (double)(1 << P));
            if (ev < 1) ev = 1; // clamp-true floor
            if (ev != 1.0)
              error("causal softmax: masked exp != 1 (public-const opt invalid) -- mask/model mismatch\n");
          }
        }
    }

    std::vector<IntFp> e(L);
    IntFp sum(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < L; ++i) {
      if (causal && (int)i >= valid_len) {
        e[i] = IntFp(static_cast<uint64_t>(1), PUBLIC); // masked: public exp_table[N-1]=1, no lookup
      } else {
        IntFp t = m + logits[i].negate(); // t = max - logit >= 0
        if (cl && !causal) { // causal valid keys are in-range by construction (self-checked)
          IntFp b = cmp->is_le(t, IntFp(Nm1, PUBLIC), 32);
          t = b * (t + IntFp(Nm1, PUBLIC).negate()) + IntFp(Nm1, PUBLIC);
        }
        e[i] = expt->lookup(t);
      }
      sum = sum + e[i];
    }

    std::vector<IntFp> out(L);
    const IntFp two_p(static_cast<uint64_t>(1) << P, PUBLIC);
    for (std::size_t i = 0; i < L; ++i)
      out[i] = divr->round_div(e[i] * two_p, sum); // round((e_i * 2^P) / sum), no MSB
    return out;
  }

  // DeepProve LogSumExp softmax: NO DIVISION.  softmax_i = exp(score_i - max - logsumexp) =
  // exp_table[(max - score_i) + shift] with shift = round(log(sum_exp) * 2^P).  The prover COMMITS
  // shift as a witness; soundness comes from proving the row sums to ~2^P (which forces shift =
  // logsumexp).
  // in_scale = the input score scale baked into the exp table (S_score). Default (<0)
  // resolves to 2^-P, the uniform bus the ONNX driver rescales to before softmax.
  std::vector<IntFp> forward_lse(std::vector<IntFp> &logits, const int64_t *rlog, RangeCheckLU *rc,
                                 uint64_t err, std::size_t wf_bits, double in_scale = -1.0) {
    const std::size_t L = logits.size();
    // DeepProve-style: NO select_max.  ALICE commits the row max as a plain upper-bound witness
    // (one commit) instead of proving it via a one-hot selector + per-element mul_gates + asserts.
    // Soundness is unchanged: each exp lookup enforces (m - logit_i + shift) in [0,N), and the
    // normalization check below forces m + shift = logsumexp -- so the softmax output depends only
    // on m+shift and is invariant to m.  Any committed m that keeps the lookups in range works.
    int64_t mref = INT64_MIN;
    if (qi->party == ALICE && rlog)
      for (std::size_t i = 0; i < L; ++i) if (rlog[i] > mref) mref = rlog[i];
    IntFp m = qi->commit(mref);   // BOB commits 0
    std::vector<IntFp> t(L);
    for (std::size_t i = 0; i < L; ++i) t[i] = m + logits[i].negate(); // t_i = max - logit_i >= 0
    int64_t shift_v = 0; // ALICE: shift = round(log(sum_exp) * 2^P)
    if (qi->party == ALICE && rlog) {
      const double isc =
          (in_scale > 0) ? in_scale : 1.0 / static_cast<double>(static_cast<uint64_t>(1) << P);
      double sum = 0; // sum_exp at the table's input scale; shift = round(log(sum)/in_scale)
      for (std::size_t i = 0; i < L; ++i)
        sum += std::exp(-static_cast<double>(mref - rlog[i]) * isc);
      shift_v = static_cast<int64_t>(llround(std::log(sum) / isc));
      if (shift_v < 0) shift_v = 0;
    }
    IntFp shift(static_cast<uint64_t>(shift_v), ALICE); // committed LSE shift witness (>= 0)
    std::vector<IntFp> keys(L);
    for (std::size_t i = 0; i < L; ++i) keys[i] = t[i] + shift;
    std::vector<IntFp> out;
    expt->lookup(out, keys);   // batched Access: ONE feed for all L returned values
    IntFp sumout(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < L; ++i) sumout = sumout + out[i];
    // normalization check (replaces the division): | sum_i out_i - 2^P | <= err  -> forces shift =
    // logsumexp
    IntFp delta = sumout + IntFp(static_cast<uint64_t>(1) << P, PUBLIC).negate();
    rc->check(delta + IntFp(err, PUBLIC), wf_bits);          // delta + err >= 0
    rc->check(IntFp(err, PUBLIC) + delta.negate(), wf_bits); // err - delta >= 0
    return out;
  }
};

// ---------------------------------------------------------------------------
// Pointwise activation via table lookup (ReLU, GeLU, ...).  idx = x + offset
// must land in [0, T); the LogUp key-value binding range-checks it.  Load the
// table with build_relu_table / build_gelu_table.
class QuantActivation {
public:
  QuantTable *tbl;
  int64_t offset;

  QuantActivation(QuantTable *tbl_, int64_t offset_) : tbl(tbl_), offset(offset_) {}

  IntFp forward(IntFp x) {
    return tbl->lookup(x + IntFp(static_cast<uint64_t>(offset), PUBLIC));
  }

  void forward(std::vector<IntFp> &out, const std::vector<IntFp> &x) {
    // Build all table keys, then ONE batched Access (one batch_feed for all values)
    // instead of a per-element lookup (per-element IntFp(.,ALICE) commit).
    const std::size_t n = x.size();
    const IntFp off(static_cast<uint64_t>(offset), PUBLIC);
    std::vector<IntFp> keys(n);
    for (std::size_t i = 0; i < n; ++i) keys[i] = x[i] + off;
    tbl->lookup(out, keys);
  }
};

// ---------------------------------------------------------------------------
// LayerNorm (DeepProve sqrt-free):
//   mu  = mean(x); c = x - mu; var = mean(c^2);
//   commit d ~ round(2^2P / sqrt(var_q)) (the 2^-P encoding of 1/sigma), proven by
//   prove_rsqrt;  y = trunc(c * d);  out = y*gamma + beta.
class QuantLayerNorm {
public:
  QuantInfer *qi;
  RangeCheckLU *rc; // well-formedness non-negativity checks (replaces MSB)
  Div *divr;        // zkGPT-style division (no MSB) for mean / variance
  int P;
  uint64_t d; // feature dimension
  int var_bits; // var_q must lie in [1, 2^var_bits)

  QuantLayerNorm(QuantInfer *qi_, RangeCheckLU *rc_, Div *divr_, int P_, uint64_t d_)
      : qi(qi_), rc(rc_), divr(divr_), P(P_), d(d_), var_bits(2 * P_ + 20) {}

  void forward(std::vector<IntFp> &out, const std::vector<IntFp> &x,
               const std::vector<IntFp> &gamma, const std::vector<IntFp> &beta) {
    const std::size_t n = x.size();

    // mu = mean(x)
    IntFp sumx(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < n; ++i) sumx = sumx + x[i];
    IntFp mu = divr->round_div(sumx, (uint64_t)d);

    // c = x - mu ;  ssq = sum c^2 ;  var_q = ssq / d   (scale 2^-2P)
    // QuickSilver degree-2 batching (CCS'21 sec 3.2): sum c_i^2 is one degree-2 polynomial, so
    // prove it with ONE fp_zkp_inner_prdt ([c...,-1] . [c...,ssq] = 0) -- ALICE commits just ssq,
    // NO per-square mult-gate commitments (was n IntFp*IntFp gates per row).
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
      // Route through the DEFERRED IPProof buffer (IP_DEFER, 32M) so these ~12.8k claims share the
      // matmul challenge instead of flushing the emp-zk 1024-poly buffer (which added ~22 rounds).
      qi->ipp->accumulate(xx.data(), yy.data(), 0, (int)(n + 1));
    }
    IntFp var_q = divr->round_div(ssq, (uint64_t)d);
    var_q = var_q + IntFp(static_cast<uint64_t>(1), PUBLIC);
    // +1 epsilon (LayerNorm eps): a CONSTANT input gives var=0, whose 1/sigma is undefined.  With
    // +1, var=0 -> dinv=2^2P; negligible (2^-2P real) for any normal variance.

    // prover commits d ~ 1/sigma encoded at scale 2^-P
    IntFp dinv = prove_rsqrt(rc, qi->party, var_q, P, var_bits);

    // y = trunc(c * dinv)   (scale 2^-P normalized) ; out = y*gamma + beta
    std::vector<IntFp> prod(n);
    for (std::size_t i = 0; i < n; ++i) prod[i] = c[i] * dinv;
    std::vector<IntFp> y;
    qi->requant_cm(y, prod, 1, P); // normalize is activation-scale (c=1)
    qi->affine(out, y, gamma, beta);
  }
};

// ---------------------------------------------------------------------------
// RMSNorm (Llama/Gemma): like LayerNorm but NO mean-subtraction and NO bias:
//   ms = mean(x^2) + 1;  commit d ~ round(2^2P/sqrt(ms)), proven by prove_rsqrt;
//   y = trunc(x*d);  out = y*gamma.   (Gemma's (1+gamma): caller pre-adds 2^P to gamma.)
// Reuses the exact sqrt-free dinv gadget -- the only change from QuantLayerNorm is dropping mu.
class QuantRMSNorm {
public:
  QuantInfer *qi;
  RangeCheckLU *rc;
  Div *divr;
  int P;
  uint64_t d;
  int var_bits; // ms must lie in [1, 2^var_bits)

  QuantRMSNorm(QuantInfer *qi_, RangeCheckLU *rc_, Div *divr_, int P_, uint64_t d_)
      : qi(qi_), rc(rc_), divr(divr_), P(P_), d(d_), var_bits(2 * P_ + 20) {}

  void forward(std::vector<IntFp> &out, const std::vector<IntFp> &x,
               const std::vector<IntFp> &gamma) {
    const std::size_t n = x.size();
    // ms = mean(x^2)  (scale 2^-2P), no centering.  sum x_i^2 is ONE degree-2 poly -> prove with a
    // single deferred IPProof ([x...,-1].[x...,ssq]=0), NO per-square mult-gate (mirrors QuantLayerNorm).
    IntFp ssq;
    if (qi->party == ALICE) {
      uint64_t s = 0;
      for (std::size_t i = 0; i < n; ++i) {
        uint64_t v = static_cast<uint64_t>(VAL(x[i].value));
        s = add_mod(s, mult_mod(v, v));
      }
      ssq = IntFp(s, ALICE);
    } else {
      ssq = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    {
      std::vector<IntFp> xx(n + 1), yy(n + 1);
      for (std::size_t i = 0; i < n; ++i) { xx[i] = x[i]; yy[i] = x[i]; }
      xx[n] = IntFp(PR - 1, PUBLIC); yy[n] = ssq;   // -1 , ssq : proves sum x_i^2 - ssq = 0
      qi->ipp->accumulate(xx.data(), yy.data(), 0, (int)(n + 1));
    }
    // +1 epsilon (as in QuantLayerNorm): an all-zero row has ms=0, whose 1/RMS is undefined
    IntFp ms = divr->round_div(ssq, (uint64_t)d) + IntFp(static_cast<uint64_t>(1), PUBLIC);

    // commit d ~ 1/RMS encoded at 2^-P
    IntFp dinv = prove_rsqrt(rc, qi->party, ms, P, var_bits);

    // y = trunc(x*dinv) ; out = y*gamma  (no beta)
    std::vector<IntFp> prod(n);
    for (std::size_t i = 0; i < n; ++i) prod[i] = x[i] * dinv;
    std::vector<IntFp> y;
    qi->requant_cm(y, prod, 1, P);
    std::vector<IntFp> gp(n);
    for (std::size_t i = 0; i < n; ++i) gp[i] = y[i] * gamma[i];
    qi->requant_cm(out, gp, 1, P);
  }
};

#endif
