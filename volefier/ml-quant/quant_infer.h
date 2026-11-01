#ifndef ML_QUANT_INFER_H
#define ML_QUANT_INFER_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup.h"     // LogUpRangeCheck
#include "volefier/linear/ip_matmul.h" // standalone (thread-able) IP matmul
#include <cstdint>
#include <vector>
#include <cmath>
#include <utility>
#include <atomic>
#include <algorithm>
using namespace emp;

// QuantInfer: building blocks for a VOLE-ZK proof of *quantized* neural-network
// inference, in the same spirit as DeepProve / zkGPT but on top of emp-zk's
// arithmetic gates plus the LogUp range-check gadgets already in this repo.
//
// Fixed-point convention (the simple power-of-two case):
//   - every tensor is a committed Q-bit signed integer with an implicit scale
//     of 2^-precision (i.e. real value  =  q * 2^-precision).
//   - a matmul of two scale-2^-p tensors yields a scale-2^-2p accumulator, so
//     re-quantization back to scale 2^-p is exactly a truncation of `precision`
//     low bits — Div_LOGUP::truncate, the same gadget the tree code uses.
//
// This is "DeepProve Case 1" (multiplier epsilon == 1, pure right shift). A
// non-power-of-two scale c*2^-e (the zkGPT style) would multiply by the integer
// mantissa c before truncating; that is a drop-in extension (see requant()).
//
// Proof model: this is VOLE-ZK (QuickSilver) — interactive, designated-verifier,
// communication grows with #mults + #lookup-rows. ALICE is the prover (knows the
// plaintext quantized model + activations); BOB is the verifier (sees only
// commitments). All correctness is enforced by the committed checks below plus
// the LogUp teardowns in destroy().
class QuantInfer {
public:
  int party;
  std::size_t precision; // fractional bits == per-layer requant shift
  std::size_t out_bits;  // quantization width of a requant output
  uint64_t out_range;    // 2^out_bits (symmetric clamp bound)
  IOChannel *io = nullptr;
  // ONE small LogUp range table (size 2^precision) backs ALL range checks --
  // truncation remainders AND output clamps -- by limb decomposition. This
  // avoids a large 2^(out_bits+1) table, whose committed multiplicities used to
  // dominate the teardown.
  LogUpRangeCheck *rc_rem = nullptr;
  // zkGPT-style weight scale mantissa: weights live at scale wc*2^-precision, so
  // a dense layer's rescale ratio is wc*2^-precision (c = wc, not 1). Activations
  // stay at 2^-precision (c=1) for residual-stream consistency.
  uint64_t wc = 1;
  // Threaded standalone IP matmul (replaces emp-zk's singleton for dense layers).
  volefier::IPProof<BoolIO> *ipp = nullptr;
  int mat_threads = 1;
  bool flag_teardown = false;

  // bio is the BoolIO channel (ios[0]); the LogUp gadgets use bio->io (NetIO).
  QuantInfer(int party, std::size_t precision, BoolIO *bio, std::size_t out_bits = 16,
             uint64_t weight_mantissa = 1, int mat_threads_ = 1)
      : party(party), precision(precision), out_bits(out_bits),
        out_range(static_cast<uint64_t>(1) << out_bits), wc(weight_mantissa), io(bio->io),
        mat_threads(mat_threads_) {
    rc_rem = new LogUpRangeCheck(static_cast<uint64_t>(1) << precision, party, io);
    rc_rem->Setup();
    ipp = new volefier::IPProof<BoolIO>(party, bio,
                                               volefier::get_ostriple(party));
    // Shared process-global Hadamard2IP: batches every RLC-then-IP claim's chi challenge --
    // comparison bitness g(g-1)=0 AND (under LOGUP_INVBATCH) the LogUp inverse checks
    // (x+w)*inv==1 across all teardowns.  Flushed once in destroy() before finalize_zk_arith().
    if (volefier::global_h2ip() == nullptr) {
      std::size_t cap = (std::size_t)32 << 20;
      if (const char *c = getenv("LOGUP_INVBATCH_CAP")) { std::size_t v = strtoull(c, nullptr, 10); if (v) cap = v; }
      volefier::global_h2ip() = new volefier::Hadamard2IP(party, io, cap);
    }
  }

  ~QuantInfer() {
    if (!flag_teardown) destroy();
  }

  // Flush the LogUp range-check proofs. Must be called before finalize_zk_arith().
  void destroy() {
    if (flag_teardown) return;
    delete ipp;
    ipp = nullptr; // flushes any residual IP batch-check (no-op; matmul flushes per call)
    rc_rem->Teardown();
    delete rc_rem;
    rc_rem = nullptr;
    // Flush the shared Hadamard2IP AFTER the last teardown fed it (bitness during run() +
    // LogUp inverse checks), before finalize_zk_arith() runs FpPolyProof's batch_check.
    if (volefier::global_h2ip()) {
      volefier::global_h2ip()->flush();
      delete volefier::global_h2ip();
      volefier::global_h2ip() = nullptr;
    }
    flag_teardown = true;
  }

  // ---- commitment helpers -------------------------------------------------
  // Commit a signed quantized integer as an ALICE-witness IntFp. Negative
  // values use the field's two's-complement image (v + PR). On BOB the value
  // is irrelevant (verifier holds only the MAC), so we commit a placeholder 0.
  IntFp commit(int64_t v) const {
    if (party == ALICE) {
      uint64_t raw =
          (v >= 0) ? static_cast<uint64_t>(v) : static_cast<uint64_t>(v + static_cast<int64_t>(PR));
      return IntFp(raw, ALICE);
    }
    return IntFp(static_cast<uint64_t>(0), ALICE);
  }

  void commit_vec(std::vector<IntFp> &out, const std::vector<int64_t> &raw) const {
    out.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) out[i] = commit(raw[i]);
  }

  void commit_mat(std::vector<std::vector<IntFp>> &out,
                  const std::vector<std::vector<int64_t>> &raw) const {
    out.resize(raw.size());
    for (std::size_t j = 0; j < raw.size(); ++j) commit_vec(out[j], raw[j]);
  }

  // Decode a revealed field element back to a signed integer.
  static int64_t decode_signed(uint64_t v) {
    if (v > (PR >> 1)) return static_cast<int64_t>(v) - static_cast<int64_t>(PR);
    return static_cast<int64_t>(v);
  }

  // ---- core gadgets -------------------------------------------------------

  // Linear layer (no requant): acc[j] = sum_i W[j][i] * x[i].
  // W is out_dim x in_dim, x is in_dim. Each product is a committed
  // multiplication; the running sum is a free linear combination. The result
  // is at scale 2^-2p (caller re-quantizes).
  void linear(std::vector<IntFp> &acc, const std::vector<std::vector<IntFp>> &W,
              const std::vector<IntFp> &x) const {
    const std::size_t out_dim = W.size();
    acc.resize(out_dim);
    for (std::size_t j = 0; j < out_dim; ++j) {
      IntFp s(static_cast<uint64_t>(0), PUBLIC);
      for (std::size_t i = 0; i < x.size(); ++i) {
        s = s + W[j][i] * x[i];
      }
      acc[j] = s;
    }
  }

  // Truncate x by `precision` bits (floor, signed) using two DIRECT LogUp range
  // checks instead of an MSB bit-decomposition (cf. zkGPT's single range
  // relation / DeepProve's clamp lookup):
  //   prover supplies y = floor(x / 2^p) and r = x - y*2^p;
  //   (1) committed identity   x == y*2^p + r,
  //   (2) LogUp-range  r in [0, 2^p),
  //   (3) LogUp-range  (y + 2^b) in [0, 2^(b+1))  -- range-checks AND clamps y.
  // The honest (y, r) is the unique solution since |x| stays well below 2^60
  // (no field wraparound), so soundness needs no sign gadget.
  // Range-check a committed NON-NEGATIVE v into [0, 2^(ceil(nbits/precision)*precision))
  // by decomposing it into precision-bit limbs, each checked against the small
  // rc_rem table -- so no large 2^nbits table is ever materialized. (Bound is
  // rounded up to a multiple of `precision`, slightly looser but still sound.)
  void range_chunks(IntFp v, std::size_t nbits) const {
    const std::size_t nchunks = (nbits + precision - 1) / precision;
    if (nchunks <= 1) {
      rc_rem->Access(v);
      return;
    } // v is the only limb
    const uint64_t mask = (static_cast<uint64_t>(1) << precision) - 1;
    std::vector<IntFp> chunk(nchunks);
    if (party == ALICE) {
      uint64_t val = static_cast<uint64_t>(VAL(v.value));
      for (std::size_t i = 0; i < nchunks; ++i)
        chunk[i] = IntFp((val >> (precision * i)) & mask, ALICE);
    } else {
      for (std::size_t i = 0; i < nchunks; ++i) chunk[i] = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    IntFp recon(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < nchunks; ++i) {
      rc_rem->Access(chunk[i]);
      recon = recon + chunk[i] * (static_cast<int64_t>(1) << (precision * i));
    }
    (recon + v.negate()).reveal_zero();
  }

  // zkGPT scale search: best (c, e) with c in [1,800], e in [-10,10] minimizing
  // |c*2^e - target|.  A scale c*2^e generalizes a pure power of two (c=1).
  static std::pair<int, int> scale_search(double target) {
    double best = 1e300;
    int bc = 1, be = 0;
    for (int e = -24; e <= 10; ++e) // wide enough for ~2^-12 rescale ratios
      for (int c = 1; c <= 800; ++c) {
        double d = std::fabs(std::ldexp(static_cast<double>(c), e) - target);
        if (d < best) {
          best = d;
          bc = c;
          be = e;
        }
      }
    return {bc, be};
  }

  // Bound on a requant OUTPUT q: |q| < 2^(q_bits-1).  The floor relation Nb = q*2^Mpad + r
  // with r in [0,2^Mpad) pins q only modulo p -- each in-range r admits its own (huge) field q --
  // so q is committed as limbs of q + 2^(q_bits-1) instead of as a free witness (DeepProve's
  // value + zeroing chunks).  Callers with wider outputs pass a larger bound per call.  24 bits
  // (two 12-bit limbs) covers the int12 activations and logits; honest GPT-2 / BERT outputs peak
  // near 20 bits.
  std::size_t q_bits = 24;

  // Effective q bound for shift Mpad: sound while 2^(qb+Mpad) < p, i.e. qb + Mpad <= 60.
  std::size_t out_qbits(std::size_t Mpad, std::size_t qb) const {
    if (Mpad + 2 > 60) error("requant: shift too large for a bounded output\n");
    return std::min(qb ? qb : q_bits, 60 - Mpad);
  }

  // q = round-half-up(acc * c / 2^M).  ONE-SIDED requant (DeepProve-style): fold round-to-nearest
  // into a +2^(M-1) bias and prove the floor by decomposing the DISCARDED low bits --
  // Nb = q*2^Mpad + r, r in [0,2^Mpad) -- with the shift padded UP to a multiple of `precision`
  // (scaling N by 2^pad, which cancels) so r decomposes into EXACT full chunks.  q itself is
  // rebuilt from bounded limbs (see q_bits), so (q, r) is the unique solution.
  IntFp requant_cm_one(IntFp acc, uint64_t c, std::size_t M, std::size_t qb = 0) const {
    IntFp out;
    requant_cm_batch(&out, &acc, c, M, 1, qb);
    return out;
  }

  // BATCHED requant of a whole tensor at ONE (c, M).  Commits ALL limbs (remainder + bounded q)
  // with ONE batch_feed and checks every floor with ONE batch_reveal_check_zero.  Public-constant
  // scaling stays local (mul_const).
  void requant_cm_batch(IntFp *out, const IntFp *acc, uint64_t c, std::size_t M,
                        std::size_t n, std::size_t qb = 0) const {
    if (n == 0) return;
    if (M == 0) {  // identity (optional public scale)
      for (std::size_t i = 0; i < n; ++i) out[i] = (c == 1) ? acc[i] : acc[i] * c;
      return;
    }
    // UPSCALE guard: callers pass M = (size_t)(-ce.second); when scale_search returns a POSITIVE
    // exponent (ratio>1, e.g. QDQ-int8 -> finer 2^-P) the shift is negative and the size_t cast
    // UNDERFLOWS to ~2^64.  An upscale by 2^e is an EXACT integer multiply (no rounding/range
    // check), so just multiply -- this is the BERT layer-1 bug (a negative shift broke the requant
    // gate).
    if (M > (static_cast<std::size_t>(1) << 40)) {
      const uint64_t f = static_cast<uint64_t>(1) << (static_cast<std::size_t>(0) - M);
      for (std::size_t i = 0; i < n; ++i) out[i] = ((c == 1) ? acc[i] : acc[i] * c) * f;
      return;
    }
    const std::size_t Mpad = ((M + precision - 1) / precision) * precision,
                      pad = Mpad - M, nchunks = Mpad / precision;
    const std::size_t QB = out_qbits(Mpad, qb), nq = (QB + precision - 1) / precision,
                      nl = nchunks + nq; // limbs per element: remainder, then q + 2^(QB-1)
    const uint64_t qoff = static_cast<uint64_t>(1) << (QB - 1);
    // top q limb also looked up scaled by 2^(nq*precision - QB): the limbs then cover exactly
    // [0, 2^QB) (DeepProve's shifted_chunk_multiplier)
    const uint64_t topmul = static_cast<uint64_t>(1) << (nq * precision - QB);
    const uint64_t bias = static_cast<uint64_t>(1) << (Mpad - 1);
    const uint64_t mask = (static_cast<uint64_t>(1) << precision) - 1;
    const uint64_t padf = static_cast<uint64_t>(1) << pad;
    std::vector<IntFp> Nb(n);
    std::vector<uint64_t> chraw(n * nl, 0);
    std::atomic<bool> overflow(false);
    // Phase 1: local -- Nb[i], and (ALICE) the raw limb values.  All lane math
    // (mul_const/add) on DISJOINT indices -> threaded ONLY for large n: requant is called
    // ~300K times/run, mostly tiny (per-head/per-row) -- raw-thread spawn there cost ~300s.
    const int _rt = (n >= (std::size_t)1 << 16) ? mat_threads : 1;
    volefier::parallel_for(n, _rt, [&](std::size_t i) {
      IntFp N = (c == 1) ? acc[i] : acc[i] * c;      // public scale: mul_const (local)
      Nb[i] = (pad ? N * padf : N) + IntFp(bias, PUBLIC);
      if (party == ALICE) {
        int64_t Nbv = decode_signed(static_cast<uint64_t>(VAL(Nb[i].value)));
        int64_t qv = Nbv >> static_cast<int>(Mpad);
        uint64_t rv = static_cast<uint64_t>(Nbv - (qv << static_cast<int>(Mpad)));
        if (qv < -static_cast<int64_t>(qoff) || qv >= static_cast<int64_t>(qoff)) overflow = true;
        const uint64_t u = static_cast<uint64_t>(qv + static_cast<int64_t>(qoff));
        for (std::size_t k = 0; k < nchunks; ++k)
          chraw[i * nl + k] = (rv >> (precision * k)) & mask;
        for (std::size_t k = 0; k < nq; ++k)
          chraw[i * nl + nchunks + k] = (u >> (precision * k)) & mask;
      }
    });
    if (overflow) error("requant: output exceeds q_bits -- pass a wider bound at this call site\n");
    // Phase 2: ONE batched commit of every limb.
    std::vector<IntFp> ch(n * nl);
    batch_feed(ch.data(), chraw.data(), (int64_t)(n * nl));
    // Phase 3: rebuild r and q from their limbs, ONE batched zero-check of the floor.
    std::vector<IntFp> zchk(n);
    const int64_t two_Mpad = static_cast<int64_t>(1) << static_cast<int>(Mpad);
    const IntFp negoff(PR - qoff, PUBLIC);
    // reconstruction lane math on disjoint indices -> threaded; the Access() pushes stay in a
    // SERIAL loop (shared witness buffer + possible auto-teardown).
    volefier::parallel_for(n, _rt, [&](std::size_t i) {
      IntFp r(static_cast<uint64_t>(0), PUBLIC), q = negoff;
      for (std::size_t k = 0; k < nchunks; ++k)
        r = r + ch[i * nl + k] * (static_cast<int64_t>(1) << (precision * k));
      for (std::size_t k = 0; k < nq; ++k)
        q = q + ch[i * nl + nchunks + k] * (static_cast<int64_t>(1) << (precision * k));
      zchk[i] = q * two_Mpad + r + Nb[i].negate();  // must equal 0: Nb == q*2^Mpad + r
      out[i] = q;
    });
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < nl; ++k) rc_rem->Access(ch[i * nl + k]);
      if (topmul != 1) { IntFp t = ch[i * nl + nl - 1] * topmul; rc_rem->Access(t); }
    }
    batch_reveal_check_zero(zchk.data(), (int64_t)n);
  }

  void requant_cm(std::vector<IntFp> &out, const std::vector<IntFp> &acc, uint64_t c,
                  std::size_t M, std::size_t qb = 0) const {
    out.resize(acc.size());
    requant_cm_batch(out.data(), acc.data(), c, M, acc.size(), qb);
  }

  // Re-quantization after a weight matmul: rescale ratio = wc*2^-precision.
  void requant(std::vector<IntFp> &out, std::vector<IntFp> &acc) const {
    requant_cm(out, acc, wc, precision);
  }

  // Fused multiplier + shift (e.g. attention's 1/sqrt(dh) folded into one
  // requant): out = round(acc * fpm / 2^shift), still the two-sided form.
  IntFp requant_fp_shift(IntFp acc, uint64_t fpm, std::size_t shift) const {
    return requant_cm_one(acc, fpm, shift);
  }

  // Committed inner product  sum_i a_i * b_i  (no requant; scale 2^-2p when both
  // operands are at scale 2^-p). Naive form: len committed multiplies, so ~len
  // field elements of communication.
  IntFp dot(const std::vector<IntFp> &a, const std::vector<IntFp> &b) const {
    IntFp s(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < a.size(); ++i) s = s + a[i] * b[i];
    return s;
  }

  // Committed inner product via emp-zk's BATCHED inner-product proof
  // (fp_zkp_inner_prdt). The whole length-m inner product is proven as ONE
  // batched-check entry instead of m committed multiplies: only the output z is
  // committed (~1 field element), the products are never materialized. This
  // turns matmul communication from O(n*m*k) into O(n*k). Soundness uses the
  // identity  [a..., -1] . [b..., z] = 0  with public constant 0 (z committed).
  IntFp dot_ip(const std::vector<IntFp> &a, const std::vector<IntFp> &b) const {
    const std::size_t m = a.size();
    IntFp z;
    if (party == ALICE) {
      uint64_t acc = 0;
      for (std::size_t i = 0; i < m; ++i)
        acc = add_mod(acc, mult_mod(static_cast<uint64_t>(VAL(a[i].value)),
                                    static_cast<uint64_t>(VAL(b[i].value))));
      z = IntFp(acc, ALICE);
    } else {
      z = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    std::vector<IntFp> xa(m + 1), yb(m + 1);
    for (std::size_t i = 0; i < m; ++i) {
      xa[i] = a[i];
      yb[i] = b[i];
    }
    xa[m] = IntFp(PR - 1, PUBLIC); // -1
    yb[m] = z;
    fp_zkp_inner_prdt(xa.data(), yb.data(), 0, m + 1);
    return z;
  }

  // Matrix x vector using the batched inner-product proof: acc[j] = W[j] . x.
  void linear_ip(std::vector<IntFp> &acc, const std::vector<std::vector<IntFp>> &W,
                 const std::vector<IntFp> &x) const {
    ipp->matmul(acc, W, x, mat_threads); // standalone threaded IP matmul
  }

  // Elementwise residual add (same scale in, same scale out).
  void add(std::vector<IntFp> &out, const std::vector<IntFp> &a,
           const std::vector<IntFp> &b) const {
    out.resize(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  }

  // Single-value re-quantization at the ACTIVATION scale (c = 1): round(acc/2^p).
  IntFp requant_one(IntFp acc) const {
    return requant_cm_one(acc, 1, precision);
  }

  // Re-quantization with a fixed-point multiplier c = fpm: round(acc*fpm/2^p).
  void requant_fp(std::vector<IntFp> &out, const std::vector<IntFp> &acc, uint64_t fpm) const {
    requant_cm(out, acc, fpm, precision);
  }

  // Affine map  out_i = requant(y_i * gamma_i) + beta_i  (scale 2^-p throughout),
  // used for LayerNorm's learned scale/shift.
  void affine(std::vector<IntFp> &out, const std::vector<IntFp> &y, const std::vector<IntFp> &gamma,
              const std::vector<IntFp> &beta) const {
    std::vector<IntFp> prod(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) prod[i] = y[i] * gamma[i];
    std::vector<IntFp> tr;
    requant_cm(tr, prod, 1, precision); // gamma is an activation-scale (c=1) param
    out.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) out[i] = tr[i] + beta[i];
  }

  // Quantized dense layer:  y = requant(W x) + bias  (scale 2^-p).
  // Activations (ReLU / GeLU) are applied separately as table lookups, matching
  // DeepProve's operator decomposition (see QuantActivation in quant_nonlinear.h).
  void dense_layer(std::vector<IntFp> &y, const std::vector<std::vector<IntFp>> &W,
                   const std::vector<IntFp> &bias, const std::vector<IntFp> &x) const {
    std::vector<IntFp> acc;
    linear_ip(acc, W, x); // batched inner-product proof (cheap communication)
    requant(y, acc);
    for (std::size_t j = 0; j < y.size(); ++j) y[j] = y[j] + bias[j];
  }

  // PER-TENSOR quantized dense layer -- the zkGPT / DeepProve method, vs
  // dense_layer's uniform 2^-P. Each activation tensor carries its OWN scale S,
  // so the rescale ratio is S_in*S_w/S_out (a non-power-of-two real), searched to
  // a c*2^e multiplier (scale_search) and applied with requant_cm. The output is
  // at the per-tensor scale S_out, NOT a global 2^-P. (Assumes a down-scale, i.e.
  // S_in*S_w/S_out < 1, so the searched exponent e <= 0 and the shift is -e.)
  void dense_layer_q(std::vector<IntFp> &y, const std::vector<std::vector<IntFp>> &W,
                     const std::vector<IntFp> &bias, const std::vector<IntFp> &x, double s_in,
                     double s_w, double s_out) const {
    std::vector<IntFp> acc;
    linear_ip(acc, W, x);
    std::pair<int, int> ce = scale_search(s_in * s_w / s_out); // ratio ~ c * 2^e
    std::vector<IntFp> rq;
    requant_cm(rq, acc, static_cast<uint64_t>(ce.first), static_cast<std::size_t>(-ce.second));
    y.resize(W.size());
    for (std::size_t j = 0; j < W.size(); ++j) y[j] = rq[j] + bias[j];
  }

  // BATCHED per-tensor dense layer: Y[t] = requant(W . X[t]) + bias for ALL
  // tokens t at once. Same proof as calling dense_layer_q per token, but the
  // whole [n_out x seq] product is one matmul_mm (a single batch_feed +
  // batch_check vs one pair per token) and the O(n_out*seq*d) field work is
  // threaded over n_out*seq -- far better load balance / fewer thread spawns
  // than threading each token's matmul separately. The requant LogUp is also
  // batched per output channel (one requant_cm over all seq entries of a row).
  //   W: [n_out][d]   X: [seq][d]   ->   Y: [seq][n_out]
  void dense_layer_q_mm(std::vector<std::vector<IntFp>> &Y,
                        const std::vector<std::vector<IntFp>> &W,
                        const std::vector<IntFp> &bias,
                        const std::vector<std::vector<IntFp>> &X, double s_in, double s_w,
                        double s_out) const {
    std::vector<std::vector<IntFp>> acc;          // [n_out][seq]: acc[j][t] = <W[j], X[t]>
    static const bool FV = !getenv("FREIVALDS") || atoi(getenv("FREIVALDS")) != 0;   // DEFAULT ON (measured -4.2% e2e @500Mbps); FREIVALDS=0 disables
    if (FV) ipp->matmul_mm_freivalds(acc, X, W, mat_threads);  // cols = X (P = seq), kernel = W (Cout = n_out)
    else    ipp->matmul_mm(acc, X, W, mat_threads);
    std::pair<int, int> ce = scale_search(s_in * s_w / s_out);
    const uint64_t c = static_cast<uint64_t>(ce.first);
    const std::size_t M = static_cast<std::size_t>(-ce.second);
    const std::size_t n_out = W.size(), seq = X.size();
    Y.assign(seq, std::vector<IntFp>(n_out));
    for (std::size_t j = 0; j < n_out; ++j) {     // requant the whole row (all tokens) at once
      std::vector<IntFp> rq;
      requant_cm(rq, acc[j], c, M);
      for (std::size_t t = 0; t < seq; ++t) Y[t][j] = rq[t] + bias[j];
    }
  }

  // The per-tensor requant multiplier (c, shift) the layer above would use --
  // exposed so a reference / caller can reproduce it exactly.
  static std::pair<uint64_t, std::size_t> requant_mult(double s_in, double s_w, double s_out) {
    std::pair<int, int> ce = scale_search(s_in * s_w / s_out);
    return {static_cast<uint64_t>(ce.first), static_cast<std::size_t>(-ce.second)};
  }
};

#endif
