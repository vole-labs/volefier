#ifndef ML_FIXED_INFER_H
#define ML_FIXED_INFER_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup.h"     // LogUpRangeCheck
#include "volefier/linear/ip_matmul.h" // standalone (thread-able) IP matmul
#include <cstdint>
#include <vector>
#include <cmath>
#include <utility>
using namespace emp;

// FixedInfer: building blocks for a VOLE-ZK proof of *fixed-point* neural-network
// inference -- the FIXED-POINT counterpart of volefier/ml-quant.
//
// Fixed-point convention: EVERYTHING (weights AND activations) shares ONE global
// scale 2^-precision (real value = q * 2^-precision). A matmul of two scale-2^-p
// tensors gives a scale-2^-2p accumulator, so rescaling back to 2^-p is a PURE
// power-of-two shift round(acc / 2^precision) -- no per-tensor weight mantissa
// (wc == 1). Non-power-of-two CONSTANTS (1/sqrt(dh), 1/sigma, gamma) are still
// represented as fixed-point values at scale 2^-p and applied with requant_cm.
//
// The only thing that differs from ml-quant: ml-quant carries a per-tensor weight
// scale wc != 1 on the dense-layer rescale (quantization); here wc == 1.
//
// Proof model: VOLE-ZK (QuickSilver), interactive/designated-verifier, MSB-free
// (all rescales via the two-sided range gadget). ALICE proves; BOB verifies.
//
// Proof model: this is VOLE-ZK (QuickSilver) — interactive, designated-verifier,
// communication grows with #mults + #lookup-rows. ALICE is the prover (knows the
// plaintext quantized model + activations); BOB is the verifier (sees only
// commitments). All correctness is enforced by the committed checks below plus
// the LogUp teardowns in destroy().
class FixedInfer {
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
  // FIXED-POINT: weights AND activations share ONE uniform scale 2^-precision, so
  // a dense-layer rescale is a pure power-of-two shift round(acc / 2^precision).
  // wc is fixed at 1 (no per-tensor weight mantissa) -- this is the only thing
  // that differs from ml-quant. Kept as a member for gadget symmetry.
  const uint64_t wc = 1;
  // Threaded standalone IP matmul (replaces emp-zk's singleton for dense layers).
  volefier::IPProof<BoolIO> *ipp = nullptr;
  int mat_threads = 1;
  bool flag_teardown = false;

  // bio is the BoolIO channel (ios[0]); the LogUp gadgets use bio->io (NetIO).
  FixedInfer(int party, std::size_t precision, BoolIO *bio, std::size_t out_bits = 16,
             int mat_threads_ = 1)
      : party(party), precision(precision), out_bits(out_bits),
        out_range(static_cast<uint64_t>(1) << out_bits), io(bio->io), mat_threads(mat_threads_) {
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

  ~FixedInfer() {
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

  // zkGPT roundLayer: prove q = round(acc * c / 2^M) by the TWO-SIDED RANGE
  // relation (no advice remainder, no MSB).  With N = acc*c:
  //   t1 = 2^M + 2N - q*2^(M+1) >= 0,   t2 = 2^M - 2N + q*2^(M+1) >= 0,
  // and t1 + t2 = 2^(M+1) (constant), so range-checking both into [0, 2^(M+1))
  // forces q = round(N / 2^M).  This is the c*2^e rescale: the rescale ratio
  // S_in*S_w/S_out is searched to c*2^-M and applied here.
  IntFp requant_cm_one(IntFp acc, uint64_t c, std::size_t M) const {
    IntFp N = (c == 1) ? acc : acc * c;   // public scale: *uint64 = mul_const (local), not a mult-gate
    IntFp q;
    if (party == ALICE) {
      int64_t Nv = decode_signed(static_cast<uint64_t>(VAL(N.value)));
      int64_t half = static_cast<int64_t>(1) << (M - 1);
      int64_t Dv = static_cast<int64_t>(1) << M;
      int64_t qv = (Nv >= 0) ? (Nv + half) / Dv : -(((-Nv) + half) / Dv);
      q = commit(qv);
    } else {
      q = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    IntFp DM(static_cast<uint64_t>(1) << M, PUBLIC);
    IntFp q2 = q * (static_cast<int64_t>(1) << (M + 1)); // q * 2^(M+1)
    range_chunks(DM + N * 2LL + q2.negate(), M + 1);     // t1 >= 0
    range_chunks(DM + (N * 2LL).negate() + q2, M + 1);   // t2 >= 0
    return q;
  }

  // Batched round-half-away requant: SAME two-sided range relation as requant_cm_one, but
  // ONE batch_feed for all quotients q + one for all the range-check chunks + ONE batched
  // zero-check, instead of a per-element commit + two per-element reveal_zeros.  Output is
  // bit-identical to the requant_cm_one loop (same rounding).
  void requant_cm_batch(std::vector<IntFp> &out, const std::vector<IntFp> &acc, uint64_t c,
                        std::size_t M) const {
    const std::size_t n = acc.size();
    out.resize(n);
    if (n == 0) return;
    const std::size_t nchunks = (M + 1 + precision - 1) / precision;
    const uint64_t mask = (static_cast<uint64_t>(1) << precision) - 1;
    const int64_t two_M = static_cast<int64_t>(1) << M;
    const int64_t two_M1 = static_cast<int64_t>(1) << (M + 1);
    std::vector<IntFp> N(n);
    std::vector<uint64_t> qraw(n, 0), c1raw(n * nchunks, 0), c2raw(n * nchunks, 0);
    // Phase 1: local -- N[i] and (ALICE) the raw q / t1,t2 chunk values.
    for (std::size_t i = 0; i < n; ++i) {
      N[i] = (c == 1) ? acc[i] : acc[i] * c;   // public scale: mul_const
      if (party == ALICE) {
        int64_t Nv = decode_signed(static_cast<uint64_t>(VAL(N[i].value)));
        int64_t half = static_cast<int64_t>(1) << (M - 1);
        int64_t qv = (Nv >= 0) ? (Nv + half) / two_M : -(((-Nv) + half) / two_M);
        qraw[i] = (qv >= 0) ? static_cast<uint64_t>(qv)
                            : static_cast<uint64_t>(qv + static_cast<int64_t>(PR));
        const int64_t t1v = two_M + 2 * Nv - qv * two_M1; // both in [0, 2^(M+1))
        const int64_t t2v = two_M - 2 * Nv + qv * two_M1;
        for (std::size_t k = 0; k < nchunks; ++k) {
          c1raw[i * nchunks + k] = (static_cast<uint64_t>(t1v) >> (precision * k)) & mask;
          c2raw[i * nchunks + k] = (static_cast<uint64_t>(t2v) >> (precision * k)) & mask;
        }
      }
    }
    // Phase 2: THREE batched commits (q + t1-chunks + t2-chunks).
    std::vector<IntFp> q(n), c1(n * nchunks), c2(n * nchunks);
    batch_feed(q.data(), qraw.data(), static_cast<int64_t>(n));
    batch_feed(c1.data(), c1raw.data(), static_cast<int64_t>(n * nchunks));
    batch_feed(c2.data(), c2raw.data(), static_cast<int64_t>(n * nchunks));
    // Phase 3: range-check chunks, rebuild t1,t2, ONE batched zero-check that recon == t.
    const IntFp DM(static_cast<uint64_t>(1) << M, PUBLIC);
    std::vector<IntFp> zchk;
    zchk.reserve(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
      IntFp q2 = q[i] * two_M1;
      IntFp t1 = DM + N[i] * 2LL + q2.negate();
      IntFp t2 = DM + (N[i] * 2LL).negate() + q2;
      IntFp r1(static_cast<uint64_t>(0), PUBLIC), r2(static_cast<uint64_t>(0), PUBLIC);
      for (std::size_t k = 0; k < nchunks; ++k) {
        rc_rem->Access(c1[i * nchunks + k]);
        rc_rem->Access(c2[i * nchunks + k]);
        const int64_t w = static_cast<int64_t>(1) << (precision * k);
        r1 = r1 + c1[i * nchunks + k] * w;
        r2 = r2 + c2[i * nchunks + k] * w;
      }
      zchk.push_back(r1 + t1.negate());
      zchk.push_back(r2 + t2.negate());
      out[i] = q[i];
    }
    batch_reveal_check_zero(zchk.data(), static_cast<int64_t>(zchk.size()));
  }

  void requant_cm(std::vector<IntFp> &out, const std::vector<IntFp> &acc, uint64_t c,
                  std::size_t M) const {
    requant_cm_batch(out, acc, c, M);
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
  // DeepProve's operator decomposition (see FixedActivation in quant_nonlinear.h).
  void dense_layer(std::vector<IntFp> &y, const std::vector<std::vector<IntFp>> &W,
                   const std::vector<IntFp> &bias, const std::vector<IntFp> &x) const {
    std::vector<IntFp> acc;
    linear_ip(acc, W, x); // batched inner-product proof (cheap communication)
    requant(y, acc);
    for (std::size_t j = 0; j < y.size(); ++j) y[j] = y[j] + bias[j];
  }

  // Whole-matrix dense (batches ALL rows): X [rows][in], W [out][in], bias [out] -> Y [rows][out].
  // Proves the entire [out x rows] product with ONE matmul_mm (a single batch_feed + batch_check)
  // instead of one dense_layer (one batch_check) per row, then a batched per-channel requant. The
  // proof + requant are bit-identical to calling dense_layer per row, but it costs ~1 network round
  // trip for the whole layer instead of `rows` -- the dominant WAN cost for GPT-2 at large seq.
  void dense_layer_mm(std::vector<std::vector<IntFp>> &Y, const std::vector<std::vector<IntFp>> &W,
                      const std::vector<IntFp> &bias, const std::vector<std::vector<IntFp>> &X) const {
    std::vector<std::vector<IntFp>> acc; // [out][rows]: acc[j][t] = <W[j], X[t]> @ 2^-2P
    ipp->matmul_mm(acc, X, W, mat_threads);
    const std::size_t n_out = W.size(), rows = X.size();
    Y.assign(rows, std::vector<IntFp>(n_out));
    for (std::size_t j = 0; j < n_out; ++j) {
      std::vector<IntFp> rq;
      requant(rq, acc[j]); // batched requant over all rows of channel j (same rounding as per-row)
      for (std::size_t t = 0; t < rows; ++t) Y[t][j] = rq[t] + bias[j];
    }
  }
};

#endif
