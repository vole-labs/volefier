#ifndef ML_FIXED_BACKWARD_H
#define ML_FIXED_BACKWARD_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-fixed/fixed_infer.h"
#include "volefier/ml-fixed/fixed_nonlinear.h"   // FixedTable + build_gelu_table (forward)
#include "volefier/ml-fixed/fixed_transformer.h" // Mat + FixedAttention (attention backward pairing)
#include "volefier/linear/ip_matmul_backward.h"  // matmul_backward (dLdA = dLdC B^T, dLdB = A^T dLdC)
#include <cmath>
#include <cstdint>
#include <vector>
using namespace emp;

// Backward-pass (proof-of-fine-tuning) gadgets for the ml-fixed track.  They
// consume COMMITTED forward-pass tensors (the values ALICE retained during the
// forward proof) and produce COMMITTED gradients, so a whole train step is one
// continuous VOLE-ZK proof.  Fixed-point convention is inherited from
// FixedInfer: everything shares the global scale 2^-precision (real = q*2^-P),
// so a product of two scale-2^-P tensors lands at 2^-2P and is rescaled back
// with the same requant used in the forward pass -- no MSB.

// ---------------------------------------------------------------------------
// Softmax backward.  Given the softmax outputs p (probabilities at scale 2^-P,
// i.e. p_real*2^P, exactly the vector FixedSoftmax::forward returns), produce
// dL/dscores.  Two entry points:
//   cross_entropy(): the softmax+cross-entropy fused special case, where the
//     upstream gradient collapses to  dL/dscores[i] = p[i] - 1{i==true_token}.
//     Pure linear arithmetic on committed values -- zero new gadgets, no mults,
//     no lookups (the one-hot "1" is the public constant 2^P).
//   general(): the full Jacobian-vector product for an arbitrary upstream
//     gradient dL/dp,  dL/dscores[i] = p[i] * (dL/dp[i] - dot),
//     dot = <p, dL/dp>.  Needs one committed inner product + one requant per
//     rescale (the p*(.) product is at 2^-2P).
class SoftmaxBackward {
public:
  FixedInfer *qi;

  explicit SoftmaxBackward(FixedInfer *qi_) : qi(qi_) {}

  // Cross-entropy special case:  dL/dscores[i] = p[i] - 1{i==true_token}.
  // p is at scale 2^-P; the subtracted one-hot "1" is the public constant 2^P.
  // Every output is a free linear combination of committed values (no mults).
  void cross_entropy(std::vector<IntFp> &dLdscores, const std::vector<IntFp> &p,
                     std::size_t true_token) const {
    const std::size_t L = p.size();
    const IntFp one(static_cast<uint64_t>(1) << qi->precision, PUBLIC); // 1.0 @ 2^-P
    dLdscores.resize(L);
    for (std::size_t i = 0; i < L; ++i)
      dLdscores[i] = (i == true_token) ? (p[i] + one.negate()) : p[i];
  }

  // Cross-entropy with a PRIVATE label: onehot is a committed 0/1 vector (scale 1) instead of a
  // public index, proven well formed here -- every entry a bit (g(g-1)=0, batched through the
  // shared Hadamard2IP) and the entries summing to 1 -- so the verifier never learns the target.
  //   dL/dscores[i] = p[i] - onehot[i] * 2^P.
  void cross_entropy_private(std::vector<IntFp> &dLdscores, const std::vector<IntFp> &p,
                             const std::vector<IntFp> &onehot) const {
    const std::size_t L = p.size();
    const uint64_t two_p = static_cast<uint64_t>(1) << qi->precision;
    std::vector<IntFp> om1(L);
    IntFp sum(static_cast<uint64_t>(0), PUBLIC);
    dLdscores.resize(L);
    for (std::size_t i = 0; i < L; ++i) {
      om1[i] = onehot[i] - static_cast<uint64_t>(1);
      sum = sum + onehot[i];
      dLdscores[i] = p[i] + (onehot[i] * two_p).negate();
    }
    volefier::global_h2ip()->add_zero(onehot.data(), om1.data(), L); // onehot[i] in {0,1}
    (sum - static_cast<uint64_t>(1)).reveal_zero();                   // exactly one 1
  }

  // General case:  dL/dscores[i] = p[i] * (dL/dp[i] - dot),  dot = <p, dL/dp>.
  // Both p and dL/dp are at scale 2^-P.  The inner product lands at 2^-2P, so it
  // is requantized back to 2^-P before the subtraction; the per-element product
  // p[i]*(dL/dp[i]-dot) is again at 2^-2P and requantized to 2^-P.
  void general(std::vector<IntFp> &dLdscores, const std::vector<IntFp> &p,
               const std::vector<IntFp> &dLdp) const {
    const std::size_t L = p.size();
    // dot = sum_j p_j * dL/dp_j   (scale 2^-2P) -> rescale to 2^-P
    IntFp dot2p = qi->dot(p, dLdp);
    IntFp dot = qi->requant_one(dot2p); // round(dot2p / 2^P), scale 2^-P
    // prod_i = p_i * (dL/dp_i - dot)   (scale 2^-2P) -> requant to 2^-P
    std::vector<IntFp> prod(L);
    for (std::size_t i = 0; i < L; ++i)
      prod[i] = p[i] * (dLdp[i] + dot.negate());
    qi->requant_cm(dLdscores, prod, 1, qi->precision);
  }

  // general() at finer scales, for long rows: with L keys the probabilities are ~1/L, so at
  // 2^-P most p[i]*(dL/dp[i]-dot) round to 0 or +-1 ulp.  Here p is at 2^-pbits, dL/dp comes in
  // at 2^-2P, dot is kept at 2^-2P, and the 2^-(2P+pbits) product is NOT requantized -- the
  // caller does that in one batched requant over all rows.
  void general_fine(std::vector<IntFp> &prod, const std::vector<IntFp> &p,
                    const std::vector<IntFp> &dLdp, std::size_t pbits) const {
    const std::size_t L = p.size();
    IntFp dot = qi->requant_cm_one(qi->dot(p, dLdp), 1, pbits); // -> 2^-2P
    prod.resize(L);
    for (std::size_t i = 0; i < L; ++i) prod[i] = p[i] * (dLdp[i] + dot.negate());
  }
};

// ---------------------------------------------------------------------------
// GELU'(x) table over inputs idx-OFFSET (scale 2^-P) -- the backward twin of
// build_gelu_table.  Same structure (offset-shifted index, output field-encoded
// because GELU' dips slightly negative), but tabulates the derivative of the
// SAME tanh approximation used in the forward table so a train step stays exact:
//   g(x)  = 0.5 x (1 + tanh(u)),  u = k (x + 0.044715 x^3),  k = sqrt(2/pi)
//   g'(x) = 0.5 (1 + tanh(u)) + 0.5 x (1 - tanh^2(u)) * u',  u' = k(1 + 0.134145 x^2)
static inline std::vector<uint64_t> build_gelu_grad_table(uint64_t T, int P, int64_t offset) {
  std::vector<uint64_t> g(T);
  const double scale = static_cast<double>(1ULL << P);
  const double k = std::sqrt(2.0 / M_PI);
  for (uint64_t i = 0; i < T; ++i) {
    double x = (static_cast<double>(static_cast<int64_t>(i) - offset)) / scale;
    double u = k * (x + 0.044715 * x * x * x);
    double th = std::tanh(u);
    double du = k * (1.0 + 3.0 * 0.044715 * x * x);
    double gp = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
    long long q = llround(gp * scale);
    g[i] =
        (q >= 0) ? static_cast<uint64_t>(q) : static_cast<uint64_t>(q + static_cast<long long>(PR));
  }
  return g;
}

// ---------------------------------------------------------------------------
// GELU backward.  Elementwise chain rule through the (tanh-approx) GELU that the
// forward pass proved as a FixedActivation table lookup:  dL/dx_i = dL/dy_i *
// GELU'(x_i).  GELU'(x_i) is a public-function table lookup (idx = x_i + offset,
// LogUp key-value binding range-checks it into [0, T)) mirroring the forward
// FixedActivation; the chain-rule product is one committed multiply per element,
// requantized from 2^-2P back to 2^-P with FixedInfer's requant.
// The class OWNS its gradient table -- call Teardown() before finalize_zk_arith.
class GELUBackward {
public:
  FixedInfer *qi;
  FixedTable *gtab = nullptr;
  int64_t offset;
  uint64_t T;

  GELUBackward(FixedInfer *qi_, uint64_t T_, int64_t offset_)
      : qi(qi_), offset(offset_), T(T_) {
    gtab = new FixedTable(T_, qi_->party, qi_->io);
    gtab->Setup(build_gelu_grad_table(T_, static_cast<int>(qi_->precision), offset_));
  }
  ~GELUBackward() { delete gtab; }

  // Flush the LogUp key-value teardown for the gradient table. MUST run before
  // finalize_zk_arith (matches how the forward FixedTable is torn down).
  void Teardown() { gtab->Teardown(); }

  // dL/dx_i = requant(dL/dy_i * GELU'(x_i)).  x and dL/dy are at scale 2^-P; the
  // looked-up GELU'(x_i) is at 2^-P, so the product is at 2^-2P -> requant to 2^-P.
  void forward(std::vector<IntFp> &dLdx, const std::vector<IntFp> &dLdy,
               const std::vector<IntFp> &x) const {
    const std::size_t n = x.size();
    // GELU'(x_i) via the shifted-index lookup (ONE batched feed for all n), same
    // idx = x_i + offset convention as the forward FixedActivation.
    std::vector<IntFp> idx(n);
    const IntFp off(static_cast<uint64_t>(offset), PUBLIC);
    for (std::size_t i = 0; i < n; ++i) idx[i] = x[i] + off;
    std::vector<IntFp> gp;
    gtab->lookup(gp, idx);
    // chain rule: prod_i = dL/dy_i * GELU'(x_i)  (scale 2^-2P) -> requant to 2^-P
    std::vector<IntFp> prod(n);
    for (std::size_t i = 0; i < n; ++i) prod[i] = dLdy[i] * gp[i];
    qi->requant(dLdx, prod); // round(prod / 2^P), wc == 1
  }
};

// ---------------------------------------------------------------------------
// RMSNorm backward (Llama/Gemma).  The forward (QuantRMSNorm / FixedLayerNorm's
// RMS twin) proved  y_i = gamma_i * x_hat_i,  x_hat_i = x_i * dinv,  dinv = 1/RMS
// committed and range-proven sqrt-free.  Backward consumes the RETAINED forward
// commitments (x_hat, gamma, dinv -- all already proven correct) plus the upstream
// dL/dy and produces committed gradients:
//   dL/dgamma_i = dL/dy_i * x_hat_i
//   dL/dx_i     = (dinv/d) * ( d*gamma_i*dL/dy_i - x_hat_i * dot ),
//                 dot = sum_j gamma_j * dL/dy_j * x_hat_j.
// Distributed as the algebraically identical  dL/dx_i = dinv*(g_i - x_hat_i*mean),
// g_i = gamma_i*dL/dy_i,  mean = dot/d -- so the only non-power-of-two division is
// ONE mean (via Div::round_div by d*2^P), mirroring the forward RMSNorm's mean;
// everything else is committed multiplies + the requant.  dinv, d
// are the forward RMS witnesses; d is the feature dimension (== x_hat.size()).
class RMSNormBackward {
public:
  FixedInfer *qi;
  Div *divr; // the single mean (dot/d) rescale, exactly like the forward RMSNorm's mean

  RMSNormBackward(FixedInfer *qi_, Div *divr_) : qi(qi_), divr(divr_) {}

  void forward(std::vector<IntFp> &dLdx, std::vector<IntFp> &dLdgamma,
               const std::vector<IntFp> &dLdy, const std::vector<IntFp> &x_hat,
               const std::vector<IntFp> &gamma, const IntFp &dinv, uint64_t d) const {
    const std::size_t n = x_hat.size();
    const std::size_t P = qi->precision;

    // dL/dgamma_i = requant(dL/dy_i * x_hat_i)   (product 2^-2P -> 2^-P)
    std::vector<IntFp> pg(n);
    for (std::size_t i = 0; i < n; ++i) pg[i] = dLdy[i] * x_hat[i];
    qi->requant_cm(dLdgamma, pg, 1, P);

    // g_i = requant(gamma_i * dL/dy_i)  ( = dL/dx_hat_i, scale 2^-P)
    std::vector<IntFp> pgd(n);
    for (std::size_t i = 0; i < n; ++i) pgd[i] = gamma[i] * dLdy[i];
    std::vector<IntFp> g;
    qi->requant_cm(g, pgd, 1, P);

    // dot = <g, x_hat> (scale 2^-2P); mean = dot/d rescaled to 2^-P by ONE round_div
    // by d*2^P (the 2^-2P->2^-P rescale and the /d folded into a single divisor).
    IntFp dot2p = qi->dot(g, x_hat);
    IntFp D(static_cast<uint64_t>(d) << P, PUBLIC); // d * 2^P, public divisor
    IntFp mean = divr->round_div(dot2p, D);         // scale 2^-P

    // C_i = g_i - requant(x_hat_i * mean) ;  dL/dx_i = requant(dinv * C_i)
    std::vector<IntFp> pq(n);
    for (std::size_t i = 0; i < n; ++i) pq[i] = x_hat[i] * mean;
    std::vector<IntFp> Q;
    qi->requant_cm(Q, pq, 1, P);
    std::vector<IntFp> pr(n);
    for (std::size_t i = 0; i < n; ++i) pr[i] = dinv * (g[i] + Q[i].negate());
    qi->requant_cm(dLdx, pr, 1, P);
  }
};

// ---------------------------------------------------------------------------
// Multi-head causal self-attention backward (pairs with FixedAttention::
// forward_save).  Consumes the RETAINED forward commitments -- Q,K,V (2^-P),
// the per (token,head) softmax probs (2^-P), ctx (2^-2P), and dL/dOut (2^-P) --
// plus the output-projection weights Wo, and produces committed gradients
// dL/dH, dL/dQ, dL/dK, dL/dV (all 2^-P) and dL/dWo (flat d*d, 2^-P).
//
// Everything is committed multiplies (via matmul_backward) + the softmax-Jacobian
// gadget + requants; no new LogUp tables, so no Teardown().
//
// Scale bookkeeping (S = 2^P):
//   * Output projection  out[i] = Wo @ ctx[i]  is proven per token with ONE
//     matmul_backward: A=Wo (dxd), B=ctx[i] (dx1).  It returns simultaneously
//       dLdA = outer(dL/dOut[i], ctx[i])       -> dL/dWo contribution (2^-3P)
//       dLdB = Wo^T @ dL/dOut[i]               -> dCtx_requant[i]      (2^-2P)
//     dCtx_requant is already at 2^-2P (matching ctx_saved), so it feeds Attn@V
//     backward directly (this is the "requant dCtx by P" the forward_save note
//     refers to -- realized as the natural 2^-2P matmul scale, not a extra shift).
//   * Attn@V backward feeds dCtx (2^-2P) x probs,V (2^-P): products land at 2^-3P
//     and are requantized by 2P back to 2^-P.
//   * The forward FUSED 1/sqrt(dh) into requant_fp_shift(dotQK@2^-2P, qinv, 2P);
//     the backward un-scale is requant_fp_shift(dScores@2^-P, qinv, P) -- shift P,
//     one 2^P less than the forward, because dScores is one scale finer than dotQK.
//   * Q@K^T backward feeds dScores (2^-P) x Q,K (2^-P): products at 2^-2P,
//     requantized by P to 2^-P.
class AttentionBackward {
public:
  FixedInfer *qi;
  FixedSoftmax *sm; // accepted for API symmetry with FixedAttention (unused: the
                    // softmax Jacobian is done by SoftmaxBackward, below)
  std::size_t h, dh;
  uint64_t qinv;
  int P;

  AttentionBackward(FixedInfer *qi_, FixedSoftmax *sm_, std::size_t h_, std::size_t dh_,
                    uint64_t qinv_, int P_)
      : qi(qi_), sm(sm_), h(h_), dh(dh_), qinv(qinv_), P(P_) {}

  void forward(Mat &dLdH, Mat &dLdQ, Mat &dLdK, Mat &dLdV, std::vector<IntFp> &dLdWo_flat,
               const Mat &dLdOut, const Mat &Q_saved, const Mat &K_saved, const Mat &V_saved,
               const std::vector<std::vector<std::vector<IntFp>>> &probs_saved, const Mat &ctx_saved,
               const Mat &Wo, uint64_t d) const {
    const std::size_t s = dLdOut.size();
    const std::size_t Pz = static_cast<std::size_t>(P);
    volefier::IPProof<BoolIO> &ipp = *qi->ipp;
    SoftmaxBackward sb(qi);
    const IntFp zero(static_cast<uint64_t>(0), PUBLIC);

    dLdQ.assign(s, std::vector<IntFp>(d, zero));
    dLdK.assign(s, std::vector<IntFp>(d, zero));
    dLdV.assign(s, std::vector<IntFp>(d, zero));
    dLdH.assign(s, std::vector<IntFp>(d, zero));

    // Wo flattened row-major (d x d) for matmul_backward.
    std::vector<IntFp> Wo_flat(d * d);
    for (std::size_t a = 0; a < d; ++a)
      for (std::size_t b = 0; b < d; ++b) Wo_flat[a * d + b] = Wo[a][b];

    // ---- STEP 1: output projection backward (one matmul_backward per token) ----
    std::vector<IntFp> dWo_acc(d * d, zero);      // dL/dWo accumulator, 2^-3P
    Mat dCtx_requant(s, std::vector<IntFp>(d));    // Wo^T @ dL/dOut, 2^-2P (== ctx_saved scale)
    for (std::size_t i = 0; i < s; ++i) {
      std::vector<IntFp> dWo_i, dCtx_i;
      volefier::matmul_backward(ipp, dWo_i, dCtx_i, Wo_flat, ctx_saved[i], dLdOut[i], (int)d, (int)d,
                                1);
      for (std::size_t t = 0; t < d * d; ++t) dWo_acc[t] = dWo_acc[t] + dWo_i[t];
      dCtx_requant[i] = dCtx_i; // 2^-2P
    }

    // ---- STEP 2: per-token per-head attention backward (causal, j <= i) ----
    for (std::size_t i = 0; i < s; ++i) {
      const std::size_t L = i + 1; // causal length
      for (std::size_t hh = 0; hh < h; ++hh) {
        const std::size_t off = hh * dh;

        // 2a. Attn@V backward: ctx_h (1 x dh) = probs_h (1 x L) @ V_h (L x dh).
        const std::vector<IntFp> &probs_h = probs_saved[i][hh]; // length L, 2^-P
        std::vector<IntFp> V_h(L * dh);
        for (std::size_t j = 0; j < L; ++j)
          for (std::size_t k = 0; k < dh; ++k) V_h[j * dh + k] = V_saved[j][off + k];
        std::vector<IntFp> dCtx_h(dh); // 2^-2P
        for (std::size_t k = 0; k < dh; ++k) dCtx_h[k] = dCtx_requant[i][off + k];
        std::vector<IntFp> dprobs_h, dV_h; // both 2^-3P
        volefier::matmul_backward(ipp, dprobs_h, dV_h, probs_h, V_h, dCtx_h, 1, (int)L, (int)dh);
        std::vector<IntFp> dprobs_P, dV_P; // requant 2P -> 2^-P
        qi->requant_cm(dprobs_P, dprobs_h, 1, 2 * Pz);
        qi->requant_cm(dV_P, dV_h, 1, 2 * Pz);
        for (std::size_t j = 0; j < L; ++j)
          for (std::size_t k = 0; k < dh; ++k)
            dLdV[j][off + k] = dLdV[j][off + k] + dV_P[j * dh + k];

        // 2b. Softmax backward (general Jacobian-vector product), stays at 2^-P.
        std::vector<IntFp> dScores_h;
        sb.general(dScores_h, probs_h, dprobs_P);

        // 2c. Undo the fused forward 1/sqrt(dh) scale: dScores (2^-P) -> d(dotQK) (2^-P).
        std::vector<IntFp> dScores_scaled(L);
        for (std::size_t j = 0; j < L; ++j)
          dScores_scaled[j] = qi->requant_fp_shift(dScores_h[j], qinv, Pz); // shift P (not 2P)

        // 2d. Q@K^T backward: scores (1 x L) = Q_h (1 x dh) @ Kh^T (dh x L).
        std::vector<IntFp> Q_h(dh);
        for (std::size_t k = 0; k < dh; ++k) Q_h[k] = Q_saved[i][off + k];
        std::vector<IntFp> KhT(dh * L); // (dh x L): KhT[k*L+j] = K[j][off+k]
        for (std::size_t k = 0; k < dh; ++k)
          for (std::size_t j = 0; j < L; ++j) KhT[k * L + j] = K_saved[j][off + k];
        std::vector<IntFp> dQ_h, dKhT; // dQ_h (1 x dh) 2^-2P ; dKhT (dh x L) = dKh^T 2^-2P
        volefier::matmul_backward(ipp, dQ_h, dKhT, Q_h, KhT, dScores_scaled, 1, (int)dh, (int)L);
        std::vector<IntFp> dQ_P, dKhT_P; // requant P -> 2^-P
        qi->requant_cm(dQ_P, dQ_h, 1, Pz);
        qi->requant_cm(dKhT_P, dKhT, 1, Pz);
        for (std::size_t k = 0; k < dh; ++k) dLdQ[i][off + k] = dLdQ[i][off + k] + dQ_P[k];
        for (std::size_t j = 0; j < L; ++j)
          for (std::size_t k = 0; k < dh; ++k)
            dLdK[j][off + k] = dLdK[j][off + k] + dKhT_P[k * L + j]; // transpose dKh^T -> dKh
      }
    }

    // ---- STEP 1 (finalize): dL/dWo = requant_2P(sum of per-token outer products) ----
    qi->requant_cm(dLdWo_flat, dWo_acc, 1, 2 * Pz);

    // ---- STEP 3: dL/dH.  The Q/K/V projection weights are not passed to this gadget
    // (only Wo is), so dL/dH is the summed projection-input gradient -- the
    // identity/residual view  dL/dH[i] = dL/dQ[i] + dL/dK[i] + dL/dV[i].  With the
    // weights available a caller would instead route each through matmul_backward. ----
    for (std::size_t i = 0; i < s; ++i)
      for (std::size_t k = 0; k < d; ++k) dLdH[i][k] = dLdQ[i][k] + dLdK[i][k] + dLdV[i][k];
  }

  // Additive extension of forward(): also backprop through the Q/K/V projections
  // (Q[i] = Wq @ H_normed[i], etc.), producing the TRUE dL/dH and the projection weight
  // gradients instead of forward()'s identity/residual dL/dH = dQ+dK+dV.  Per token:
  //   dL/dWq += outer(dL/dQ[i], H_normed[i]) ,  dL/dH += Wq^T @ dL/dQ[i]   (one matmul_backward)
  // H_normed and dL/dQ are BOTH at 2^-P, so every product lands at 2^-2P and is requantized
  // by P (NOT 2P -- that is the output-projection case, whose ctx operand is 2^-2P giving
  // 2^-3P; here H_normed is 2^-P).  dL/dW{q,k,v} are returned both flat (d*d) and as Mat.
  void forward_full(Mat &dLdH, Mat &dLdWq, Mat &dLdWk, Mat &dLdWv, std::vector<IntFp> &dLdWq_flat,
                    std::vector<IntFp> &dLdWk_flat, std::vector<IntFp> &dLdWv_flat, Mat &dLdQ,
                    Mat &dLdK, Mat &dLdV, std::vector<IntFp> &dLdWo_flat, const Mat &dLdOut,
                    const Mat &Q_saved, const Mat &K_saved, const Mat &V_saved,
                    const std::vector<std::vector<std::vector<IntFp>>> &probs_saved,
                    const Mat &ctx_saved, const Mat &Wo, const Mat &Wq, const Mat &Wk,
                    const Mat &Wv, const Mat &H_normed, uint64_t d) const {
    // 1. attention-inner backward: fills dLdQ, dLdK, dLdV, dLdWo_flat.  Its identity-view
    //    dL/dH is discarded; the true projection gradient replaces it below.
    Mat dLdH_inner;
    forward(dLdH_inner, dLdQ, dLdK, dLdV, dLdWo_flat, dLdOut, Q_saved, K_saved, V_saved,
            probs_saved, ctx_saved, Wo, d);

    // 2. backprop through the Q/K/V projections.
    const std::size_t s = dLdOut.size();
    const std::size_t Pz = static_cast<std::size_t>(P);
    volefier::IPProof<BoolIO> &ipp = *qi->ipp;
    const IntFp zero(static_cast<uint64_t>(0), PUBLIC);

    std::vector<IntFp> Wq_flat(d * d), Wk_flat(d * d), Wv_flat(d * d);
    for (std::size_t a = 0; a < d; ++a)
      for (std::size_t b = 0; b < d; ++b) {
        Wq_flat[a * d + b] = Wq[a][b];
        Wk_flat[a * d + b] = Wk[a][b];
        Wv_flat[a * d + b] = Wv[a][b];
      }

    std::vector<IntFp> dWq_acc(d * d, zero), dWk_acc(d * d, zero), dWv_acc(d * d, zero); // 2^-2P
    dLdH.assign(s, std::vector<IntFp>(d, zero));
    for (std::size_t i = 0; i < s; ++i) {
      std::vector<IntFp> dWq_i, dHq_i, dWk_i, dHk_i, dWv_i, dHv_i;
      volefier::matmul_backward(ipp, dWq_i, dHq_i, Wq_flat, H_normed[i], dLdQ[i], (int)d, (int)d, 1);
      volefier::matmul_backward(ipp, dWk_i, dHk_i, Wk_flat, H_normed[i], dLdK[i], (int)d, (int)d, 1);
      volefier::matmul_backward(ipp, dWv_i, dHv_i, Wv_flat, H_normed[i], dLdV[i], (int)d, (int)d, 1);
      for (std::size_t t = 0; t < d * d; ++t) {
        dWq_acc[t] = dWq_acc[t] + dWq_i[t];
        dWk_acc[t] = dWk_acc[t] + dWk_i[t];
        dWv_acc[t] = dWv_acc[t] + dWv_i[t];
      }
      // dL/dH[i] = requant_P( (Wq^T + Wk^T + Wv^T) @ dL/d{Q,K,V}[i] )   (2^-2P -> 2^-P)
      std::vector<IntFp> hsum(d);
      for (std::size_t k = 0; k < d; ++k) hsum[k] = dHq_i[k] + dHk_i[k] + dHv_i[k];
      qi->requant_cm(dLdH[i], hsum, 1, Pz);
    }
    // weight-grad accumulators 2^-2P -> 2^-P (requant P; H_normed operand is 2^-P)
    qi->requant_cm(dLdWq_flat, dWq_acc, 1, Pz);
    qi->requant_cm(dLdWk_flat, dWk_acc, 1, Pz);
    qi->requant_cm(dLdWv_flat, dWv_acc, 1, Pz);
    // reshape flat (d*d) -> Mat (d x d)
    auto reshape = [&](Mat &M, const std::vector<IntFp> &f) {
      M.assign(d, std::vector<IntFp>(d));
      for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b) M[a][b] = f[a * d + b];
    };
    reshape(dLdWq, dLdWq_flat);
    reshape(dLdWk, dLdWk_flat);
    reshape(dLdWv, dLdWv_flat);
  }
};

// ---------------------------------------------------------------------------
// LayerNorm backward.  The forward (FixedLayerNorm) proved  y_i = x_hat_i =
// (x_i - mu)*dinv,  out_i = gamma_i*y_i + beta_i,  dinv = 1/sigma committed and
// range-proven sqrt-free.  Backward consumes the RETAINED forward commitments --
// c = x - mu (2^-P), gamma (2^-P), dinv (2^-P, the SAME committed witness, never
// recomputed) -- plus upstream dL/dy and produces committed gradients:
//   dL/dgamma_i = dL/dy_i * x_hat_i
//   dL/dbeta_i  = dL/dy_i                         (free -- no multiply, no requant)
//   dL/dx_i     = dinv*( g_i - mean(g) - x_hat_i*mean(g (.) x_hat) ),  g = dL/dy (.) gamma.
// Unlike RMSNorm, LayerNorm subtracts BOTH the mean of g (the shift direction,
// dot2) and the g-component aligned with x_hat (the scale direction, dot1); both
// are means (the /n on dot1 is what a naive <g,y> would miss).  The two means are
// the only non-power-of-two divisions, done on the single shared Div gadget:
//   dot1 = <g,y>/n  via round_div by n*2^P (folds the 2^-2P->2^-P rescale and /n),
//   dot2 = sum(g)/n via round_div by n     (g is already 2^-P).
// x_hat = y = requant(c*dinv) is computed ONCE and reused. No LogUp tables -> no Teardown.
class LayerNormBackward {
public:
  FixedInfer *qi;
  Div *divr; // the two means (dot1, dot2); one instance, two round_div calls

  LayerNormBackward(FixedInfer *qi_, Div *divr_) : qi(qi_), divr(divr_) {}

  void forward(std::vector<IntFp> &dLdx, std::vector<IntFp> &dLdgamma, std::vector<IntFp> &dLdbeta,
               const std::vector<IntFp> &dLdy, const std::vector<IntFp> &c,
               const std::vector<IntFp> &gamma, const IntFp &dinv, uint64_t n) const {
    const std::size_t N = c.size();
    const std::size_t P = qi->precision;

    // x_hat = y_i = requant(c_i * dinv)   (2^-2P -> 2^-P), reused in STEPs 1,3,5.
    std::vector<IntFp> cd(N);
    for (std::size_t i = 0; i < N; ++i) cd[i] = c[i] * dinv;
    std::vector<IntFp> y;
    qi->requant_cm(y, cd, 1, P);

    // STEP 1: dL/dgamma_i = requant(dL/dy_i * y_i) ;  dL/dbeta_i = dL/dy_i (free copy).
    std::vector<IntFp> pgam(N);
    for (std::size_t i = 0; i < N; ++i) pgam[i] = dLdy[i] * y[i];
    qi->requant_cm(dLdgamma, pgam, 1, P);
    dLdbeta = dLdy; // dL/dbeta_i == dL/dy_i, no multiply, no requant

    // STEP 2: g_i = requant(dL/dy_i * gamma_i)   ( = dL/dx_hat_i, scale 2^-P)
    std::vector<IntFp> pg(N);
    for (std::size_t i = 0; i < N; ++i) pg[i] = dLdy[i] * gamma[i];
    std::vector<IntFp> g;
    qi->requant_cm(g, pg, 1, P);

    // STEP 3: dot1 = mean(g (.) x_hat) = <g,y>/n.  <g,y> is 2^-2P; dividing by n*2^P
    // both rescales to 2^-P and applies the /n (mirrors RMSNormBackward's mean).
    IntFp dot1 = divr->round_div(qi->dot(g, y), IntFp(static_cast<uint64_t>(n) << P, PUBLIC));

    // STEP 4: dot2 = mean(g) = sum(g)/n.  sum(g) is already 2^-P, so divisor is n.
    IntFp sum_g(static_cast<uint64_t>(0), PUBLIC);
    for (std::size_t i = 0; i < N; ++i) sum_g = sum_g + g[i];
    IntFp dot2 = divr->round_div(sum_g, IntFp(static_cast<uint64_t>(n), PUBLIC));

    // STEP 5: dL/dx_i = requant( dinv * (g_i - dot2 - requant(y_i*dot1)) ).
    std::vector<IntFp> py(N);
    for (std::size_t i = 0; i < N; ++i) py[i] = y[i] * dot1; // 2^-2P
    std::vector<IntFp> yprod;
    qi->requant_cm(yprod, py, 1, P); // requant(y_i*dot1) -> 2^-P
    std::vector<IntFp> pterm(N);
    for (std::size_t i = 0; i < N; ++i)
      pterm[i] = dinv * (g[i] + dot2.negate() + yprod[i].negate()); // 2^-2P
    qi->requant_cm(dLdx, pterm, 1, P);
  }
};

// ---------------------------------------------------------------------------
// SGD parameter update (the train-step tail of proof-of-fine-tuning):
//   W_new = W_old - lr * dW.
// The learning rate is a PUBLIC hyperparameter both parties know, pre-quantized
// to lr_q = round(lr * 2^P).  lr_q * dW_i lands at 2^-2P (both operands 2^-P) and
// is requantized back to 2^-P; the subtraction is a free linear combination.
// One committed multiply per parameter, nothing else -- no divisions, no lookups,
// no Teardown.  A matrix variant is provided for LoRA adapter weights.
class SGDOptimizer {
public:
  FixedInfer *qi;
  uint64_t lr_q; // learning rate at scale 2^-P (public constant), = round(lr * 2^P)

  SGDOptimizer(FixedInfer *qi_, uint64_t lr_q_) : qi(qi_), lr_q(lr_q_) {}

  // W_new_i = W_old_i - requant(lr_q * dW_i).
  void forward(std::vector<IntFp> &W_new, const std::vector<IntFp> &W_old,
               const std::vector<IntFp> &dW) const {
    const std::size_t n = W_old.size();
    const IntFp lr(lr_q, PUBLIC); // public hyperparameter -- not committed
    std::vector<IntFp> prod(n);
    for (std::size_t i = 0; i < n; ++i) prod[i] = dW[i] * lr; // 2^-2P
    std::vector<IntFp> update;
    qi->requant_cm(update, prod, 1, qi->precision); // round(lr_q * dW / 2^P), 2^-P
    W_new.resize(n);
    for (std::size_t i = 0; i < n; ++i) W_new[i] = W_old[i] + update[i].negate(); // free subtract
  }

  // Matrix update (LoRA adapters): flatten row-major, update, reshape.
  void forward_mat(Mat &W_new, const Mat &W_old, const Mat &dW) const {
    const std::size_t rows = W_old.size();
    const std::size_t cols = rows ? W_old[0].size() : 0;
    std::vector<IntFp> Wf, dWf;
    Wf.reserve(rows * cols);
    dWf.reserve(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < cols; ++c) {
        Wf.push_back(W_old[r][c]);
        dWf.push_back(dW[r][c]);
      }
    }
    std::vector<IntFp> Wnf;
    forward(Wnf, Wf, dWf);
    W_new.assign(rows, std::vector<IntFp>(cols));
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < cols; ++c) W_new[r][c] = Wnf[r * cols + c];
  }
};

// ---------------------------------------------------------------------------
// FFN (GPT-2 MLP) backward.  The forward per token is
//   hidden  = requant(W1 @ n2) + b1        (d -> ffn, 2^-P)
//   geluout = GELU(hidden)                 (table lookup, 2^-P)
//   ff      = requant(W2 @ geluout) + b2   (ffn -> d, 2^-P)
// Given upstream dL/dff (2^-P) plus the RETAINED n2 (LN2 output), hidden (pre-GELU),
// and geluout, produce dL/dn2 (2^-P) and the parameter grads dL/dW1, dL/db1,
// dL/dW2, dL/db2.  Chain: W2 dense backward (matmul_backward) -> GELUBackward ->
// W1 dense backward.  Every operand is 2^-P, so each matmul_backward product lands
// at 2^-2P and is requantized by P; bias grads are the free per-token sum of the
// layer's output grad.  Takes a caller-owned GELUBackward (which owns the GELU'
// table and must be torn down before finalize_zk_arith) -- FFNBackward itself owns
// no LogUp tables, so no Teardown of its own.
class FFNBackward {
public:
  FixedInfer *qi;
  GELUBackward *gb; // GELU' lookup (caller owns + Teardown()s it)

  FFNBackward(FixedInfer *qi_, GELUBackward *gb_) : qi(qi_), gb(gb_) {}

  void forward(Mat &dLdn2, std::vector<IntFp> &dLdW1_flat, std::vector<IntFp> &dLdb1,
               std::vector<IntFp> &dLdW2_flat, std::vector<IntFp> &dLdb2, const Mat &dLdff,
               const Mat &n2, const Mat &hidden, const Mat &geluout, const Mat &W1, const Mat &W2,
               uint64_t d, uint64_t ffn) const {
    const std::size_t s = dLdff.size();
    const std::size_t Pz = qi->precision;
    volefier::IPProof<BoolIO> &ipp = *qi->ipp;
    const IntFp zero(static_cast<uint64_t>(0), PUBLIC);

    // W1 is [ffn][d], W2 is [d][ffn]; flatten row-major.
    std::vector<IntFp> W1_flat(ffn * d), W2_flat(d * ffn);
    for (std::size_t a = 0; a < ffn; ++a)
      for (std::size_t b = 0; b < d; ++b) W1_flat[a * d + b] = W1[a][b];
    for (std::size_t a = 0; a < d; ++a)
      for (std::size_t b = 0; b < ffn; ++b) W2_flat[a * ffn + b] = W2[a][b];

    std::vector<IntFp> dW2_acc(d * ffn, zero), dW1_acc(ffn * d, zero); // 2^-2P
    std::vector<IntFp> db2_acc(d, zero), db1_acc(ffn, zero);          // 2^-P (free sums)
    dLdn2.assign(s, std::vector<IntFp>(d));

    for (std::size_t i = 0; i < s; ++i) {
      // W2 dense backward: ff = requant(W2 @ geluout) + b2.
      std::vector<IntFp> dW2_i, dGelu_i; // dW2_i (d x ffn) 2^-2P ; dGelu_i (ffn) 2^-2P
      volefier::matmul_backward(ipp, dW2_i, dGelu_i, W2_flat, geluout[i], dLdff[i], (int)d, (int)ffn,
                                1);
      for (std::size_t t = 0; t < d * ffn; ++t) dW2_acc[t] = dW2_acc[t] + dW2_i[t];
      for (std::size_t j = 0; j < d; ++j) db2_acc[j] = db2_acc[j] + dLdff[i][j]; // dL/db2 = sum_i dL/dff
      std::vector<IntFp> dGelu_P;
      qi->requant_cm(dGelu_P, dGelu_i, 1, Pz); // 2^-2P -> 2^-P

      // GELU backward: dL/dhidden = dL/dgeluout (.) GELU'(hidden).
      std::vector<IntFp> dHidden_i;
      gb->forward(dHidden_i, dGelu_P, hidden[i]); // 2^-P
      for (std::size_t j = 0; j < ffn; ++j) db1_acc[j] = db1_acc[j] + dHidden_i[j]; // dL/db1 = sum_i dL/dhidden

      // W1 dense backward: hidden = requant(W1 @ n2) + b1.
      std::vector<IntFp> dW1_i, dN2_i; // dW1_i (ffn x d) 2^-2P ; dN2_i (d) 2^-2P
      volefier::matmul_backward(ipp, dW1_i, dN2_i, W1_flat, n2[i], dHidden_i, (int)ffn, (int)d, 1);
      for (std::size_t t = 0; t < ffn * d; ++t) dW1_acc[t] = dW1_acc[t] + dW1_i[t];
      qi->requant_cm(dLdn2[i], dN2_i, 1, Pz); // 2^-2P -> 2^-P
    }
    qi->requant_cm(dLdW2_flat, dW2_acc, 1, Pz); // 2^-2P -> 2^-P
    qi->requant_cm(dLdW1_flat, dW1_acc, 1, Pz);
    dLdb2 = db2_acc; // already 2^-P
    dLdb1 = db1_acc;
  }
};

// LM-head (unembedding) backward for a single position -- the output side of
// proof-of-fine-tuning.  Forward:
//   logits = requant(We @ H) + be        (We: vocab x d, H: d, logits: vocab)
// The training loss is softmax(logits) + cross-entropy, whose gradient is
// dL/dlogits = p - onehot(true_token)  (via FixedSoftmax::forward then
// SoftmaxBackward::cross_entropy -- both existing).  This gadget turns that into
// the parameter/input grads:
//   dL/dWe = outer(dL/dlogits, H)   dL/dH = We^T @ dL/dlogits   dL/dbe = dL/dlogits.
// H and dL/dlogits are both 2^-P, so the one matmul_backward's products land at
// 2^-2P and are requantized by P; dL/dbe is the free copy.  No tables, no Teardown.
class LMHeadBackward {
public:
  FixedInfer *qi;

  explicit LMHeadBackward(FixedInfer *qi_) : qi(qi_) {}

  void forward(std::vector<IntFp> &dLdH, std::vector<IntFp> &dLdWe_flat, std::vector<IntFp> &dLdbe,
               const std::vector<IntFp> &dLdlogits, const std::vector<IntFp> &H, const Mat &We,
               uint64_t vocab, uint64_t d) const {
    const std::size_t P = qi->precision;
    volefier::IPProof<BoolIO> &ipp = *qi->ipp;
    std::vector<IntFp> We_flat(vocab * d);
    for (std::size_t v = 0; v < vocab; ++v)
      for (std::size_t k = 0; k < d; ++k) We_flat[v * d + k] = We[v][k];
    std::vector<IntFp> dWe_i, dH_i; // dWe_i (vocab x d) 2^-2P ; dH_i (d) 2^-2P
    volefier::matmul_backward(ipp, dWe_i, dH_i, We_flat, H, dLdlogits, (int)vocab, (int)d, 1);
    qi->requant_cm(dLdWe_flat, dWe_i, 1, P); // 2^-2P -> 2^-P
    qi->requant_cm(dLdH, dH_i, 1, P);
    dLdbe = dLdlogits; // dL/dbe = dL/dlogits, free
  }

  // Frozen head, many positions (LoRA fine-tuning):  dL/dH = dL/dlogits @ We for every loss
  // position at once with ONE whole-sequence matmul; no dL/dWe.  dLdlogits: T x vocab, 2^-P.
  void input_grad(Mat &dLdH, const Mat &dLdlogits, const Mat &We) const {
    const std::size_t T = dLdlogits.size(), vocab = We.size(), d = vocab ? We[0].size() : 0;
    std::vector<IntFp> raw, out;
    mm_seq(qi, raw, flat_of(dLdlogits), flat_of(We), T, d, vocab); // 2^-2P
    qi->requant_cm(out, raw, 1, qi->precision);
    dLdH.assign(T, std::vector<IntFp>(d));
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t k = 0; k < d; ++k) dLdH[t][k] = out[t * d + k];
  }
};

// ---------------------------------------------------------------------------
// LoRA adapter on one frozen projection W (dout x din):  W_eff = W + (alpha/r) * B @ A, with
// B: dout x r and A: r x din, both flat row-major.  The adapters live at a FINER master scale
// 2^-(P+E) (E extra fractional bits, like fp32 master weights in mixed precision) so a small SGD
// step does not round to zero; W_eff is produced at the activation scale 2^-P.  Both directions
// are proven from the COMMITTED adapters, so the forward pass is bound to them:
//   effective():  W_eff = W + round(B@A * s_c / 2^s_M)              (B@A at 2^-2(P+E))
//   sgd_step():   B' = B - round((dW_eff @ A^T) * g_c / 2^g_M)
//                 A' = A - round((B^T @ dW_eff) * g_c / 2^g_M)      (both at 2^-(2P+E+G))
// with s_c/2^s_M = (alpha/r) * 2^-(P+2E) and g_c/2^g_M = eta * (alpha/r) * 2^-(P+G), eta the
// step's learning rate and dW_eff taken at the finer gradient scale 2^-(P+G) that
// LoRABlockBackward produces.  (alpha/r) enters both directions exactly as in standard LoRA.
class LoRAAdapter {
public:
  FixedInfer *qi;
  std::size_t dout, din, r, E, G;
  uint64_t s_c, g_c;
  std::size_t s_M, g_M;

  LoRAAdapter(FixedInfer *qi_, std::size_t dout_, std::size_t din_, std::size_t r_,
              std::size_t E_, std::size_t G_, double scale, double eta)
      : qi(qi_), dout(dout_), din(din_), r(r_), E(E_), G(G_) {
    const std::size_t P = qi->precision;
    to_fixed(scale, s_c, s_M);
    s_M += P + 2 * E;
    to_fixed(eta * scale, g_c, g_M);
    g_M += P + G;
  }

  // x ~= c / 2^M with c in [2^9, 2^10): 10 significant bits keeps acc*c far below 2^60.
  static void to_fixed(double x, uint64_t &c, std::size_t &M) {
    int e;
    std::frexp(x, &e); // x = m * 2^e, m in [0.5, 1)
    M = static_cast<std::size_t>(10 - e);
    c = static_cast<uint64_t>(llround(std::ldexp(x, static_cast<int>(M))));
  }

  void effective(Mat &W_eff, const Mat &W, const std::vector<IntFp> &B,
                 const std::vector<IntFp> &A) const {
    std::vector<IntFp> raw, delta;
    mm_seq(qi, raw, B, A, dout, din, r);
    qi->requant_cm(delta, raw, s_c, s_M);
    W_eff.assign(dout, std::vector<IntFp>(din));
    for (std::size_t i = 0; i < dout; ++i)
      for (std::size_t j = 0; j < din; ++j) W_eff[i][j] = W[i][j] + delta[i * din + j];
  }

  // Unscaled adapter gradients from dL/dW_eff (dout x din, 2^-(P+G)), at 2^-(2P+E+G):
  // rawB = dW_eff @ A^T, rawA = B^T @ dW_eff.  dL/dB = (alpha/r) rawB, dL/dA = (alpha/r) rawA.
  void raw_grads(std::vector<IntFp> &rawB, std::vector<IntFp> &rawA, const std::vector<IntFp> &B,
                 const std::vector<IntFp> &A, const std::vector<IntFp> &dWeff) const {
    mm_seq(qi, rawB, dWeff, transpose_of(A, r, din), dout, r, din);
    mm_seq(qi, rawA, transpose_of(B, dout, r), dWeff, r, din, dout);
  }

  // One SGD step on both adapters from dL/dW_eff (dout x din, 2^-(P+G)); both use the OLD B, A.
  void sgd_step(std::vector<IntFp> &B_new, std::vector<IntFp> &A_new, const std::vector<IntFp> &B,
                const std::vector<IntFp> &A, const std::vector<IntFp> &dWeff) const {
    std::vector<IntFp> rawB, rawA, uB, uA;
    raw_grads(rawB, rawA, B, A, dWeff);
    qi->requant_cm(uB, rawB, g_c, g_M);
    qi->requant_cm(uA, rawA, g_c, g_M);
    B_new.resize(B.size());
    A_new.resize(A.size());
    for (std::size_t t = 0; t < B.size(); ++t) B_new[t] = B[t] + uB[t].negate();
    for (std::size_t t = 0; t < A.size(); ++t) A_new[t] = A[t] + uA[t].negate();
  }
};

// ---------------------------------------------------------------------------
// AdamW on committed parameters theta (scale 2^-Ft) with committed BIAS-CORRECTED moments
// m (= m_hat, 2^-Fg) and v (= v_hat, 2^-Fv), carried from step to step.  Updating m_hat directly,
//   m_hat_t = m_hat_{t-1} + a_t (g - m_hat_{t-1}),  a_t = (1-b1) / (1-b1^t)   (likewise v_hat, b2),
// is the same recurrence as AdamW's m_t = b1 m_{t-1} + (1-b1) g followed by m_t / (1-b1^t), and
// keeps v_hat ~ g^2 from step 1 instead of (1-b2) g^2, so its fixed-point resolution is not lost.
// Per element, from a raw gradient at 2^-Fin whose real value is gscale * raw:
//   g      = round(raw * gscale * 2^(Fg-Fin))                  |g| < 2^(29-Fg)
//   m'     = m + round(a1_t (g - m))
//   v'     = v + round(a2_t (round(g^2 / 2^(2Fg-Fv)) - v))      v' in [0, 2^Vb), Fv = 2Fg - 10
//   s      = round(sqrt(v' 2^2K))                               sqrt(v') at 2^-(Fv/2+K)
//   r      = round(m' 2^Fr / (s + e))                           m'/(sqrt(v')+eps) at 2^-Fr
//   theta' = theta - round(r * lr 2^(Ft-Fr)) - round(theta * lr wd)
// Fg (default 28) places the window of representable gradient magnitudes, ~[2^-(Fg-5), 2^(29-Fg)),
// like a loss scale; a gradient past its top makes the proof fail.  Every rescale is
// FixedInfer::requant_cm; s is proven as in prove_rsqrt (|v' 2^2K - s^2| <= s, exact range
// checks, s^2 < 2^60), r by Div::round_div with the committed divisor s + e.  a1_t, a2_t, lr and
// lr wd are 10-significant-bit constants (LoRAAdapter::to_fixed); e is eps at the scale of s,
// at least 1.
class FixedAdamW {
public:
  FixedInfer *qi;
  RangeCheckLU *rc;
  Div *div;
  std::size_t Ft, Fin, Fg, Fv, Vb = 48, K, Sb, Fr = 16;
  double lr, b1, b2, eps, wd;
  uint64_t cu, cw = 0, e;
  std::size_t Mu, Mw = 0;

  // the moment coefficients a1_t, a2_t of step t
  struct StepConst { uint64_t cg, c1, c2; std::size_t Mg, M1, M2; };

  FixedAdamW(FixedInfer *qi_, RangeCheckLU *rc_, std::size_t Ft_, std::size_t Fin_, double lr_,
             double b1_ = 0.9, double b2_ = 0.999, double eps_ = 1e-8, double wd_ = 0.01,
             std::size_t Fg_ = 28)
      : qi(qi_), rc(rc_), Ft(Ft_), Fin(Fin_), Fg(Fg_), Fv(2 * Fg_ - 10), lr(lr_), b1(b1_), b2(b2_), eps(eps_), wd(wd_) {
    K = 29 - (Vb + 1) / 2;   // s < 2^Sb = 2^30, s^2 < 2^60; v' 2^2K < 2^58
    Sb = (Vb + 1) / 2 + K + 1;
    LoRAAdapter::to_fixed(lr * std::ldexp(1.0, (int)Ft - (int)Fr), cu, Mu);
    if (wd > 0) LoRAAdapter::to_fixed(lr * wd, cw, Mw);
    e = std::max<uint64_t>(1, llround(eps * std::ldexp(1.0, (int)(Fv / 2 + K))));
    div = new Div(qi->party, qi->io, Sb + 2);   // divisor s + e, 2(s + e) < 2^(Sb+2)
  }
  ~FixedAdamW() { delete div; }
  void destroy() { div->destroy(); }   // before finalize_zk_arith

  StepConst consts(double gscale, int t) const {
    StepConst k{};
    LoRAAdapter::to_fixed(gscale * std::ldexp(1.0, (int)Fg - (int)Fin), k.cg, k.Mg);
    LoRAAdapter::to_fixed((1 - b1) / (1 - std::pow(b1, t)), k.c1, k.M1);
    LoRAAdapter::to_fixed((1 - b2) / (1 - std::pow(b2, t)), k.c2, k.M2);
    return k;
  }

  // One step on a parameter vector; m and v are updated in place.
  void step(std::vector<IntFp> &th_new, std::vector<IntFp> &m, std::vector<IntFp> &v,
            const std::vector<IntFp> &th, const std::vector<IntFp> &raw, double gscale, int t) const {
    const std::size_t n = th.size();
    const StepConst k = consts(gscale, t);
    std::vector<IntFp> g, d, upd, gsq, sq(n), s(n);
    qi->requant_cm(g, raw, k.cg, k.Mg, 30);                          // |g| < 2^29: g^2 < 2^58
    d.resize(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = g[i] + m[i].negate();
    qi->requant_cm(upd, d, k.c1, k.M1);
    for (std::size_t i = 0; i < n; ++i) { m[i] = m[i] + upd[i]; sq[i] = g[i] * g[i]; }
    qi->requant_cm(gsq, sq, 1, 2 * Fg - Fv, Vb + 2);                // up to 2^48, like v
    for (std::size_t i = 0; i < n; ++i) d[i] = gsq[i] + v[i].negate();
    qi->requant_cm(upd, d, k.c2, k.M2, Vb + 2);
    for (std::size_t i = 0; i < n; ++i) v[i] = v[i] + upd[i];
    // s = round(sqrt(v * 2^2K))
    std::vector<uint64_t> sraw(n, 0);
    if (qi->party == ALICE)
      for (std::size_t i = 0; i < n; ++i) {
        const int64_t vv = FixedInfer::decode_signed(static_cast<uint64_t>(VAL(v[i].value)));
        if (vv < 0 || vv >> Vb) error("FixedAdamW: v outside [0, 2^Vb)\n");
        sraw[i] = round_sqrt(static_cast<uint64_t>(vv) << (2 * K));
      }
    batch_feed(s.data(), sraw.data(), static_cast<int64_t>(n));
    std::vector<IntFp> ep(n), en(n);
    for (std::size_t i = 0; i < n; ++i) {
      const IntFp e = v[i] * (static_cast<uint64_t>(1) << (2 * K)) + (s[i] * s[i]).negate();
      ep[i] = s[i] + e; en[i] = s[i] + e.negate();
    }
    rc->check_batch(v.data(), n, Vb, true);
    rc->check_batch(s.data(), n, Sb, true);
    rc->check_batch(ep.data(), n, Sb + 1, true);   // |v 2^2K - s^2| <= s
    rc->check_batch(en.data(), n, Sb + 1, true);
    // r = round(m 2^Fr / (s + e)), then the parameter update
    std::vector<IntFp> r(n), wdv;
    const IntFp ep_(e, PUBLIC);
    for (std::size_t i = 0; i < n; ++i) r[i] = div->round_div(m[i] * (static_cast<uint64_t>(1) << Fr), s[i] + ep_);
    qi->requant_cm(upd, r, cu, Mu);
    if (wd > 0) qi->requant_cm(wdv, th, cw, Mw);
    th_new.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      th_new[i] = th[i] + upd[i].negate();
      if (wd > 0) th_new[i] = th_new[i] + wdv[i].negate();
    }
  }

  static uint64_t round_sqrt(uint64_t X) {
    uint64_t r = static_cast<uint64_t>(sqrtl(static_cast<long double>(X)));
    while (r * r > X) --r;
    while ((r + 1) * (r + 1) <= X) ++r;
    return (X - r * r > r) ? r + 1 : r;   // round: X > r^2 + r  <=>  sqrt(X) > r + 1/2
  }

  // Plaintext replay of step() on decoded integers (prover-side checks).
  void replay(std::vector<int64_t> &th_new, std::vector<int64_t> &m, std::vector<int64_t> &v,
              const std::vector<int64_t> &th, const std::vector<int64_t> &raw, double gscale, int t) const {
    const StepConst k = consts(gscale, t);
    auto rhu = [](__int128 N, std::size_t M) { return (int64_t)((N + ((__int128)1 << (M - 1))) >> M); };
    th_new.resize(th.size());
    for (std::size_t i = 0; i < th.size(); ++i) {
      const int64_t g = rhu((__int128)raw[i] * k.cg, k.Mg);
      m[i] += rhu((__int128)(g - m[i]) * k.c1, k.M1);
      const int64_t gs = rhu((__int128)g * g, 2 * Fg - Fv);
      v[i] += rhu((__int128)(gs - v[i]) * k.c2, k.M2);
      const int64_t sd = (int64_t)round_sqrt((uint64_t)v[i] << (2 * K)) + (int64_t)e;
      const int64_t N = m[i] * ((int64_t)1 << Fr);
      const int64_t r = N >= 0 ? (N + sd / 2) / sd : -((-N + sd / 2) / sd);   // Div: half away
      th_new[i] = th[i] - rhu((__int128)r * cu, Mu) - (wd > 0 ? rhu((__int128)th[i] * cw, Mw) : 0);
    }
  }
};

// ---------------------------------------------------------------------------
// Transformer-block backward over a whole sequence for FROZEN base weights (LoRA fine-tuning).
// Same math and scales as FFNBackward + AttentionBackward::forward_full + LayerNormBackward, but
//   * every matmul is one whole-sequence mm_seq (per head for the attention core) instead of
//     per-token matmul_backward calls, and
//   * frozen weights get no gradient: the only weight gradients produced are dL/dW_q and dL/dW_v
//     (the adapted projections, taken w.r.t. W_eff), each one d x d matmul over the sequence,
//     kept at the finer scale 2^-(P+G) -- they only feed LoRAAdapter::sgd_step, and at 2^-P a
//     small gradient (e.g. W_q's under near-uniform attention) would round to zero.
// The per-token gadgets commit an s x dout x din outer-product tensor for EVERY weight, which at
// GPT-2 size (s=512) is ~3.6e9 committed values per block.  Consumes the forward caches of the
// block's FixedLayerNorm / FixedAttention / FixedTransformerBlock (save_intermediates = true).
class LoRABlockBackward {
public:
  FixedInfer *qi;
  LayerNormBackward *lnb;
  GELUBackward *gb; // GELU' lookup (caller owns + Teardown()s it)
  std::size_t d, h, dh, ffn, G;
  uint64_t qinv;

  LoRABlockBackward(FixedInfer *qi_, LayerNormBackward *lnb_, GELUBackward *gb_, std::size_t d_,
                    std::size_t h_, std::size_t ffn_, uint64_t qinv_, std::size_t G_)
      : qi(qi_), lnb(lnb_), gb(gb_), d(d_), h(h_), dh(d_ / h_), ffn(ffn_), G(G_), qinv(qinv_) {}

  // dLdout: s x d (2^-P).  Wq / Wv are the EFFECTIVE weights used in the forward pass.
  // dWq / dWv come out at 2^-(P+G).
  void forward(Mat &dLdx, std::vector<IntFp> &dWq, std::vector<IntFp> &dWv, const Mat &dLdout,
               FixedLayerNorm &ln1, FixedLayerNorm &ln2, FixedAttention &attn,
               FixedTransformerBlock &blk, const Mat &Wq, const Mat &Wk, const Mat &Wv,
               const Mat &Wo, const Mat &W1, const Mat &W2, const std::vector<IntFp> &g1,
               const std::vector<IntFp> &g2) const {
    const std::size_t s = dLdout.size(), P = qi->precision;
    const IntFp zero(static_cast<uint64_t>(0), PUBLIC);
    std::vector<IntFp> raw;

    // ---- FFN:  dGelu = dOut @ W2,  dHidden = dGelu (.) GELU'(hidden),  dN2 = dHidden @ W1 ----
    std::vector<IntFp> dGelu, dHid, dN2;
    mm_seq(qi, raw, flat_of(dLdout), flat_of(W2), s, ffn, d);
    qi->requant_cm(dGelu, raw, 1, P);
    gb->forward(dHid, dGelu, flat_of(blk.cache["hidden"]));
    mm_seq(qi, raw, dHid, flat_of(W1), s, d, ffn);
    qi->requant_cm(dN2, raw, 1, P);

    // ---- LN2 backward + residual:  dL/da = dL/dout + LN2'(dN2) ----
    Mat dLda;
    ln_backward(dLda, dLdout, dN2, ln2, g2);

    // ---- attention.  dCtx = dL/da @ Wo stays at the 2^-2P ctx scale (as in AttentionBackward) ----
    std::vector<IntFp> dCtx;
    mm_seq(qi, dCtx, flat_of(dLda), flat_of(Wo), s, d, d);
    Mat &Q = attn.cache["Q"], &K = attn.cache["K"], &V = attn.cache["V"];
    Mat &probs = attn.cache["probs"]; // row i*h+hh: length i+1, 2^-(P+X)
    const std::size_t X = static_cast<std::size_t>(attn.prob_extra());
    std::vector<IntFp> dQ(s * d), dK(s * d), dV(s * d);
    SoftmaxBackward sb(qi);
    for (std::size_t hh = 0; hh < h; ++hh) {
      const std::size_t off = hh * dh;
      std::vector<IntFp> Qh(s * dh), Kh(s * dh), VhT(dh * s), dCh(s * dh), Pr(s * s, zero);
      for (std::size_t i = 0; i < s; ++i) {
        for (std::size_t k = 0; k < dh; ++k) {
          Qh[i * dh + k] = Q[i][off + k];
          Kh[i * dh + k] = K[i][off + k];
          VhT[k * s + i] = V[i][off + k];
          dCh[i * dh + k] = dCtx[i * d + off + k];
        }
        for (std::size_t j = 0; j <= i; ++j) Pr[i * s + j] = probs[i * h + hh][j];
      }
      // dprobs = dCtx_h @ V_h^T (2^-3P -> 2^-2P); only the causal j <= i entries are kept.
      // The softmax Jacobian and dS stay at 2^-2P: at 2^-P the ~1/L-sized entries of dS
      // round away and dQ (a sum over L keys) loses accuracy as the sequence grows.
      mm_seq(qi, raw, dCh, VhT, s, s, dh);
      std::vector<IntFp> causal, dpr;
      causal.reserve(s * (s + 1) / 2);
      for (std::size_t i = 0; i < s; ++i)
        for (std::size_t j = 0; j <= i; ++j) causal.push_back(raw[i * s + j]);
      qi->requant_cm(dpr, causal, 1, P);
      // dV_h = probs_h^T @ dCtx_h (2^-(3P+X) -> 2^-2P, kept fine for dW_v like dQ)
      std::vector<IntFp> dVh;
      mm_seq(qi, raw, transpose_of(Pr, s, s), dCh, s, dh, s);
      qi->requant_cm(dVh, raw, 1, P + X);
      // softmax Jacobian per query row (2^-(3P+X)) -> dS at 2^-2P
      std::vector<IntFp> dsc, dsq;
      dsc.reserve(causal.size());
      for (std::size_t i = 0, base = 0; i < s; base += i + 1, ++i) {
        std::vector<IntFp> dp(dpr.begin() + base, dpr.begin() + base + i + 1), ds;
        sb.general_fine(ds, probs[i * h + hh], dp, P + X);
        dsc.insert(dsc.end(), ds.begin(), ds.end());
      }
      qi->requant_cm(dsq, dsc, 1, P + X);
      std::vector<IntFp> dS(s * s, zero);
      for (std::size_t i = 0, t = 0; i < s; ++i)
        for (std::size_t j = 0; j <= i; ++j) dS[i * s + j] = dsq[t++];
      // dQ_h = dS @ K_h,  dK_h = dS^T @ Q_h  (2^-3P), each times the fused forward 1/sqrt(dh)
      // (qinv, 2^-P) in its requant: dQ to 2^-2P (kept fine for dW_q), dK to 2^-P.  Applying
      // qinv here rather than on dS keeps every intermediate ~ (true value) * 2^48.
      std::vector<IntFp> dQh, dKh;
      mm_seq(qi, raw, dS, Kh, s, dh, s);
      qi->requant_cm(dQh, raw, qinv, 2 * P);
      mm_seq(qi, raw, transpose_of(dS, s, s), Qh, s, dh, s);
      qi->requant_cm(dKh, raw, qinv, 3 * P);
      for (std::size_t i = 0; i < s; ++i)
        for (std::size_t k = 0; k < dh; ++k) {
          dQ[i * d + off + k] = dQh[i * dh + k];
          dK[i * d + off + k] = dKh[i * dh + k];
          dV[i * d + off + k] = dVh[i * dh + k];
        }
    }

    // ---- projections:  dL/dnormed = dQ@Wq + dK@Wk + dV@Wv,  dW{q,v} = dX^T @ normed ----
    // dQ and dV are at 2^-2P (dK at 2^-P): their products land at 2^-3P and are brought to
    // 2^-2P first, in one requant.
    std::vector<IntFp> rq, rk, rv, rqv, dNrm;
    mm_seq(qi, rq, dQ, flat_of(Wq), s, d, d);
    mm_seq(qi, rv, dV, flat_of(Wv), s, d, d);
    for (std::size_t t = 0; t < s * d; ++t) rq[t] = rq[t] + rv[t];
    qi->requant_cm(rqv, rq, 1, P);
    mm_seq(qi, rk, dK, flat_of(Wk), s, d, d);
    for (std::size_t t = 0; t < s * d; ++t) rk[t] = rqv[t] + rk[t];
    qi->requant_cm(dNrm, rk, 1, P);
    const std::vector<IntFp> nrm = flat_of(blk.cache["normed"]);
    mm_seq(qi, raw, transpose_of(dQ, s, d), nrm, d, d, s);
    qi->requant_cm(dWq, raw, 1, 2 * P - G); // 2^-3P -> 2^-(P+G)
    mm_seq(qi, raw, transpose_of(dV, s, d), nrm, d, d, s);
    qi->requant_cm(dWv, raw, 1, 2 * P - G);

    // ---- LN1 backward + residual:  dL/dx = dL/da + LN1'(dNrm) ----
    ln_backward(dLdx, dLda, dNrm, ln1, g1);
  }

private:
  // out[i] = resid[i] + LayerNorm'(dy[i]) per token, from the LN's cached c / dinv.  gamma is
  // frozen, so its gradient (computed by LayerNormBackward anyway) is dropped.
  void ln_backward(Mat &out, const Mat &resid, const std::vector<IntFp> &dy, FixedLayerNorm &ln,
                   const std::vector<IntFp> &g) const {
    Mat &c = ln.cache_mat["c"];
    std::vector<IntFp> &dinv = ln.cache_scalar_vec["dinv"];
    const std::size_t s = resid.size();
    out.assign(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      std::vector<IntFp> dyi(dy.begin() + i * d, dy.begin() + (i + 1) * d), dxi, dgi, dbi;
      lnb->forward(dxi, dgi, dbi, dyi, c[i], g, dinv[i], d);
      for (std::size_t k = 0; k < d; ++k) out[i][k] = resid[i][k] + dxi[k];
    }
  }
};

#endif
