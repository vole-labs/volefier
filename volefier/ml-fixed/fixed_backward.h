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
// with the same two-sided-range requant used in the forward pass -- no MSB.

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
// requantized from 2^-2P back to 2^-P with FixedInfer's two-sided-range requant.
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
// everything else is committed multiplies + the two-sided-range requant.  dinv, d
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
// gadget + two-sided-range requants; no new LogUp tables, so no Teardown().
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

// ---------------------------------------------------------------------------
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
};

#endif
