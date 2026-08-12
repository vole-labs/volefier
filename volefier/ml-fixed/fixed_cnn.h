#ifndef ML_FIXED_CNN_H
#define ML_FIXED_CNN_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-fixed/fixed_infer.h"
#include "volefier/nonlinear/comparison.h" // CompareLU (max pool)
#include "volefier/nonlinear/div.h"        // Div (avg pool)
#include "volefier/linear/conv2d.h"
#include <vector>
#include <cstdint>
using namespace emp;

// CNN / ResNet building blocks on top of the quantized-inference gadgets.
// Feature maps are volefier::Tensor3 (C x H x W) or FMap (C x P = channel x
// flattened spatial). Convolution reuses the threaded IP matmul; everything
// else is a thin wrapper over requant / CompareLU / Div / add. ReLU is the
// existing FixedActivation (build_relu_table), applied per channel.

using Tensor3 = volefier::Tensor3;
using FMap = volefier::FMap;

// Conv2D + per-channel bias + requant (scale 2^-P out). Activations applied
// separately (DeepProve-style operator decomposition).
class FixedConv2D {
public:
  FixedInfer *qi;
  int C_in, H, W, C_out, K, stride, pad, H_out, W_out, P;

  FixedConv2D(FixedInfer *qi_, int C_in_, int H_, int W_, int C_out_, int K_, int stride_, int pad_)
      : qi(qi_), C_in(C_in_), H(H_), W(W_), C_out(C_out_), K(K_), stride(stride_), pad(pad_) {
    H_out = volefier::conv_out_dim(H, K, stride, pad);
    W_out = volefier::conv_out_dim(W, K, stride, pad);
    P = H_out * W_out;
  }

  // input: C_in x H x W.  kernel: C_out x (C_in*K*K).  bias: length C_out.
  // out: C_out x P at scale 2^-P.
  void forward(FMap &out, const Tensor3 &input, const std::vector<std::vector<IntFp>> &kernel,
               const std::vector<IntFp> &bias) {
    FMap raw;
    volefier::conv2d<BoolIO>(*qi->ipp, raw, input, kernel, C_in, H, W, C_out, K, stride, pad,
                                    qi->mat_threads);
    out.assign(C_out, {});
    for (int co = 0; co < C_out; ++co) {
      qi->requant(out[co], raw[co]); // round(W x * wc / 2^P)
      for (int p = 0; p < P; ++p) out[co][p] = out[co][p] + bias[co];
    }
  }
};

// Inference BatchNorm folded to a per-channel affine: out[c] = requant(x[c]*g[c]) + b[c]
// (g = gamma/sqrt(var+eps), b = beta - g*mean, both precomputed & committed per channel).
class FixedBatchNorm {
public:
  FixedInfer *qi;
  int C, P;
  FixedBatchNorm(FixedInfer *qi_, int C_, int P_) : qi(qi_), C(C_), P(P_) {}

  void forward(FMap &out, const FMap &x, const std::vector<IntFp> &gamma,
               const std::vector<IntFp> &beta) {
    out.assign(C, std::vector<IntFp>(P));
    for (int c = 0; c < C; ++c) {
      std::vector<IntFp> prod(P);
      for (int p = 0; p < P; ++p) prod[p] = x[c][p] * gamma[c]; // per-channel committed scale
      std::vector<IntFp> tr;
      qi->requant_cm(tr, prod, 1, qi->precision);
      for (int p = 0; p < P; ++p) out[c][p] = tr[p] + beta[c];
    }
  }
};

// Max pooling: per-channel max over each K x K window. Out-of-bounds positions
// are omitted (== padding with -inf, which never wins). Uses CompareLU.
class FixedMaxPool {
public:
  FixedInfer *qi;
  CompareLU *cmp;
  int C, H, W, K, stride, pad, H_out, W_out;
  std::size_t val_bits; // bit width of pooled values (for the max comparison)

  FixedMaxPool(FixedInfer *qi_, CompareLU *cmp_, int C_, int H_, int W_, int K_, int stride_,
               int pad_, std::size_t val_bits_)
      : qi(qi_), cmp(cmp_), C(C_), H(H_), W(W_), K(K_), stride(stride_), pad(pad_),
        val_bits(val_bits_) {
    H_out = volefier::conv_out_dim(H, K, stride, pad);
    W_out = volefier::conv_out_dim(W, K, stride, pad);
  }

  void forward(FMap &out, const Tensor3 &in) {
    out.assign(C, std::vector<IntFp>(H_out * W_out));
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < H_out; ++oh)
        for (int ow = 0; ow < W_out; ++ow) {
          std::vector<IntFp> win;
          for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj) {
              const int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) win.push_back(in[c][ih][iw]);
            }
          out[c][oh * W_out + ow] = cmp->max_value(win, val_bits);
        }
  }
};

// Average pooling: per-channel rounded mean over each window (Div::round_div),
// and global average pooling (mean over the whole map -> one value per channel).
class FixedAvgPool {
public:
  FixedInfer *qi;
  Div *divr;
  int C, H, W, K, stride, pad, H_out, W_out;

  FixedAvgPool(FixedInfer *qi_, Div *divr_, int C_, int H_, int W_, int K_, int stride_, int pad_)
      : qi(qi_), divr(divr_), C(C_), H(H_), W(W_), K(K_), stride(stride_), pad(pad_) {
    H_out = volefier::conv_out_dim(H, K, stride, pad);
    W_out = volefier::conv_out_dim(W, K, stride, pad);
  }

  void forward(FMap &out, const Tensor3 &in) {
    out.assign(C, std::vector<IntFp>(H_out * W_out));
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < H_out; ++oh)
        for (int ow = 0; ow < W_out; ++ow) {
          IntFp sum(static_cast<uint64_t>(0), PUBLIC);
          int cnt = 0;
          for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj) {
              const int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum = sum + in[c][ih][iw];
                ++cnt;
              }
            }
          out[c][oh * W_out + ow] = divr->round_div(sum, (uint64_t)cnt);  // public-D overload: no q*D mult-gate
        }
  }

  // Global average pool: out[c] = round(mean_p x[c][p]).  x is C x P.
  void global(std::vector<IntFp> &out, const FMap &x, int P) {
    out.resize(x.size());
    for (std::size_t c = 0; c < x.size(); ++c) {
      IntFp sum(static_cast<uint64_t>(0), PUBLIC);
      for (int p = 0; p < P; ++p) sum = sum + x[c][p];
      out[c] = divr->round_div(sum, (uint64_t)P);  // public-D overload: no q*D mult-gate
    }
  }
};

#endif
