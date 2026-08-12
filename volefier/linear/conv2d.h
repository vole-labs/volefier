#ifndef LINEAR_CONV2D_H
#define LINEAR_CONV2D_H

#include "emp-zk/emp-zk.h"
#include "volefier/linear/ip_matmul.h"
#include <vector>
#include <cstdint>
using namespace emp;

// 2D convolution proven by lowering to the standalone threaded IP matmul.
// A conv output pixel is an inner product over the receptive field:
//   out[co,h,w] = sum_{ci,ki,kj} in[ci, h*s-p+ki, w*s-p+kj] * ker[co, ci,ki,kj]
// im2col rearranges the committed input into a (P x M) patch matrix (P =
// H_out*W_out, M = C_in*K*K) -- overlapping fields REUSE the same committed
// IntFps (no new commitment), out-of-bounds positions are a public 0. The conv
// is then one IP matmul per output channel: out[co] = cols . ker[co].

namespace volefier {

using Tensor3 = std::vector<std::vector<std::vector<IntFp>>>;  // C x H x W feature map
using FMap = std::vector<std::vector<IntFp>>;                  // C x P (channel x flat-spatial)

static inline int conv_out_dim(int in, int K, int stride, int pad) {
  return (in + 2 * pad - K) / stride + 1;
}

// Build the (P x M) im2col patch matrix from the committed input.
static inline void im2col(std::vector<std::vector<IntFp>> &cols, const Tensor3 &input,
                          int C_in, int H, int W, int K, int stride, int pad) {
  const int H_out = conv_out_dim(H, K, stride, pad);
  const int W_out = conv_out_dim(W, K, stride, pad);
  const int M = C_in * K * K, P = H_out * W_out;
  IntFp zero(static_cast<uint64_t>(0), PUBLIC);
  cols.assign(P, std::vector<IntFp>(M, zero));
  for (int oh = 0; oh < H_out; ++oh)
    for (int ow = 0; ow < W_out; ++ow) {
      const int p = oh * W_out + ow;
      for (int ci = 0; ci < C_in; ++ci)
        for (int ki = 0; ki < K; ++ki)
          for (int kj = 0; kj < K; ++kj) {
            const int ih = oh * stride - pad + ki;
            const int iw = ow * stride - pad + kj;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
              cols[p][ci * K * K + ki * K + kj] = input[ci][ih][iw];
          }
    }
}

// Raw conv (scale 2^-2P; caller re-quantizes). kernel rows are flattened in the
// same (ci,ki,kj) order as im2col. One threaded IP matmul per output channel.
template <typename IO>
static inline void conv2d(IPProof<IO> &ip, FMap &out, const Tensor3 &input,
                          const std::vector<std::vector<IntFp>> &kernel,
                          int C_in, int H, int W, int C_out, int K, int stride, int pad,
                          int nthreads = 1) {
  std::vector<std::vector<IntFp>> cols;
  im2col(cols, input, C_in, H, W, K, stride, pad);
  // ONE batched IP proof for the whole layer: out[co][p] = <ker[co], cols[p]>.
  // (Was a per-channel loop -> C_out network round-pairs per layer; now exactly one.)
  ip.matmul_mm(out, cols, kernel, nthreads);
}

}  // namespace volefier

#endif
