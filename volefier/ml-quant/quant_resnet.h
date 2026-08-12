#ifndef ML_QUANT_RESNET_H
#define ML_QUANT_RESNET_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-quant/quant_cnn.h"
#include "volefier/nonlinear/comparison.h"
#include "volefier/nonlinear/div.h"
#include <vector>
#include <cstdint>
#include <string>
#include <functional>
using namespace emp;

// ResNet-18/34/50/101/152 assembled from the CNN building blocks. Convolution
// rides the threaded IP matmul; BN/ReLU/pool/FC are the MSB-free gadgets. This
// is a self-checking harness: it commits random weights AND tracks a parallel
// plaintext forward (ALICE), so the test can compare revealed logits to the
// integer reference. A real prover would commit loaded weights instead.

using IT3 = std::vector<std::vector<std::vector<int64_t>>>; // C x H x W plaintext

// committed + plaintext feature-map state
// PER-TENSOR quantization: each activation tensor carries its own scale `s`
// (zkGPT/DeepProve method) -- each conv/fc requants by S_in*S_w/S_out.
struct St {
  Tensor3 X;
  IT3 Xr;
  int C, H, W;
  double s = 1.0 / 256;
};
struct VecSt {
  std::vector<IntFp> X;
  std::vector<int64_t> Xr;
  double s = 1.0 / 256;
};

struct ResNetCfg {
  bool bottleneck;
  int blocks[4];
  int base;
  int classes;
  int in_ch;
  int in_hw;
  std::string name;
};
static inline ResNetCfg resnet_cfg(int depth, int base, int classes, int in_hw) {
  switch (depth) {
  case 18: return {false, {2, 2, 2, 2}, base, classes, 3, in_hw, "resnet18"};
  case 34: return {false, {3, 4, 6, 3}, base, classes, 3, in_hw, "resnet34"};
  case 50: return {true, {3, 4, 6, 3}, base, classes, 3, in_hw, "resnet50"};
  case 101: return {true, {3, 4, 23, 3}, base, classes, 3, in_hw, "resnet101"};
  default: return {true, {3, 8, 36, 3}, base, classes, 3, in_hw, "resnet152"};
  }
}

class ResNet {
public:
  QuantInfer *qi;
  CompareLU *cmp;
  Div *divr;
  PRG *prg;
  int party;
  std::size_t act_bits; // activation magnitude bound (ReLU/clamp comparisons)
  // per-tensor scales: S_res = residual-stream scale (shared so residual adds
  // stay valid), S_int = block-internal activation scale, S_w = weight scale,
  // S_fc = final logits scale. Each conv/fc requants by S_in*S_w/S_out -> c!=1.
  // ratios S_in*S_w/S_out land ~1/2400..1/3750 (bounded activations) and differ
  // per conv (S_res<->S_int) so each gets its own non-power-of-two multiplier.
  double S_res = 1.0 / 256, S_int = 1.0 / 320, S_w = 1.0 / 3000, S_fc = 1.0 / 256;

  ResNet(QuantInfer *qi_, CompareLU *cmp_, Div *divr_, PRG *prg_, int party_,
         std::size_t act_bits_ = 28)
      : qi(qi_), cmp(cmp_), divr(divr_), prg(prg_), party(party_), act_bits(act_bits_) {}

  int64_t rnd(int64_t lo, int64_t hi) {
    if (party != ALICE) return 0;
    uint64_t r;
    prg->random_data(&r, sizeof(r));
    return lo + (int64_t)(r % (uint64_t)(hi - lo + 1));
  }
  static int64_t rcm(int64_t acc, int64_t c, int M) {
    int64_t N = acc * c, h = (int64_t)1 << (M - 1), D = (int64_t)1 << M;
    return N >= 0 ? (N + h) / D : -(((-N) + h) / D);
  }
  static int64_t rdiv(int64_t N, int64_t D) {
    return N >= 0 ? (N + D / 2) / D : -(((-N) + D / 2) / D);
  }
  IntFp zero() const {
    return IntFp((uint64_t)0, PUBLIC);
  }

  // ---- profiling: set commf to a comm-byte reader to enable per-op accounting ----
  struct Phase {
    double us = 0;
    uint64_t bytes = 0;
  };
  Phase ph_commit, ph_conv, ph_bn, ph_relu, ph_pool, ph_fc;
  std::function<uint64_t()> commf = nullptr;
  uint64_t cb() const {
    return commf ? commf() : 0;
  }
  template <class F> void timed(Phase &p, F &&f) {
    auto t = clock_start();
    uint64_t b = cb();
    f();
    p.us += time_from(t);
    p.bytes += cb() - b;
  }

  // ---- Conv2D (+ per-channel bias + requant) : St -> St ----
  St conv(const St &in, int Co, int K, int s, int pd, double s_out) {
    const int Ho = volefier::conv_out_dim(in.H, K, s, pd),
              Wo = volefier::conv_out_dim(in.W, K, s, pd);
    const int M = in.C * K * K, Pn = Ho * Wo;
    // PER-TENSOR requant multiplier (c, shift) = scale_search(S_in * S_w / S_out)
    std::pair<uint64_t, std::size_t> mm = QuantInfer::requant_mult(in.s, S_w, s_out);
    const uint64_t cmant = mm.first;
    const std::size_t shift = mm.second;
    std::vector<std::vector<int64_t>> ker(Co, std::vector<int64_t>(M));
    std::vector<int64_t> bias(Co);
    for (auto &r : ker)
      for (auto &v : r) v = rnd(-3, 3);
    for (auto &v : bias) v = rnd(-30, 30);
    std::vector<std::vector<IntFp>> cker(Co);
    std::vector<IntFp> cbias;
    timed(ph_commit, [&] {
      for (int co = 0; co < Co; ++co) qi->commit_vec(cker[co], ker[co]);
      qi->commit_vec(cbias, bias);
    });

    volefier::FMap raw;
    St o;
    o.C = Co;
    o.H = Ho;
    o.W = Wo;
    o.s = s_out;
    o.X.assign(Co, std::vector<std::vector<IntFp>>(Ho, std::vector<IntFp>(Wo)));
    timed(ph_conv, [&] {
      volefier::conv2d<BoolIO>(*qi->ipp, raw, in.X, cker, in.C, in.H, in.W, Co, K, s, pd,
                                      qi->mat_threads);
      for (int co = 0; co < Co; ++co) {
        std::vector<IntFp> rq;
        qi->requant_cm(rq, raw[co], cmant, shift); // per-tensor c*2^-shift
        for (int p = 0; p < Pn; ++p) o.X[co][p / Wo][p % Wo] = rq[p] + cbias[co];
      }
    });
    if (party == ALICE) {
      o.Xr.assign(Co, std::vector<std::vector<int64_t>>(Ho, std::vector<int64_t>(Wo)));
      for (int co = 0; co < Co; ++co)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            int64_t acc = 0;
            for (int ci = 0; ci < in.C; ++ci)
              for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj) {
                  int ih = oh * s - pd + ki, iw = ow * s - pd + kj;
                  if (ih >= 0 && ih < in.H && iw >= 0 && iw < in.W)
                    acc += in.Xr[ci][ih][iw] * ker[co][ci * K * K + ki * K + kj];
                }
            o.Xr[co][oh][ow] = rcm(acc, (int64_t)cmant, (int)shift) + bias[co];
          }
    }
    return o;
  }

  // ---- BatchNorm (per-channel affine: requant(x*gamma)+beta) : St -> St ----
  St bn(const St &in) {
    const int P0 = (int)qi->precision;
    std::vector<int64_t> g(in.C), b(in.C);
    for (int c = 0; c < in.C; ++c) {
      g[c] = (1 << P0) + rnd(-100, 100);
      b[c] = rnd(-30, 30);
    }
    std::vector<IntFp> cg, cb;
    timed(ph_commit, [&] {
      qi->commit_vec(cg, g);
      qi->commit_vec(cb, b);
    });
    St o = in;
    o.X.assign(in.C, std::vector<std::vector<IntFp>>(in.H, std::vector<IntFp>(in.W)));
    timed(ph_bn, [&] {
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w)
            o.X[c][h][w] = qi->requant_cm_one(in.X[c][h][w] * cg[c], 1, P0) + cb[c];
    });
    if (party == ALICE) {
      o.Xr.assign(in.C, std::vector<std::vector<int64_t>>(in.H, std::vector<int64_t>(in.W)));
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.Xr[c][h][w] = rcm(in.Xr[c][h][w] * g[c], 1, P0) + b[c];
    }
    return o;
  }

  // ---- ReLU (MSB-free: b=1{x>=0}, out=b*x) : St -> St ----
  St relu(const St &in) {
    St o = in;
    o.X = in.X;
    timed(ph_relu, [&] {
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) {
            IntFp x = in.X[c][h][w];
            IntFp ge = cmp->is_le(zero(), x, act_bits); // 1{0 <= x}
            o.X[c][h][w] = ge * x;
          }
    });
    if (party == ALICE)
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.Xr[c][h][w] = in.Xr[c][h][w] > 0 ? in.Xr[c][h][w] : 0;
    return o;
  }

  // ---- residual add ----
  St add(const St &a, const St &b) {
    St o = a;
    for (int c = 0; c < a.C; ++c)
      for (int h = 0; h < a.H; ++h)
        for (int w = 0; w < a.W; ++w) o.X[c][h][w] = a.X[c][h][w] + b.X[c][h][w];
    if (party == ALICE)
      for (int c = 0; c < a.C; ++c)
        for (int h = 0; h < a.H; ++h)
          for (int w = 0; w < a.W; ++w) o.Xr[c][h][w] = a.Xr[c][h][w] + b.Xr[c][h][w];
    return o;
  }

  // ---- MaxPool : St -> St ----
  St maxpool(const St &in, int K, int s, int pd) {
    const int Ho = volefier::conv_out_dim(in.H, K, s, pd),
              Wo = volefier::conv_out_dim(in.W, K, s, pd);
    St o;
    o.C = in.C;
    o.H = Ho;
    o.W = Wo;
    o.s = in.s; // pool is scale-preserving
    o.X.assign(in.C, std::vector<std::vector<IntFp>>(Ho, std::vector<IntFp>(Wo)));
    if (party == ALICE)
      o.Xr.assign(in.C, std::vector<std::vector<int64_t>>(Ho, std::vector<int64_t>(Wo)));
    timed(ph_pool, [&] {
      for (int c = 0; c < in.C; ++c)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            std::vector<IntFp> win;
            int64_t mx = INT64_MIN;
            for (int ki = 0; ki < K; ++ki)
              for (int kj = 0; kj < K; ++kj) {
                int ih = oh * s - pd + ki, iw = ow * s - pd + kj;
                if (ih >= 0 && ih < in.H && iw >= 0 && iw < in.W) {
                  win.push_back(in.X[c][ih][iw]);
                  if (party == ALICE) mx = std::max(mx, in.Xr[c][ih][iw]);
                }
              }
            o.X[c][oh][ow] = cmp->max_value(win, act_bits);
            if (party == ALICE) o.Xr[c][oh][ow] = mx;
          }
    });
    return o;
  }

  // ---- global average pool : St -> VecSt (one value per channel) ----
  VecSt global_avg(const St &in) {
    VecSt o;
    o.X.resize(in.C);
    o.s = in.s;
    if (party == ALICE) o.Xr.resize(in.C);
    const int Pn = in.H * in.W;
    timed(ph_pool, [&] {
      for (int c = 0; c < in.C; ++c) {
        IntFp sum = zero();
        int64_t sr = 0;
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) {
            sum = sum + in.X[c][h][w];
            if (party == ALICE) sr += in.Xr[c][h][w];
          }
        o.X[c] = divr->round_div(sum, (uint64_t)Pn);
        if (party == ALICE) o.Xr[c] = rdiv(sr, Pn);
      }
    });
    return o;
  }

  // ---- fully connected : VecSt -> VecSt (matmul + requant + bias) ----
  VecSt fc(const VecSt &in, int out_dim, double s_out) {
    const int M = (int)in.X.size();
    std::pair<uint64_t, std::size_t> mm = QuantInfer::requant_mult(in.s, S_w, s_out);
    const uint64_t cmant = mm.first;
    const std::size_t shift = mm.second;
    std::vector<std::vector<int64_t>> W(out_dim, std::vector<int64_t>(M));
    std::vector<int64_t> bias(out_dim);
    for (auto &r : W)
      for (auto &v : r) v = rnd(-3, 3);
    for (auto &v : bias) v = rnd(-30, 30);
    std::vector<std::vector<IntFp>> cW(out_dim);
    std::vector<IntFp> cb;
    timed(ph_commit, [&] {
      for (int j = 0; j < out_dim; ++j) qi->commit_vec(cW[j], W[j]);
      qi->commit_vec(cb, bias);
    });
    VecSt o;
    o.s = s_out;
    timed(ph_fc, [&] { qi->dense_layer_q(o.X, cW, cb, in.X, in.s, S_w, s_out); });
    if (party == ALICE) {
      o.Xr.resize(out_dim);
      for (int j = 0; j < out_dim; ++j) {
        int64_t acc = 0;
        for (int i = 0; i < M; ++i) acc += W[j][i] * in.Xr[i];
        o.Xr[j] = rcm(acc, (int64_t)cmant, (int)shift) + bias[j];
      }
    }
    return o;
  }

  // ---- flatten C x H x W -> vector (for VGG / classifier head) ----
  VecSt flatten(const St &in) {
    VecSt o;
    o.s = in.s;
    for (int c = 0; c < in.C; ++c)
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) {
          o.X.push_back(in.X[c][h][w]);
          if (party == ALICE) o.Xr.push_back(in.Xr[c][h][w]);
        }
    return o;
  }

  // ---- ReLU on a vector (between FC layers) ----
  VecSt relu_vec(const VecSt &in) {
    VecSt o;
    o.X.resize(in.X.size());
    o.s = in.s;
    if (party == ALICE) o.Xr.resize(in.Xr.size());
    timed(ph_relu, [&] {
      for (std::size_t i = 0; i < in.X.size(); ++i) {
        IntFp ge = cmp->is_le(zero(), in.X[i], act_bits);
        o.X[i] = ge * in.X[i];
        if (party == ALICE) o.Xr[i] = in.Xr[i] > 0 ? in.Xr[i] : 0;
      }
    });
    return o;
  }

  // ---- residual blocks ----
  St basic_block(const St &in, int out_ch, int stride) {
    St m = relu(bn(conv(in, out_ch, 3, stride, 1, S_int))); // internal -> S_int
    m = bn(conv(m, out_ch, 3, 1, 1, S_res));                // -> residual scale
    St skip = (stride != 1 || in.C != out_ch) ? bn(conv(in, out_ch, 1, stride, 0, S_res)) : in;
    return relu(add(m, skip)); // both at S_res
  }
  St bottleneck(const St &in, int mid, int out_ch, int stride) {
    St m = relu(bn(conv(in, mid, 1, 1, 0, S_int)));
    m = relu(bn(conv(m, mid, 3, stride, 1, S_int)));
    m = bn(conv(m, out_ch, 1, 1, 0, S_res)); // -> residual scale
    St skip = (stride != 1 || in.C != out_ch) ? bn(conv(in, out_ch, 1, stride, 0, S_res)) : in;
    return relu(add(m, skip));
  }

  // ---- full forward : input St -> logits VecSt ----
  VecSt forward(const ResNetCfg &cfg, const St &input) {
    St x = relu(bn(conv(input, cfg.base, 7, 2, 3, S_res))); // stem conv -> residual scale
    x = maxpool(x, 3, 2, 1);                                // maxpool
    for (int i = 0; i < 4; ++i) {
      int mid = cfg.base << i;
      int out_ch = cfg.bottleneck ? mid * 4 : mid;
      for (int b = 0; b < cfg.blocks[i]; ++b) {
        int stride = (b == 0 && i > 0) ? 2 : 1;
        x = cfg.bottleneck ? bottleneck(x, mid, out_ch, stride) : basic_block(x, out_ch, stride);
      }
    }
    return fc(global_avg(x), cfg.classes, S_fc);
  }
};

#endif
