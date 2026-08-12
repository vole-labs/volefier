#ifndef ML_FLOAT_RESNET_H
#define ML_FLOAT_RESNET_H

#include "emp-zk/emp-zk.h"
#include "emp-tool/emp-tool.h"
#include "volefier/ml-float/float_nonlinear.h"
#include "volefier/linear/conv2d.h"
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// HIGH-PRECISION ResNet (Mystique hybrid): LINEAR layers (conv / fc) are proven
// in the fixed-point FIELD (scale 2^S) with the inner-product matmul; NON-LINEAR
// layers (BatchNorm with sqrt, ReLU, max/avg pool, residual add) are computed in
// IEEE-754 FLOAT Boolean circuits.  At every linear<->non-linear boundary the
// activation crosses representations via emp-zk's edaBit (arith2bool/bool2arith)
// + Mystique Fix<->Float (FloatToInt62 / Int62ToFloat):
//
//   float act --FloatToInt62(S)--> [fixed-point IntFp] --conv2d(field)-->
//   [accumulator IntFp, scale 2^2S] --Int62ToFloat(2S)--> float act --BN/ReLU/...
//
// Linear stays in the field because float matmul in Boolean circuits is
// prohibitive; only the activations are converted (the Mystique design).
// ---------------------------------------------------------------------------

using namespace emp::rec;  // Integer/Float/Bit moved to emp::rec in new emp
namespace volefier {

template <typename IO> class FloatResNet {
public:
  using FT = std::vector<std::vector<std::vector<Float>>>;  // C x H x W (committed)
  using DT = std::vector<std::vector<std::vector<double>>>; // plaintext reference
  struct St {
    FT X;
    DT Xr;
    int C, H, W;
  };
  struct VSt {
    std::vector<Float> X;
    std::vector<double> Xr;
  };

  IPProof<IO> *ipp;
  int party, S, mat_threads;
  PRG *prg;
  long n_conv_acts = 0; // count of per-activation Fix<->Float conversions (the cost driver)

  FloatResNet(IPProof<IO> *ipp_, int party_, int S_, PRG *prg_, int mt = 1)
      : ipp(ipp_), party(party_), S(S_), prg(prg_), mat_threads(mt) {}

  double rnd(double lo, double hi) {
    if (party != ALICE) return 0.0;
    uint64_t r;
    prg->random_data(&r, sizeof(r));
    return lo + (hi - lo) * ((double)(r >> 11) / (double)(1ULL << 53));
  }
  // He-style fan-in scaled weight init: keeps per-layer gain ~1 so activations
  // stay O(1) over many layers and the fixed-point accumulator never overflows.
  double winit(int fan_in) {
    return rnd(-1.0, 1.0) * std::sqrt(3.0 / (double)fan_in);
  }

  // ---- representation conversions (the linear<->non-linear bridge) ----
  IntFp f2fix(const Float &f) {
    ++n_conv_acts;
    Integer ib = FloatToInt62(f, S);
    return bool2arith<IO>(ib);
  } // float -> fixed IntFp (2^S)
  Float fix2f(IntFp &acc, int scale) {
    ++n_conv_acts;
    Integer ib = arith2bool<IO>(acc);
    return Int62ToFloat(ib, scale);
  }

  // BATCHED conversions: the per-element float circuits (FloatToInt62/Int62ToFloat)
  // stay local, but the edaBit arith<->bool step opens the WHOLE tensor in one
  // batched call (one amortized network round instead of one per activation).
  std::vector<IntFp> f2fix_batch(const std::vector<Float> &fs) { // float[] -> fixed IntFp[]
    const int n = (int)fs.size();
    std::vector<Integer> ib;
    ib.reserve(n);
    for (const auto &f : fs) ib.push_back(FloatToInt62(f, S));
    std::vector<IntFp> out(n);
    bool2arith<IO>(out.data(), ib.data(), n);
    n_conv_acts += n;
    return out;
  }
  std::vector<Float> fix2f_batch(std::vector<IntFp> &accs, int scale) { // IntFp[] -> float[]
    const int n = (int)accs.size();
    std::vector<Integer> ob(n);
    arith2bool<IO>(ob.data(), accs.data(), n);
    std::vector<Float> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(Int62ToFloat(ob[i], scale));
    n_conv_acts += n;
    return out;
  }
  IntFp commit_q(int64_t q) {
    uint64_t v = q >= 0 ? (uint64_t)q : PR - (uint64_t)(-q);
    return IntFp(v, ALICE);
  }
  Float commit_f(double v) {
    return Float((float)v, ALICE);
  }
  static Float maxf(const Float &a, const Float &b) {
    Float am = a;
    return am.If(am.less_than(b), b);
  }

  // ---- Conv2D: float feature map -> fixed-point field matmul -> float ----
  St conv(const St &in, int Co, int K, int stride, int pd) {
    const int Ho = conv_out_dim(in.H, K, stride, pd), Wo = conv_out_dim(in.W, K, stride, pd);
    const int M = in.C * K * K, Pn = Ho * Wo;
    // (1) float input -> fixed-point IntFp tensor (BATCHED edaBit)
    std::vector<Float> infs;
    infs.reserve(in.C * in.H * in.W);
    for (int c = 0; c < in.C; ++c)
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) infs.push_back(in.X[c][h][w]);
    std::vector<IntFp> flat = f2fix_batch(infs);
    Tensor3 X(in.C, std::vector<std::vector<IntFp>>(in.H, std::vector<IntFp>(in.W)));
    {
      int idx = 0;
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) X[c][h][w] = flat[idx++];
    }
    // (2) random weights committed at scale 2^S
    std::vector<std::vector<int64_t>> qw(Co, std::vector<int64_t>(M));
    std::vector<std::vector<IntFp>> cker(Co);
    for (int co = 0; co < Co; ++co)
      for (int m = 0; m < M; ++m) {
        int64_t q = (int64_t)llround(winit(M) * (double)(1LL << S));
        qw[co][m] = q;
        cker[co].push_back(commit_q(q));
      }
    // (3) field conv -> accumulators at scale 2^2S
    FMap raw;
    conv2d<IO>(*ipp, raw, X, cker, in.C, in.H, in.W, Co, K, stride, pd, mat_threads);
    // (4) accumulators -> float (BATCHED edaBit)
    std::vector<IntFp> accs;
    accs.reserve(Co * Pn);
    for (int co = 0; co < Co; ++co)
      for (int p = 0; p < Pn; ++p) accs.push_back(raw[co][p]);
    std::vector<Float> ofs = fix2f_batch(accs, 2 * S);
    St o;
    o.C = Co;
    o.H = Ho;
    o.W = Wo;
    o.X.assign(Co, std::vector<std::vector<Float>>(Ho, std::vector<Float>(Wo)));
    {
      int idx = 0;
      for (int co = 0; co < Co; ++co)
        for (int p = 0; p < Pn; ++p) o.X[co][p / Wo][p % Wo] = ofs[idx++];
    }
    if (party == ALICE) {
      o.Xr.assign(Co, std::vector<std::vector<double>>(Ho, std::vector<double>(Wo)));
      for (int co = 0; co < Co; ++co)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            double acc = 0;
            for (int ci = 0; ci < in.C; ++ci)
              for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj) {
                  int ih = oh * stride - pd + ki, iw = ow * stride - pd + kj;
                  if (ih >= 0 && ih < in.H && iw >= 0 && iw < in.W) {
                    int64_t qx = (int64_t)llround(in.Xr[ci][ih][iw] * (double)(1LL << S));
                    acc += (double)(qx * qw[co][ci * K * K + ki * K + kj]);
                  }
                }
            o.Xr[co][oh][ow] = acc / std::pow(2.0, 2 * S);
          }
    }
    return o;
  }

  // ---- BatchNorm (float, real normalization with sqrt) ----
  St bn(const St &in) {
    St o = in;
    o.X.assign(in.C, std::vector<std::vector<Float>>(in.H, std::vector<Float>(in.W)));
    Float eps(1e-5f, PUBLIC);
    for (int c = 0; c < in.C; ++c) {
      double gr = rnd(0.8, 1.2), br = rnd(-0.2, 0.2), mr = rnd(-0.3, 0.3), vr = rnd(0.7, 1.3);
      Float g = commit_f(gr), b = commit_f(br), mu = commit_f(mr), var = commit_f(vr);
      Float denom = (var + eps).sqrt();
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) o.X[c][h][w] = g * (in.X[c][h][w] - mu) / denom + b;
      if (party == ALICE) {
        double dn = std::sqrt(vr + 1e-5);
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.Xr[c][h][w] = gr * (in.Xr[c][h][w] - mr) / dn + br;
      }
    }
    return o;
  }

  St relu(const St &in) {
    St o = in;
    for (int c = 0; c < in.C; ++c)
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) o.X[c][h][w] = FloatNonlinear::relu(in.X[c][h][w]);
    if (party == ALICE)
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.Xr[c][h][w] = in.Xr[c][h][w] > 0 ? in.Xr[c][h][w] : 0.0;
    return o;
  }

  St add(const St &a, const St &b) { // residual add (float)
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

  St maxpool(const St &in, int K, int stride, int pd) {
    const int Ho = conv_out_dim(in.H, K, stride, pd), Wo = conv_out_dim(in.W, K, stride, pd);
    St o;
    o.C = in.C;
    o.H = Ho;
    o.W = Wo;
    o.X.assign(in.C, std::vector<std::vector<Float>>(Ho, std::vector<Float>(Wo)));
    if (party == ALICE)
      o.Xr.assign(in.C, std::vector<std::vector<double>>(Ho, std::vector<double>(Wo)));
    for (int c = 0; c < in.C; ++c)
      for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow) {
          Float m;
          double mr = -1e300;
          bool any = false;
          for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj) {
              int ih = oh * stride - pd + ki, iw = ow * stride - pd + kj;
              if (ih >= 0 && ih < in.H && iw >= 0 && iw < in.W) {
                m = any ? maxf(m, in.X[c][ih][iw]) : in.X[c][ih][iw];
                any = true;
                if (party == ALICE) mr = std::max(mr, in.Xr[c][ih][iw]);
              }
            }
          o.X[c][oh][ow] = m;
          if (party == ALICE) o.Xr[c][oh][ow] = mr;
        }
    return o;
  }

  VSt global_avg(const St &in) {
    VSt o;
    o.X.resize(in.C);
    if (party == ALICE) o.Xr.resize(in.C);
    Float ninv((float)(1.0 / (in.H * in.W)), PUBLIC);
    for (int c = 0; c < in.C; ++c) {
      Float s(0.0f, PUBLIC);
      double sr = 0;
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) {
          s = s + in.X[c][h][w];
          if (party == ALICE) sr += in.Xr[c][h][w];
        }
      o.X[c] = s * ninv;
      if (party == ALICE) o.Xr[c] = sr / (in.H * in.W);
    }
    return o;
  }

  VSt fc(const VSt &in, int out_dim) { // float -> fixed-point field matmul -> float
    const int M = (int)in.X.size();
    std::vector<IntFp> x = f2fix_batch(in.X);
    std::vector<std::vector<int64_t>> qw(out_dim, std::vector<int64_t>(M));
    std::vector<std::vector<IntFp>> cW(out_dim);
    for (int j = 0; j < out_dim; ++j)
      for (int i = 0; i < M; ++i) {
        int64_t q = (int64_t)llround(winit(M) * (double)(1LL << S));
        qw[j][i] = q;
        cW[j].push_back(commit_q(q));
      }
    std::vector<IntFp> acc;
    ipp->matmul(acc, cW, x, mat_threads);
    VSt o;
    o.X = fix2f_batch(acc, 2 * S);
    if (party == ALICE) {
      o.Xr.resize(out_dim);
      for (int j = 0; j < out_dim; ++j) {
        double a = 0;
        for (int i = 0; i < M; ++i) {
          int64_t qx = (int64_t)llround(in.Xr[i] * (double)(1LL << S));
          a += (double)(qx * qw[j][i]);
        }
        o.Xr[j] = a / std::pow(2.0, 2 * S);
      }
    }
    return o;
  }

  // ---- basic block (two 3x3 convs + identity residual) ----
  St basic_block(const St &x, int ch) {
    St m = bn(conv(relu(bn(conv(x, ch, 3, 1, 1))), ch, 3, 1, 1));
    return relu(add(m, x));
  }
  // ---- downsample block (stride-2 main path + 1x1 stride-2 skip) ----
  St downsample_block(const St &x, int ch) {
    St m = bn(conv(relu(bn(conv(x, ch, 3, 2, 1))), ch, 3, 1, 1));
    St skip = bn(conv(x, ch, 1, 2, 0));
    return relu(add(m, skip));
  }

  // ---- bottleneck block (1x1 reduce -> 3x3 -> 1x1 expand, ResNet-50 style) ----
  St bottleneck(const St &x, int mid, int out_ch, int stride) {
    St m = relu(bn(conv(x, mid, 1, 1, 0)));   // 1x1 reduce
    m = relu(bn(conv(m, mid, 3, stride, 1))); // 3x3
    m = bn(conv(m, out_ch, 1, 1, 0));         // 1x1 expand
    St skip = (stride != 1 || x.C != out_ch) ? bn(conv(x, out_ch, 1, stride, 0)) : x;
    return relu(add(m, skip));
  }

  // ---- configurable ResNet: stem -> [stages of basic blocks w/ downsample] -> gap -> fc ----
  //   blocks = #basic blocks per stage (e.g. {2,2,2,2} ~ ResNet-18 structure).
  VSt forward(const St &input, int base, int classes, const std::vector<int> &blocks) {
    St x = relu(bn(conv(input, base, 3, 1, 1))); // stem
    int ch = base;
    for (std::size_t s = 0; s < blocks.size(); ++s) {
      if (s > 0) {
        ch *= 2;
        x = downsample_block(x, ch);
      } // halve spatial, double channels
      for (int b = 0; b < blocks[s]; ++b) x = basic_block(x, ch);
    }
    return fc(global_avg(x), classes);
  }

  // ---- full ResNet-50: stem -> 4 stages of bottleneck blocks {3,4,6,3} -> gap -> fc ----
  //   base=64 => mid channels 64/128/256/512, out channels 256/512/1024/2048.
  VSt forward_resnet50(const St &input, int base, int classes) {
    St x = relu(bn(conv(input, base, 3, 1, 1))); // CIFAR stem (3x3)
    const int blocks[4] = {3, 4, 6, 3};
    for (int s = 0; s < 4; ++s) {
      int mid = base << s, out_ch = mid * 4; // 64..512 mid, 256..2048 out
      for (int b = 0; b < blocks[s]; ++b) {
        int stride = (b == 0 && s > 0) ? 2 : 1; // downsample at the start of stages 1..3
        x = bottleneck(x, mid, out_ch, stride);
      }
    }
    return fc(global_avg(x), classes);
  }
};

} // namespace volefier

#endif // ML_FLOAT_RESNET_H
