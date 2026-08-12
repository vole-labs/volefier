#ifndef ML_FLOAT_FIELD_RESNET_H
#define ML_FLOAT_FIELD_RESNET_H

#include "emp-zk/emp-zk.h"
#include "emp-tool/emp-tool.h"
#include "volefier/linear/conv2d.h"
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// FIELD-NATIVE high-precision ResNet -- Mystique's actual architecture.
// Activations live in the fixed-point FIELD (IntFp, scale 2^S). Linear layers
// (conv/fc) run in the field; ReLU is an INTEGER comparison via edaBit
// (arith2bool -> integer geq/select -> bool2arith, mirroring wolverine_relu);
// residual add is a FREE field add; truncation (2^2S -> 2^S) uses the Mystique
// float-rescale (Int62ToFloat(2S) -> FloatToInt62(S)); and ONLY BatchNorm
// (which needs sqrt) crosses into IEEE-754 float circuits.
//
// This is the corrected design vs the float-native float_resnet.h, which put
// EVERY activation in float and converted around every conv. Here float work is
// confined to (a) the per-conv truncation rescale and (b) BatchNorm's sqrt.
// ---------------------------------------------------------------------------

using namespace emp::rec;  // Integer/Float/Bit moved to emp::rec in new emp
namespace volefier {

template <typename IO> class FieldResNet {
public:
  using DT = std::vector<std::vector<std::vector<double>>>;
  struct St {
    Tensor3 X;
    DT Xr;
    int C, H, W;
    int sx = 0;
  }; // X = IntFp at scale 2^(S+sx): sx=0 truncated, sx=S double-scale (Mystique deferred)
  struct VSt {
    std::vector<IntFp> X;
    std::vector<double> Xr;
  };

  IPProof<IO> *ipp;
  int party, S, mat_threads;
  PRG *prg;
  long n_eda = 0; // # of edaBit arith<->bool element conversions (cost driver)
  long n_flt = 0; // # of elements pushed through a FLOAT circuit (rescale or sqrt-BN)
  // microbenchmark accumulators (microseconds): field matmul, edaBit A2B/B2A,
  // float circuits (rescale/sqrt/relu-compare), field multiplies, commits.
  double t_matmul = 0, t_eda = 0, t_float = 0, t_fieldmul = 0, t_commit = 0;

  FieldResNet(IPProof<IO> *ipp_, int party_, int S_, PRG *prg_, int mt = 1)
      : ipp(ipp_), party(party_), S(S_), prg(prg_), mat_threads(mt) {}

  double rnd(double lo, double hi) {
    if (party != ALICE) return 0.0;
    uint64_t r;
    prg->random_data(&r, sizeof(r));
    return lo + (hi - lo) * ((double)(r >> 11) / (double)(1ULL << 53));
  }
  double winit(int fan) {
    return rnd(-1.0, 1.0) * std::sqrt(3.0 / (double)fan);
  }
  IntFp commit_q(int64_t q) {
    uint64_t v = q >= 0 ? (uint64_t)q : PR - (uint64_t)(-q);
    return IntFp(v, ALICE);
  }
  static int64_t dec(uint64_t v) {
    return v > PR / 2 ? (int64_t)v - (int64_t)PR : (int64_t)v;
  }

  // truncate flat accumulators (scale 2^2S) -> field (2^S) via batched float-rescale
  // Truncate v (scale 2^2S) -> 2^S via Mystique's FIELD range-check
  // (_inner_proof_and_check): prover commits the truncated value t, then proves
  // |v - t*2^S| < 2^S with one arith2bool + two integer geq. NO float rescale.
  void truncate(std::vector<IntFp> &v) {
    const int n = (int)v.size();
    std::vector<uint64_t> tval(n, 0);
    if (party == ALICE)
      for (int i = 0; i < n; ++i) {
        int64_t acc = dec((uint64_t)(v[i].value >> 64));
        int64_t tq = (int64_t)llround((double)acc / (double)(1LL << S));
        tval[i] = tq >= 0 ? (uint64_t)tq : PR - (uint64_t)(-tq);
      }
    std::vector<IntFp> t(n);
    batch_feed(t.data(), tval.data(), n); // prover commits truncated values
    std::vector<IntFp> diff(n);
    const uint64_t neg2S = PR - (1ULL << S);
    for (int i = 0; i < n; ++i) diff[i] = v[i] + t[i] * neg2S; // diff = v - t*2^S
    std::vector<Integer> bd(n);
    auto _t = clock_start();
    arith2bool<IO>(bd.data(), diff.data(), n);
    t_eda += time_from(_t);
    n_eda += n;
    Integer POS(62, (uint64_t)(1ULL << S), PUBLIC), NEGB(62, PR - (uint64_t)(1ULL << S), PUBLIC);
    Bit allbad(false, PUBLIC);
    _t = clock_start();
    for (int i = 0; i < n; ++i) {
      Bit p = bd[i].geq(POS);
      Bit ng = bd[i].geq(NEGB);
      allbad = allbad | (p & (!ng));
    }
    t_float += time_from(_t);
    n_flt += n; // range-check circuit (no float)
    if (allbad.reveal<bool>(PUBLIC) && party == ALICE) error("truncation range-check failed\n");
    v = t;
  }

  // ---- Conv2D: field input -> field matmul -> truncate -> field (2^S) ----
  St conv(const St &in, int Co, int K, int stride, int pd) {
    const int Ho = conv_out_dim(in.H, K, stride, pd), Wo = conv_out_dim(in.W, K, stride, pd);
    const int M = in.C * K * K, Pn = Ho * Wo;
    std::vector<std::vector<int64_t>> qw(Co, std::vector<int64_t>(M));
    std::vector<std::vector<IntFp>> cker(Co);
    auto _t = clock_start();
    for (int co = 0; co < Co; ++co)
      for (int m = 0; m < M; ++m) {
        int64_t q = (int64_t)llround(winit(M) * (double)(1LL << S));
        qw[co][m] = q;
        cker[co].push_back(commit_q(q));
      }
    t_commit += time_from(_t);
    FMap raw;
    _t = clock_start();
    conv2d<IO>(*ipp, raw, in.X, cker, in.C, in.H, in.W, Co, K, stride, pd, mat_threads);
    t_matmul += time_from(_t);
    std::vector<IntFp> flat;
    flat.reserve(Co * Pn);
    for (int co = 0; co < Co; ++co)
      for (int p = 0; p < Pn; ++p) flat.push_back(raw[co][p]);
    truncate(flat); // 2^2S -> 2^S (field input for BN)
    St o;
    o.C = Co;
    o.H = Ho;
    o.W = Wo;
    o.X.assign(Co, std::vector<std::vector<IntFp>>(Ho, std::vector<IntFp>(Wo)));
    {
      int idx = 0;
      for (int co = 0; co < Co; ++co)
        for (int p = 0; p < Pn; ++p) o.X[co][p / Wo][p % Wo] = flat[idx++];
    }
    if (party == ALICE) {
      o.Xr.assign(Co, std::vector<std::vector<double>>(Ho, std::vector<double>(Wo)));
      const double sc = std::pow(2.0, S);
      for (int co = 0; co < Co; ++co)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            double acc = 0;
            for (int ci = 0; ci < in.C; ++ci)
              for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj) {
                  int ih = oh * stride - pd + ki, iw = ow * stride - pd + kj;
                  if (ih >= 0 && ih < in.H && iw >= 0 && iw < in.W) {
                    int64_t qx = (int64_t)llround(in.Xr[ci][ih][iw] * sc);
                    acc += (double)(qx * qw[co][ci * K * K + ki * K + kj]);
                  }
                }
            o.Xr[co][oh][ow] = std::round(acc / sc) / sc; // matches the 2^2S->2^S truncate
          }
    }
    return o;
  }

  // ---- BatchNorm EXACTLY as Mystique's SecureFusedBatchNormOp ----
  //   per-channel:  new_var = var+eps; norm = Rsqrt(new_var) [FLOAT]; scaling = norm*gamma [field
  //   mul] per-element:  x_centered = x-mean [field]; x_scaled = x_centered*scaling [field
  //   mul+truncate]; + beta [field]
  // The ONLY float work is the per-channel Rsqrt; the affine is field arithmetic.
  St bn(const St &in) { // in @ 2^S -> out @ 2^S
    const int C = in.C, HW = in.H * in.W;
    const double sc = std::pow(2.0, S);
    std::vector<double> gr(C), mr(C), br(C), vr(C);
    for (int c = 0; c < C; ++c) {
      gr[c] = rnd(0.8, 1.2);
      br[c] = rnd(-0.2, 0.2);
      mr[c] = rnd(-0.3, 0.3);
      vr[c] = rnd(0.7, 1.3);
    }
    // per-channel Rsqrt(var+eps) in FLOAT
    std::vector<IntFp> nv(C);
    for (int c = 0; c < C; ++c) nv[c] = commit_q((int64_t)llround((vr[c] + 1e-5) * sc));
    std::vector<Integer> nb(C);
    auto _t = clock_start();
    arith2bool<IO>(nb.data(), nv.data(), C);
    t_eda += time_from(_t);
    n_eda += C;
    Float one(1.0f, PUBLIC);
    _t = clock_start();
    for (int c = 0; c < C; ++c) {
      Float f = Int62ToFloat(nb[c], S);
      f = one / f.sqrt();
      nb[c] = FloatToInt62(f, S);
    }
    t_float += time_from(_t);
    n_flt += C;
    std::vector<IntFp> norm(C);
    _t = clock_start();
    bool2arith<IO>(norm.data(), nb.data(), C);
    t_eda += time_from(_t);
    n_eda += C;
    // scaling = norm * gamma  (field mul -> 2^2S -> truncate -> 2^S)
    std::vector<IntFp> scaling(C);
    _t = clock_start();
    for (int c = 0; c < C; ++c) scaling[c] = norm[c] * commit_q((int64_t)llround(gr[c] * sc));
    t_fieldmul += time_from(_t);
    truncate(scaling);
    std::vector<int64_t> negmeanq(C), betaq(C);
    for (int c = 0; c < C; ++c) {
      negmeanq[c] = (int64_t)llround(-mr[c] * sc);
      betaq[c] = (int64_t)llround(br[c] * sc * sc);
    } // beta @ 2^2S
    // per-element: (x - mean) * scaling -> 2^2S, then + beta @ 2^2S.  NO truncate
    // here -- Mystique's Mul leaves the output un-truncated (double-scale) and the
    // following ReLU absorbs the truncation. (IntFp has no operator-, add neg-mean.)
    std::vector<IntFp> prod;
    prod.reserve(C * HW);
    _t = clock_start();
    for (int c = 0; c < C; ++c) {
      IntFp negm = commit_q(negmeanq[c]);
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) prod.push_back((in.X[c][h][w] + negm) * scaling[c]);
    }
    t_fieldmul += time_from(_t);
    St o;
    o.C = C;
    o.H = in.H;
    o.W = in.W;
    o.sx = S; // double-scale, deferred truncate
    o.X.assign(C, std::vector<std::vector<IntFp>>(in.H, std::vector<IntFp>(in.W)));
    {
      int idx = 0;
      for (int c = 0; c < C; ++c) {
        IntFp bt = commit_q(betaq[c]);
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.X[c][h][w] = prod[idx++] + bt;
      }
    }
    if (party == ALICE) {
      o.Xr.assign(C, std::vector<std::vector<double>>(in.H, std::vector<double>(in.W)));
      for (int c = 0; c < C; ++c) {
        double dn = std::sqrt(vr[c] + 1e-5);
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w)
            o.Xr[c][h][w] = gr[c] * (in.Xr[c][h][w] - mr[c]) / dn + br[c];
      }
    }
    return o;
  }

  // ---- ReLU: INTEGER comparison in the bool domain, absorbs the deferred
  //      truncation via a free bit-shift (mirrors Mystique wolverine_relu scale_count==2) ----
  St relu(const St &in) {
    const int n = in.C * in.H * in.W;
    std::vector<IntFp> flat;
    flat.reserve(n);
    for (int c = 0; c < in.C; ++c)
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) flat.push_back(in.X[c][h][w]);
    const int shift = in.sx; // absorb the deferred truncation: 2^(S+sx) -> 2^S
    std::vector<Integer> b(n);
    auto _t = clock_start();
    arith2bool<IO>(b.data(), flat.data(), n);
    t_eda += time_from(_t);
    n_eda += n;
    Integer zero(62, 0, PUBLIC), smallest_neg(62, (uint64_t)((PR - 1) / 2) + 1, PUBLIC);
    _t = clock_start();
    for (int i = 0; i < n; ++i) { // neg? 0 : floor(x / 2^shift)
      Bit neg = b[i].geq(smallest_neg);
      Integer t = shift > 0 ? (b[i] >> shift) : b[i];
      b[i] = t.select(neg, zero);
    }
    t_float += time_from(_t);
    n_flt += n;
    _t = clock_start();
    bool2arith<IO>(flat.data(), b.data(), n);
    t_eda += time_from(_t);
    n_eda += n;
    St o;
    o.C = in.C;
    o.H = in.H;
    o.W = in.W;
    o.X.assign(in.C, std::vector<std::vector<IntFp>>(in.H, std::vector<IntFp>(in.W)));
    {
      int k = 0;
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.X[c][h][w] = flat[k++];
    }
    if (party == ALICE) {
      o.Xr.assign(in.C, std::vector<std::vector<double>>(in.H, std::vector<double>(in.W)));
      for (int c = 0; c < in.C; ++c)
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) o.Xr[c][h][w] = in.Xr[c][h][w] > 0 ? in.Xr[c][h][w] : 0.0;
    }
    return o;
  }

  // ---- residual add: FREE field add (no conversion, no circuit) ----
  St add(const St &a, const St &b) {
    const int target = std::max(a.sx, b.sx); // reconcile scales (free public scale-up)
    const uint64_t ua = 1ULL << (target - a.sx), ub = 1ULL << (target - b.sx);
    St o;
    o.C = a.C;
    o.H = a.H;
    o.W = a.W;
    o.sx = target;
    o.X.assign(a.C, std::vector<std::vector<IntFp>>(a.H, std::vector<IntFp>(a.W)));
    for (int c = 0; c < a.C; ++c)
      for (int h = 0; h < a.H; ++h)
        for (int w = 0; w < a.W; ++w) o.X[c][h][w] = a.X[c][h][w] * ua + b.X[c][h][w] * ub;
    if (party == ALICE) {
      o.Xr.assign(a.C, std::vector<std::vector<double>>(a.H, std::vector<double>(a.W)));
      for (int c = 0; c < a.C; ++c)
        for (int h = 0; h < a.H; ++h)
          for (int w = 0; w < a.W; ++w) o.Xr[c][h][w] = a.Xr[c][h][w] + b.Xr[c][h][w];
    }
    return o;
  }

  // ---- global average pool, EXACTLY as Mystique's Mean: field sum + field
  //      const-multiply by (1/N) + one truncate (no per-element float divide) ----
  VSt global_avg(const St &in) {
    const int HW = in.H * in.W;
    const double sc = std::pow(2.0, S);
    std::vector<IntFp> sums(in.C);
    for (int c = 0; c < in.C; ++c) {
      IntFp s((uint64_t)0, PUBLIC);
      for (int h = 0; h < in.H; ++h)
        for (int w = 0; w < in.W; ++w) s = s + in.X[c][h][w];
      sums[c] = s;
    }
    IntFp inv_n((uint64_t)llround((1.0 / HW) * sc), PUBLIC);  // 1/N at scale 2^S
    for (int c = 0; c < in.C; ++c) sums[c] = sums[c] * inv_n; // field const-mul -> 2^2S
    truncate(sums);                                           // -> 2^S
    VSt o;
    o.X = sums;
    if (party == ALICE) {
      o.Xr.resize(in.C);
      for (int c = 0; c < in.C; ++c) {
        double s = 0;
        for (int h = 0; h < in.H; ++h)
          for (int w = 0; w < in.W; ++w) s += in.Xr[c][h][w];
        o.Xr[c] = s / HW;
      }
    }
    return o;
  }

  // ---- FC: field matmul -> truncate -> field logits (2^S) ----
  VSt fc(const VSt &in, int out_dim) {
    const int M = (int)in.X.size();
    std::vector<std::vector<int64_t>> qw(out_dim, std::vector<int64_t>(M));
    std::vector<std::vector<IntFp>> cW(out_dim);
    auto _t = clock_start();
    for (int j = 0; j < out_dim; ++j)
      for (int i = 0; i < M; ++i) {
        int64_t q = (int64_t)llround(winit(M) * (double)(1LL << S));
        qw[j][i] = q;
        cW[j].push_back(commit_q(q));
      }
    t_commit += time_from(_t);
    VSt o;
    _t = clock_start();
    ipp->matmul(o.X, cW, in.X, mat_threads);
    t_matmul += time_from(_t);
    truncate(o.X);
    if (party == ALICE) {
      o.Xr.resize(out_dim);
      const double sc = std::pow(2.0, S);
      for (int j = 0; j < out_dim; ++j) {
        double a = 0;
        for (int i = 0; i < M; ++i) {
          int64_t qx = (int64_t)llround(in.Xr[i] * sc);
          a += (double)(qx * qw[j][i]);
        }
        o.Xr[j] = std::round(a / sc) / sc;
      }
    }
    return o;
  }

  St basic_block(const St &x, int ch) {
    St m = bn(conv(relu(bn(conv(x, ch, 3, 1, 1))), ch, 3, 1, 1));
    return relu(add(m, x));
  }
  St bottleneck(const St &x, int mid, int out_ch, int stride) {
    St m = relu(bn(conv(x, mid, 1, 1, 0)));
    m = relu(bn(conv(m, mid, 3, stride, 1)));
    m = bn(conv(m, out_ch, 1, 1, 0));
    St skip = (stride != 1 || x.C != out_ch) ? bn(conv(x, out_ch, 1, stride, 0)) : x;
    return relu(add(m, skip));
  }

  VSt forward(const St &input, int base, int classes, const std::vector<int> &blocks) {
    St x = relu(bn(conv(input, base, 3, 1, 1)));
    int ch = base;
    for (std::size_t s = 0; s < blocks.size(); ++s) {
      if (s > 0) {
        ch *= 2;
        St m = bn(conv(relu(bn(conv(x, ch, 3, 2, 1))), ch, 3, 1, 1));
        St sk = bn(conv(x, ch, 1, 2, 0));
        x = relu(add(m, sk));
      }
      for (int b = 0; b < blocks[s]; ++b) x = basic_block(x, ch);
    }
    return fc(global_avg(x), classes);
  }
  VSt forward_resnet50(const St &input, int base, int classes) {
    St x = relu(bn(conv(input, base, 3, 1, 1)));
    const int blk[4] = {3, 4, 6, 3};
    for (int s = 0; s < 4; ++s) {
      int mid = base << s, out_ch = mid * 4;
      for (int b = 0; b < blk[s]; ++b) x = bottleneck(x, mid, out_ch, (b == 0 && s > 0) ? 2 : 1);
    }
    return fc(global_avg(x), classes);
  }
};

} // namespace volefier

#endif // ML_FLOAT_FIELD_RESNET_H
