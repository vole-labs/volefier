#ifndef ML_IO_ONNX_CALIB_H
#define ML_IO_ONNX_CALIB_H

// Static calibration pass.  After fold+fuse with the model inputs fixed to
// concrete public constants, the WHOLE forward is a function of constants +
// weights, so it can be evaluated on the host in float.  We run that forward,
// observe each tensor's max-abs magnitude, and emit a per-tensor scale that
// maps its range into a target integer domain -- exactly the post-training
// quantization calibration DeepProve/EZKL (and TensorRT/ONNXRuntime) do.

#include "volefier/io/onnx_fold.h"
#include <cmath>

namespace volefier {

// host eval of the heavy compute ops (structural/elementwise fall back to fold_eval)
inline bool calib_eval(const OnnxNode &n, const std::map<std::string, HT> &V, HT &res) {
  auto in = [&](int i) -> const HT & { return V.at(n.input[i]); };
  auto aI = [&](const char *k, int d) {
    auto it = n.attr.find(k);
    return it != n.attr.end() ? (int)it->second.i : d;
  };
  auto aF = [&](const char *k, double d) {
    auto it = n.attr.find(k);
    return it != n.attr.end() ? (double)it->second.f : d;
  };
  const std::string &op = n.op_type;

  // Gemm, or MatMul with a 2-D right operand: [...,K] @ [K,N] (W maybe transposed)
  if (op == "Gemm" || (op == "MatMul" && in(1).shape.size() == 2)) {
    const HT &A = in(0);
    const HT &W = in(1);
    int tb = (op == "Gemm") ? aI("transB", 0) : 0;
    int K = (int)A.shape.back();
    int R = (int)(A.numel() / std::max(1, K));
    int N = tb ? (int)W.shape[0] : (int)W.shape[1];
    res.dt = 1;
    res.shape = A.shape;
    res.shape.back() = N;
    res.v.assign((size_t)R * N, 0.0);
    for (int r = 0; r < R; ++r)
      for (int j = 0; j < N; ++j) {
        double s = 0;
        for (int k = 0; k < K; ++k)
          s += A.v[(size_t)r * K + k] * (tb ? W.v[(size_t)j * K + k] : W.v[(size_t)k * N + j]);
        res.v[(size_t)r * N + j] = s;
      }
    if (op == "Gemm" && n.input.size() > 2) {
      const HT &b = in(2);
      for (int r = 0; r < R; ++r)
        for (int j = 0; j < N; ++j) res.v[(size_t)r * N + j] += b.v[j];
    }
    return true;
  }
  // batched MatMul: [G,M,K] @ [(G,)K,N]  (attention)
  if (op == "MatMul") {
    const HT &A = in(0);
    const HT &B = in(1);
    int K = (int)A.shape.back();
    int N = (int)B.shape.back();
    int M = A.shape.size() >= 2 ? (int)A.shape[A.shape.size() - 2] : 1;
    int G = (int)(A.numel() / ((size_t)M * K));
    bool bb = B.shape.size() >= 3;
    res.dt = 1;
    res.shape = A.shape;
    res.shape.back() = N;
    res.v.assign((size_t)G * M * N, 0.0);
    for (int g = 0; g < G; ++g)
      for (int m = 0; m < M; ++m)
        for (int j = 0; j < N; ++j) {
          double s = 0;
          for (int k = 0; k < K; ++k)
            s += A.v[((size_t)g * M + m) * K + k] *
                 B.v[(bb ? (size_t)g * K * N : 0) + (size_t)k * N + j];
          res.v[((size_t)g * M + m) * N + j] = s;
        }
    return true;
  }
  if (op == "LayerNormalization") {
    const HT &x = in(0), &gm = in(1);
    const HT *bt = n.input.size() > 2 ? &in(2) : nullptr;
    int D = (int)x.shape.back();
    int R = (int)(x.numel() / std::max(1, D));
    double eps = aF("epsilon", 1e-5);
    res = x;
    for (int r = 0; r < R; ++r) {
      double mu = 0;
      for (int k = 0; k < D; ++k) mu += x.v[(size_t)r * D + k];
      mu /= D;
      double var = 0;
      for (int k = 0; k < D; ++k) {
        double d = x.v[(size_t)r * D + k] - mu;
        var += d * d;
      }
      var /= D;
      double inv = 1.0 / std::sqrt(var + eps);
      for (int k = 0; k < D; ++k)
        res.v[(size_t)r * D + k] =
            (x.v[(size_t)r * D + k] - mu) * inv * gm.v[k] + (bt ? bt->v[k] : 0.0);
    }
    return true;
  }
  if (op == "Relu") {
    res = in(0);
    for (auto &x : res.v) x = std::max(0.0, x);
    return true;
  }
  if (op == "Conv") {
    const HT &X = in(0), &W = in(1);
    if (X.shape.size() < 4 || W.shape.size() < 4) return false;
    const HT *B = n.input.size() > 2 ? &in(2) : nullptr;
    int C = (int)X.shape[1], H = (int)X.shape[2], Wd = (int)X.shape[3];
    int Cout = (int)W.shape[0], Cin = (int)W.shape[1], kh = (int)W.shape[2], kw = (int)W.shape[3];
    int sh = 1, sw = 1, ph = 0, pw = 0;
    auto it = n.attr.find("strides");
    if (it != n.attr.end() && it->second.ints.size() >= 2) {
      sh = (int)it->second.ints[0];
      sw = (int)it->second.ints[1];
    }
    it = n.attr.find("pads");
    if (it != n.attr.end() && it->second.ints.size() >= 4) {
      ph = (int)it->second.ints[0];
      pw = (int)it->second.ints[1];
    }
    int g_ = 1;
    it = n.attr.find("group");
    if (it != n.attr.end()) g_ = (int)it->second.i;
    int Ho = (H + 2 * ph - kh) / sh + 1, Wo = (Wd + 2 * pw - kw) / sw + 1;
    int cpg = C / g_, opg = Cout / g_;
    res.dt = 1;
    res.shape = {1, Cout, Ho, Wo};
    res.v.assign((size_t)Cout * Ho * Wo, 0.0);
    for (int co = 0; co < Cout; ++co) {
      int grp = co / opg;
      for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow) {
          double s = B ? B->v[co] : 0;
          for (int ci = 0; ci < Cin; ++ci)
            for (int r = 0; r < kh; ++r)
              for (int c = 0; c < kw; ++c) {
                int ih = oh * sh - ph + r, iw = ow * sw - pw + c;
                if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;
                s += X.v[(((size_t)(grp * cpg + ci)) * H + ih) * Wd + iw] *
                     W.v[((((size_t)co * Cin + ci)) * kh + r) * kw + c];
              }
          res.v[((size_t)co * Ho + oh) * Wo + ow] = s;
        }
    }
    return true;
  }
  if (op == "MaxPool" || op == "AveragePool") {
    const HT &X = in(0);
    int C = (int)X.shape[1], H = (int)X.shape[2], Wd = (int)X.shape[3];
    int kh = 2, kw = 2, sh = 2, sw = 2, ph = 0, pw = 0;
    auto it = n.attr.find("kernel_shape");
    if (it != n.attr.end() && it->second.ints.size() >= 2) {
      kh = (int)it->second.ints[0];
      kw = (int)it->second.ints[1];
    }
    it = n.attr.find("strides");
    if (it != n.attr.end() && it->second.ints.size() >= 2) {
      sh = (int)it->second.ints[0];
      sw = (int)it->second.ints[1];
    }
    it = n.attr.find("pads");
    if (it != n.attr.end() && it->second.ints.size() >= 2) {
      ph = (int)it->second.ints[0];
      pw = (int)it->second.ints[1];
    }
    bool avg = op == "AveragePool";
    int Ho = (H + 2 * ph - kh) / sh + 1, Wo = (Wd + 2 * pw - kw) / sw + 1;
    res.dt = 1;
    res.shape = {1, C, Ho, Wo};
    res.v.assign((size_t)C * Ho * Wo, 0.0);
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow) {
          double m = avg ? 0 : -1e300;
          int cnt = 0;
          for (int r = 0; r < kh; ++r)
            for (int cc = 0; cc < kw; ++cc) {
              int ih = oh * sh - ph + r, iw = ow * sw - pw + cc;
              if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;
              double v = X.v[((size_t)c * H + ih) * Wd + iw];
              if (avg)
                m += v;
              else
                m = std::max(m, v);
              ++cnt;
            }
          res.v[((size_t)c * Ho + oh) * Wo + ow] = avg ? m / std::max(1, cnt) : m;
        }
    return true;
  }
  if (op == "GlobalAveragePool") {
    const HT &X = in(0);
    int C = (int)X.shape[1];
    int HW = (int)(X.numel() / std::max(1, C));
    res.dt = 1;
    res.shape = {1, C, 1, 1};
    res.v.assign(C, 0.0);
    for (int c = 0; c < C; ++c) {
      double s = 0;
      for (int p = 0; p < HW; ++p) s += X.v[(size_t)c * HW + p];
      res.v[c] = s / std::max(1, HW);
    }
    return true;
  }
  if (op == "Gelu") {
    res = in(0);
    for (auto &x : res.v) x = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    return true;
  }
  if (op == "Erf") {
    res = in(0);
    for (auto &x : res.v) x = std::erf(x);
    return true;
  }
  if (op == "Tanh") {
    res = in(0);
    for (auto &x : res.v) x = std::tanh(x);
    return true;
  }
  if (op == "Sigmoid") {
    res = in(0);
    for (auto &x : res.v) x = 1.0 / (1.0 + std::exp(-x));
    return true;
  }
  if (op == "QuantizeLinear") {
    const HT &x = in(0);
    if (in(1).v.empty()) {
      res = x;
      return true;
    }
    double sc = in(1).v[0];
    double zp = (n.input.size() > 2 && !in(2).v.empty()) ? in(2).v[0] : 0;
    res = x;
    double lo = (n.input.size() > 2 && zp == 0) ? -128 : 0,
           hi = (n.input.size() > 2 && zp == 0) ? 127 : 255;
    for (auto &v : res.v) v = std::max(lo, std::min(hi, std::round(v / sc) + zp));
    return true;
  }
  if (op == "DequantizeLinear") {
    const HT &x = in(0);
    if (in(1).v.empty()) {
      res = x;
      return true;
    }
    size_t nc = in(1).v.size();
    double zp = (n.input.size() > 2 && !in(2).v.empty()) ? in(2).v[0] : 0;
    res = x;
    if (nc == 1) {
      double sc = in(1).v[0];
      for (auto &v : res.v) v = (v - zp) * sc;
    } // per-tensor
    else {
      int64_t inner = (int64_t)(res.v.size() / nc);
      for (size_t k = 0; k < res.v.size(); ++k)
        res.v[k] = (res.v[k] - zp) * in(1).v[(inner > 0) ? (k / inner) % nc : 0];
    } // per-channel (axis 0)
    return true;
  }
  if (op == "Softmax") {
    const HT &x = in(0);
    int D = (int)x.shape.back();
    int R = (int)(x.numel() / std::max(1, D));
    res = x;
    for (int r = 0; r < R; ++r) {
      double mx = -1e300;
      for (int k = 0; k < D; ++k) mx = std::max(mx, x.v[(size_t)r * D + k]);
      double sm = 0;
      for (int k = 0; k < D; ++k) {
        res.v[(size_t)r * D + k] = std::exp(x.v[(size_t)r * D + k] - mx);
        sm += res.v[(size_t)r * D + k];
      }
      for (int k = 0; k < D; ++k) res.v[(size_t)r * D + k] /= sm;
    }
    return true;
  }
  if (op == "Slice" && n.input.size() >= 3) { // general N-D slice (fold_eval is 1-D only)
    const HT &x = in(0);
    int nd = (int)x.shape.size();
    auto getv = [&](int i) {
      std::vector<int64_t> r;
      if (i < (int)n.input.size() && V.count(n.input[i]))
        for (double d : V.at(n.input[i]).v) r.push_back((int64_t)d);
      return r;
    };
    auto starts = getv(1), ends = getv(2), axes = getv(3), steps = getv(4);
    if (axes.empty())
      for (int i = 0; i < (int)starts.size(); ++i) axes.push_back(i);
    std::vector<int64_t> lo(nd, 0), hi(x.shape), stp(nd, 1);
    for (size_t a = 0; a < axes.size(); ++a) {
      int ax = (int)(axes[a] < 0 ? axes[a] + nd : axes[a]);
      int64_t d = x.shape[ax];
      int64_t s = starts[a] < 0 ? starts[a] + d : starts[a],
              e = ends[a] < 0 ? ends[a] + d : ends[a];
      lo[ax] = std::max((int64_t)0, std::min(s, d));
      hi[ax] = std::max((int64_t)0, std::min(e, d));
      if (a < steps.size()) stp[ax] = steps[a];
    }
    std::vector<int64_t> ist(nd, 1);
    for (int i = nd - 2; i >= 0; --i) ist[i] = ist[i + 1] * x.shape[i + 1];
    std::vector<int64_t> ns(nd);
    for (int i = 0; i < nd; ++i) ns[i] = stp[i] > 0 ? (hi[i] - lo[i] + stp[i] - 1) / stp[i] : 0;
    std::vector<int64_t> ost(nd, 1);
    for (int i = nd - 2; i >= 0; --i) ost[i] = ost[i + 1] * ns[i + 1];
    int64_t tot = 1;
    for (auto d : ns) tot *= d;
    res.dt = x.dt;
    res.shape = ns;
    res.v.resize(tot);
    std::vector<int> id(nd);
    for (int64_t o = 0; o < tot; ++o) {
      int64_t rem = o, src = 0;
      for (int i = 0; i < nd; ++i) {
        id[i] = (int)(rem / ost[i]);
        rem %= ost[i];
        src += (lo[i] + (int64_t)id[i] * stp[i]) * ist[i];
      }
      res.v[o] = x.v[src];
    }
    return true;
  }
  return fold_eval(n, V, res);
}

// Run the full host forward; return per-tensor max-abs magnitude.
inline std::map<std::string, double> calibrate_ranges(const OnnxGraph &g) {
  std::map<std::string, HT> V;
  for (const auto &kv : g.initializer) V[kv.first] = ht_from_init(kv.second);
  std::map<std::string, double> rng;
  auto rec = [&](const std::string &nm, const HT &h) {
    double m = 0;
    for (double x : h.v) {
      double a = std::fabs(x);
      if (a > m) m = a;
    }
    rng[nm] = m;
  };
  for (const auto &kv : V) rec(kv.first, kv.second);
  for (const auto &n : g.nodes) {
    if (n.op_type == "Split") { // multi-output: split along axis
      auto it = V.find(n.input[0]);
      if (it == V.end()) continue;
      const HT &x = it->second;
      int nd = (int)x.shape.size();
      int axis = n.attr.count("axis") ? (int)n.attr.at("axis").i : 0;
      if (axis < 0) axis += nd;
      int nout = (int)n.output.size();
      std::vector<int64_t> sizes;
      if (n.input.size() > 1 && V.count(n.input[1])) {
        for (double d : V.at(n.input[1]).v) sizes.push_back((int64_t)d);
      } else if (n.attr.count("split")) {
        for (auto s : n.attr.at("split").ints) sizes.push_back(s);
      } else {
        int64_t each = x.shape[axis] / std::max(1, nout);
        for (int o = 0; o < nout; ++o) sizes.push_back(each);
      }
      int64_t outer = 1;
      for (int i = 0; i < axis; ++i) outer *= x.shape[i];
      int64_t inner = 1;
      for (int i = axis + 1; i < nd; ++i) inner *= x.shape[i];
      int64_t off = 0;
      for (int o = 0; o < nout; ++o) {
        int64_t as = sizes[o];
        HT p;
        p.dt = x.dt;
        p.shape = x.shape;
        p.shape[axis] = as;
        p.v.resize((size_t)outer * as * inner);
        for (int64_t oo = 0; oo < outer; ++oo)
          for (int64_t a = 0; a < as; ++a)
            for (int64_t k = 0; k < inner; ++k)
              p.v[((size_t)oo * as + a) * inner + k] =
                  x.v[((size_t)oo * x.shape[axis] + off + a) * inner + k];
        off += as;
        V[n.output[o]] = p;
        rec(n.output[o], p);
      }
      continue;
    }
    HT res;
    bool ok = false;
    try {
      ok = calib_eval(n, V, res);
    } catch (...) {
      ok = false;
    }
    if (ok && !n.output.empty()) {
      V[n.output[0]] = res;
      rec(n.output[0], res);
    }
  }
  return rng;
}

// Convert observed ranges to per-tensor scales: scale = max-abs / 2^(bits-1),
// rounded UP to a power of two (so requant is an exact shift).  A tensor whose
// values fit [-2^bits/2, +2^bits/2) at this scale stays inside the LUT domain.
inline std::map<std::string, double> calibrate_scales(const OnnxGraph &g, int bits = 12) {
  auto rng = calibrate_ranges(g);
  std::map<std::string, double> sc;
  double cap = (double)(1LL << (bits - 1));
  for (auto &kv : rng) {
    double a = kv.second;
    if (!std::isfinite(a) || a <= 0) {
      sc[kv.first] = 1.0 / (double)(1LL << bits);
      continue;
    }
    double s = a / cap;
    double e = std::ceil(std::log2(s));
    sc[kv.first] = std::pow(2.0, e);
  } // power-of-two scale
  return sc;
}

} // namespace volefier
#endif
