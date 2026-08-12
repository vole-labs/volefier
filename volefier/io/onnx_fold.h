#ifndef ML_IO_ONNX_FOLD_H
#define ML_IO_ONNX_FOLD_H

// Constant-folding + shape-inference pass for ONNX graphs.
//
// A real transformer export (e.g. gpt2) is dominated by shape/constant plumbing
// (Constant/Shape/Unsqueeze/Concat/Gather/Slice/Cast) that computes tensor shapes
// and index constants -- it does NOT depend on the runtime input, so it can be
// evaluated on the host and replaced by initializers. With a CONCRETE input shape
// fixed, this pass:
//   (1) propagates shapes through every node (shape inference),
//   (2) evaluates any node whose inputs are all static (incl. Shape, which only
//       needs the input's shape) on the host, turning it into an initializer,
//   (3) drops the folded nodes.
// It is SAFE: a node is folded only if it can be evaluated exactly; otherwise it
// is kept. The result is the same graph with the plumbing collapsed, leaving the
// real tensor compute for the prover.

#include "volefier/io/onnx.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace volefier {

// Saturating double -> int64: the C++ cast is UB when the value is outside int64 range
// (x86 cvttsd2si yields INT64_MIN, arm64 saturates) -- make the saturation explicit.
inline int64_t sat_i64(double x) {
  if (std::isnan(x)) return 0;
  if (x >= 9223372036854775296.0) return INT64_MAX;   // largest double < 2^63
  if (x <= -9223372036854775808.0) return INT64_MIN;
  return (int64_t)x;
}

struct HT {   // host tensor (values known at fold time)
  int dt = 7; // 1=float, 7=int64
  std::vector<int64_t> shape;
  std::vector<double> v; // unified numeric storage
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

inline HT ht_from_init(const OnnxTensor &t) {
  HT h;
  h.dt = t.dtype;
  h.shape = t.dims;
  // integer-like dtypes (int64=7, int32=6, bool=9, ...) store their value in i64; floats in data.
  // Reading i64 only for dtype==7 silently dropped bool/int32 constants (e.g. ConstantOfShape's
  // bool True fill -> 0), which broke the causal mask. Use i64 whenever it's populated.
  if (!t.i64.empty()) {
    h.v.resize(t.i64.size());
    for (size_t i = 0; i < t.i64.size(); ++i) h.v[i] = (double)t.i64[i];
  } else {
    h.v.resize(t.data.size());
    for (size_t i = 0; i < t.data.size(); ++i) h.v[i] = (double)t.data[i];
  }
  return h;
}
inline OnnxTensor ht_to_init(const std::string &name, const HT &h) {
  OnnxTensor t;
  t.name = name;
  t.dims = h.shape;
  t.dtype = h.dt;
  if (h.dt == 7) {
    t.i64.resize(h.v.size());
    for (size_t i = 0; i < h.v.size(); ++i) t.i64[i] = (int64_t)llround(h.v[i]);
  } else {
    t.data.resize(h.v.size());
    for (size_t i = 0; i < h.v.size(); ++i) t.data[i] = (float)h.v[i];
  }
  return t;
}

inline int norm_axis(int a, int nd) {
  return a < 0 ? a + nd : a;
}

// numpy-style broadcasting binary op. op: +,-,*,/,=,p(ow),m(in),M(ax),<,>
inline HT bcast(const HT &a, const HT &b, char op) {
  int nd = (int)std::max(a.shape.size(), b.shape.size());
  std::vector<int64_t> as(nd, 1), bs(nd, 1), os(nd, 1), ast(nd), bst(nd), ost(nd);
  for (int i = 0; i < (int)a.shape.size(); ++i) as[nd - (int)a.shape.size() + i] = a.shape[i];
  for (int i = 0; i < (int)b.shape.size(); ++i) bs[nd - (int)b.shape.size() + i] = b.shape[i];
  for (int i = 0; i < nd; ++i) os[i] = std::max(as[i], bs[i]);
  int64_t sa = 1, sb = 1, so = 1;
  for (int i = nd - 1; i >= 0; --i) {
    ast[i] = as[i] == 1 ? 0 : sa;
    sa *= as[i];
    bst[i] = bs[i] == 1 ? 0 : sb;
    sb *= bs[i];
    ost[i] = so;
    so *= os[i];
  }
  HT r;
  r.shape = os;
  bool cmp =
      (op == '=' || op == '<' || op == '>' || op == 'l' || op == 'g' || op == 'a' || op == 'o');
  r.dt = cmp ? 7 : ((a.dt == 1 || b.dt == 1) ? 1 : 7);
  if ((int64_t)a.v.size() < sa || (int64_t)b.v.size() < sb) {
    r.shape.clear();
    r.v.clear();
    return r;
  } // operand not materialized -> signal "cannot fold"
  r.v.resize(so);
  for (int64_t k = 0; k < so; ++k) {
    int64_t ai = 0, bi = 0, rem = k;
    for (int i = 0; i < nd; ++i) {
      int64_t idx = rem / ost[i];
      rem %= ost[i];
      ai += idx * ast[i];
      bi += idx * bst[i];
    }
    // bounds-safe: a malformed/dead constant subgraph can yield sa/sb <= 0 (e.g. a negative dim)
    // that slips past the size guard; read 0 for any out-of-range index instead of OOB (UB: harmless
    // on libc++/arm64, SIGSEGV on libstdc++/x86). Does not change the fold on a well-formed operand.
    double x = ((size_t)ai < a.v.size()) ? a.v[ai] : 0.0,
           y = ((size_t)bi < b.v.size()) ? b.v[bi] : 0.0, z = 0;
    switch (op) {
    case '+': z = x + y; break;
    case '-': z = x - y; break;
    case '*': z = x * y; break;
    case '/': z = r.dt == 7 ? (double)((int64_t)x / (int64_t)y) : x / y; break;
    case '=': z = (x == y) ? 1 : 0; break;
    case '<': z = (x < y) ? 1 : 0; break;
    case '>': z = (x > y) ? 1 : 0; break;
    case 'l': z = (x <= y) ? 1 : 0; break;
    case 'g': z = (x >= y) ? 1 : 0; break;
    case 'a': z = (x != 0 && y != 0) ? 1 : 0; break;
    case 'o': z = (x != 0 || y != 0) ? 1 : 0; break;
    case 'p': z = std::pow(x, y); break;
    case 'm': z = std::min(x, y); break;
    case 'M': z = std::max(x, y); break;
    }
    r.v[k] = z;
  }
  return r;
}

// Evaluate a node whose inputs are all static. Returns true + fills `res`.
inline bool fold_eval(const OnnxNode &n, const std::map<std::string, HT> &V, HT &res) {
  auto in = [&](int i) -> const HT & { return V.at(n.input[i]); };
  auto ai = [&](const char *k, int d) {
    auto it = n.attr.find(k);
    return (it != n.attr.end() && !it->second.ints.empty()) ? (int)it->second.ints[0] : d;
  };
  const std::string &op = n.op_type;

  if (op == "Constant") {
    auto it = n.attr.find("value");
    if (it != n.attr.end() && it->second.has_t) {
      res = ht_from_init(it->second.t);
      return true;
    }
    auto vi = n.attr.find("value_ints");
    if (vi != n.attr.end()) {
      res.dt = 7;
      res.shape = {(int64_t)vi->second.ints.size()};
      for (auto x : vi->second.ints) res.v.push_back((double)x);
      return true;
    }
    auto vint = n.attr.find("value_int");
    if (vint != n.attr.end()) {
      res.dt = 7;
      res.shape = {};
      res.v = {(double)vint->second.i};
      return true;
    }
    auto vf = n.attr.find("value_float");
    if (vf != n.attr.end()) {
      res.dt = 1;
      res.shape = {};
      res.v = {(double)vf->second.f};
      return true;
    }
    return false;
  }
  if (op == "Identity" || op == "Dropout") {
    res = in(0);
    return true;
  }
  if (op == "Cast") {
    res = in(0);
    res.dt = ai("to", res.dt);
    if (res.dt == 7)
      for (auto &x : res.v) x = (double)sat_i64(x);
    return true;
  }
  if (op == "Add") {
    res = bcast(in(0), in(1), '+');
    return !res.v.empty();
  }
  if (op == "Sub") {
    res = bcast(in(0), in(1), '-');
    return !res.v.empty();
  }
  if (op == "Mul") {
    res = bcast(in(0), in(1), '*');
    return !res.v.empty();
  }
  if (op == "Div") {
    res = bcast(in(0), in(1), '/');
    return !res.v.empty();
  }
  if (op == "Pow") {
    res = bcast(in(0), in(1), 'p');
    return !res.v.empty();
  }
  if (op == "Equal") {
    res = bcast(in(0), in(1), '=');
    return !res.v.empty();
  }
  if (op == "Less") {
    res = bcast(in(0), in(1), '<');
    return !res.v.empty();
  }
  if (op == "Greater") {
    res = bcast(in(0), in(1), '>');
    return !res.v.empty();
  }
  if (op == "LessOrEqual") {
    res = bcast(in(0), in(1), 'l');
    return !res.v.empty();
  }
  if (op == "GreaterOrEqual") {
    res = bcast(in(0), in(1), 'g');
    return !res.v.empty();
  }
  if (op == "And") {
    res = bcast(in(0), in(1), 'a');
    return !res.v.empty();
  }
  if (op == "Or") {
    res = bcast(in(0), in(1), 'o');
    return !res.v.empty();
  }
  if (op == "Not") {
    res = in(0);
    res.dt = 7;
    for (auto &x : res.v) x = (x == 0) ? 1 : 0;
    return true;
  }
  if (op == "IsNaN") {
    res = in(0);
    res.dt = 7;
    for (auto &x : res.v) x = std::isnan(x) ? 1 : 0;
    return true;
  }
  if (op == "Expand") {
    const HT &x = in(0);
    std::vector<int64_t> tgt(in(1).v.begin(), in(1).v.end());
    HT one;
    one.dt = x.dt;
    one.shape = tgt;
    one.v.assign(
        [&] {
          int64_t p = 1;
          for (auto d : tgt) p *= d;
          return p;
        }(),
        0);
    res = bcast(x, one, '+');
    return !res.v.empty();
  }
  if (op == "Min") {
    res = bcast(in(0), in(1), 'm');
    return !res.v.empty();
  }
  if (op == "Max") {
    res = bcast(in(0), in(1), 'M');
    return !res.v.empty();
  }
  if (op == "Sqrt") {
    res = in(0);
    for (auto &x : res.v) x = std::sqrt(x);
    return true;
  }
  if (op == "Neg") {
    res = in(0);
    for (auto &x : res.v) x = -x;
    return true;
  }
  if (op == "Where") {
    const HT &c = in(0), &x = in(1), &y = in(2);
    res = bcast(x, y, '+');
    if (res.v.empty() || c.v.empty() || x.v.empty() || y.v.empty()) {
      res.v.clear();
      return false;
    }
    for (int64_t k = 0; k < res.numel(); ++k)
      res.v[k] = c.v[c.numel() == 1 ? 0 : k] != 0 ? x.v[x.numel() == 1 ? 0 : k]
                                                  : y.v[y.numel() == 1 ? 0 : k];
    res.dt = x.dt;
    return true;
  }
  if (op == "Range") {
    double s = in(0).v[0], lim = in(1).v[0], d = in(2).v[0];
    res.dt = in(0).dt;
    for (double x = s; (d > 0 ? x < lim : x > lim); x += d) res.v.push_back(x);
    res.shape = {(int64_t)res.v.size()};
    return true;
  }
  if (op == "ConstantOfShape") {
    const HT &s = in(0);
    res.shape.assign(s.v.begin(), s.v.end() == s.v.begin() ? s.v.end() : s.v.end());
    res.shape.clear();
    for (double d : s.v) res.shape.push_back((int64_t)d);
    double fill = 0;
    res.dt = 7;
    auto it = n.attr.find("value");
    if (it != n.attr.end() && it->second.has_t) {
      HT vt = ht_from_init(it->second.t);
      if (!vt.v.empty()) fill = vt.v[0];
      res.dt = vt.dt;
    }
    res.v.assign(res.numel(), fill);
    return true;
  }
  if (op == "Unsqueeze") {
    res = in(0);
    std::vector<int> ax;
    if (n.input.size() > 1)
      for (double d : in(1).v) ax.push_back((int)d);
    else {
      auto it = n.attr.find("axes");
      if (it != n.attr.end())
        for (auto a : it->second.ints) ax.push_back((int)a);
    }
    int nd = (int)res.shape.size() + (int)ax.size();
    for (auto &a : ax) a = norm_axis(a, nd);
    std::sort(ax.begin(), ax.end());
    for (int a : ax) res.shape.insert(res.shape.begin() + a, 1);
    return true;
  }
  if (op == "Squeeze") {
    res = in(0);
    std::vector<int> ax;
    if (n.input.size() > 1)
      for (double d : in(1).v) ax.push_back(norm_axis((int)d, res.shape.size()));
    else {
      auto it = n.attr.find("axes");
      if (it != n.attr.end())
        for (auto a : it->second.ints) ax.push_back(norm_axis((int)a, res.shape.size()));
    }
    if (ax.empty()) {
      std::vector<int64_t> ns;
      for (auto d : res.shape)
        if (d != 1) ns.push_back(d);
      res.shape = ns;
    } else {
      std::sort(ax.rbegin(), ax.rend());
      for (int a : ax)
        if (a < (int)res.shape.size()) res.shape.erase(res.shape.begin() + a);
    }
    return true;
  }
  if (op == "Concat") {
    int axis = ai("axis", 0);
    res.dt = in(0).dt;
    res.shape = in(0).shape;
    axis = norm_axis(axis, res.shape.size());
    // only 1-D concat (the shape-vector case) handled exactly
    if (res.shape.size() == 1) {
      res.v.clear();
      int64_t tot = 0;
      for (size_t j = 0; j < n.input.size(); ++j) {
        for (double x : in(j).v) res.v.push_back(x);
        tot += in(j).numel();
      }
      res.shape = {tot};
      return true;
    }
    return false;
  }
  if (op == "Gather") {
    const HT &d = in(0), &idx = in(1);
    if (d.shape.empty()) return false;
    int axis = norm_axis(ai("axis", 0), d.shape.size());
    if (axis < 0 || axis >= (int)d.shape.size()) return false;
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= d.shape[i];
    int64_t ad = d.shape[axis];
    int64_t inner = 1;
    for (int i = axis + 1; i < (int)d.shape.size(); ++i) inner *= d.shape[i];
    int64_t ni = idx.numel();
    res.dt = d.dt;
    res.shape.clear();
    for (int i = 0; i < axis; ++i) res.shape.push_back(d.shape[i]);
    for (auto s : idx.shape) res.shape.push_back(s);
    for (int i = axis + 1; i < (int)d.shape.size(); ++i) res.shape.push_back(d.shape[i]);
    res.v.resize(outer * ni * inner);
    for (int64_t o = 0; o < outer; ++o)
      for (int64_t j = 0; j < ni; ++j) {
        int64_t ix = (int64_t)idx.v[j];
        if (ix < 0) ix += ad;
        for (int64_t k = 0; k < inner; ++k)
          res.v[(o * ni + j) * inner + k] = d.v[(o * ad + ix) * inner + k];
      }
    return true;
  }
  if (op == "Flatten") {
    res = in(0);
    int ax = n.attr.count("axis") ? (int)n.attr.at("axis").i : 1;
    int nd = (int)res.shape.size();
    if (ax < 0) ax += nd;
    int64_t d0 = 1, d1 = 1;
    for (int i = 0; i < ax; ++i) d0 *= res.shape[i];
    for (int i = ax; i < nd; ++i) d1 *= res.shape[i];
    res.shape = {d0, d1};
    return true;
  }
  if (op == "Reshape") {
    res = in(0);
    std::vector<int64_t> ns;
    int64_t known = 1, neg = -1;
    const HT &sh = in(1);
    for (size_t i = 0; i < sh.v.size(); ++i) {
      int64_t d = (int64_t)sh.v[i];
      if (d == -1) {
        neg = (int64_t)ns.size();
        ns.push_back(-1);
      } else if (d == 0 && i < res.shape.size()) {
        ns.push_back(res.shape[i]);
        known *= res.shape[i];
      } else {
        ns.push_back(d);
        known *= (d == 0 ? 1 : d);
      }
    }
    if (neg >= 0 && known) ns[neg] = res.numel() / known;
    res.shape = ns;
    return true;
  }
  if (op == "Transpose") {
    const HT &x = in(0);
    int nd = (int)x.shape.size();
    std::vector<int> perm;
    auto it = n.attr.find("perm");
    if (it != n.attr.end() && !it->second.ints.empty())
      for (auto p : it->second.ints) perm.push_back((int)p);
    else
      for (int i = nd - 1; i >= 0; --i) perm.push_back(i);
    std::vector<int64_t> ns(nd), ist(nd, 1), ost(nd, 1);
    for (int i = 0; i < nd; ++i) ns[i] = x.shape[perm[i]];
    for (int i = nd - 2; i >= 0; --i) {
      ist[i] = ist[i + 1] * x.shape[i + 1];
      ost[i] = ost[i + 1] * ns[i + 1];
    }
    res.dt = x.dt;
    res.shape = ns;
    res.v.resize(x.numel());
    std::vector<int> id(nd);
    for (int64_t o = 0; o < x.numel(); ++o) {
      int64_t rem = o, src = 0;
      for (int i = 0; i < nd; ++i) {
        id[i] = rem / ost[i];
        rem %= ost[i];
      }
      for (int i = 0; i < nd; ++i) src += id[i] * ist[perm[i]];
      res.v[o] = x.v[src];
    }
    return true;
  }
  if (op == "Slice" && n.input.size() >= 3) { // starts,ends,(axes),(steps) as inputs (opset 10+)
    const HT &x = in(0);
    int nd = (int)x.shape.size();
    // HF exports use ends = INT64_MAX; as a double that is 2^63, and the implicit
    // (int64_t)double conversion of an out-of-range value is UB -- x86 yields INT64_MIN
    // (-> empty slice -> the folded attention scale became 1/0 = inf) while arm64
    // saturates to INT64_MAX (correct). Saturate explicitly.
    std::vector<int64_t> starts, ends;
    for (double v : in(1).v) starts.push_back(sat_i64(v));
    for (double v : in(2).v) ends.push_back(sat_i64(v));
    std::vector<int> axes;
    if (n.input.size() > 3)
      for (double a : in(3).v) axes.push_back(norm_axis((int)a, nd));
    else
      for (int i = 0; i < (int)starts.size(); ++i) axes.push_back(i);
    std::vector<int64_t> steps(axes.size(), 1);
    if (n.input.size() > 4)
      for (size_t i = 0; i < in(4).v.size(); ++i) steps[i] = sat_i64(in(4).v[i]);
    std::vector<int64_t> lo(nd, 0), hi(x.shape), stp(nd, 1);
    for (size_t a = 0; a < axes.size(); ++a) {
      int ax = axes[a];
      int64_t d = x.shape[ax];
      int64_t s = starts[a] < 0 ? starts[a] + d : starts[a],
              e = ends[a] < 0 ? ends[a] + d : ends[a];
      s = std::max((int64_t)0, std::min(s, d));
      e = std::max((int64_t)0, std::min(e, d));
      lo[ax] = s;
      hi[ax] = e;
      stp[ax] = steps[a];
    }
    std::vector<int64_t> ns(nd);
    for (int i = 0; i < nd; ++i) ns[i] = stp[i] > 0 ? (hi[i] - lo[i] + stp[i] - 1) / stp[i] : 0;
    if (nd != 1) return false; // exact only for 1-D shape slices
    res.dt = x.dt;
    res.shape = ns;
    for (int64_t i = lo[0]; i < hi[0]; i += stp[0]) res.v.push_back(x.v[i]);
    return true;
  }
  return false;
}

// best-effort output-shape inference for a KEPT (non-folded) node
inline void fold_infer_shape(const OnnxNode &n, std::map<std::string, std::vector<int64_t>> &S,
                             const std::map<std::string, HT> &V) {
  auto has = [&](const std::string &t) { return S.count(t) > 0; };
  auto sh = [&](int i) -> const std::vector<int64_t> & { return S.at(n.input[i]); };
  const std::string &op = n.op_type;
  const std::string &o = n.output.empty() ? std::string() : n.output[0];
  if (o.empty()) return;
  // elementwise / same-shape
  static const std::map<std::string, int> same = {{"Relu", 1},
                                                  {"Sigmoid", 1},
                                                  {"Tanh", 1},
                                                  {"Erf", 1},
                                                  {"Sqrt", 1},
                                                  {"Softmax", 1},
                                                  {"LayerNormalization", 1},
                                                  {"BatchNormalization", 1},
                                                  {"Dropout", 1},
                                                  {"Identity", 1},
                                                  {"Cast", 1},
                                                  {"Not", 1},
                                                  {"IsNaN", 1},
                                                  {"Neg", 1},
                                                  {"Abs", 1},
                                                  {"Exp", 1},
                                                  {"Log", 1},
                                                  {"Reciprocal", 1},
                                                  {"Floor", 1},
                                                  {"Ceil", 1},
                                                  {"Gelu", 1},
                                                  {"QuantizeLinear", 1},
                                                  {"DequantizeLinear", 1}};
  if (same.count(op) && has(n.input[0])) {
    S[o] = sh(0);
    return;
  }
  static const std::map<std::string, int> ewise = {
      {"Add", 1},   {"Mul", 1},  {"Sub", 1},     {"Div", 1},         {"Pow", 1},
      {"Equal", 1}, {"Less", 1}, {"Greater", 1}, {"LessOrEqual", 1}, {"GreaterOrEqual", 1},
      {"And", 1},   {"Or", 1},   {"Min", 1},     {"Max", 1}};
  if (ewise.count(op) && has(n.input[0])) {
    if (n.input.size() > 1 && has(n.input[1])) {
      const auto &a = sh(0), &b = sh(1);
      S[o] = a.size() >= b.size() ? a : b;
    } else
      S[o] = sh(0);
    return;
  }
  if (op == "Where" && has(n.input[1])) {
    const auto &x = sh(1);
    std::vector<int64_t> r = x;
    for (int k = 0; k < 3; ++k)
      if (has(n.input[k]) && sh(k).size() > r.size()) r = sh(k);
    S[o] = r;
    return;
  }
  if (op == "Expand" && V.count(n.input[1])) {
    const HT &t = V.at(n.input[1]);
    std::vector<int64_t> ns(t.v.begin(), t.v.end());
    if (has(n.input[0])) {
      const auto &a = sh(0);
      for (size_t i = 0; i < ns.size() && i < a.size(); ++i) {
        size_t ai = a.size() - 1 - i, ni = ns.size() - 1 - i;
        ns[ni] = std::max(ns[ni], a[ai]);
      }
    }
    S[o] = ns;
    return;
  }
  if ((op == "MatMul" || op == "Gemm") && has(n.input[0])) {
    if (op == "Gemm") {
      int tb = n.attr.count("transB") ? (int)n.attr.at("transB").i : 0;
      const auto &w = V.count(n.input[1]) ? std::vector<int64_t>() : std::vector<int64_t>();
      (void)w;
      auto wit = S.count(n.input[1]) ? sh(1) : std::vector<int64_t>();
      int outd = wit.empty() ? 0 : (int)(tb ? wit[0] : wit[1]);
      auto a = sh(0);
      S[o] = {a.size() >= 2 ? a[a.size() - 2] : 1, (int64_t)outd};
      return;
    }
    if (has(n.input[1])) {
      auto a = sh(0), b = sh(1);
      if (a.empty() || b.empty()) return;
      std::vector<int64_t> r = a;
      r.back() = b.back();
      S[o] = r;
      return;
    }
  }
  if (op == "Reshape" && V.count(n.input[1]) && has(n.input[0])) {
    const HT &t = V.at(n.input[1]);
    const auto &a = sh(0);
    int64_t tot = 1;
    for (auto d : a) tot *= d;
    std::vector<int64_t> ns;
    int64_t known = 1, neg = -1;
    for (size_t i = 0; i < t.v.size(); ++i) {
      int64_t d = (int64_t)t.v[i];
      if (d == -1) {
        neg = ns.size();
        ns.push_back(-1);
      } else if (d == 0 && i < a.size()) {
        ns.push_back(a[i]);
        known *= a[i];
      } else {
        ns.push_back(d);
        known *= (d == 0 ? 1 : d);
      }
    }
    if (neg >= 0 && known) ns[neg] = tot / known;
    S[o] = ns;
    return;
  }
  if (op == "Transpose" && has(n.input[0])) {
    auto a = sh(0);
    int nd = (int)a.size();
    std::vector<int> perm;
    auto it = n.attr.find("perm");
    if (it != n.attr.end() && !it->second.ints.empty())
      for (auto p : it->second.ints) perm.push_back((int)p);
    else
      for (int i = nd - 1; i >= 0; --i) perm.push_back(i);
    std::vector<int64_t> ns(nd);
    for (int i = 0; i < nd; ++i) ns[i] = a[perm[i]];
    S[o] = ns;
    return;
  }
  if (op == "Gather" && has(n.input[0])) {
    int axis = norm_axis(n.attr.count("axis") ? (int)n.attr.at("axis").i : 0, sh(0).size());
    auto d = sh(0);
    std::vector<int64_t> is =
        has(n.input[1]) ? sh(1)
                        : (V.count(n.input[1]) ? V.at(n.input[1]).shape : std::vector<int64_t>());
    std::vector<int64_t> ns;
    for (int i = 0; i < axis; ++i) ns.push_back(d[i]);
    for (auto s : is) ns.push_back(s);
    for (int i = axis + 1; i < (int)d.size(); ++i) ns.push_back(d[i]);
    S[o] = ns;
    return;
  }
  if (op == "Concat" && has(n.input[0])) {
    auto ns = sh(0);
    int axis = norm_axis(n.attr.count("axis") ? (int)n.attr.at("axis").i : 0, ns.size());
    if (axis < 0 || axis >= (int)ns.size()) return;
    for (size_t j = 1; j < n.input.size(); ++j)
      if (has(n.input[j]) && axis < (int)sh(j).size()) ns[axis] += sh(j)[axis];
    S[o] = ns;
    return;
  }
  if (op == "ReduceMean" && has(n.input[0])) {
    auto a = sh(0);
    int keep = n.attr.count("keepdims") ? (int)n.attr.at("keepdims").i : 1;
    std::vector<int> axes;
    auto it = n.attr.find("axes");
    if (it != n.attr.end())
      for (auto x : it->second.ints) axes.push_back(norm_axis((int)x, a.size()));
    for (int ax : axes) {
      if (keep)
        a[ax] = 1;
      else
        a[ax] = -1;
    }
    if (!keep) {
      std::vector<int64_t> r;
      for (auto d : a)
        if (d != -1) r.push_back(d);
      a = r;
    }
    S[o] = a;
    return;
  }
  if (op == "Flatten" && has(n.input[0])) {
    auto a = sh(0);
    int ax = n.attr.count("axis") ? (int)n.attr.at("axis").i : 1;
    int64_t d0 = 1, d1 = 1;
    for (int i = 0; i < ax; ++i) d0 *= a[i];
    for (int i = ax; i < (int)a.size(); ++i) d1 *= a[i];
    S[o] = {d0, d1};
    return;
  }
  if (op == "Split" && has(n.input[0])) {
    int axis = norm_axis(n.attr.count("axis") ? (int)n.attr.at("axis").i : 0, sh(0).size());
    auto a = sh(0);
    if (axis >= 0 && axis < (int)a.size()) {
      a[axis] /= (int64_t)std::max((size_t)1, n.output.size());
      for (auto &oo : n.output) S[oo] = a;
    }
    return;
  }
  if ((op == "Unsqueeze" || op == "Squeeze") &&
      has(n.input[0])) { /* handled by fold if static; else approx keep */
    S[o] = sh(0);
    return;
  }
  // unknown: leave output shape unknown (downstream Shape on it won't fold -> stays a node, safe)
}

struct FoldStats {
  int before = 0, after = 0, folded = 0;
};

inline OnnxGraph fold_constants(const OnnxGraph &g, FoldStats &st) {
  std::map<std::string, HT> V;
  std::map<std::string, std::vector<int64_t>> S;
  OnnxGraph out;
  out.inputs = g.inputs;
  out.outputs = g.outputs;
  out.input_shapes = g.input_shapes;
  out.initializer = g.initializer;
  // seed shapes for ALL initializers, but only small ones as host VALUES (fold
  // only needs shape/constant tensors; weight matrices are never folded and
  // materializing them as host doubles is wasteful + crash-prone).
  for (const auto &kv : g.initializer) {
    S[kv.first] = kv.second.dims;
    int64_t ne = 1;
    for (auto d : kv.second.dims) ne *= d;
    if (ne <= 65536) V[kv.first] = ht_from_init(kv.second);
  }
  for (const auto &kv : g.input_shapes)
    S[kv.first] = kv.second; // concrete input shape seeds inference
  st.before = (int)g.nodes.size();
  int _dbg = getenv("FOLD_DBG") ? 1 : 0;
  int _i = 0;
  for (const auto &n : g.nodes) {
    if (_dbg) {
      fprintf(stderr, "[%d] %s -> %s\n", _i++, n.op_type.c_str(),
              n.output.empty() ? "" : n.output[0].c_str());
      fflush(stderr);
    }
    bool all_static = true; // a node with no inputs (Constant) is a constant source
    for (const auto &i : n.input)
      if (!i.empty() && !V.count(i)) {
        all_static = false;
        break;
      }
    HT res;
    bool ok = false;
    try {
      if (n.op_type == "Shape") {
        auto it = S.find(n.input[0]);
        if (it != S.end()) {
          res.dt = 7;
          res.shape = {(int64_t)it->second.size()};
          res.v.assign(it->second.begin(), it->second.end());
          ok = true;
        }
      } else if (all_static)
        ok = fold_eval(n, V, res);
    } catch (...) {
      ok = false;
    }
    if (ok && !n.output.empty()) {
      V[n.output[0]] = res;
      S[n.output[0]] = res.shape;
      out.initializer[n.output[0]] = ht_to_init(n.output[0], res);
      st.folded++;
      continue;
    }
    out.nodes.push_back(n);
    try {
      fold_infer_shape(n, S, V);
    } catch (...) {}
  }
  st.after = (int)out.nodes.size();
  return out;
}

} // namespace volefier
#endif // ML_IO_ONNX_FOLD_H
