#ifndef ML_FLOAT_GRAPH_INFER_H
#define ML_FLOAT_GRAPH_INFER_H

// Graph-driven inference prover for the ml-float (Mystique) backend. Same ONNX
// front-end (parse/fold/fuse) and same fixed-point IntFp activations as
// FixedGraphInfer, but every gadget is the float/Boolean one:
//   - truncation     : field range-check (arith2bool + integer geq), not LogUp
//   - ReLU / MaxPool : Boolean domain (arith2bool -> integer compare -> bool2arith)
//   - BatchNorm      : per-channel rsqrt in IEEE-754 float (Int62ToFloat/sqrt)
//   - Sigmoid / Gelu : the function evaluated in an IEEE-754 float circuit
// Linear layers (Conv/Gemm/MatMul) are the same committed field matmul.
//
// Requires the dual setup: setup_zk_bool THEN setup_zk_arith(..., enable_conversion=true).

#include "emp-zk/emp-zk.h"
#include "emp-tool/emp-tool.h"
#include "volefier/io/onnx.h"
#include "volefier/linear/conv2d.h"
#include "volefier/ml-float/float_convert.h"  // Int62ToFloat / FloatToInt62 (ported) + ZK type aliases
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace volefier {

template <typename IO> class FloatGraphInfer {
public:
  IPProof<IO> *ipp;
  emp::ZKBoolSession *bsess;   // Boolean session: field<->bool conversion, float circuits, reveals
  int party, S, mat_threads;
  std::vector<std::string> ops_run;
  long n_eda = 0, n_flt = 0;

  FloatGraphInfer(IPProof<IO> *ipp_, emp::ZKBoolSession *bsess_, int party_, int S_, int mt = 1)
      : ipp(ipp_), bsess(bsess_), party(party_), S(S_), mat_threads(mt) {}

  Ctx &ctx() { return bsess->ctx(); }
  // public 62-bit ZKInt constant (replaces old Integer(62, v, PUBLIC)).
  ZKInt kint(uint64_t v) { return bsess->input<ZKInt>(PUBLIC, (int64_t)v, 62); }

  std::vector<IntFp> commit_vec(const std::vector<int64_t> &q) {
    std::vector<IntFp> c(q.size());
    for (size_t i = 0; i < q.size(); ++i) c[i] = commit_q(q[i]);
    return c;
  }
  static int64_t decode(uint64_t v) {
    return dec(v);
  }

  static bool supported(const std::string &op) {
    static const std::set<std::string> ok = {
        "Gemm",    "MatMul",  "Conv",        "BatchNormalization", "Relu", "Sigmoid", "Gelu",
        "Erf",     "MaxPool", "AveragePool", "GlobalAveragePool",  "Add",  "Flatten", "Reshape",
        "Dropout", "Identity"};
    return ok.count(op) > 0;
  }
  static void check_ops(const OnnxGraph &g) {
    for (const auto &n : g.nodes)
      if (!supported(n.op_type))
        throw std::runtime_error("ml-float: unsupported operator '" + n.op_type + "'");
  }

  std::vector<IntFp> run(const OnnxGraph &g, const std::vector<IntFp> &input,
                         const std::vector<int64_t> &input_ref, std::vector<int64_t> &out_ref) {
    g_ = &g;
    const std::string in0 = g.inputs[0];
    act_[in0] = input;
    ref_[in0] = input_ref;
    std::vector<int> s;
    auto it = g.input_shapes.find(in0);
    if (it != g.input_shapes.end())
      for (size_t i = (it->second.size() > 1 ? 1 : 0); i < it->second.size(); ++i)
        s.push_back((int)it->second[i]);
    if (s.empty()) s.push_back((int)input.size());
    shp_[in0] = s;
    // layer-by-layer STREAMING: free each tensor's state after its last consumer runs (peak mem ~=
    // a few layers).
    const bool stream = !getenv("NO_STREAM");
    const bool free_init =
        stream && getenv("FREE_INIT"); // also free the fp32 weight after last use
                                       // (single-inference: mutates caller graph)
    std::map<std::string, int> last_use;
    if (stream) {
      for (int k = 0; k < (int)g.nodes.size(); ++k)
        for (const auto &in : g.nodes[k].input)
          if (!in.empty()) last_use[in] = k;
    }
    std::set<std::string> keep(g.outputs.begin(), g.outputs.end());
    int ni = -1;
    for (const auto &n : g.nodes) {
      ++ni;
      ops_run.push_back(n.op_type);
      const std::string &op = n.op_type;
      if (op == "Gemm" || op == "MatMul")
        gemm(n);
      else if (op == "Conv")
        conv(n);
      else if (op == "BatchNormalization")
        batchnorm(n);
      else if (op == "Relu")
        relu(n);
      else if (op == "Sigmoid")
        floatfn(n, 0);
      else if (op == "Gelu" || op == "Erf")
        floatfn(n, 1);
      else if (op == "MaxPool")
        pool(n, true);
      else if (op == "AveragePool")
        pool(n, false);
      else if (op == "GlobalAveragePool")
        global_avgpool(n);
      else if (op == "Add")
        add(n);
      else if (op == "Flatten" || op == "Reshape")
        reshape(n);
      else if (op == "Dropout" || op == "Identity")
        identity(n);
      else
        throw std::runtime_error("ml-float: unsupported op " + op);
      // free every input whose LAST use is this node (skip graph outputs)
      if (stream)
        for (const auto &in : n.input)
          if (!in.empty() && last_use[in] == ni && !keep.count(in)) {
            act_.erase(in);
            ref_.erase(in);
            shp_.erase(in);
            if (free_init)
              const_cast<OnnxGraph &>(*g_).initializer.erase(
                  in); // free the fp32 weight (graph is caller-owned non-const)
          }
    }
    out_ref = ref_.count(g.outputs[0]) ? ref_[g.outputs[0]] : std::vector<int64_t>{};
    return act_[g.outputs[0]];
  }

private:
  const OnnxGraph *g_ = nullptr;
  std::map<std::string, std::vector<IntFp>> act_;
  std::map<std::string, std::vector<int>> shp_;
  std::map<std::string, std::vector<int64_t>> ref_;

  static int64_t dec(uint64_t v) {
    return v > PR / 2 ? (int64_t)v - (int64_t)PR : (int64_t)v;
  }
  IntFp commit_q(int64_t q) {
    return IntFp(q >= 0 ? (uint64_t)q : PR - (uint64_t)(-q), ALICE);
  }
  IntFp zero() {
    return IntFp((uint64_t)0, PUBLIC);
  }
  static int attr1(const OnnxNode &n, const char *k, int d) {
    auto it = n.attr.find(k);
    return (it != n.attr.end() && !it->second.ints.empty()) ? (int)it->second.ints[0] : d;
  }
  static int prod(const std::vector<int> &s) {
    int n = 1;
    for (int d : s) n *= d;
    return n;
  }
  static int64_t rdiv(int64_t a, int64_t D) {
    int64_t h = D / 2;
    return a >= 0 ? (a + h) / D : -(((-a) + h) / D);
  }
  int64_t qz(double v) {
    return (int64_t)llround(v * (double)(1LL << S));
  }

  // ---- truncation: v @ 2^2S -> 2^S, remainder |v - t*2^S| < 2^S -----------------
  // Default is an Appenzeller-to-Brie style "length check" (CCS'21, Fig 12): shift
  // the remainder r = v - t*2^S to the non-negative window r' = r + 2^S in
  // [0, 2^{S+1}) and prove r' fits in S+1 bits by asserting all bits >= S+1 are
  // zero.  The length check IS the range enforcement, so it replaces the two
  // 62-bit range comparisons of the legacy Mystique check (bd>=POS, bd>=NEGB)
  // with one high-bit OR-reduction -- ~7.5% fewer AND gates end-to-end on ResNet.
  // LEGACY_TRUNC=1 restores the old two-comparison range check.
  void truncate(std::vector<IntFp> &v) {
    const int n = (int)v.size();
    std::vector<uint64_t> tval(n, 0);
    if (party == ALICE)
      for (int i = 0; i < n; ++i) {
        int64_t acc = dec((uint64_t)VAL(v[i].value));   // low 64 = value (new emp IntFp layout)
        int64_t tq = (int64_t)llround((double)acc / (double)(1LL << S));
        tval[i] = tq >= 0 ? (uint64_t)tq : PR - (uint64_t)(-tq);
      }
    std::vector<IntFp> t(n);
    batch_feed(t.data(), tval.data(), n);
    const uint64_t neg2S = PR - (1ULL << S);

    static const bool legacy_trunc = getenv("LEGACY_TRUNC") != nullptr;
    if (!legacy_trunc) {
      std::vector<IntFp> rp(n);
      const IntFp shift((uint64_t)(1ULL << S), PUBLIC); // + 2^S
      for (int i = 0; i < n; ++i) rp[i] = v[i] + t[i] * neg2S + shift;
      std::vector<ZKInt> bd(n);
      arith2bool(bd.data(), rp.data(), n);
      n_eda += n;
      Bit allbad = Bit::constant(ctx(), false);
      for (int i = 0; i < n; ++i) {
        Bit hi = bd[i][S + 1];               // high bits [S+1 .. 61] must be zero
        for (int b = S + 2; b < 62; ++b) hi = hi | bd[i][b];
        allbad = allbad | hi;
      }
      if (bsess->reveal(allbad, PUBLIC).value_or(false) && party == ALICE)
        error("ml-float truncation length-check failed\n");
      v = t;
      return;
    }

    std::vector<IntFp> diff(n);
    for (int i = 0; i < n; ++i) diff[i] = v[i] + t[i] * neg2S;
    std::vector<ZKInt> bd(n);
    arith2bool(bd.data(), diff.data(), n);
    n_eda += n;
    ZKInt POS = kint((uint64_t)(1ULL << S)), NEGB = kint(PR - (uint64_t)(1ULL << S));
    Bit allbad = Bit::constant(ctx(), false);
    for (int i = 0; i < n; ++i) {
      Bit p = bd[i] >= POS;
      Bit ng = bd[i] >= NEGB;
      allbad = allbad | (p & (!ng));
    }
    if (bsess->reveal(allbad, PUBLIC).value_or(false) && party == ALICE)
      error("ml-float truncation range-check failed\n");
    v = t;
  }

  // ---- Gemm / MatMul: committed field matmul + truncate + bias ----
  void gemm(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    const OnnxTensor &W = g_->initializer.at(n.input[1]);
    const bool isgemm = n.op_type == "Gemm";
    const int transB = isgemm ? (n.attr.count("transB") ? (int)n.attr.at("transB").i : 0) : 0;
    int out = (isgemm ? (transB ? W.dims[0] : W.dims[1]) : W.dims[1]),
        in = (isgemm ? (transB ? W.dims[1] : W.dims[0]) : W.dims[0]);
    std::vector<std::vector<int64_t>> qW(out, std::vector<int64_t>(in));
    std::vector<std::vector<IntFp>> cW(out);
    for (int o = 0; o < out; ++o) {
      cW[o].resize(in);
      for (int i = 0; i < in; ++i) {
        qW[o][i] = qz((isgemm && !transB) || (!isgemm) ? W.data[(int64_t)i * out + o]
                                                       : W.data[(int64_t)o * in + i]);
        cW[o][i] = commit_q(qW[o][i]);
      }
    }
    std::vector<int64_t> qb(out, 0);
    if (n.input.size() > 2 && g_->initializer.count(n.input[2])) {
      const auto &B = g_->initializer.at(n.input[2]);
      for (int o = 0; o < out; ++o) qb[o] = qz(B.data[o]);
    }
    std::vector<IntFp> cb(out);
    for (int o = 0; o < out; ++o) cb[o] = commit_q(qb[o]);
    const int M = (int)x.size() / in;
    std::vector<IntFp> raw((size_t)M * out);
    for (int m = 0; m < M; ++m)
      for (int o = 0; o < out; ++o) {
        IntFp acc = zero();
        for (int i = 0; i < in; ++i) acc = acc + cW[o][i] * x[(int64_t)m * in + i];
        raw[(int64_t)m * out + o] = acc;
      }
    truncate(raw);
    for (int m = 0; m < M; ++m)
      for (int o = 0; o < out; ++o) raw[(int64_t)m * out + o] = raw[(int64_t)m * out + o] + cb[o];
    act_[n.output[0]] = std::move(raw);
    shp_[n.output[0]] = M > 1 ? std::vector<int>{M, out} : std::vector<int>{out};
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r((size_t)M * out);
      const int64_t D = 1LL << S;
      for (int m = 0; m < M; ++m)
        for (int o = 0; o < out; ++o) {
          int64_t a = 0;
          for (int i = 0; i < in; ++i) a += qW[o][i] * xr[(int64_t)m * in + i];
          r[(int64_t)m * out + o] = rdiv(a, D) + qb[o];
        }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- Conv: conv2d (field) + truncate + per-channel bias ----
  void conv(const OnnxNode &n) {
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int Cin = xs[0], H = xs[1], Wd = xs[2];
    const OnnxTensor &W = g_->initializer.at(n.input[1]);
    const int Cout = W.dims[0], K = W.dims[2], stride = attr1(n, "strides", 1),
              pad = attr1(n, "pads", 0);
    const int Ho = conv_out_dim(H, K, stride, pad), Wo = conv_out_dim(Wd, K, stride, pad),
              Msz = Cin * K * K, Pn = Ho * Wo;
    std::vector<std::vector<int64_t>> qK(Cout, std::vector<int64_t>(Msz));
    std::vector<std::vector<IntFp>> cK(Cout);
    for (int co = 0; co < Cout; ++co) {
      cK[co].resize(Msz);
      for (int m = 0; m < Msz; ++m) {
        qK[co][m] = qz(W.data[(int64_t)co * Msz + m]);
        cK[co][m] = commit_q(qK[co][m]);
      }
    }
    std::vector<int64_t> qb(Cout, 0);
    if (n.input.size() > 2 && g_->initializer.count(n.input[2])) {
      const auto &B = g_->initializer.at(n.input[2]);
      for (int co = 0; co < Cout; ++co) qb[co] = qz(B.data[co]);
    }
    Tensor3 X(Cin, std::vector<std::vector<IntFp>>(H, std::vector<IntFp>(Wd)));
    for (int c = 0; c < Cin; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < Wd; ++w) X[c][h][w] = xf[(int64_t)c * H * Wd + h * Wd + w];
    FMap raw;
    conv2d<IO>(*ipp, raw, X, cK, Cin, H, Wd, Cout, K, stride, pad, mat_threads);
    std::vector<IntFp> flat;
    flat.reserve((size_t)Cout * Pn);
    for (int co = 0; co < Cout; ++co)
      for (int p = 0; p < Pn; ++p) flat.push_back(raw[co][p]);
    truncate(flat);
    std::vector<IntFp> out((size_t)Cout * Pn);
    for (int co = 0; co < Cout; ++co)
      for (int p = 0; p < Pn; ++p)
        out[(int64_t)co * Pn + p] = flat[(int64_t)co * Pn + p] + commit_q(qb[co]);
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {Cout, Ho, Wo};
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r((size_t)Cout * Pn);
      const int64_t D = 1LL << S;
      for (int co = 0; co < Cout; ++co)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            int64_t a = 0;
            for (int ci = 0; ci < Cin; ++ci)
              for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj) {
                  int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
                  if (ih >= 0 && ih < H && iw >= 0 && iw < Wd)
                    a += xr[(int64_t)ci * H * Wd + ih * Wd + iw] * qK[co][ci * K * K + ki * K + kj];
                }
            r[(int64_t)co * Pn + oh * Wo + ow] = rdiv(a, D) + qb[co];
          }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- ReLU: Boolean domain (arith2bool -> sign -> select 0 -> bool2arith) ----
  void relu(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    const int nn = (int)x.size();
    std::vector<IntFp> flat = x;
    std::vector<ZKInt> b(nn);
    arith2bool(b.data(), flat.data(), nn);
    n_eda += nn;
    ZKInt zero62 = kint(0), smallest_neg = kint((uint64_t)((PR - 1) / 2) + 1);
    for (int i = 0; i < nn; ++i) {
      Bit neg = b[i] >= smallest_neg;      // field-rep >= PR/2  =>  represents a negative
      b[i] = b[i].select(neg, zero62);     // neg ? 0 : b[i]
    }
    bool2arith(flat.data(), b.data(), nn);
    n_eda += nn;
    act_[n.output[0]] = std::move(flat);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r(xr.size());
      for (size_t k = 0; k < xr.size(); ++k) r[k] = xr[k] > 0 ? xr[k] : 0;
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- Sigmoid / Gelu: need exp/tanh IEEE-754 float circuits, which the new emp
  // removed (only sqrt/rsqrt/recip survive).  ResNet never uses these; keep them
  // stubbed until the transcendentals are reimplemented (transformer track).
  void floatfn(const OnnxNode &, int) {
    throw std::runtime_error(
        "ml-float: Sigmoid/Gelu not supported (exp/tanh float circuits removed from new emp)");
  }

  // ---- BatchNorm: per-channel rsqrt in FLOAT, then field affine + truncate ----
  void batchnorm(const OnnxNode &n) {
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int C = xs[0], HW = prod(xs) / C;
    const auto &sc = g_->initializer.at(n.input[1]), &be = g_->initializer.at(n.input[2]),
               &mn = g_->initializer.at(n.input[3]), &vr = g_->initializer.at(n.input[4]);
    const double eps = n.attr.count("epsilon") ? n.attr.at("epsilon").f : 1e-5;
    std::vector<IntFp> nv(C);
    for (int c = 0; c < C; ++c) nv[c] = commit_q(qz(vr.data[c] + eps));
    std::vector<ZKInt> nb(C);
    arith2bool(nb.data(), nv.data(), C);
    n_eda += C;
    for (int c = 0; c < C; ++c) {
      FloatZK f = Int62ToFloat(ctx(), nb[c], S);
      f = f.rsqrt();               // 1 / sqrt(var + eps)
      nb[c] = FloatToInt62(ctx(), f, S);
    }
    n_flt += C;
    std::vector<IntFp> rs(C);
    bool2arith(rs.data(), nb.data(), C);
    n_eda += C;
    std::vector<IntFp> scaling(C);
    for (int c = 0; c < C; ++c) scaling[c] = rs[c] * commit_q(qz(sc.data[c]));
    truncate(scaling); // gamma*rsqrt @ 2^S
    std::vector<IntFp> out((size_t)C * HW);
    for (int c = 0; c < C; ++c) {
      IntFp negm = commit_q(qz(-mn.data[c]));
      std::vector<IntFp> prodv(HW);
      for (int p = 0; p < HW; ++p) prodv[p] = (xf[(int64_t)c * HW + p] + negm) * scaling[c];
      truncate(prodv);
      IntFp bt = commit_q(qz(be.data[c]));
      for (int p = 0; p < HW; ++p) out[(int64_t)c * HW + p] = prodv[p] + bt;
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = xs;
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r((size_t)C * HW);
      const int64_t D = 1LL << S;
      for (int c = 0; c < C; ++c) {
        double a = sc.data[c] / std::sqrt(vr.data[c] + eps);
        int64_t qa = rdiv((int64_t)llround(a * (1LL << S)) * 1, 1);
        int64_t qsc = (int64_t)llround(a * (double)(1LL << S));
        int64_t qb = qz(be.data[c]);
        int64_t qm = qz(-mn.data[c]);
        for (int p = 0; p < HW; ++p)
          r[(int64_t)c * HW + p] = rdiv((xr[(int64_t)c * HW + p] + qm) * qsc, D) + qb;
        (void)qa;
      }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- pooling ----
  void pool(const OnnxNode &n, bool ismax) {
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int C = xs[0], H = xs[1], Wd = xs[2];
    const int K = attr1(n, "kernel_shape", 2), stride = attr1(n, "strides", K),
              pad = attr1(n, "pads", 0);
    const int Ho = conv_out_dim(H, K, stride, pad), Wo = conv_out_dim(Wd, K, stride, pad);
    bool hr = party == ALICE && ref_.count(n.input[0]);
    const auto *xr = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<IntFp> out((size_t)C * Ho * Wo);
    std::vector<int64_t> r;
    if (hr) r.resize((size_t)C * Ho * Wo);
    if (ismax) { // Boolean-domain max (MaxPool follows ReLU in ResNet, so field-rep >= is real >=)
      std::vector<ZKInt> ball((size_t)C * H * Wd);
      std::vector<IntFp> flat(xf);
      arith2bool(ball.data(), flat.data(), (int)flat.size());
      n_eda += (int)flat.size();
      std::vector<ZKInt> bout((size_t)C * Ho * Wo);
      for (int c = 0; c < C; ++c)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            ZKInt mx;
            bool first = true;
            int64_t mr = INT64_MIN;
            for (int ki = 0; ki < K; ++ki)
              for (int kj = 0; kj < K; ++kj) {
                int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
                if (ih >= 0 && ih < H && iw >= 0 && iw < Wd) {
                  ZKInt &val = ball[(int64_t)c * H * Wd + ih * Wd + iw];
                  if (first) {
                    mx = val;
                    first = false;
                  } else {
                    Bit gt = val >= mx;
                    mx = mx.select(gt, val);   // gt ? val : mx
                  }
                  if (hr) mr = std::max(mr, (*xr)[(int64_t)c * H * Wd + ih * Wd + iw]);
                }
              }
            bout[(int64_t)c * Ho * Wo + oh * Wo + ow] = mx;
            if (hr) r[(int64_t)c * Ho * Wo + oh * Wo + ow] = mr;
          }
      bool2arith(out.data(), bout.data(), (int)out.size());
      n_eda += (int)out.size();
    } else {
      avg_window(xf, xr, C, H, Wd, K, stride, pad, Ho, Wo, out, r, hr);
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {C, Ho, Wo};
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void avg_window(const std::vector<IntFp> &xf, const std::vector<int64_t> *xr, int C, int H,
                  int Wd, int K, int stride, int pad, int Ho, int Wo, std::vector<IntFp> &out,
                  std::vector<int64_t> &r, bool hr) {
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow) {
          IntFp s = zero();
          int cnt = 0;
          int64_t sm = 0;
          for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj) {
              int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
              if (ih >= 0 && ih < H && iw >= 0 && iw < Wd) {
                int64_t idx = (int64_t)c * H * Wd + ih * Wd + iw;
                s = s + xf[idx];
                ++cnt;
                if (hr) sm += (*xr)[idx];
              }
            }
          out[(int64_t)c * Ho * Wo + oh * Wo + ow] = float_div(s, cnt);
          if (hr) r[(int64_t)c * Ho * Wo + oh * Wo + ow] = rdiv(sm, cnt);
        }
  }
  void global_avgpool(const OnnxNode &n) {
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int C = xs[0], HW = prod(xs) / C;
    bool hr = party == ALICE && ref_.count(n.input[0]);
    const auto *xr = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<IntFp> out(C);
    std::vector<int64_t> r;
    if (hr) r.resize(C);
    for (int c = 0; c < C; ++c) {
      IntFp s = zero();
      int64_t sm = 0;
      for (int p = 0; p < HW; ++p) {
        s = s + xf[(int64_t)c * HW + p];
        if (hr) sm += (*xr)[(int64_t)c * HW + p];
      }
      out[c] = float_div(s, HW);
      if (hr) r[c] = rdiv(sm, HW);
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {C, 1, 1};
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  // divide a single committed value by a public count via a FLOAT circuit
  IntFp float_div(IntFp s, int cnt) {
    std::vector<ZKInt> bb(1);
    std::vector<IntFp> sv(1, s);
    arith2bool(bb.data(), sv.data(), 1);
    n_eda += 1;
    FloatZK f = Int62ToFloat(ctx(), bb[0], S);
    f = f / FloatZK::constant(ctx(), (float)cnt);
    bb[0] = FloatToInt62(ctx(), f, S);
    n_flt += 1;
    std::vector<IntFp> ov(1);
    bool2arith(ov.data(), bb.data(), 1);
    n_eda += 1;
    return ov[0];
  }

  void add(const OnnxNode &n) {
    const auto &A = act_.at(n.input[0]);
    std::vector<IntFp> B;
    std::vector<int64_t> Br;
    bool hr = party == ALICE && ref_.count(n.input[0]);
    if (g_->initializer.count(n.input[1])) {
      const auto &T = g_->initializer.at(n.input[1]);
      B.resize(T.data.size());
      Br.resize(T.data.size());
      for (size_t i = 0; i < T.data.size(); ++i) {
        Br[i] = qz(T.data[i]);
        B[i] = commit_q(Br[i]);
      }
    } else {
      B = act_.at(n.input[1]);
      if (hr && ref_.count(n.input[1]))
        Br = ref_.at(n.input[1]);
      else
        hr = false;
    }
    const size_t na = A.size(), nb = B.size();
    const auto &sA = shp_.at(n.input[0]);
    const auto *Ar = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<IntFp> y(na);
    std::vector<int64_t> r;
    if (hr) r.resize(na);
    auto bidx = [&](size_t i) -> size_t {
      if (nb == na) return i;
      if (nb == 1) return 0;
      if (!sA.empty() && (int)nb == sA[0]) return i / (na / nb);
      return i % nb;
    };
    for (size_t i = 0; i < na; ++i) {
      size_t j = bidx(i);
      y[i] = A[i] + B[j];
      if (hr) r[i] = (*Ar)[i] + Br[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = sA;
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void reshape(const OnnxNode &n) {
    act_[n.output[0]] = act_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) ref_[n.output[0]] = ref_.at(n.input[0]);
    int total = prod(shp_.at(n.input[0]));
    std::vector<int> ns;
    if (n.op_type == "Reshape" && n.input.size() > 1 && g_->initializer.count(n.input[1])) {
      const auto &sh = g_->initializer.at(n.input[1]).i64;
      int known = 1, neg = -1;
      for (size_t i = 1; i < sh.size(); ++i) {
        if (sh[i] == -1)
          neg = (int)ns.size();
        else
          known *= (int)sh[i];
        ns.push_back((int)sh[i]);
      }
      if (neg >= 0 && known) ns[neg] = total / known;
    }
    if (ns.empty()) ns = {total};
    shp_[n.output[0]] = ns;
  }
  void identity(const OnnxNode &n) {
    act_[n.output[0]] = act_.at(n.input[0]);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) ref_[n.output[0]] = ref_.at(n.input[0]);
  }
};

} // namespace volefier
#endif // ML_FLOAT_GRAPH_INFER_H
