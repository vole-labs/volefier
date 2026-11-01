#ifndef ML_FIXED_GRAPH_INFER_H
#define ML_FIXED_GRAPH_INFER_H

// Graph-driven fixed-point inference prover. Walks a parsed model graph, threads
// committed activations (with shape) between nodes, and dispatches each operator
// to a fixed-point ZKP component. Aborts on any operator with no handler, or any
// handler whose required component was not configured.
//
// Supported ops:
//   compute  : Gemm, MatMul, Conv, BatchNormalization, LayerNormalization
//   nonlinear: Relu, Sigmoid, Gelu/Erf, Softmax, MaxPool, AveragePool, GlobalAveragePool
//   tensor   : Add, Flatten, Reshape, Dropout, Identity
// Covers MLP / VGG / ResNet fully, AND a full transformer ONNX export (GPT-2):
//   transformer: two-activation MatMul (attention), Gather (embeddings), Transpose,
//                Split, Softmax, LayerNorm, Gelu, plus masked-softmax glue
//                (Where, IsNaN, And, Sub, Concat, Slice, Expand, Cast/Unsqueeze/Squeeze).
// GPT-2 proves end-to-end (uniform 2^-P fixed-point, committed weights).

#include "volefier/io/onnx.h"
#include "volefier/ml-fixed/fixed_infer.h"
#include "volefier/ml-fixed/fixed_nonlinear.h"
#include "volefier/nonlinear/comparison.h"
#include "volefier/nonlinear/range_check.h"
#include "volefier/nonlinear/div.h"
#include "volefier/linear/conv2d.h"
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace emp;

namespace volefier {

class FixedGraphInfer {
public:
  FixedInfer *qi;
  CompareLU *cmp;
  int party, P;
  std::size_t act_bits;
  // optional components (set before run() for the ops that need them)
  Div *divr = nullptr;
  RangeCheckLU *rcmp = nullptr;
  FixedTable *exp_tab = nullptr, *gelu_tab = nullptr, *sig_tab = nullptr, *tanh_tab = nullptr;
  int64_t gelu_off = 0, sig_off = 0, tanh_off = 0;
  std::size_t score_bits = 16;
  bool lse_softmax = false; // DeepProve LogSumExp softmax (no max_value, no division)
  int lse_extra_bits = 0;   // LSE: probabilities at 2^-(P+lse_extra_bits) (exp_tab built to match);
                            // the extra bits are dropped by the consuming attn@V MatMul's requant
  double lse_max_slack = 0; // LSE: refuse rows whose normalization slack exceeds this (0 = off)
  std::vector<std::string> ops_run;

  FixedGraphInfer(FixedInfer *qi_, CompareLU *cmp_, int party_, int P_, std::size_t ab = 30)
      : qi(qi_), cmp(cmp_), party(party_), P(P_), act_bits(ab) {}

  static bool supported(const std::string &op) {
    static const std::set<std::string> ok = {
        "Gemm",
        "MatMul",
        "Conv",
        "BatchNormalization",
        "LayerNormalization",
        "Relu",
        "Sigmoid",
        "Gelu",
        "Erf",
        "Softmax",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Add",
        "Mul",
        "Flatten",
        "Reshape",
        "Dropout",
        "Identity",
        "Gather",
        "Transpose",
        "Split",
        "Sub",
        "Where",
        "IsNaN",
        "And",
        "Concat",
        "Slice",
        "Expand",
        "Cast",
        "Unsqueeze",
        "Squeeze",
        "Tanh",
        "Div"}; // Tanh: BERT pooler;  Div: BERT attention score / sqrt(d_head)
    return ok.count(op) > 0;
  }
  static void check_ops(const OnnxGraph &g) {
    for (const auto &n : g.nodes)
      if (!supported(n.op_type))
        throw std::runtime_error("model not provable: unsupported operator '" + n.op_type +
                                 "' (node '" + n.name + "')");
  }

  std::vector<IntFp> run(const OnnxGraph &g, const std::vector<IntFp> &input,
                         const std::vector<int64_t> &input_ref, std::vector<int64_t> &out_ref) {
    if (g.inputs.empty()) throw std::runtime_error("graph has no input");
    const std::string in0 = g.inputs[0];
    g_ = &g;
    // Point comparison's bitness g(g-1)=0 / selector u(1-u)=0 checks at the shared process-global
    // Hadamard2IP (created by FixedInfer, flushed in qi->destroy()); one RLC challenge for the
    // whole batch across every comparison + LogUp site.
    if (cmp)
      cmp->h2ip = volefier::global_h2ip();
    act_[in0] = input;
    ref_[in0] = input_ref;
    std::vector<int> s;
    auto it = g.input_shapes.find(in0);
    if (it != g.input_shapes.end())
      for (size_t i = (it->second.size() > 1 ? 1 : 0); i < it->second.size(); ++i)
        s.push_back((int)it->second[i]);
    if (s.empty()) s.push_back((int)input.size());
    shp_[in0] = s;

    const char *sa_env = getenv("STOP_AFTER");
    const size_t stop_after = sa_env ? (size_t)atoi(sa_env) : 0;
    // layer-by-layer STREAMING: free each tensor's state after its last consumer runs, so peak
    // memory ~= a few layers, not the whole graph (else act_/ref_ accumulate for the entire model).
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
      if (stop_after && ops_run.size() >= stop_after) {
        fprintf(stderr, "[stop_after %zu] halt before %s %s\n", stop_after, n.op_type.c_str(),
                n.output.empty() ? "" : n.output[0].c_str());
        break;
      }
      ops_run.push_back(n.op_type);
      const std::string &op = n.op_type;
      if (!n.output.empty()) last_out_ = n.output[0];
      // tensors carrying extra fractional bits (the LSE softmax output) flow unchanged through
      // pure data movement and into a batched MatMul, which absorbs them in its requant; any
      // other consumer gets them requantized to 2^-P first.
      static const std::set<std::string> movers = {"Transpose", "Flatten", "Reshape", "Dropout",
                                                   "Identity", "Cast", "Unsqueeze", "Squeeze"};
      const bool mover = movers.count(op) > 0;
      if (!mover && op != "MatMul")
        for (const auto &in : n.input) settle(in);
      // commit any SMALL folded-constant input that flows into a compute op (mask/glue
      // constants); large weights are read directly from the initializer by their handlers.
      for (const auto &in : n.input)
        if (!in.empty() && !act_.count(in) && g.initializer.count(in)) {
          const auto &t = g.initializer.at(in);
          int64_t ne = 1;
          for (auto d : t.dims) ne *= d;
          if (ne <= 65536) ensure_act(in);
        }
      try {
        if (op == "Gemm")
          gemm(n);
        else if (op == "MatMul")
          matmul(n);
        else if (op == "Conv")
          conv(n);
        else if (op == "BatchNormalization")
          batchnorm(n);
        else if (op == "LayerNormalization")
          layernorm(n);
        else if (op == "Relu")
          relu(n);
        else if (op == "Sigmoid")
          table_act(n, sig_tab, sig_off, "Sigmoid");
        else if (op == "Gelu" || op == "Erf")
          table_act(n, gelu_tab, gelu_off, "Gelu");
        else if (op == "Tanh")
          table_act(n, tanh_tab, tanh_off, "Tanh");
        else if (op == "Softmax")
          softmax(n);
        else if (op == "MaxPool")
          pool(n, true);
        else if (op == "AveragePool")
          pool(n, false);
        else if (op == "GlobalAveragePool")
          global_avgpool(n);
        else if (op == "Add")
          add(n);
        else if (op == "Mul")
          mul(n);
        else if (op == "Div")
          div_(n);
        else if (op == "Gather")
          gather(n);
        else if (op == "Transpose")
          transpose(n);
        else if (op == "Split")
          split(n);
        else if (op == "Flatten" || op == "Reshape")
          reshape(n);
        else if (op == "Dropout" || op == "Identity")
          identity(n);
        else if (op == "Sub")
          sub(n);
        else if (op == "Where")
          where(n);
        else if (op == "IsNaN")
          isnan(n);
        else if (op == "And")
          and_(n);
        else if (op == "Concat")
          concat(n);
        else if (op == "Slice")
          slice(n);
        else if (op == "Expand")
          expand(n);
        else if (op == "Cast" || op == "Unsqueeze" || op == "Squeeze")
          passthru(n);
        else
          throw std::runtime_error("unsupported operator: '" + op + "' (node '" + n.name +
                                   "'); aborting");
      } catch (const std::exception &e) {
        fprintf(stderr, "FAIL node %s '%s' in=[", op.c_str(), n.name.c_str());
        for (auto &i : n.input) fprintf(stderr, "%s,", i.c_str());
        fprintf(stderr, "]: %s\n", e.what());
        throw;
      }
      if (mover && !n.input.empty() && xbits_.count(n.input[0]))
        for (const auto &o : n.output) xbits_[o] = xbits_.at(n.input[0]);
      if (stream)
        for (const auto &in : n.input)
          if (!in.empty() && last_use[in] == ni && !keep.count(in) && !ctab_.count(in) &&
              !wteT_src_.count(in)) {
            act_.erase(in);
            ref_.erase(in);
            shp_.erase(in);
            xbits_.erase(in);
            if (free_init)
              const_cast<OnnxGraph &>(*g_).initializer.erase(
                  in); // free the fp32 weight (graph is caller-owned non-const)
          }
    }
    qi->ipp->batch_check();   // flush deferred IP-matmul claims (IP_DEFER) before reveals/teardowns/finalize
    const std::string ok = act_.count(g.outputs.empty() ? "" : g.outputs[0])
                               ? g.outputs[0]
                               : last_out_; // STOP_AFTER may halt early
    if (ok.empty() || !act_.count(ok)) throw std::runtime_error("graph output not produced");
    out_ref = ref_.count(ok) ? ref_[ok] : std::vector<int64_t>{};
    return act_[ok];
  }

private:
  std::string last_out_;
  const OnnxGraph *g_ = nullptr;
  // Shared committed weight table (LM head wte): committed ONCE flat [V*D], consumed as rows by
  // the IP-proof matmul (transpose-free). logit[s][v] = <wte_row_v, hidden_s>.
  std::map<std::string, std::vector<IntFp>> ctab_;
  std::map<std::string, std::vector<int64_t>> ctab_ref_;
  std::map<std::string, std::pair<int, int>> ctab_dim_; // name -> (V, D)
  std::map<std::string, std::string> wteT_src_; // transposed-handle -> underlying table name
  std::map<std::string, std::vector<IntFp>> act_;
  std::map<std::string, int> xbits_; // tensors carrying extra fractional bits (see xbits())
  bool lse_slack_reported_ = false;
  std::map<std::string, std::vector<int>> shp_;
  std::map<std::string, std::vector<int64_t>> ref_;

  IntFp zero() {
    return IntFp((uint64_t)0, PUBLIC);
  }
  static int attr1(const OnnxNode &n, const char *k, int dflt) {
    auto it = n.attr.find(k);
    return (it != n.attr.end() && !it->second.ints.empty()) ? (int)it->second.ints[0] : dflt;
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
  // FixedInfer::requant_cm's rounding (half up); D is a power of two
  static int64_t rhu(int64_t a, int64_t D) { return (a + D / 2) >> __builtin_ctzll((uint64_t)D); }
  int64_t qz(double v) {
    return (int64_t)llround(v * (double)(1LL << P));
  }
  // A folded constant flowing into an activation op (mask / glue), quantized like qz but clamped
  // to half the exp-table domain (2^(score_bits-1), real 256 at the default score_bits = P+9) so a
  // mask's -inf / finfo.min stays representable and inside the table.  A fixed 2^15 was real 256
  // at P=7 but only real 8 at P=12, where exp(-8) would leak attention to future tokens.
  int64_t qz_const(double v) const {
    const int64_t CL = (int64_t)1 << (score_bits - 1);
    v *= (double)(1LL << P);
    if (!std::isfinite(v)) return v < 0 ? -CL : CL;
    return (int64_t)std::max(-(double)CL, std::min((double)CL, (double)llround(v)));
  }

  // ---- Gemm: y = x @ W^T + b ----
  void gemm(const OnnxNode &n) {
    if (n.attr.count("alpha") && n.attr.at("alpha").f != 1.0f)
      throw std::runtime_error("Gemm alpha != 1");
    if (n.attr.count("beta") && n.attr.at("beta").f != 1.0f)
      throw std::runtime_error("Gemm beta != 1");
    const auto &x = act_.at(n.input[0]);
    const OnnxTensor &W = g_->initializer.at(n.input[1]);
    const int transB = n.attr.count("transB") ? (int)n.attr.at("transB").i : 0;
    int out = transB ? (int)W.dims[0] : (int)W.dims[1],
        in = transB ? (int)W.dims[1] : (int)W.dims[0];
    std::vector<int64_t> qb(out, 0);
    if (n.input.size() > 2 && g_->initializer.count(n.input[2])) {
      const OnnxTensor &B = g_->initializer.at(n.input[2]);
      for (int o = 0; o < out; ++o) qb[o] = qz(B.data[o]);
    }
    linear(n.output[0], x, ref_.count(n.input[0]) ? &ref_.at(n.input[0]) : nullptr, W, transB != 0,
           qb, out, in);
  }
  // ---- MatMul: weight matrix (init), or two activations (attention) ----
  // LM head over the shared committed wte: logit[s][v] = <wte_row_v, hidden_s> via the
  // QuickSilver flat IP-proof (one batched check per row, threaded) -- no wte^T, no naive loop.
  void matmul_wte(const OnnxNode &n) {
    const std::string &xin = n.input[0];
    const std::string &tn = wteT_src_.at(n.input[1]);
    const auto &xf = act_.at(xin);
    const int V = ctab_dim_.at(tn).first, D = ctab_dim_.at(tn).second;
    const auto &c = ctab_.at(tn);
    const auto &cr = ctab_ref_.at(tn);
    const int R = D > 0 ? (int)(xf.size() / D) : 1;
    bool hr = party == ALICE && ref_.count(xin);
    const auto *xr = hr ? &ref_.at(xin) : nullptr;
    std::vector<IntFp> y((size_t)R * V);
    std::vector<int64_t> r;
    if (hr) r.resize((size_t)R * V);
    for (int rr = 0; rr < R; ++rr) {
      std::vector<IntFp> xrow(xf.begin() + (int64_t)rr * D, xf.begin() + (int64_t)(rr + 1) * D),
          outr;
      qi->ipp->matmul(outr, c, (size_t)V, (size_t)D, xrow,
                      qi->mat_threads); // outr[v] = <wte_row_v, hidden_rr> @ 2^-2P
      std::vector<IntFp> rq;
      qi->requant_cm(rq, outr, 1, qi->precision); // -> 2^-P
      for (int v = 0; v < V; ++v) {
        y[(int64_t)rr * V + v] = rq[v];
        if (hr) {
          int64_t a = 0;
          for (int d = 0; d < D; ++d) a += (*xr)[(int64_t)rr * D + d] * cr[(int64_t)v * D + d];
          r[(int64_t)rr * V + v] = rhu(a, 1LL << P);
        }
      }
    }
    std::vector<int> ns = shp_.at(xin);
    if (!ns.empty())
      ns.back() = V;
    else
      ns = {V};
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = ns;
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void matmul(const OnnxNode &n) {
    if (wteT_src_.count(n.input[1]) || g_->initializer.count(n.input[1]))
      for (const auto &in : n.input) settle(in);
    if (wteT_src_.count(n.input[1])) {
      matmul_wte(n);
      return;
    }                                        // LM head over shared committed wte
    if (g_->initializer.count(n.input[1])) { // A @ W  (W initializer [in,out])
      const auto &x = act_.at(n.input[0]);
      const OnnxTensor &W = g_->initializer.at(n.input[1]);
      linear(n.output[0], x, ref_.count(n.input[0]) ? &ref_.at(n.input[0]) : nullptr, W, false,
             std::vector<int64_t>((int)W.dims[1], 0), (int)W.dims[1], (int)W.dims[0]);
      return;
    }
    // [G,M,K] @ [G,K,N] (or [M,K]@[K,N]): batched committed matmul (per-head Q.K^T, attn.V)
    const auto &A = act_.at(n.input[0]);
    const auto &sA = shp_.at(n.input[0]);
    const auto &Bv = act_.at(n.input[1]);
    const auto &sB = shp_.at(n.input[1]);
    const int K = sA.back(), M = sA.size() >= 2 ? sA[sA.size() - 2] : 1, N = sB.back();
    int G = 1;
    for (size_t i = 0; i + 2 < sA.size(); ++i) G *= sA[i]; // leading batch dims of A
    const bool Bb = sB.size() >= 3;                        // B carries its own batch
    // Each score = ONE QuickSilver inner-product proof (dot_ip) instead of K naive committed
    // multiplies -> K-fold less mult-check communication (the seq^2 attention is the comm hog).
    // dot_ip batches into the global IP buffer (few network rounds); tiny reused K-length temps,
    // so no per-(g) allocation blow-up.  Numerically identical to the naive Sum_k A*B accumulator.
    // Threaded QuickSilver IP matmul on the flat activations (no nested-vector copies):
    // raw[g,m,j] = <A[g,m,:], B[g,:,j]>, one batched IP entry per score, one batch_check per head.
    std::vector<IntFp> raw;
    qi->ipp->matmul_attn(raw, A.data(), Bv.data(), G, M, N, K, Bb, qi->mat_threads);
    std::vector<IntFp> out;
    const int X = xbits(n.input[0]) + xbits(n.input[1]); // e.g. LSE probs @ V
    qi->requant_cm(out, raw, 1, qi->precision + X);     // 2^-(2P+X) -> 2^-P
    std::vector<int> ns;
    if (sA.size() >= 3) {
      ns = sA;
      ns.back() = N;
    } else if (M > 1)
      ns = {M, N};
    else
      ns = {N};
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = ns;
    if (party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1])) {
      const auto &ar = ref_.at(n.input[0]), &br = ref_.at(n.input[1]);
      std::vector<int64_t> r((size_t)G * M * N);
      const int64_t D = 1LL << (P + X);
      for (int g = 0; g < G; ++g)
        for (int m = 0; m < M; ++m)
          for (int j = 0; j < N; ++j) {
            int64_t a = 0;
            for (int k = 0; k < K; ++k)
              a += ar[((int64_t)g * M + m) * K + k] *
                   br[(Bb ? (int64_t)g * K * N : 0) + (int64_t)k * N + j];
            r[((int64_t)g * M + m) * N + j] = rhu(a, D);
          }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- Gather: select rows of a (committed) table at public indices (embeddings) ----
  void gather(const OnnxNode &n) {
    if (!g_->initializer.count(n.input[1]))
      throw std::runtime_error("Gather with non-constant indices not supported");
    const auto &idx = g_->initializer.at(n.input[1]).i64;
    const int S = (int)idx.size();
    if (g_->initializer.count(
            n.input[0])) { // EMBEDDING gather: constant table, axis 0 (GPT-2 path -- unchanged)
      const int axis = n.attr.count("axis") ? (int)n.attr.at("axis").i : 0;
      if (axis != 0) throw std::runtime_error("Gather (table) axis != 0 not supported");
      const OnnxTensor &data = g_->initializer.at(n.input[0]);
      const int D = (int)(data.dims.size() > 1 ? data.dims[1] : 1);
      std::vector<int64_t> q((size_t)S * D);
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d) q[(int64_t)s * D + d] = qz(data.data[(int64_t)idx[s] * D + d]);
      std::vector<IntFp> out;
      qi->commit_vec(out, q);
      act_[n.output[0]] = std::move(out);
      shp_[n.output[0]] = {S, D};
      if (party == ALICE) ref_[n.output[0]] = std::move(q);
    } else { // ACTIVATION gather along `axis` (BERT pooler: select [CLS] at axis 1)
      const auto &A = act_.at(n.input[0]);
      const auto &sA = shp_.at(n.input[0]);
      int ax = n.attr.count("axis") ? (int)n.attr.at("axis").i : 0;
      if (ax < 0) ax += (int)sA.size();
      // ONNX axes count the leading batch dim, but activations may be stored batch-squeezed
      // ([seq,hid] for a [1,seq,hid] tensor -- e.g. the plain-MatMul BERT export, whose LN out
      // is 2-D here; the Gemm+Reshape export keeps [1,seq,hid]). If the stored shape has no
      // leading 1, shift the axis down -- else axis=1 gathers along HIDDEN and the pooler
      // silently emits a wrong-size (then empty) tail.
      if (ax > 0 && !sA.empty() && sA[0] != 1) --ax;
      int outer = 1;
      for (int i = 0; i < ax; ++i) outer *= sA[i];
      const int axdim = ax < (int)sA.size() ? sA[ax] : 1;
      int inner = 1;
      for (int i = ax + 1; i < (int)sA.size(); ++i) inner *= sA[i];
      const bool hr = party == ALICE && ref_.count(n.input[0]);
      const auto *Ar = hr ? &ref_.at(n.input[0]) : nullptr;
      std::vector<IntFp> out((size_t)outer * S * inner);
      std::vector<int64_t> rr;
      if (hr) rr.resize(out.size());
      for (int o = 0; o < outer; ++o)
        for (int s = 0; s < S; ++s)
          for (int in = 0; in < inner; ++in) {
            int64_t src = ((int64_t)o * axdim + idx[s]) * inner + in,
                    dst = ((int64_t)o * S + s) * inner + in;
            out[dst] = A[src];
            if (hr) rr[dst] = (*Ar)[src];
          }
      std::vector<int> ns;
      for (int i = 0; i < (int)sA.size(); ++i) ns.push_back(i == ax ? S : sA[i]);
      act_[n.output[0]] = std::move(out);
      shp_[n.output[0]] = ns;
      if (hr) ref_[n.output[0]] = std::move(rr);
    }
  }

  // ---- Transpose: permute dims (default reverse) ----
  void transpose(const OnnxNode &n) {
    const std::string &tin = n.input[0];
    if (!act_.count(tin) &&
        g_->initializer.count(tin)) { // transpose of a 2-D weight initializer (LM head wte^T)
      const OnnxTensor &t = g_->initializer.at(tin);
      if (t.dims.size() == 2) { // NO-OP: commit wte ONCE flat [V,D], consumed as rows by the IP
                                // matmul (transpose-free)
        if (!ctab_.count(tin)) {
          int V = (int)t.dims[0], D = (int)t.dims[1];
          std::vector<int64_t> q((size_t)V * D);
          for (size_t k = 0; k < q.size(); ++k) q[k] = qz(t.data[k]);
          std::vector<IntFp> c;
          qi->commit_vec(c, q);
          ctab_ref_[tin] = q;
          ctab_[tin] = std::move(c);
          ctab_dim_[tin] = {V, D};
        }
        wteT_src_[n.output[0]] = tin;
        shp_[n.output[0]] = {(int)t.dims[1], (int)t.dims[0]};
        return;
      }
      throw std::runtime_error("Transpose of non-2D weight initializer unsupported");
    }
    const auto &x = act_.at(tin);
    const auto &xs = shp_.at(tin);
    const int nd = (int)xs.size();
    std::vector<int> perm;
    auto it = n.attr.find("perm");
    if (it != n.attr.end() && !it->second.ints.empty())
      for (auto p : it->second.ints) perm.push_back((int)p);
    else
      for (int i = nd - 1; i >= 0; --i) perm.push_back(i);
    std::vector<int> ns(nd), ist(nd, 1), ost(nd, 1);
    for (int i = 0; i < nd; ++i) ns[i] = xs[perm[i]];
    for (int i = nd - 2; i >= 0; --i) {
      ist[i] = ist[i + 1] * xs[i + 1];
      ost[i] = ost[i + 1] * ns[i + 1];
    }
    const int total = prod(xs);
    bool hr = party == ALICE && ref_.count(n.input[0]);
    const auto *xr = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<IntFp> y(total);
    std::vector<int64_t> r;
    if (hr) r.resize(total);
    std::vector<int> idx(nd);
    for (int o = 0; o < total; ++o) {
      int rem = o, in = 0;
      for (int i = 0; i < nd; ++i) {
        idx[i] = rem / ost[i];
        rem %= ost[i];
      }
      for (int i = 0; i < nd; ++i) in += idx[i] * ist[perm[i]];
      y[o] = x[in];
      if (hr) r[o] = (*xr)[in];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = ns;
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- Split: equal split of the last axis into |outputs| parts (QKV) ----
  void split(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int nout = (int)n.output.size(), last = xs.back(), part = last / nout,
              outer = prod(xs) / last;
    bool hr = party == ALICE && ref_.count(n.input[0]);
    const auto *xr = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<int> ns = xs;
    ns.back() = part;
    for (int o = 0; o < nout; ++o) {
      std::vector<IntFp> y((size_t)outer * part);
      std::vector<int64_t> r;
      if (hr) r.resize((size_t)outer * part);
      for (int ro = 0; ro < outer; ++ro)
        for (int p = 0; p < part; ++p) {
          int64_t src = (int64_t)ro * last + o * part + p;
          y[(int64_t)ro * part + p] = x[src];
          if (hr) r[(int64_t)ro * part + p] = (*xr)[src];
        }
      act_[n.output[o]] = std::move(y);
      shp_[n.output[o]] = ns;
      if (hr) ref_[n.output[o]] = std::move(r);
    }
  }

  // ---- Mul by a constant (scalar / per-feature): scale, e.g. 1/sqrt(d_head) ----
  void mul(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    if (!g_->initializer.count(n.input[1]))
      throw std::runtime_error("Mul of two activations not supported");
    const OnnxTensor &T = g_->initializer.at(n.input[1]);
    const size_t nb = T.data.size();
    std::vector<int64_t> qm(nb);
    for (size_t i = 0; i < nb; ++i) qm[i] = qz(T.data[i]);
    std::vector<IntFp> cm;
    qi->commit_vec(cm, qm);
    std::vector<IntFp> raw(x.size());
    for (size_t i = 0; i < x.size(); ++i) raw[i] = x[i] * cm[nb == 1 ? 0 : i % nb];
    std::vector<IntFp> out;
    qi->requant_cm(out, raw, 1, qi->precision);
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r(xr.size());
      const int64_t D = 1LL << P;
      for (size_t i = 0; i < xr.size(); ++i) r[i] = rhu(xr[i] * qm[nb == 1 ? 0 : i % nb], D);
      ref_[n.output[0]] = std::move(r);
    }
  }
  // Div by a CONSTANT (BERT attention score / sqrt(d_head)): multiply by the reciprocal 1/const.
  void div_(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    if (!g_->initializer.count(n.input[1]))
      throw std::runtime_error("Div by an activation not supported");
    const OnnxTensor &T = g_->initializer.at(n.input[1]);
    const size_t nb = T.data.size();
    std::vector<int64_t> qm(nb);
    for (size_t i = 0; i < nb; ++i) qm[i] = qz(1.0 / T.data[i]); // reciprocal @ 2^-P
    std::vector<IntFp> cm;
    qi->commit_vec(cm, qm);
    std::vector<IntFp> raw(x.size());
    for (size_t i = 0; i < x.size(); ++i) raw[i] = x[i] * cm[nb == 1 ? 0 : i % nb];
    std::vector<IntFp> out;
    qi->requant_cm(out, raw, 1, qi->precision);
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r(xr.size());
      const int64_t D = 1LL << P;
      for (size_t i = 0; i < xr.size(); ++i) r[i] = rhu(xr[i] * qm[nb == 1 ? 0 : i % nb], D);
      ref_[n.output[0]] = std::move(r);
    }
  }
  // shared dense: per row of x (M = |x|/in rows), out[m][o] = requant(sum_i x*W) + b
  void linear(const std::string &oname, const std::vector<IntFp> &x, const std::vector<int64_t> *xr,
              const OnnxTensor &W, bool transB, const std::vector<int64_t> &qb, int out, int in) {
    const int M = (int)x.size() / in;
    std::vector<std::vector<int64_t>> qW(out, std::vector<int64_t>(in));
    for (int o = 0; o < out; ++o)
      for (int i = 0; i < in; ++i)
        qW[o][i] = qz(transB ? W.data[(int64_t)o * in + i] : W.data[(int64_t)i * out + o]);
    std::vector<std::vector<IntFp>> cW;
    qi->commit_mat(cW, qW);
    std::vector<IntFp> cb;
    qi->commit_vec(cb, qb);
    std::vector<IntFp> y((size_t)M * out);
    std::vector<std::vector<IntFp>> X((size_t)M);
    for (int m = 0; m < M; ++m)
      X[m].assign(x.begin() + (int64_t)m * in, x.begin() + (int64_t)(m + 1) * in);
    std::vector<std::vector<IntFp>> Y;
    qi->dense_layer_mm(Y, cW, cb, X); // ONE matmul_mm for ALL rows (was dense_layer per row -> per-row flush)
    for (int m = 0; m < M; ++m)
      for (int o = 0; o < out; ++o) y[(int64_t)m * out + o] = Y[m][o];
    act_[oname] = std::move(y);
    shp_[oname] = M > 1 ? std::vector<int>{M, out} : std::vector<int>{out};
    if (party == ALICE && xr) {
      std::vector<int64_t> r((size_t)M * out);
      const int64_t D = 1LL << P;
      for (int m = 0; m < M; ++m)
        for (int o = 0; o < out; ++o) {
          int64_t a = 0;
          for (int i = 0; i < in; ++i) a += qW[o][i] * (*xr)[(int64_t)m * in + i];
          r[(int64_t)m * out + o] = rhu(a, D) + qb[o];
        }
      ref_[oname] = std::move(r);
    }
  }

  void relu(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    std::vector<IntFp> y(x.size());
    for (size_t k = 0; k < x.size(); ++k) {
      IntFp ge = cmp->is_le(zero(), x[k], act_bits);
      y[k] = ge * x[k];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r(xr.size());
      for (size_t k = 0; k < xr.size(); ++k) r[k] = xr[k] > 0 ? xr[k] : 0;
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- Sigmoid / Gelu: clamp into the table domain (these saturate, so the
  //      clamp is exact at the ends) then public-function lookup ----
  void table_act(const OnnxNode &n, FixedTable *tab, int64_t off, const char *name) {
    if (!tab)
      throw std::runtime_error(std::string(name) + " requires a lookup table (not configured)");
    const auto &x = act_.at(n.input[0]);
    const int64_t T = (int64_t)tab->T;
    IntFp lo((uint64_t)(PR - (uint64_t)off), PUBLIC), hi((uint64_t)(T - 1 - off), PUBLIC),
        neg1((uint64_t)(PR - 1), PUBLIC), coff((uint64_t)off, PUBLIC);
    auto sel = [&](IntFp bit, IntFp a, IntFp b) { return b + bit * (a + b * neg1); }; // bit ? a : b
    std::vector<IntFp> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      IntFp v = x[i];
      v = sel(cmp->is_le(lo, v, act_bits), v, lo); // max(v, lo)
      v = sel(cmp->is_le(v, hi, act_bits), v, hi); // min(v, hi)
      if (party == ALICE) {
        uint64_t hv = (uint64_t)(VAL(v.value));
        int64_t sv = (hv > (uint64_t)(PR >> 1)) ? (int64_t)(hv - PR) : (int64_t)hv;
        int64_t idx = sv + off;
        if (idx < 0 || idx >= T)
          throw std::runtime_error(std::string(name) + " '" + n.name + "': table index " +
                                   std::to_string(idx) + " OOB [0," + std::to_string(T) +
                                   ") -- input " + std::to_string(sv) +
                                   " escaped clamp (act_bits=" + std::to_string(act_bits) + ")");
      }
      y[i] = tab->lookup(v + coff);
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    // table-based; exact reference omitted (the table IS the function)
  }

  // ---- Conv: im2col field matmul + requant + per-channel bias ----
  void conv(const OnnxNode &n) {
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int Cin = xs[0], H = xs[1], Wd = xs[2];
    const OnnxTensor &W = g_->initializer.at(n.input[1]);
    const int Cout = (int)W.dims[0], K = (int)W.dims[2], stride = attr1(n, "strides", 1),
              pad = attr1(n, "pads", 0);
    const int Ho = volefier::conv_out_dim(H, K, stride, pad),
              Wo = volefier::conv_out_dim(Wd, K, stride, pad), M = Cin * K * K, Pn = Ho * Wo;
    std::vector<std::vector<int64_t>> qK(Cout, std::vector<int64_t>(M));
    std::vector<std::vector<IntFp>> cK(Cout);
    for (int co = 0; co < Cout; ++co) {
      for (int m = 0; m < M; ++m) qK[co][m] = qz(W.data[(int64_t)co * M + m]);
      std::vector<IntFp> c;
      qi->commit_vec(c, qK[co]);
      cK[co] = c;
    }
    std::vector<int64_t> qb(Cout, 0);
    if (n.input.size() > 2 && g_->initializer.count(n.input[2])) {
      const OnnxTensor &B = g_->initializer.at(n.input[2]);
      for (int co = 0; co < Cout; ++co) qb[co] = qz(B.data[co]);
    }
    std::vector<IntFp> cb;
    qi->commit_vec(cb, qb);
    Tensor3 X(Cin, std::vector<std::vector<IntFp>>(H, std::vector<IntFp>(Wd)));
    for (int c = 0; c < Cin; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < Wd; ++w) X[c][h][w] = xf[(int64_t)c * H * Wd + h * Wd + w];
    FMap raw;
    volefier::conv2d<BoolIO>(*qi->ipp, raw, X, cK, Cin, H, Wd, Cout, K, stride, pad,
                                    qi->mat_threads);
    std::vector<IntFp> out((size_t)Cout * Pn);
    for (int co = 0; co < Cout; ++co) {
      std::vector<IntFp> rq;
      qi->requant(rq, raw[co]);
      for (int p = 0; p < Pn; ++p) out[(int64_t)co * Pn + p] = rq[p] + cb[co];
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {Cout, Ho, Wo};
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r((size_t)Cout * Pn);
      const int64_t D = 1LL << P;
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
            r[(int64_t)co * Pn + oh * Wo + ow] = rhu(a, D) + qb[co];
          }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- BatchNormalization (inference): fold to per-channel affine, then requant ----
  void batchnorm(const OnnxNode &n) {
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int C = xs[0], HW = prod(xs) / C;
    const OnnxTensor &sc = g_->initializer.at(n.input[1]), &be = g_->initializer.at(n.input[2]),
                     &mn = g_->initializer.at(n.input[3]), &vr = g_->initializer.at(n.input[4]);
    const double eps = n.attr.count("epsilon") ? n.attr.at("epsilon").f : 1e-5;
    std::vector<int64_t> qg(C), qb(C);
    for (int c = 0; c < C; ++c) {
      double g = sc.data[c] / std::sqrt(vr.data[c] + eps);
      qg[c] = qz(g);
      qb[c] = qz(be.data[c] - g * mn.data[c]);
    }
    std::vector<IntFp> cg;
    qi->commit_vec(cg, qg);
    std::vector<IntFp> cb;
    qi->commit_vec(cb, qb);
    std::vector<IntFp> out((size_t)C * HW);
    for (int c = 0; c < C; ++c) {
      std::vector<IntFp> prodv(HW);
      for (int p = 0; p < HW; ++p) prodv[p] = xf[(int64_t)c * HW + p] * cg[c];
      std::vector<IntFp> tr;
      qi->requant_cm(tr, prodv, 1, qi->precision);
      for (int p = 0; p < HW; ++p) out[(int64_t)c * HW + p] = tr[p] + cb[c];
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = xs;
    if (party == ALICE && ref_.count(n.input[0])) {
      const auto &xr = ref_.at(n.input[0]);
      std::vector<int64_t> r((size_t)C * HW);
      const int64_t D = 1LL << P;
      for (int c = 0; c < C; ++c)
        for (int p = 0; p < HW; ++p)
          r[(int64_t)c * HW + p] = rhu(xr[(int64_t)c * HW + p] * qg[c], D) + qb[c];
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- Max / Average pooling ----
  void pool(const OnnxNode &n, bool ismax) {
    if (!ismax && !divr) throw std::runtime_error("AveragePool requires Div (not configured)");
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int C = xs[0], H = xs[1], Wd = xs[2];
    const int K = attr1(n, "kernel_shape", 2), stride = attr1(n, "strides", K),
              pad = attr1(n, "pads", 0);
    const int Ho = volefier::conv_out_dim(H, K, stride, pad),
              Wo = volefier::conv_out_dim(Wd, K, stride, pad);
    std::vector<IntFp> out((size_t)C * Ho * Wo);
    bool haveref = party == ALICE && ref_.count(n.input[0]);
    std::vector<int64_t> r;
    if (haveref) r.resize((size_t)C * Ho * Wo);
    const auto *xr = haveref ? &ref_.at(n.input[0]) : nullptr;
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow) {
          std::vector<IntFp> win;
          IntFp sum = zero();
          int cnt = 0;
          int64_t mx = INT64_MIN, sm = 0;
          for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj) {
              int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
              if (ih >= 0 && ih < H && iw >= 0 && iw < Wd) {
                int64_t idx = (int64_t)c * H * Wd + ih * Wd + iw;
                win.push_back(xf[idx]);
                sum = sum + xf[idx];
                ++cnt;
                if (haveref) {
                  mx = std::max(mx, (*xr)[idx]);
                  sm += (*xr)[idx];
                }
              }
            }
          int64_t oidx = (int64_t)c * Ho * Wo + oh * Wo + ow;
          out[oidx] = ismax ? cmp->max_value(win, act_bits)
                            : divr->round_div(sum, (uint64_t)((uint64_t)cnt));
          if (haveref) r[oidx] = ismax ? mx : rdiv(sm, cnt);
        }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {C, Ho, Wo};
    if (haveref) ref_[n.output[0]] = std::move(r);
  }
  void global_avgpool(const OnnxNode &n) {
    if (!divr) throw std::runtime_error("GlobalAveragePool requires Div (not configured)");
    const auto &xf = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int C = xs[0], HW = prod(xs) / C;
    std::vector<IntFp> out(C);
    bool haveref = party == ALICE && ref_.count(n.input[0]);
    std::vector<int64_t> r;
    if (haveref) r.resize(C);
    const auto *xr = haveref ? &ref_.at(n.input[0]) : nullptr;
    for (int c = 0; c < C; ++c) {
      IntFp s = zero();
      int64_t sm = 0;
      for (int p = 0; p < HW; ++p) {
        s = s + xf[(int64_t)c * HW + p];
        if (haveref) sm += (*xr)[(int64_t)c * HW + p];
      }
      out[c] = divr->round_div(s, (uint64_t)((uint64_t)HW));
      if (haveref) r[c] = rdiv(sm, HW);
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {C, 1, 1};
    if (haveref) ref_[n.output[0]] = std::move(r);
  }

  // ---- Add: residual (equal size) or broadcast bias (per-channel block / per-feature tile) ----
  void add(const OnnxNode &n) {
    const auto &A = act_.at(n.input[0]);
    const auto &sA = shp_.at(n.input[0]);
    std::vector<IntFp> B;
    std::vector<int64_t> Br;
    std::vector<int> sB;
    bool haveref = party == ALICE && ref_.count(n.input[0]);
    if (g_->initializer.count(n.input[1])) {
      const OnnxTensor &T = g_->initializer.at(n.input[1]);
      std::vector<int64_t> q(T.data.size());
      for (size_t i = 0; i < T.data.size(); ++i) q[i] = qz_const(T.data[i]);
      qi->commit_vec(B, q);
      Br = q;
      sB = {(int)q.size()};
    } else {
      B = act_.at(n.input[1]);
      sB = shp_.at(n.input[1]);
      if (haveref && ref_.count(n.input[1]))
        Br = ref_.at(n.input[1]);
      else
        haveref = false;
    }
    // Broadcast to the LARGER operand so a [768] bias added to [16,768] (bias-first, BERT) tiles
    // over all rows.  When na>=nb (GPT-2/CNN/ResNet: activation is input[0]) big=sA and this
    // matches the old code.
    const size_t na = A.size(), nb = B.size(), nout = na >= nb ? na : nb;
    const std::vector<int> &big = na >= nb ? sA : sB;
    std::vector<IntFp> y(nout);
    std::vector<int64_t> r;
    if (haveref) r.resize(nout);
    const auto *Ar = haveref ? &ref_.at(n.input[0]) : nullptr;
    auto idx = [&](size_t i, size_t s) -> size_t { // broadcast an operand of size s into [0,nout)
      if (s == nout) return i;
      if (s == 1) return 0;
      if (!big.empty() && (int)s == big[0]) return i / (nout / s); // per-row/channel block
      return i % s;                                                // per-feature tile
    };
    for (size_t i = 0; i < nout; ++i) {
      size_t ia = idx(i, na), ib = idx(i, nb);
      y[i] = A[ia] + B[ib];
      if (haveref) r[i] = (*Ar)[ia] + Br[ib];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = big;
    if (haveref) ref_[n.output[0]] = std::move(r);
  }

  // ---- Softmax (over last dim) ----
  void softmax(const OnnxNode &n) {
    if (!exp_tab || !divr)
      throw std::runtime_error("Softmax requires exp table + Div (not configured)");
    const auto &x = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int d = xs.back(), rows = prod(xs) / d;
    FixedSoftmax sm(qi, cmp, exp_tab, divr, P, score_bits);
    sm.lse = lse_softmax;
    const int X = lse_softmax ? lse_extra_bits : 0;
    sm.out_bits = P + X;
    sm.max_slack = lse_softmax ? lse_max_slack : 0;
    if (lse_softmax && party == ALICE && !lse_slack_reported_) {
      lse_slack_reported_ = true;
      printf("[1] LSE softmax: probs at 2^-%d, normalization slack %.4f%% for rows of %d\n", P + X,
             100.0 * sm.lse_slack(d), d);
    }
    std::vector<IntFp> y(x.size());
    for (int ri = 0; ri < rows; ++ri) {
      std::vector<IntFp> row(x.begin() + (int64_t)ri * d, x.begin() + (int64_t)(ri + 1) * d);
      std::vector<IntFp> o = sm.forward(row);
      for (int j = 0; j < d; ++j) y[(int64_t)ri * d + j] = o[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = xs; // table/exp-based; exact ref omitted
    if (X) xbits_[n.output[0]] = X;
  }

  // extra fractional bits a tensor carries beyond 2^-P (0 for all but the LSE softmax output)
  int xbits(const std::string &nm) const {
    auto it = xbits_.find(nm);
    return it == xbits_.end() ? 0 : it->second;
  }
  // bring a tensor carrying extra bits back to 2^-P in place (non-MatMul consumers)
  void settle(const std::string &nm) {
    auto it = xbits_.find(nm);
    if (it == xbits_.end()) return;
    std::vector<IntFp> &v = act_.at(nm);
    std::vector<IntFp> out;
    qi->requant_cm(out, v, 1, it->second);
    v = std::move(out);
    xbits_.erase(it);
  }

  // ---- LayerNormalization (over last dim) ----
  void layernorm(const OnnxNode &n) {
    if (!divr || !rcmp)
      throw std::runtime_error("LayerNormalization requires Div + RangeCheckLU (not configured)");
    const auto &x = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    const int d = xs.back(), rows = prod(xs) / d;
    const OnnxTensor &G = g_->initializer.at(n.input[1]), &Bt = g_->initializer.at(n.input[2]);
    if ((int)G.data.size() < d)
      throw std::runtime_error("LayerNorm '" + n.name + "': d=" + std::to_string(d) +
                               " (input shape last-dim) > gamma size " +
                               std::to_string(G.data.size()) + " -- input shape mismatch");
    {
      std::string ss;
      for (int v : xs) ss += std::to_string(v) + ",";
      if (rows * d != (int)x.size())
        throw std::runtime_error("LayerNorm '" + n.name + "' in='" + n.input[0] + "': shape [" +
                                 ss + "] prod=" + std::to_string(rows * d) +
                                 " (d=" + std::to_string(d) + " rows=" + std::to_string(rows) +
                                 ") != actual size " + std::to_string(x.size()));
    }
    std::vector<int64_t> qg(d), qb(d);
    for (int i = 0; i < d; ++i) {
      qg[i] = qz(G.data[i]);
      qb[i] = qz(Bt.data[i]);
    }
    std::vector<IntFp> cg;
    qi->commit_vec(cg, qg);
    std::vector<IntFp> cb;
    qi->commit_vec(cb, qb);
    FixedLayerNorm ln(qi, rcmp, divr, P, d);
    std::vector<IntFp> y(x.size());
    for (int ri = 0; ri < rows; ++ri) {
      std::vector<IntFp> row(x.begin() + (int64_t)ri * d, x.begin() + (int64_t)(ri + 1) * d), o;
      ln.forward(o, row, cg, cb);
      for (int j = 0; j < d; ++j) y[(int64_t)ri * d + j] = o[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = xs; // sqrt-free; exact ref omitted
  }

  // ---- Flatten / Reshape: data unchanged, shape recomputed from the target (keep ALL dims,
  //      incl. a leading batch=1, so 4-D attention transposes get the right rank) ----
  void reshape(const OnnxNode &n) {
    act_[n.output[0]] = act_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) ref_[n.output[0]] = ref_.at(n.input[0]);
    const int64_t total = (int64_t)act_.at(n.input[0]).size();
    std::vector<int64_t> tgt;
    if (n.op_type == "Reshape" && n.input.size() > 1 && g_->initializer.count(n.input[1]))
      for (auto z : g_->initializer.at(n.input[1]).i64) tgt.push_back(z);
    if (tgt.empty()) {
      shp_[n.output[0]] = {(int)total};
      return;
    }
    const auto &is = shp_.at(n.input[0]);
    int64_t known = 1;
    int neg = -1;
    for (size_t i = 0; i < tgt.size(); ++i) {
      if (tgt[i] == -1)
        neg = (int)i;
      else {
        if (tgt[i] == 0) tgt[i] = (i < is.size() ? is[i] : 1);
        known *= tgt[i];
      }
    }
    if (neg >= 0 && known) tgt[neg] = total / known;
    std::vector<int> ns;
    for (auto d : tgt) ns.push_back((int)d);
    shp_[n.output[0]] = ns;
  }
  void identity(const OnnxNode &n) {
    act_[n.output[0]] = act_.at(n.input[0]);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0])) ref_[n.output[0]] = ref_.at(n.input[0]);
  }

  // ---- transformer glue ops (masked softmax + broadcasting), ported from ml-quant ----
  IntFp pub(int64_t v) {
    return IntFp(v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v), PUBLIC);
  }
  // Ensure a (small folded-constant) initializer is committed into act_: float -> scale 2^-P,
  // int -> raw; magnitude clamped so the causal mask's -inf stays inside the exp-table domain.
  const std::vector<IntFp> &ensure_act(const std::string &nm) {
    if (act_.count(nm)) return act_.at(nm);
    const OnnxTensor &t = g_->initializer.at(nm);
    size_t cnt = std::max(t.data.size(), t.i64.size());
    const int64_t CL = (int64_t)1 << (score_bits - 1);
    std::vector<int64_t> q(cnt);
    for (size_t i = 0; i < cnt; ++i)
      q[i] = t.dtype == 1 ? qz_const(t.data[i])
                          : std::max(-CL, std::min(CL, (int64_t)t.i64[i]));
    std::vector<IntFp> c;
    qi->commit_vec(c, q);
    shp_[nm] = std::vector<int>(t.dims.begin(), t.dims.end());
    if (party == ALICE) ref_[nm] = q;
    act_[nm] = std::move(c);
    return act_.at(nm);
  }
  void passthru(const OnnxNode &n) {
    identity(n);
  } // Cast / Unsqueeze / Squeeze: data unchanged

  void sub(const OnnxNode &n) {
    const auto &A = ensure_act(n.input[0]), &B = ensure_act(n.input[1]);
    size_t na = A.size(), nb = B.size();
    bool hr = party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1]);
    const auto *ar = hr ? &ref_.at(n.input[0]) : nullptr, *br = hr ? &ref_.at(n.input[1]) : nullptr;
    std::vector<IntFp> y(na);
    std::vector<int64_t> r;
    if (hr) r.resize(na);
    for (size_t i = 0; i < na; ++i) {
      size_t j = nb == na ? i : (nb == 1 ? 0 : i % nb);
      y[i] = A[i] + B[j] * pub(-1);
      if (hr) r[i] = (*ar)[i] - (*br)[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void where(const OnnxNode &n) {
    const auto &C = ensure_act(n.input[0]), &A = ensure_act(n.input[1]),
               &B = ensure_act(n.input[2]);
    size_t na = A.size(), nb = B.size(), nc = C.size(), no = std::max(std::max(na, nb), nc);
    const std::string &shapeop = na == no ? n.input[1] : (nb == no ? n.input[2] : n.input[0]);
    bool hr = party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1]) &&
              ref_.count(n.input[2]);
    const auto *cr = hr ? &ref_.at(n.input[0]) : nullptr, *ar = hr ? &ref_.at(n.input[1]) : nullptr,
               *br = hr ? &ref_.at(n.input[2]) : nullptr;
    std::vector<IntFp> y(no);
    std::vector<int64_t> r;
    if (hr) r.resize(no);
    for (size_t i = 0; i < no; ++i) {
      size_t ic = nc == no ? i : (nc == 1 ? 0 : i % nc), ia = na == no ? i : (na == 1 ? 0 : i % na),
             ib = nb == no ? i : (nb == 1 ? 0 : i % nb);
      y[i] = B[ib] + C[ic] * (A[ia] + B[ib] * pub(-1));
      if (hr) r[i] = (*cr)[ic] ? (*ar)[ia] : (*br)[ib];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(shapeop);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void isnan(const OnnxNode &n) { // committed inputs are never NaN -> committed zeros
    const auto &x = act_.at(n.input[0]);
    std::vector<int64_t> z(x.size(), 0);
    std::vector<IntFp> y;
    qi->commit_vec(y, z);
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE) ref_[n.output[0]] = std::move(z);
  }
  void and_(const OnnxNode &n) {
    const auto &A = ensure_act(n.input[0]), &B = ensure_act(n.input[1]);
    size_t nb = B.size();
    std::vector<IntFp> y(A.size());
    for (size_t i = 0; i < A.size(); ++i)
      y[i] = A[i] * B[nb == A.size() ? i : (nb == 1 ? 0 : i % nb)];
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    if (party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1])) {
      const auto &ar = ref_.at(n.input[0]), &br = ref_.at(n.input[1]);
      std::vector<int64_t> r(ar.size());
      for (size_t i = 0; i < ar.size(); ++i)
        r[i] = ar[i] && br[nb == ar.size() ? i : i % nb] ? 1 : 0;
      ref_[n.output[0]] = std::move(r);
    }
  }
  void concat(const OnnxNode &n) {
    int axis = n.attr.count("axis") ? (int)n.attr.at("axis").i : 0;
    const auto &s0 = shp_.at(n.input[0]);
    int nd = (int)s0.size();
    if (axis < 0) axis += nd;
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= s0[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < nd; ++i) inner *= s0[i];
    std::vector<int> ns = s0;
    int axsum = 0;
    std::vector<IntFp> y;
    std::vector<int64_t> r;
    bool hr = party == ALICE;
    for (auto &in : n.input) hr = hr && ref_.count(in);
    for (int64_t o = 0; o < outer; ++o)
      for (auto &in : n.input) {
        const auto &x = act_.at(in);
        int as = (int)shp_.at(in)[axis];
        for (int a = 0; a < as; ++a)
          for (int64_t k = 0; k < inner; ++k) {
            y.push_back(x[(o * as + a) * inner + k]);
            if (hr) r.push_back(ref_.at(in)[(o * as + a) * inner + k]);
          }
      }
    for (auto &in : n.input) axsum += (int)shp_.at(in)[axis];
    ns[axis] = axsum;
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = ns;
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void slice(const OnnxNode &n) {
    const auto &x = ensure_act(n.input[0]);   // materialize large folded constants (e.g. the seq^2 causal
    const auto &xs = shp_.at(n.input[0]);      // mask) on demand -- the pre-commit guard skips >65536 elems

    int nd = (int)xs.size();
    auto getv = [&](int i) {
      std::vector<int64_t> v;
      if (i < (int)n.input.size() && g_->initializer.count(n.input[i]))
        for (auto z : g_->initializer.at(n.input[i]).i64) v.push_back(z);
      return v;
    };
    auto starts = getv(1), ends = getv(2), axes = getv(3);
    if (axes.empty())
      for (int i = 0; i < (int)starts.size(); ++i) axes.push_back(i);
    std::vector<int64_t> lo(nd, 0), hi(xs.begin(), xs.end());
    for (size_t a = 0; a < axes.size(); ++a) {
      int ax = (int)(axes[a] < 0 ? axes[a] + nd : axes[a]);
      int64_t d = xs[ax];
      int64_t s = starts[a] < 0 ? starts[a] + d : starts[a],
              e = ends[a] < 0 ? ends[a] + d : ends[a];
      lo[ax] = std::max((int64_t)0, std::min(s, d));
      hi[ax] = std::max((int64_t)0, std::min(e, d));
    }
    std::vector<int64_t> ist(nd, 1);
    for (int i = nd - 2; i >= 0; --i) ist[i] = ist[i + 1] * xs[i + 1];
    std::vector<int> ns(nd);
    int64_t tot = 1;
    for (int i = 0; i < nd; ++i) {
      ns[i] = (int)(hi[i] - lo[i]);
      tot *= ns[i];
    }
    std::vector<int64_t> ost(nd, 1);
    for (int i = nd - 2; i >= 0; --i) ost[i] = ost[i + 1] * ns[i + 1];
    bool hr = party == ALICE && ref_.count(n.input[0]);
    const auto *xr = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<IntFp> y(tot);
    std::vector<int64_t> r;
    if (hr) r.resize(tot);
    std::vector<int> id(nd);
    for (int64_t o = 0; o < tot; ++o) {
      int64_t rem = o, src = 0;
      for (int i = 0; i < nd; ++i) {
        id[i] = (int)(rem / ost[i]);
        rem %= ost[i];
        src += (lo[i] + id[i]) * ist[i];
      }
      y[o] = x[src];
      if (hr) r[o] = (*xr)[src];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = ns;
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  void expand(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
    std::vector<int64_t> tgt;
    if (g_->initializer.count(n.input[1]))
      for (auto z : g_->initializer.at(n.input[1]).i64) tgt.push_back(z);
    int nd = (int)std::max(xs.size(), tgt.size());
    std::vector<int64_t> as(nd, 1), os(nd, 1);
    for (int i = 0; i < (int)xs.size(); ++i) as[nd - xs.size() + i] = xs[i];
    for (int i = 0; i < (int)tgt.size(); ++i) os[nd - tgt.size() + i] = tgt[i];
    for (int i = 0; i < nd; ++i) os[i] = std::max(as[i], os[i]);
    std::vector<int64_t> ast(nd), ost(nd);
    int64_t sa = 1, so = 1;
    for (int i = nd - 1; i >= 0; --i) {
      ast[i] = as[i] == 1 ? 0 : sa;
      sa *= as[i];
      ost[i] = so;
      so *= os[i];
    }
    bool hr = party == ALICE && ref_.count(n.input[0]);
    const auto *xr = hr ? &ref_.at(n.input[0]) : nullptr;
    std::vector<IntFp> y(so);
    std::vector<int64_t> r;
    if (hr) r.resize(so);
    for (int64_t o = 0; o < so; ++o) {
      int64_t rem = o, src = 0;
      for (int i = 0; i < nd; ++i) {
        int64_t idx = rem / ost[i];
        rem %= ost[i];
        src += idx * ast[i];
      }
      y[o] = x[src];
      if (hr) r[o] = (*xr)[src];
    }
    std::vector<int> ns(os.begin(), os.end());
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = ns;
    if (hr) ref_[n.output[0]] = std::move(r);
  }
};

} // namespace volefier
#endif // ML_FIXED_GRAPH_INFER_H
