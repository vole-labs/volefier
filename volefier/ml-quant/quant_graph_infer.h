#ifndef ML_QUANT_GRAPH_INFER_H
#define ML_QUANT_GRAPH_INFER_H

// Graph-driven prover for QDQ (QuantizeLinear/DequantizeLinear) int8 ONNX models
// on the ml-quant (per-tensor) backend. Each activation is tracked as a committed
// INTEGER quantized value plus its per-tensor (scale, zero-point). Dense-matmul and
// Conv weights are COMMITTED witness by default (private model; see commit_weights),
// so the verifier learns neither the model nor the activations; the per-tensor scales
// and zero-points remain public metadata. A QuantizeLinear is the
// per-tensor requant by c*2^e (scale_search + requant_cm, MSB-free LogUp). The
// integer arithmetic is deterministic, so it validates EXACTLY against a
// plaintext int64 reference.
//
// y_q = clamp( round( (x_s*w_s/y_s) * (sum x_q*w_q + bias_eff) ) + y_zp )
//   with bias_eff = bias_q - x_zp * sum(w_q)   (folds the asymmetric zero-point)

#include "volefier/io/onnx.h"
#include "volefier/ml-quant/quant_infer.h"
#include "volefier/ml-quant/quant_nonlinear.h" // QuantSoftmax / QuantActivation / QuantLayerNorm / QuantTable
#include "volefier/nonlinear/comparison.h"
#include "volefier/nonlinear/range_check.h"
#include "volefier/nonlinear/div.h"
#include "volefier/linear/conv2d.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace emp;

namespace volefier {

class QuantGraphInfer {
public:
  // Debug: dump a named tensor's DEQUANTIZED (ref*scale) values as float32, to compare intermediates
  // against an fp32/ORT reference (run with NO_STREAM=1 so ref_ survives).
  void tap(const std::string &name, const std::string &file) {
    if (shp_.count(name)) { fprintf(stderr, "[tap] shp[%s]=[", name.c_str()); for (int d : shp_.at(name)) fprintf(stderr, "%d,", d); fprintf(stderr, "] act=%zu\n", act_.count(name) ? act_.at(name).size() : 0); }
    if (!ref_.count(name)) { fprintf(stderr, "[tap] no ref for '%s'\n", name.c_str()); return; }
    const auto &r = ref_.at(name);
    std::ofstream f(file, std::ios::binary);
    if (rscale_.count(name) && !rscale_.at(name).empty()) { // per-token (per-row) scales
      const auto &rs = rscale_.at(name);
      const auto &sh = shp_.count(name) ? shp_.at(name) : std::vector<int>{};
      int d = sh.empty() ? (int)r.size() : sh.back();
      for (int i = 0; i < (int)r.size(); ++i) {
        int row = d > 0 ? i / d : 0;
        double s = rs[std::min((int)rs.size() - 1, row)];
        float fv = (float)((double)r[i] * s); f.write((char *)&fv, 4);
      }
      fprintf(stderr, "[tap] %s: %zu vals PER-ROW(%zu scales) -> %s\n", name.c_str(), r.size(), rs.size(), file.c_str());
    } else {
      const double s = scale_.count(name) ? scale_.at(name) : 1.0;
      for (int64_t v : r) { float fv = (float)((double)v * s); f.write((char *)&fv, 4); }
      fprintf(stderr, "[tap] %s: %zu vals scale=%.3e -> %s\n", name.c_str(), r.size(), s, file.c_str());
    }
  }
  QuantInfer *qi;
  CompareLU *cmp;
  Div *divr;
  int party;
  // optional transformer components (set before run() to enable transformer ops)
  RangeCheckLU *rcmp = nullptr;
  QuantTable *exp_tab = nullptr, *gelu_tab = nullptr, *tanh_tab = nullptr, *silu_tab = nullptr;
  QuantTable *clamp_tab = nullptr;
  int64_t clamp_off =
      0; // optional: QuantizeLinear [0,255] saturation as ONE LogUp lookup (DeepProve-style)
  int P = 12;
  int64_t gelu_off = 0, tanh_off = 0, silu_off = 0;
  std::size_t score_bits = 16;
  bool uniform_requant = false; // fp32 path (no Q/DQ): requant each matmul output back to 2^-P so
                                // scales don't compound
  bool commit_weights = true; // PRIVATE MODEL (default): dense-matmul weights are committed witness
                              // (committed*committed). Set false for PUBLIC MODEL: weights are
                              // pub() constants (committed*public = free linear).
  int weight_max =
      511; // per-tensor symmetric weight range (int10). int8 (=127) is too lossy past ~6
           // transformer layers; int10 recovers fp32 (validated in plaintext_int8_gpt2) with
           // a smaller accumulator than int12 (int10 suffices here).
  int sym_act_bits =
      0; // 0 = uniform 2^-P fixed-point activation bus; >0 = DeepProve-style
         // SYMMETRIC int-N activation quantization (zp=0, BIT_LEN=N, typically 12).
  std::map<std::string, std::vector<double>>
      rscale_; // PER-TOKEN (per-row) activation scales when sym_act_bits>0
               // (handles per-position outliers -> 99% vs 87% for per-tensor; per-channel weight
               // stays per-tensor).
  // STATIC calibration (DeepProve-style): fixed per-tensor scales read from a calibration file
  // instead of the DYNAMIC per-token max.  calib_ non-empty => use the fixed scale for any node
  // present (no runtime scale_search on a dynamic value).  calib_write_ => ALICE records each
  // node's per-tensor real max so the file can be generated.  With static scales the prover and the
  // (same-scale) plaintext mimic are bit-exact.
  std::map<std::string, double> calib_; // node -> fixed per-tensor activation scale (static)
  std::map<std::string, double>
      calib_max_; // node -> accumulated per-tensor real max (calibration write)
  bool calib_write_ = false;
  // DeepProve-style scale-baked FUSED nonlinear lookups: per-op tables whose input is the
  // PER-TENSOR int (no 2^-P rescale) and whose output is the activation requanted to s_out +
  // clamped -- one lookup does activation + requant + clamp.  Built by the driver from the calib
  // scales; keyed by the node OUTPUT name.
  std::map<std::string, QuantTable *> gelu_tabs_;
  std::map<std::string, std::pair<double, double>>
      gelu_scales_; // (s_in, s_out) per fused-gelu op (for the ref)
  int64_t gelu_fused_off = 0;
  std::map<std::string, QuantTable *>
      tanh_tabs_; // BERT pooler tanh: asymmetric-uint8 scale-baked fused lookup
  std::map<std::string, std::array<double, 4>>
      tanh_scales_;           // {s_in, zp_in, s_out, zp_out} per fused-tanh op (for the ref)
  int64_t tanh_fused_off = 0; // 0: the committed uint8 q is the index directly
  std::map<std::string, QuantTable *>
      gelu_asym_tabs_; // BERT gelu: asymmetric-uint8 scale-baked fused lookup (vs GPT-2's symmetric
                       // gelu_tabs_)
  std::map<std::string, std::array<double, 4>>
      gelu_asym_scales_; // {s_in, zp_in, s_out, zp_out} per fused-gelu op (for the ref)
  std::vector<std::string> ops_run;
  std::string last_out_;
  std::vector<int> lmhead_pred_;   // STREAM_LMHEAD: per-position argmax (predicted token ids)
  bool streamed_lmhead_ = false;   // set when lm_head_stream_argmax ran (test reads lmhead_pred_)

  QuantGraphInfer(QuantInfer *qi_, CompareLU *cmp_, Div *divr_, int party_)
      : qi(qi_), cmp(cmp_), divr(divr_), party(party_) {}

  // DeepProve-style SYMMETRIC int-N activation quantization (zp=0).  Quantize a committed op output
  // (accumulator scale s_acc, ALICE plaintext ref) to int-N with a PER-TENSOR power-of-2 scale 2^k,
  // k = ceil(log2(max|real|/(2^(N-1)-1))) computed by ALICE and broadcast to BOB.  No zero-point ->
  // no -zp*Sw correction (the entire asymmetric-uint8 bug class disappears).  k is chosen so the
  // quantized value fits in [-2^(N-1), 2^(N-1)-1] exactly (dynamic per-tensor scale), so NO clamp.
  // PER-TOKEN (per-row): each row r gets its own ARBITRARY scale s_out[r] = (max|int| in
  // row)*s_in[r]/LIM, broadcast via the per-row max int (committed reveal).  Per-token handles
  // GPT-2's per-position outliers (99% vs 87% per-tensor).  Returns per-row out scales; ref
  // requanted identically (round-half-up).
  void sym_quant_act(std::vector<IntFp> &y, std::vector<int64_t> *ref,
                     const std::vector<double> &sin, int d, std::vector<double> &sout,
                     bool pt_out = false, double fixed = -1) {
    const int64_t LIM = ((int64_t)1 << (sym_act_bits - 1)) - 1;
    if (d <= 0) d = (int)y.size();
    int rows = d > 0 ? (int)(y.size() / d) : 1;
    if (rows < 1) rows = 1;
    auto sinr = [&](int r) { return sin.empty() ? 1.0 : sin[std::min((int)sin.size() - 1, r)]; };
    // STATIC: a fixed per-tensor scale (from the calibration file) -> no per-row max, no reveal, no
    // dynamic scale_search. One public scale for the whole tensor; the (same-scale) mimic requants
    // identically.
    const bool stat = fixed > 0;
    std::vector<int64_t> mxis;
    if (!stat) {
      mxis.assign(rows, 0);
      for (int r = 0; r < rows; ++r) {
        int64_t mxi = 0;
        if (party == ALICE && ref)
          for (int j = 0; j < d; ++j) {
            int64_t a = std::llabs((*ref)[(size_t)r * d + j]);
            if (a > mxi) mxi = a;
          }
        mxis[r] = (int64_t)IntFp((int64_t)mxi, ALICE).reveal();
        if (mxis[r] < 1) mxis[r] = 1;
      }
    }                  // broadcast each row's max int
    double s_glob = 0; // pt_out: ONE scale (attention region feeding shape ops); else per-token
    if (stat)
      s_glob = fixed;
    else if (pt_out) {
      for (int r = 0; r < rows; ++r) {
        double g = (double)mxis[r] * sinr(r);
        if (g > s_glob) s_glob = g;
      }
      s_glob /= (double)LIM;
    }
    sout.assign((pt_out || stat) ? 1 : rows, s_glob);
    for (int r = 0; r < rows; ++r) {
      double s_in = sinr(r),
             s_o = stat ? fixed : (pt_out ? s_glob : s_in * (double)mxis[r] / (double)LIM);
      if (!pt_out && !stat) sout[r] = s_o;
      auto ce = QuantInfer::scale_search(s_in / s_o);
      int64_t c = ce.first;
      int e = ce.second;
      // BATCHED requant of this row's d elements (single c,-e): 2 batch_feeds + 1 batched
      // zero-check vs 2*d per-element feeds + d reveals.  In-place safe (acc read into Nb before
      // out is written).  This is the dominant prover cost -- every op's output is requanted here.
      qi->requant_cm_batch(&y[(size_t)r * d], &y[(size_t)r * d], (uint64_t)c, (std::size_t)(-e),
                           (size_t)d);
      if (party == ALICE && ref)
        for (int j = 0; j < d; ++j) {
          size_t idx = (size_t)r * d + j;
          int64_t N = c * (*ref)[idx];
          (*ref)[idx] =
              (e > 0) ? (N << e) : (e == 0 ? N : ((N + ((int64_t)1 << (-e - 1))) >> (-e)));
        }
    }
  }

  static bool supported(const std::string &op) {
    static const std::set<std::string> ok = {"QuantizeLinear",
                                             "DequantizeLinear",
                                             "Conv",
                                             "MatMul",
                                             "Gemm",
                                             "Relu",
                                             "MaxPool",
                                             "GlobalAveragePool",
                                             "Add",
                                             "Flatten",
                                             "Reshape",
                                             "Dropout",
                                             "Identity",
                                             "Softmax",
                                             "LayerNormalization",
                                             "RMSNormalization",
                                             "RotaryEmbedding",
                                             "Gelu",
                                             "SiLU",
                                             "Erf",
                                             "Tanh",
                                             "Transpose",
                                             "Split",
                                             "Gather",
                                             "Mul",
                                             "Div",
                                             "Concat",
                                             "Slice",
                                             "Cast",
                                             "Where",
                                             "Unsqueeze",
                                             "Squeeze",
                                             "Expand",
                                             "Sub",
                                             "And",
                                             "IsNaN"};
    return ok.count(op) > 0;
  }
  static void check_ops(const OnnxGraph &g) {
    for (const auto &n : g.nodes)
      if (!supported(n.op_type))
        throw std::runtime_error("ml-quant: unsupported op '" + n.op_type + "'");
  }

  // input is the committed uint8 image (values 0..255); input_ref the same int64.
  // W_PRECOMMIT=1: quantize + COMMIT all dense weights up front, before the node loop.  The
  // ~0.7 GB of weight-commit corrections then stream (ASYNC_SEND) through the run's opening
  // compute-heavy stretch, which otherwise leaves the NIC idle -- the reservoir starts empty
  // and no ASYNC_CAP can bank bytes that don't exist yet.  Same commitments, same order-
  // independence w.r.t. every later challenge; only the schedule moves.  Keyed per NODE (the
  // handler's widx layout depends on transB); biases stay in the handler (input-scale-dependent).
  std::map<std::string, std::vector<std::vector<IntFp>>> wpre_;
  void precommit_weights(const OnnxGraph &g) {
    if (!commit_weights || !getenv("W_PRECOMMIT")) return;
    for (const auto &n : g.nodes) {
      if (n.op_type != "Gemm" && n.op_type != "MatMul") continue;
      if (n.input.size() < 2 || !g.initializer.count(n.input[1])) continue;
      if (wteT_src_.count(n.input[1])) continue;                      // LM head path commits its own table
      if (n.input[1].find("hadamard") != std::string::npos) continue; // public rotation
      const auto &Wt = g.initializer.at(n.input[1]);
      if (Wt.data.empty() || wpre_.count(n.name)) continue;
      if (!cref_.count(n.input[1])) {   // twin of the matmul() handler's quantize block
        double mxw = 1e-12;
        for (float v : Wt.data) { double a = v < 0 ? -(double)v : (double)v; if (a > mxw) mxw = a; }
        const int64_t WM = weight_max;
        const double swq = mxw / (double)WM;
        std::vector<int64_t> q(Wt.data.size());
        for (size_t i = 0; i < Wt.data.size(); ++i)
          q[i] = std::max(-WM, std::min(WM, (int64_t)llround((double)Wt.data[i] / swq)));
        cref_[n.input[1]] = std::move(q);
        scale_[n.input[1]] = swq;
        shp_[n.input[1]] = std::vector<int>(Wt.dims.begin(), Wt.dims.end());
      }
      const auto &W = cref_.at(n.input[1]);
      const auto &wsh = shp_.at(n.input[1]);
      const bool gemm = n.op_type == "Gemm";
      const int tb = gemm && n.attr.count("transB") ? (int)n.attr.at("transB").i : 0;
      const int out = tb ? wsh[0] : wsh[1], in = tb ? wsh[1] : wsh[0];
      std::vector<std::vector<IntFp>> Wrows((size_t)out, std::vector<IntFp>((size_t)in));
      std::vector<uint64_t> wraw((size_t)in, 0);
      for (int o = 0; o < out; ++o) {
        if (party == ALICE)
          for (int i = 0; i < in; ++i) { int64_t v = W[tb ? (int64_t)o * in + i : (int64_t)i * out + o];
            wraw[(size_t)i] = v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v); }
        batch_feed(Wrows[(size_t)o].data(), wraw.data(), (int64_t)in);
      }
      wpre_[n.name] = std::move(Wrows);
    }
  }

  std::vector<IntFp> run(const OnnxGraph &g, const std::vector<IntFp> &input,
                         const std::vector<int64_t> &input_ref, double in_scale, int in_zp,
                         std::vector<int64_t> &out_ref) {
    g_ = &g;
    if (getenv("DBG_OPS")) fprintf(stderr, "[pre] enter run: inputs=%zu nodes=%zu init=%zu\n", g.inputs.size(), g.nodes.size(), g.initializer.size());
    // TIED-EMBEDDING DEDUP: a fused LLM graph carries the embedding twice -- once as the Gather table
    // [V,D] and once as the LM-head weight [D,V] (== embed^T).  Committing both doubles the (huge) embed
    // footprint (Gemma: 2x 2.7 GB committed + 2x 1.3 GB plaintext).  If the LM-head weight is exactly the
    // transpose of a Gather'd embedding, mark it as a wte-transpose handle: the LM head then reuses the
    // Gather's single committed table (matmul_wte), and precommit_weights auto-skips it (wteT_src_ guard).
    if (!getenv("NO_TIE_WTE")) {
      std::set<std::string> gather_tabs; // embedding initializers indexed by a Gather
      for (const auto &nn : g.nodes)
        if (nn.op_type == "Gather" && nn.input.size() >= 1 && g.initializer.count(nn.input[0]))
          gather_tabs.insert(nn.input[0]);
      for (const auto &nn : g.nodes) {
        if (nn.op_type != "MatMul" && nn.op_type != "Gemm") continue;
        if (nn.input.size() < 2 || !g.initializer.count(nn.input[1]) || wteT_src_.count(nn.input[1])) continue;
        const auto &W = g.initializer.at(nn.input[1]);
        // under LAZY_INIT the fp32 is still on disk (data empty); sample via peek_ext_f32 so tie
        // detection never forces a full decode of the (multi-GB) LM-head weight or embedding.
        if (W.dims.size() != 2 || (W.data.empty() && W.ext_path.empty())) continue;
        const int64_t Wd0 = W.dims[0], Wd1 = W.dims[1]; // LM-head weight [D, V]
        for (const auto &em : gather_tabs) {
          const auto &E = g.initializer.at(em);            // embed [V, D]
          if (E.dims.size() != 2 || (E.data.empty() && E.ext_path.empty())) continue;
          if (E.dims[0] != Wd1 || E.dims[1] != Wd0) continue; // shapes must be transpose-compatible
          bool tied = true;                                 // sample-verify W[d,v] == E[v,d]
          const int64_t V = E.dims[0], D = E.dims[1];
          for (int64_t t = 0; t < 4096 && tied; ++t) {
            int64_t v = (t * 2654435761u) % V, d = (t * 40503u) % D;
            if (peek_ext_f32(W, (int64_t)d * V + v) != peek_ext_f32(E, (int64_t)v * D + d)) tied = false;
          }
          if (tied) {
            wteT_src_[nn.input[1]] = em;
            shp_[nn.input[1]] = {(int)Wd0, (int)Wd1};
            scale_[nn.input[1]] = 1.0 / (double)(1LL << P);
            if (getenv("DBG_OPS")) fprintf(stderr, "[pre] tied LM head %s -> shared table %s\n", nn.input[1].c_str(), em.c_str());
            break;
          }
        }
      }
    }
    precommit_weights(g);
    if (getenv("DBG_OPS")) fprintf(stderr, "[pre] after precommit_weights\n");
    // Point comparison's bitness g(g-1)=0 checks at the shared process-global Hadamard2IP
    // (created by QuantInfer, flushed in qi->destroy()); one RLC challenge for the whole batch.
    if (cmp)
      cmp->h2ip = volefier::global_h2ip();
    const std::string in0 = g.inputs[0];
    act_[in0] = input;
    ref_[in0] = input_ref;
    scale_[in0] = in_scale;
    zp_[in0] = in_zp;
    std::vector<int> s;
    auto it = g.input_shapes.find(in0);
    if (it != g.input_shapes.end())
      for (size_t i = (it->second.size() > 1 ? 1 : 0); i < it->second.size(); ++i)
        s.push_back((int)it->second[i]);
    shp_[in0] = s;

    const char *sa_env = getenv("STOP_AFTER");
    const size_t stop_after = sa_env ? (size_t)atoi(sa_env) : 0;
    // ---- weight/activation STREAMING: free each tensor's committed/quantized state after its
    // LAST consumer runs, so peak memory ~= a few layers, not the whole model. Without this,
    // cref_ (int weights) + act_ accumulate for the entire graph (OK for GPT-2, OOMs at Llama
    // scale).
    const bool stream = !getenv("NO_STREAM");
    const bool free_init =
        stream && getenv("FREE_INIT"); // also free the fp32 weight from the graph after its last
                                       // use (single-inference: mutates the caller's graph, can't
                                       // re-prove without reload)
    std::map<std::string, int> last_use;
    if (stream) {
      for (int ni = 0; ni < (int)g.nodes.size(); ++ni)
        for (const auto &in : g.nodes[ni].input)
          if (!in.empty()) last_use[in] = ni;
    }
    std::set<std::string> keep(g.outputs.begin(), g.outputs.end());
    // SYMMETRIC int-N: the attention SCORE region (QK^T scores -> mask -> softmax) stays at gadget
    // precision (DeepProve keeps softmax internals un-quantized); int-N-quantizing the masked
    // scores
    // (-256 mask) overflows the exp-LUT/score range.  Skip sym-quant for any tensor feeding a
    // Softmax.
    std::set<std::string> sm_skip;
    if (sym_act_bits > 0)
      for (const auto &nn : g.nodes)
        if (nn.op_type == "Softmax" || nn.op_type == "Where")
          for (const auto &in : nn.input) sm_skip.insert(in);
    // PER-TOKEN: activations feeding a SHAPE op (Reshape/Transpose/Split/Concat/Slice) get a SINGLE
    // per-tensor scale (the attention region) -- per-token's win is the residual+MLP, and this
    // avoids per-row scale propagation through row-permuting shape ops (validated:
    // attention-per-tensor == full per-token accuracy).
    std::set<std::string> pt_skip;
    if (sym_act_bits > 0)
      for (const auto &nn : g.nodes)
        if (nn.op_type == "Reshape" || nn.op_type == "Transpose" || nn.op_type == "Split" ||
            nn.op_type == "Concat" || nn.op_type == "Slice" || nn.op_type == "Unsqueeze" ||
            nn.op_type == "Expand")   // GQA repeat_kv (Unsqueeze->Expand->Reshape) also permutes rows
          for (const auto &in : nn.input) pt_skip.insert(in);
    // HAD_RESID: Hadamard-wrapped requant for the residual-stream norms (hidden-dim RMSNorm inputs,
    // NOT q/k-norm) -- for massive-activation models (Gemma). See hadamard_requant().
    const bool had_resid = getenv("HAD_RESID") != nullptr;
    std::set<std::string> hrot;
    if (had_resid && sym_act_bits > 0)
      for (const auto &nn : g.nodes)
        if (nn.op_type == "RMSNormalization" && !nn.input.empty() &&
            nn.name.find("q_norm") == std::string::npos && nn.name.find("k_norm") == std::string::npos)
          hrot.insert(nn.input[0]);
    const bool memtrack = getenv("MEMTRACK");
    size_t peak_live = 0;
    int ni = -1;
    const bool TM = getenv("TIMING");
    // ROUND_PROF: attribute NetIO round-trips to the op type that triggered them (same
    // close-previous-span accounting as TIMING, so requant/refill rounds between dispatches
    // bill to the producing op). Includes inline sVOLE refills in non-BG mode.
    const bool RP = getenv("ROUND_PROF") != nullptr;
    emp::NetIO *_nio = RP ? static_cast<emp::NetIO *>(qi->io) : nullptr;
    uint64_t _r0 = 0, _rloop0 = RP ? _nio->rounds : 0;
    std::string _rprev;
    std::map<std::string, uint64_t> g_rounds;
    // COMM_PROF: attribute main-socket COMMUNICATION (bytes sent+recv) to the op type that produced
    // it -- same close-previous-span accounting as ROUND_PROF/TIMING (requant/lookup bytes between
    // dispatches bill to the producing op). Shows where the forward's communication actually goes.
    const bool CP = getenv("COMM_PROF") != nullptr;
    uint64_t _c0 = CP ? io_counter(qi->io) : 0, _cloop0 = _c0;
    std::string _cprev;
    std::map<std::string, uint64_t> g_comm;
    std::map<std::string, double> g_t;   // per-op-type dispatch time
    std::map<int, double> g_layer;       // per-transformer-layer FULL time (dispatch + requant + range-check)
    std::chrono::steady_clock::time_point _t0; int _Lprev = -1; bool _started = false;
    for (const auto &n : g.nodes) {
      ++ni;
      if (stop_after && ops_run.size() >= stop_after) {
        fprintf(stderr, "[stop_after %zu] halt before %s %s\n", stop_after, n.op_type.c_str(),
                n.output.empty() ? "" : n.output[0].c_str());
        break;
      }
      ops_run.push_back(n.op_type);
      const std::string &op = n.op_type;
      // LAZY_INIT: decode every still-external initializer this op reads, right before it runs (so ALL
      // handlers see resolved .data).  FREE_INIT then frees each after its last use -> resident fp32 is
      // O(the current op's weights), not the whole model.  Central hook, so no handler is missed.
      for (const auto &in : n.input)
        if (!in.empty()) { auto iit = g_->initializer.find(in);
          if (iit != g_->initializer.end() && iit->second.ext && !iit->second.ext_path.empty())
            resolve_ext(const_cast<OnnxTensor &>(iit->second)); }
      if (getenv("DBG_OPS")) {
        printf("[op %zu] %-20s out=%s\n", ops_run.size(), op.c_str(),
               (n.output.empty() ? n.name : n.output[0]).c_str());
        fflush(stdout);
      }
      if (TM) {  // per-op profiler: close the PREVIOUS op's full span (dispatch+requant), open this op's
        if (_started) g_layer[_Lprev] += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
        const std::string &nm = n.output.empty() ? n.name : n.output[0];
        auto _p = nm.find("/h."); _Lprev = (_p != std::string::npos) ? atoi(nm.c_str() + _p + 3) : -1;
        _t0 = std::chrono::steady_clock::now(); _started = true;
      }
      if (RP) {  // close the previous op's round span, open this op's
        if (!_rprev.empty()) g_rounds[_rprev] += _nio->rounds - _r0;
        _rprev = op; _r0 = _nio->rounds;
      }
      if (CP) {  // close the previous op's comm span, open this op's
        uint64_t _cn = io_counter(qi->io);
        if (!_cprev.empty()) g_comm[_cprev] += _cn - _c0;
        _cprev = op; _c0 = _cn;
      }
      // commit any SMALL folded-constant input that flows into a compute op (large
      // weights are handled separately via cref_; threshold avoids committing them).
      for (const auto &in : n.input)
        if (!in.empty() && !act_.count(in) && g.initializer.count(in)) {
          const auto &t = g.initializer.at(in);
          int64_t ne = 1;
          for (auto d : t.dims) ne *= d;
          if (ne <= 65536) commit_const(in);
        }
      try {
        if (op == "DequantizeLinear")
          dequant(n);
        else if (op == "QuantizeLinear")
          quant(n);
        else if (op == "Conv")
          conv(n);
        else if (op == "MatMul" || op == "Gemm") {
          // STREAM_LMHEAD: the final vocab-projection matmul is computed in vocab blocks with a
          // fused public-winner argmax -- never materializing the seq*vocab logits or the whole
          // vocab*hidden committed weight (constant memory).  Untied head only (weight is a plain
          // fp32 initializer, not the shared wte table); tied heads keep matmul_wte.
          if (getenv("STREAM_LMHEAD") && !n.output.empty() && n.output[0] == g.outputs[0] &&
              n.input.size() >= 2 && g.initializer.count(n.input[1]) && !wteT_src_.count(n.input[1]) &&
              n.input[1].find("hadamard") == std::string::npos &&
              g.initializer.at(n.input[1]).dims.size() == 2 &&
              (int)g.initializer.at(n.input[1]).dims[1] >= 16384) {
            lm_head_stream_argmax(n);
          } else
            matmul(n);
        }
        else if (op == "Relu")
          relu(n);
        else if (op == "MaxPool")
          maxpool(n);
        else if (op == "GlobalAveragePool")
          gap(n);
        else if (op == "Add")
          add(n);
        else if (op == "Softmax")
          softmax(n);
        else if (op == "LayerNormalization")
          layernorm(n);
        else if (op == "RMSNormalization")
          rmsnorm(n);
        else if (op == "RotaryEmbedding")
          rope(n);
        else if (op == "Gelu" || op == "Erf")
          gelu(n);
        else if (op == "SiLU")
          silu(n);
        else if (op == "Tanh")
          tanh_act(n);
        else if (op == "Transpose")
          transpose(n);
        else if (op == "Split")
          split(n);
        else if (op == "Gather")
          gather(n);
        else if (op == "Mul")
          mul(n);
        else if (op == "Div")
          div_(n);
        else if (op == "Cast")
          passthru(n); // dtype no-op
        else if (op == "Unsqueeze")
          unsqueeze_op(n);
        else if (op == "Squeeze")
          squeeze_op(n);
        else if (op == "Concat")
          concat(n);
        else if (op == "Slice")
          slice(n);
        else if (op == "Expand")
          expand(n);
        else if (op == "Sub")
          sub(n);
        else if (op == "Where")
          where(n);
        else if (op == "IsNaN")
          isnan(n);
        else if (op == "And")
          and_(n);
        else if (op == "Reshape")
          reshape(n);
        else if (op == "Flatten" || op == "Dropout" || op == "Identity")
          passthru(n);
        else
          throw std::runtime_error("ml-quant: unsupported op " + op);
      } catch (const std::exception &e) {
        fprintf(stderr, "FAIL node %s '%s' in=[", op.c_str(), n.name.c_str());
        for (auto &i : n.input) fprintf(stderr, "%s,", i.c_str());
        fprintf(stderr, "]: %s\n", e.what());
        throw;
      }
      if (TM) g_t[op] += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
      if (!n.output.empty()) last_out_ = n.output[0];
      // DeepProve SYMMETRIC int-N activation quantization (zp=0): quantize each op's activation
      // output to int-N with a per-tensor power-of-2 scale, replacing the fixed-point 2^-P bus.
      // Re-quantizing a pass-through output is a safe re-normalize (real values preserved).  NO
      // zero-points anywhere.
      // DATA-MOVEMENT ops (Reshape/Transpose/Split/Cast/Unsqueeze/Squeeze/Flatten/Dropout/Identity)
      // alias or reorder their input WITHOUT arithmetic: act_[out] == act_[in] at scale_[in], which
      // is already int-N (its producer requanted it).  Re-quantizing it is pure waste (~124M chunk
      // range-checks); skip it and let the output carry the input's committed values + scale.
      static const std::set<std::string> kSkipRequant = {
          "Reshape", "Transpose", "Split", "Cast", "Unsqueeze", "Squeeze", "Flatten", "Dropout",
          "Identity"};
      if (sym_act_bits > 0 && !n.output.empty() && act_.count(n.output[0]) && op != "Softmax" &&
          op != "Where" && !kSkipRequant.count(op) && !sm_skip.count(n.output[0]) &&
          !keep.count(n.output[0]) &&
          !gelu_tabs_.count(n.output[0])) { // skip softmax, final logits, and scale-baked
                                            // nonlinears (already requanted)
        auto &yq = act_[n.output[0]];
        const auto &sh = shp_.at(n.output[0]);
        int d = sh.empty() ? (int)yq.size() : sh.back();
        int rows = d > 0 ? (int)(yq.size() / d) : 1;
        std::vector<int64_t> *rf =
            (party == ALICE && ref_.count(n.output[0])) ? &ref_[n.output[0]] : nullptr;
        bool pt = pt_skip.count(n.output[0]) > 0 ||
                  calib_write_; // feeds a shape op -> single per-tensor scale;
                                // calib write runs PER-TENSOR so the recorded max matches the
                                // static run's activations
        std::vector<double> sin = rscale_.count(n.output[0])
                                      ? rscale_[n.output[0]]
                                      : std::vector<double>(rows, scale_.at(n.output[0]));
        std::vector<double> sout;
        // CALIBRATION WRITE: record this tensor's per-tensor real max (max|ref*scale|), accumulated
        // across inputs.
        if (calib_write_ && party == ALICE && rf) {
          double mx = 0;
          for (size_t i = 0; i < rf->size(); ++i) {
            double v = std::fabs((double)(*rf)[i]) *
                       sin[std::min((int)sin.size() - 1, (int)(i / std::max(1, d)))];
            if (v > mx) mx = v;
          }
          double &slot = calib_max_[n.output[0]];
          if (mx > slot) slot = mx;
        }
        // CALIBRATION READ: static fixed per-tensor scale for this node (no dynamic per-token max).
        const int64_t LIM_ = ((int64_t)1 << (sym_act_bits - 1)) - 1;
        double fixed = calib_.count(n.output[0])
                           ? std::max(calib_.at(n.output[0]) / (double)LIM_, 1e-30)
                           : -1.0;
        bool stat = fixed > 0;
        if (had_resid && hrot.count(n.output[0]))
          hadamard_requant(yq, rf, sin, d, sout, pt || stat, fixed);
        else
          sym_quant_act(yq, rf, sin, d, sout, pt || stat, fixed);
        if ((pt || stat) && !(had_resid && hrot.count(n.output[0]))) {
          rscale_.erase(n.output[0]);
        } else
          rscale_[n.output[0]] = sout;
        scale_[n.output[0]] = sout.empty() ? 1.0 : sout[0];
        zp_[n.output[0]] = 0;
      }
      // MIMIC_DUMP: ALICE prints the REAL per-op range (ref * per-row scale) so it can be diffed
      // against the numpy mimic node-by-node (the FIRST node whose range diverges from the mimic is
      // the bug).
      if (getenv("MIMIC_DUMP") && party == ALICE && !n.output.empty() && ref_.count(n.output[0]) &&
          scale_.count(n.output[0])) {
        const auto &rf = ref_.at(n.output[0]);
        const auto &sh = shp_.count(n.output[0]) ? shp_.at(n.output[0]) : std::vector<int>{};
        int d = sh.empty() ? (int)rf.size() : sh.back();
        if (d < 1) d = 1;
        const std::vector<double> *rs =
            rscale_.count(n.output[0]) ? &rscale_.at(n.output[0]) : nullptr;
        double sc = scale_.at(n.output[0]);
        double mn = 1e18, mx = -1e18, sum = 0;
        for (size_t i = 0; i < rf.size(); ++i) {
          double s = rs ? (*rs)[std::min((int)rs->size() - 1, (int)(i / d))] : sc;
          double v = (double)rf[i] * s;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
          sum += v;
        }
        double s0 =
            rs ? (*rs)[0]
               : sc; // row-0 scale, for the first few elements (exact per-element check vs mimic)
        fprintf(stderr, "[MIMIC] %-46s mean=%9.4f e0..3=[%.4f %.4f %.4f %.4f] rng[%.2f,%.2f]\n",
                n.output[0].c_str(), sum / std::max((size_t)1, rf.size()),
                rf.size() > 0 ? rf[0] * s0 : 0.0, rf.size() > 1 ? rf[1] * s0 : 0.0,
                rf.size() > 2 ? rf[2] * s0 : 0.0, rf.size() > 3 ? rf[3] * s0 : 0.0, mn, mx);
      }
      // STOP_AFTER_NAME: halt right AFTER the node whose output contains this substring, so the
      // driver can finalize the ZK proof at an exact LAYER boundary (clean: tests soundness of
      // ops 0..here, no gate-buffer-fill confound). e.g. "layer.1/output/LayerNorm" = after
      // layer 1.
      {
        static const char *san = getenv("STOP_AFTER_NAME");
        if (san && !n.output.empty() && n.output[0].find(san) != std::string::npos) {
          fprintf(stderr, "[stop_after_name] halt after %s\n", n.output[0].c_str());
          break;
        }
      }
      // DEBUG_REVEAL: open each committed output mid-proof and diff vs ref_ (ALICE's exact-integer
      // mimic computed inline by the handler).  The FIRST op with maxerr>0 is where the committed
      // witness diverges from the intended integer -> pinpoints the faulty handler (a mult-gate
      // mismatch).
      if (getenv("DEBUG_REVEAL") && !n.output.empty() && act_.count(n.output[0])) {
        const std::string &o = n.output[0];
        const auto &v = act_.at(o);
        size_t K = std::min((size_t)4096, v.size());
        bool hr = ref_.count(o);
        int64_t maxerr = 0;
        for (size_t i = 0; i < K; ++i) {
          IntFp vi = v[i];
          int64_t got = QuantInfer::decode_signed(vi.reveal());
          if (party == ALICE && hr) {
            int64_t e = std::llabs(got - ref_.at(o)[i]);
            if (e > maxerr) maxerr = e;
          }
        }
        std::string st =
            !hr ? "(no ref)" : (maxerr ? "DIVERGE maxerr=" + std::to_string(maxerr) : "ok");
        if (party == ALICE)
          fprintf(stderr, "[REVEAL] %-18s %-58s %s\n", op.c_str(), o.substr(0, 56).c_str(),
                  st.c_str());
      }
      // free every input whose LAST use is this node (and isn't a graph output / shared wte table)
      if (stream)
        for (const auto &in : n.input)
          if (!in.empty() && last_use[in] == ni && !keep.count(in) && !ctab_.count(in) &&
              !wteT_src_.count(in)) {
            act_.erase(in);
            ref_.erase(in);
            shp_.erase(in);
            scale_.erase(in);
            zp_.erase(in);
            cref_.erase(in);
            if (free_init)
              const_cast<OnnxGraph &>(*g_).initializer.erase(
                  in); // free the fp32 weight (graph is caller-owned non-const)
          }
      // Free a committed embedding TABLE (ctab_) once the input Gather is its last use AND no LM head
      // reuses it (UNTIED models: QuaRot rotates the LM-head weight, so it != embed^T and the shared
      // table is dead after the input embedding).  Huge at vocab scale: Llama-1B frees 4.2 GB (ctab_) +
      // 2.1 GB (ctab_ref_) that would otherwise sit resident through the whole forward.  Tied models keep
      // it (a wteT_src_ handle points here -> the LM-head matmul_wte still needs it).
      if (stream)
        for (const auto &in : n.input)
          if (!in.empty() && last_use[in] == ni && ctab_.count(in) && !keep.count(in)) {
            bool reused = false;
            for (const auto &kv : wteT_src_) if (kv.second == in) { reused = true; break; }
            if (!reused) { ctab_.erase(in); ctab_ref_.erase(in); ctab_dim_.erase(in); }
          }
      if (memtrack) {
        size_t b = 0;
        for (auto &kv : act_) b += kv.second.size() * 16;
        for (auto &kv : cref_) b += kv.second.size() * 8;
        if (b > peak_live) peak_live = b;
      }
    }
    if (TM) {  // close the last op's span, then dump per-op-type (dispatch) and per-layer (FULL) profiles
      if (_started) g_layer[_Lprev] += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
      fprintf(stderr, "[TIMING] "); for (auto &kv : g_t) fprintf(stderr, "%s=%.1fs ", kv.first.c_str(), kv.second);
      fprintf(stderr, "\n[LAYER] "); for (auto &kv : g_layer) { if (kv.first < 0) fprintf(stderr, "other=%.1fs ", kv.second); else fprintf(stderr, "h.%d=%.1fs ", kv.first, kv.second); } fprintf(stderr, "\n");
    }
    if (RP) {  // close the last op's round span, dump per-op-type rounds (desc)
      if (!_rprev.empty()) g_rounds[_rprev] += _nio->rounds - _r0;
      std::vector<std::pair<std::string, uint64_t>> rs(g_rounds.begin(), g_rounds.end());
      std::sort(rs.begin(), rs.end(), [](auto &a, auto &b) { return a.second > b.second; });
      uint64_t inloop = _nio->rounds - _rloop0;
      fprintf(stderr, "[ROUNDS] loop-entry=%llu in-loop=%llu\n",
              (unsigned long long)_rloop0, (unsigned long long)inloop);
      for (auto &kv : rs)
        if (kv.second)
          fprintf(stderr, "[ROUNDS] %-20s %8llu\n", kv.first.c_str(), (unsigned long long)kv.second);
    }
    if (CP) {  // close the last op's comm span, dump per-op-type communication (MB, desc)
      if (!_cprev.empty()) g_comm[_cprev] += io_counter(qi->io) - _c0;
      std::vector<std::pair<std::string, uint64_t>> cs(g_comm.begin(), g_comm.end());
      std::sort(cs.begin(), cs.end(), [](auto &a, auto &b) { return a.second > b.second; });
      fprintf(stderr, "[COMM] loop-total=%.1f MB\n", (double)(io_counter(qi->io) - _cloop0) / 1e6);
      for (auto &kv : cs)
        if (kv.second)
          fprintf(stderr, "[COMM] %-20s %10.2f MB\n", kv.first.c_str(), (double)kv.second / 1e6);
    }
    // cmp->h2ip keeps pointing at the shared global Hadamard2IP (owned + flushed by qi->destroy())
    // so the post-run argmax (select_max_rows) defers its selector bitness into the same batch.
    qi->ipp->batch_check();   // flush deferred IP-matmul claims (IP_DEFER) before reveals/teardowns/finalize
    if (memtrack)
      fprintf(stderr, "[MEMTRACK] peak live act_+cref_ = %zu MB (streaming %s)\n",
              peak_live / (1024 * 1024), stream ? "ON" : "OFF");
    const std::string &ok = act_.count(g.outputs[0])
                                ? g.outputs[0]
                                : last_out_; // stop_after may halt before graph output
    // out_ref: the final logits' plaintext (ALICE; one int64 per logit -- 1+ GB at a large vocab).
    // Only the OREF_DUMP accuracy path needs it, so drop it otherwise, and MOVE (not copy) when we
    // keep it -- else the big plaintext logits sit in two places (ref_[ok] and out_ref) past the run.
    auto _rit = ref_.find(ok);
    if (getenv("OREF_DUMP") && party == ALICE && _rit != ref_.end())
      out_ref = std::move(_rit->second);
    else
      out_ref.clear();
    if (_rit != ref_.end()) ref_.erase(_rit);   // free the big plaintext logits from the ref map
    // MOVE the committed logits out of act_ -- else the returned copy AND act_[ok] both stay resident
    // (2x the seq*vocab tensor) through the whole argmax, since `gi` outlives run().  argmax reads only
    // the returned vector, and nothing else touches act_[ok] afterward.
    std::vector<IntFp> out = std::move(act_[ok]);
    act_.erase(ok);
    return out;
  }

  // Free everything the forward pass accumulated -- committed LM-head table (ctab_/ctab_ref_, several
  // GB at a large vocab), pre-committed dense weights (wpre_), plaintext weights + leftover activations
  // (cref_/act_/ref_), and the small caches.  Safe to call the moment run() returns: the deferred IP
  // claims are already flushed (batch_check in run()), and the post-run argmax + ZK teardown read only
  // the returned logits vector and the lookup gadgets, never this object's maps.  Cuts the argmax-time
  // peak by the whole weight+table footprint (Gemma: ctab_ 2.7 GB + ctab_ref_ 1.3 GB + wpre_/cref_).
  void release_forward_state() {
    ctab_.clear(); ctab_ref_.clear(); ctab_dim_.clear(); wteT_src_.clear();
    wpre_.clear(); cref_.clear(); act_.clear(); ref_.clear();
    add_const_cache_.clear(); had_.clear(); shp_.clear(); scale_.clear(); zp_.clear();
  }

private:
  const OnnxGraph *g_ = nullptr;
  std::map<std::string, std::vector<IntFp>> act_;
  std::map<std::string, std::pair<std::vector<int64_t>, std::vector<IntFp>>>
      add_const_cache_; // (const@scale) -> (quantized bq[], public-label[]); causal mask reused across layers
  std::map<std::string, std::vector<int64_t>> ref_; // plaintext quantized ints (ALICE)
  std::map<std::string, std::vector<int>> shp_;
  std::map<std::string, double> scale_; // per-tensor scale
  std::map<std::string, int> zp_;       // per-tensor zero-point

  IntFp pub(int64_t v) {
    return IntFp(v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v), PUBLIC);
  }
  IntFp commit_a(int64_t v) {
    return IntFp(v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v), ALICE);
  }
  static int prod(const std::vector<int> &s) {
    int n = 1;
    for (int d : s) n *= d;
    return n;
  }
  double initf(const std::string &nm) {
    const auto &t = g_->initializer.at(nm);
    return t.dtype == 1 ? (t.data.empty() ? 0.0 : t.data[0])
                        : (t.i64.empty() ? 0.0 : (double)t.i64[0]);
  }
  int initi(const std::string &nm) {
    if (!g_->initializer.count(nm)) return 0;
    const auto &t = g_->initializer.at(nm);
    return t.i64.empty() ? 0 : (int)t.i64[0];
  }

  // requant `v` (committed) by ratio M (=c*2^e), round-to-nearest, via LogUp two-sided range
  IntFp requant(IntFp v, double M, int64_t &refv_io, bool haveref) {
    auto ce = QuantInfer::scale_search(M); // M ~ c * 2^e
    IntFp out = qi->requant_cm_one(v, (uint64_t)ce.first, (std::size_t)(-ce.second));
    if (haveref) {
      int64_t a = refv_io;
      int sh = -ce.second;
      int64_t c = ce.first;
      int64_t num = c * a;
      int64_t D = (int64_t)1 << sh;
      refv_io = (num + D / 2) >> sh /* round-half-up: matches requant_cm_one */;
    }
    return out;
  }

  // Rescale a committed activation from its (scale, zp) to the nonlinear-gadget convention
  // (scale 2^-P, zero-point 0): q' = round((q - zp) * scale * 2^P). NO-OP when already there
  // (the uniform/fp32 path), so fp32 proofs are unaffected. QDQ activations carry arbitrary
  // per-tensor scales/zps; the LUT/LayerNorm gadgets assume 2^-P, hence this bridge.
  std::vector<IntFp> rescale_to_P(const std::string &in, std::vector<int64_t> *refout = nullptr) {
    const auto &x = act_.at(in);
    const int z = zp_.at(in);
    const double sP = 1.0 / (double)(1LL << P);
    bool hr = party == ALICE && ref_.count(in);
    const auto *xr = hr ? &ref_.at(in) : nullptr;
    const std::vector<double> *rs =
        (sym_act_bits > 0 && rscale_.count(in)) ? &rscale_.at(in) : nullptr;
    if (!rs) { // per-TENSOR (batched) -- unchanged
      const double sin = scale_.at(in);
      if (sin == sP && z == 0) {
        if (refout && hr) *refout = *xr;
        return x;
      }
      auto ce = QuantInfer::scale_search(sin / sP);
      std::vector<IntFp> shifted(x.size());
      for (size_t i = 0; i < x.size(); ++i) shifted[i] = x[i] + pub(-z);
      std::vector<IntFp> out;
      qi->requant_cm(out, shifted, (uint64_t)ce.first, (std::size_t)(-ce.second));
      if (refout && hr) {
        refout->resize(xr->size());
        int sh = -ce.second;
        int64_t c = ce.first, D = (int64_t)1 << sh;
        for (size_t i = 0; i < xr->size(); ++i) {
          int64_t num = c * ((*xr)[i] - z);
          (*refout)[i] = (num + D / 2) >> sh;
        }
      }
      return out;
    }
    // PER-ROW (per-token): each row rescaled by its own s_row -> 2^-P
    int d = (shp_.count(in) && !shp_.at(in).empty()) ? shp_.at(in).back() : (int)x.size();
    int rows = d > 0 ? (int)(x.size() / d) : 1;
    std::vector<IntFp> out(x.size());
    if (refout && hr) refout->resize(x.size());
    for (int r = 0; r < rows; ++r) {
      double s = (*rs)[std::min((int)rs->size() - 1, r)];
      auto ce = QuantInfer::scale_search(s / sP);
      int64_t c = ce.first;
      int e = ce.second;
      for (int j = 0; j < d; ++j) {
        size_t i = (size_t)r * d + j;
        out[i] = qi->requant_cm_one(x[i] + pub(-z), (uint64_t)c, (std::size_t)(-e));
        if (refout && hr) {
          int64_t N = c * ((*xr)[i] - z);
          (*refout)[i] =
              (e > 0) ? (N << e) : (e == 0 ? N : ((N + ((int64_t)1 << (-e - 1))) >> (-e)));
        }
      }
    }
    return out;
  }

  // ---- DequantizeLinear: pass the committed int through, attach (scale, zp) ----
  void dequant(const OnnxNode &n) {
    const std::string &in = n.input[0];
    if (g_->initializer.count(
            in)) { // a constant weight/bias being dequantized -> keep as a public-int tensor
      const auto &t = g_->initializer.at(in);
      std::vector<int64_t> v(t.i64.begin(), t.i64.end());
      cref_[n.output[0]] = v;
      scale_[n.output[0]] = initf(n.input[1]);
      zp_[n.output[0]] = initi(n.input[2]);
      shp_[n.output[0]] = std::vector<int>(t.dims.begin(), t.dims.end());
      return;
    }
    act_[n.output[0]] = act_.at(in);
    if (ref_.count(in)) ref_[n.output[0]] = ref_.at(in);
    shp_[n.output[0]] = shp_.at(in);
    scale_[n.output[0]] = initf(n.input[1]);
    zp_[n.output[0]] = initi(n.input[2]);
  }

  // ---- QuantizeLinear: requant (v - in_zp) by in_scale/out_scale, + out_zp, clamp [0,255] ----
  void quant(const OnnxNode &n) {
    const std::string &in = n.input[0];
    const auto &x = act_.at(in);
    const double sin = scale_.at(in), sout = initf(n.input[1]);
    const int zin = zp_.at(in), zout = initi(n.input[2]);
    // REDUNDANT-Q SKIP: a QuantizeLinear whose (scale,zp) equal the input's is a Q that undoes a DQ
    // across a scale-transparent data-movement op (Reshape/Transpose) -- an identity requant on
    // already-clamped int8.  Alias the input and skip the requant + clamp lookup entirely.  (In
    // bert512_qdq this is 132 of 392 Q's -- all the Reshape/Transpose ones.)
    if (std::abs(sin - sout) < 1e-9 * (std::abs(sout) + 1e-30) && zin == zout) {
      act_[n.output[0]] = x;
      if (ref_.count(in)) ref_[n.output[0]] = ref_.at(in);
      shp_[n.output[0]] = shp_.at(in);
      scale_[n.output[0]] = sout;
      zp_[n.output[0]] = zout;
      return;
    }
    const double M = sin / sout;
    const int n_el = (int)x.size();
    auto ce = QuantInfer::scale_search(M);
    const uint64_t cc = (uint64_t)ce.first;
    const int sh = -ce.second;
    const int64_t D = (int64_t)1 << sh; // ONCE per node (M is constant), NOT per element
    std::vector<IntFp> y(n_el);
    bool hr = party == ALICE && ref_.count(in);
    std::vector<int64_t> r;
    if (hr) r.resize(n_el);
    const auto *xr = hr ? &ref_.at(in) : nullptr;
    int64_t maxpc = 0, minpc = 0, maxin = 0;
    // BATCHED QuantizeLinear (mirrors GPT-2's batched sym_quant_act + gelu-lookup batching): shift
    // by -in_zp (local), ONE batched requant (requant_cm_batch: 2 batch_feeds + 1 batched zero-check
    // vs 2*n per-element feeds + n reveals), +out_zp (local), then ONE batched clamp lookup vs a
    // per-element table Access.  This is the dominant BERT-QDQ cost.
    std::vector<IntFp> v(n_el);
    for (int i = 0; i < n_el; ++i) v[i] = x[i] + pub(-zin); // v - in_zp
    std::vector<IntFp> q(n_el);
    qi->requant_cm_batch(q.data(), v.data(), cc, (std::size_t)sh, (std::size_t)n_el);
    const IntFp zoutp = pub(zout);
    for (int i = 0; i < n_el; ++i) q[i] = q[i] + zoutp;
    // clamp to [0,255]
    if (clamp_tab) {
      const IntFp coff = pub(clamp_off);
      std::vector<IntFp> keys(n_el);
      for (int i = 0; i < n_el; ++i) keys[i] = q[i] + coff; // key-bind range-checks into [0,T)
      clamp_tab->lookup(y, keys);                           // BATCHED saturation lookup
    } else {
      // batched [0,255] clamp: each is_le's g(g-1)=0 bitness proven as ONE RLC inner product
      // (is_le_batch) instead of 2*n_el per-element mult-gate + reveal_zero pairs.  The le255
      // comparison depends on qq, so it is a second batched pass (not one big batch).
      std::vector<IntFp> zeros((size_t)n_el, pub(0)), c255((size_t)n_el, pub(255));
      std::vector<IntFp> qvec(q.begin(), q.begin() + n_el), ge0v, le255v;
      cmp->is_le_batch(ge0v, zeros, qvec, 30);                 // ge0[i] = 1{0 <= q[i]}
      std::vector<IntFp> qq((size_t)n_el);
      for (int i = 0; i < n_el; ++i) qq[i] = sel(ge0v[i], q[i], pub(0));
      cmp->is_le_batch(le255v, qq, c255, 30);                  // le255[i] = 1{qq[i] <= 255}
      for (int i = 0; i < n_el; ++i) y[i] = sel(le255v[i], qq[i], pub(255));
    }
    if (hr)
      for (int i = 0; i < n_el; ++i) {
        int64_t num = (int64_t)cc * ((*xr)[i] - zin);
        int64_t rv = (num + D / 2) >> sh; // round-half-up: matches requant_cm
        int64_t pc = rv + zout;
        if (pc > maxpc) maxpc = pc;
        if (pc < minpc) minpc = pc;
        if (std::llabs((*xr)[i] - zin) > maxin) maxin = std::llabs((*xr)[i] - zin);
        r[i] = pc < 0 ? 0 : (pc > 255 ? 255 : pc);
      }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(in);
    scale_[n.output[0]] = sout;
    zp_[n.output[0]] = zout;
    if (hr) ref_[n.output[0]] = std::move(r);
    if (hr && getenv("DEBUG_REVEAL")) {
      static int64_t gmin = 0, gmax = 0;
      if (minpc < gmin) gmin = minpc;
      if (maxpc > gmax) gmax = maxpc;
      fprintf(stderr, "[PRECLAMP] node[%lld,%lld]  GLOBAL[%lld,%lld]\n", (long long)minpc,
              (long long)maxpc, (long long)gmin, (long long)gmax);
    }
  }
  IntFp sel(IntFp bit, IntFp a, IntFp b) {
    return b + bit * (a + b * pub(-1));
  } // bit ? a : b

  // ---- Conv: im2col + threaded matmul (conv2d) for sum x_q*w_q, then add the
  //      zero-point-folded bias. out scale = x_s*w_s, zp 0. ----
  void conv(const OnnxNode &n) {
    const std::string &xin = n.input[0];
    const auto &xf = act_.at(xin);
    const auto &xs = shp_.at(xin);
    const int Cin = xs[0], H = xs[1], Wd = xs[2];
    const int xz = zp_.at(xin);
    const double sx = scale_.at(xin);
    const auto &W = cref_.at(n.input[1]);
    const double sw = scale_.at(n.input[1]);
    const auto &wshape = shp_.at(n.input[1]);
    const int Cout = wshape[0], K = wshape[2];
    const int stride = n.attr.count("strides") && !n.attr.at("strides").ints.empty()
                           ? (int)n.attr.at("strides").ints[0]
                           : 1;
    const int pad = n.attr.count("pads") && !n.attr.at("pads").ints.empty()
                        ? (int)n.attr.at("pads").ints[0]
                        : 0;
    const int Ho = conv_out_dim(H, K, stride, pad), Wo = conv_out_dim(Wd, K, stride, pad),
              Msz = Cin * K * K, Pn = Ho * Wo;
    // bias_eff[co] = bias_q[co] - xz * sum_k w_q[co][k]   (folds the asymmetric input zero-point)
    std::vector<int64_t> beff(Cout, 0);
    if (n.input.size() > 2 && cref_.count(n.input[2])) {
      const auto &b = cref_.at(n.input[2]);
      for (int co = 0; co < Cout; ++co) beff[co] = (co < (int)b.size() ? b[co] : 0);
    } // int32 bias (DQ'd into cref_)
    std::vector<std::vector<IntFp>> cK(Cout);
    for (int co = 0; co < Cout; ++co) {
      cK[co].resize(Msz);
      int64_t ws = 0;
      for (int m = 0; m < Msz; ++m) {
        int64_t w = W[(int64_t)co * Msz + m];
        cK[co][m] = commit_a(w);
        ws += w;
      }
      beff[co] -= (int64_t)xz * ws;
    }
    Tensor3 X(Cin, std::vector<std::vector<IntFp>>(H, std::vector<IntFp>(Wd)));
    for (int c = 0; c < Cin; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < Wd; ++w) X[c][h][w] = xf[(int64_t)c * H * Wd + h * Wd + w];
    FMap raw;
    conv2d<BoolIO>(*qi->ipp, raw, X, cK, Cin, H, Wd, Cout, K, stride, pad, qi->mat_threads);
    std::vector<IntFp> out((size_t)Cout * Pn);
    for (int co = 0; co < Cout; ++co) {
      IntFp bc = pub(beff[co]);
      for (int p = 0; p < Pn; ++p) out[(int64_t)co * Pn + p] = raw[co][p] + bc;
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {Cout, Ho, Wo};
    scale_[n.output[0]] = sx * sw;
    zp_[n.output[0]] = 0;
    if (party == ALICE && ref_.count(xin)) {
      const auto &xr = ref_.at(xin);
      std::vector<int64_t> r((size_t)Cout * Pn);
      for (int co = 0; co < Cout; ++co)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            int64_t a = beff[co];
            for (int ci = 0; ci < Cin; ++ci)
              for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj) {
                  int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
                  if (ih >= 0 && ih < H && iw >= 0 && iw < Wd)
                    a += xr[(int64_t)ci * H * Wd + ih * Wd + iw] *
                         W[(int64_t)co * Msz + ci * K * K + ki * K + kj];
                }
            r[(int64_t)co * Pn + oh * Wo + ow] = a;
          }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // LM head over the shared committed table: logit[s][v] = <wte_row_v, hidden_s>. Consumes the
  // committed wte in its natural [V,D] row layout -- NO wte^T materialized, NO re-commitment.
  void matmul_wte(const OnnxNode &n) {
    const std::string &xin = n.input[0];
    const std::string &tname = wteT_src_.at(n.input[1]);
    auto _wt = std::chrono::steady_clock::now();
    commit_table(tname);
    if (getenv("WTE_PROF"))
      fprintf(stderr, "[WTE_COMMIT] %s: %.2fs\n", tname.c_str(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - _wt).count());
    const auto &xf = act_.at(xin);
    const double sx = scale_.at(xin);
    const double sw = 1.0 / (double)(1LL << P);
    const int V = ctab_dim_.at(tname).first, D = ctab_dim_.at(tname).second; // wte is [V, D]
    const auto &c = ctab_.at(tname);
    const int R = D > 0 ? (int)(xf.size() / D) : 1; // # positions (rows of hidden [.,D])
    // The LM-head output's plaintext ref (acc -> r) is consumed ONLY by OREF_DUMP / CALIB_WRITE:
    // argmax reads the committed logits' own cleartext, not ref_, and run() erases this ref anyway.
    // Skip it otherwise -- saves the R*V acc (uint64) + r (int64) buffers (~2 GB at vocab 262144).
    bool hr = party == ALICE && ref_.count(xin) && (getenv("OREF_DUMP") || getenv("CALIB_WRITE"));
    // Chunk the LM head over ROW-BLOCKS so the matmul+requant working set is O(rblk*V), not O(R*V).
    // Single-shot at R=512, V=262144 allocates acc_raw (1 GB) + the committed block + a batch-commit
    // buffer + (pre-requant) a dozen full-length requant scratch buffers ALL at once (~8 GB transient
    // -- the measured forward peak).  Per 64-row block that working set is ~1/8th; only the full
    // committed logits y (which argmax needs) is an R*V allocation that persists.  Each block is an
    // independent Freivalds/IP matmul proof + one requant zero-check (a few extra batched rounds).
    const size_t Vs = (size_t)V;
    std::vector<IntFp> y((size_t)R * Vs);
    std::vector<int64_t> r;
    if (hr) r.resize((size_t)R * Vs);
    static const bool FVH = !getenv("FREIVALDS") || atoi(getenv("FREIVALDS")) != 0;   // DEFAULT ON; FREIVALDS=0 disables
    const double sout = uniform_requant ? 1.0 / (double)(1LL << P) : sx * sw;
    const bool do_rq = uniform_requant && sout != sx * sw;
    auto ce = QuantInfer::scale_search((sx * sw) / sout); // pure/cheap; used only when do_rq
    size_t rblk = 64;
    if (const char *e = getenv("WTE_ROW_BLK")) rblk = (size_t)atoll(e);
    if (rblk == 0 || rblk > (size_t)R) rblk = (size_t)R;
    std::vector<IntFp> xblk, yb;
    std::vector<uint64_t> accb;
    for (size_t r0 = 0; r0 < (size_t)R; r0 += rblk) {
      const size_t rb = std::min(rblk, (size_t)R - r0);
      xblk.assign(xf.begin() + (long)(r0 * D), xf.begin() + (long)((r0 + rb) * D)); // block of hidden rows
      if (FVH)
        qi->ipp->matmul_mm_flat_freivalds(yb, c, Vs, xblk, rb, (size_t)D, qi->mat_threads, hr ? &accb : nullptr);
      else
        qi->ipp->matmul_mm_flat(yb, c, Vs, xblk, rb, (size_t)D, qi->mat_threads, hr ? &accb : nullptr);
      if (do_rq) // in-place block requant (requant_cm_batch reads its input into scratch before writing)
        qi->requant_cm_batch(yb.data(), yb.data(), (uint64_t)ce.first, (std::size_t)(-ce.second), rb * Vs);
      std::move(yb.begin(), yb.end(), y.begin() + (long)(r0 * Vs));
      if (hr) {
        const int sh = -ce.second; const int64_t cc = ce.first;
        for (size_t i = 0; i < rb * Vs; ++i) {
          int64_t a = QuantInfer::decode_signed(accb[i]);
          r[r0 * Vs + i] = do_rq ? ((cc * a + ((int64_t)1 << (sh - 1))) >> sh) : a; // round-half-up
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
    scale_[n.output[0]] = sout;
    zp_[n.output[0]] = 0;
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- Fast structured Hadamard: Hd = kron(O_m, S_p2/sqrt(p2)) (S = +/-1 Sylvester) ----
  // z @ Hd via per-group butterfly WHT (integer +/-1, O(p2 log p2)) + a tiny m x m public O_m mix.
  // Committed*public throughout (free ZK) and NEVER materializes/quantizes the dense I x I matrix
  // -- ~180x fewer IntFp ops than the dense public-weight loop.  Output obeys matmul()'s exact
  // act_/scale_/rscale_/ref_ contract with effective weight-scale sw = so/sqrt(p2) (== dense max|Hd|/WM).
  // mode 0 = SQUARE kron(O_m, S_p2/sqrt(p2))  (Qwen/Llama residual-free R4).
  // mode 1 = RECTANGULAR padded pure Sylvester S_out[:in,:]/sqrt(out), in<out=2^k (GPT-2 HADAMARD=1).
  struct HadStruct { int mode, in, out, m, p2; double so, invsqrt; std::vector<int64_t> Oq; };
  std::map<std::string, HadStruct> had_;
  const HadStruct &had_extract(const std::string &wn) {
    auto it = had_.find(wn);
    if (it != had_.end()) return it->second;
    const auto &Wt = g_->initializer.at(wn);
    const int in = (int)Wt.dims[0], out = (int)Wt.dims[1];
    HadStruct h; h.in = in; h.out = out;
    if (in != out) {                 // RECTANGULAR: pad in->out, one WHT_out, scale 1/sqrt(out); no O_m
      h.mode = 1; h.m = 1; h.p2 = out; h.so = 1.0; h.invsqrt = 1.0 / std::sqrt((double)out);
      return had_.emplace(wn, std::move(h)).first->second;
    }
    const int I = in, p2 = I & (-I), m = I / p2;   // SQUARE kron
    h.mode = 0; h.m = m; h.p2 = p2; h.invsqrt = 1.0 / std::sqrt((double)p2);
    const double sq = std::sqrt((double)p2);
    std::vector<double> O((size_t)m * m); double mx = 1e-30;
    for (int a = 0; a < m; ++a)      // O_m[a][a'] = Hd[a*p2, a'*p2] * sqrt(p2) (undo S[0,0]=1/sqrt)
      for (int ap = 0; ap < m; ++ap) {
        double v = (double)Wt.data[(size_t)(a * p2) * I + (ap * p2)] * sq;
        O[(size_t)a * m + ap] = v; double av = v < 0 ? -v : v; if (av > mx) mx = av;
      }
    h.so = mx / (double)weight_max; h.Oq.resize((size_t)m * m);
    for (size_t i = 0; i < O.size(); ++i)
      h.Oq[i] = std::max((int64_t)-weight_max,
                         std::min((int64_t)weight_max, (int64_t)llround(O[i] / h.so)));
    return had_.emplace(wn, std::move(h)).first->second;
  }
  static void wht_ip(IntFp *v, int p2) { // in-place natural-order (Sylvester) WHT, +/-1 combine
    for (int L = 1; L < p2; L <<= 1)
      for (int i = 0; i < p2; i += L << 1)
        for (int j = i; j < i + L; ++j) { IntFp a = v[j], b = v[j + L]; v[j] = a + b; v[j + L] = a + b.negate(); }
  }
  static void wht_ip_i(int64_t *v, int p2) {
    for (int L = 1; L < p2; L <<= 1)
      for (int i = 0; i < p2; i += L << 1)
        for (int j = i; j < i + L; ++j) { int64_t a = v[j], b = v[j + L]; v[j] = a + b; v[j + L] = a - b; }
  }
  void fast_hadamard(const OnnxNode &n) {
    const std::string &xin = n.input[0];
    const auto &xf = act_.at(xin);
    const double sx = scale_.at(xin);
    const HadStruct &h = had_extract(n.input[1]);
    if (getenv("DBG_HAD")) fprintf(stderr, "[fast_had] %s mode=%s in=%d out=%d\n",
        n.output[0].c_str(), h.mode ? "RECT-Sylvester" : "SQUARE-kron", h.in, h.out);
    const std::vector<double> *sxr = (sym_act_bits > 0 && rscale_.count(xin)) ? &rscale_.at(xin) : nullptr;
    auto srow = [&](int rr) { return sxr ? (*sxr)[std::min((int)sxr->size() - 1, rr)] : sx; };
    const bool hr = party == ALICE && ref_.count(xin);
    const auto *xr = hr ? &ref_.at(xin) : nullptr;
    // common act_/scale_/rscale_/ref_ contract (== matmul()'s), output width `od`, weight-scale `sw`
    auto finish = [&](std::vector<IntFp> &&y, std::vector<int64_t> &&r, int od, int R, double sw) {
      std::vector<int> ns = shp_.at(xin);
      if (!ns.empty()) ns.back() = od; else ns = {od};
      act_[n.output[0]] = std::move(y);
      shp_[n.output[0]] = std::move(ns);
      scale_[n.output[0]] = sx * sw;
      zp_[n.output[0]] = 0;
      if (hr) ref_[n.output[0]] = std::move(r);
      if (sym_act_bits > 0) { std::vector<double> ro(R);
        for (int rr = 0; rr < R; ++rr) ro[rr] = srow(rr) * sw; rscale_[n.output[0]] = std::move(ro); }
    };
    if (h.mode == 1) { // RECTANGULAR pure Sylvester: zero-pad z[in] -> out, one WHT_out, scale sx/sqrt(out)
      const int in = h.in, out = h.out, R = in > 0 ? (int)(xf.size() / in) : 1;
      std::vector<IntFp> y((size_t)R * out), v((size_t)out);
      std::vector<int64_t> r, vi;
      if (hr) { r.resize((size_t)R * out); vi.resize((size_t)out); }
      for (int rr = 0; rr < R; ++rr) {
        for (int i = 0; i < out; ++i) v[i] = (i < in) ? xf[(size_t)rr * in + i] : pub(0);
        wht_ip(v.data(), out);
        for (int o = 0; o < out; ++o) y[(size_t)rr * out + o] = v[o];
        if (hr) { for (int i = 0; i < out; ++i) vi[i] = (i < in) ? (*xr)[(size_t)rr * in + i] : 0;
                  wht_ip_i(vi.data(), out);
                  for (int o = 0; o < out; ++o) r[(size_t)rr * out + o] = vi[o]; }
      }
      finish(std::move(y), std::move(r), out, R, h.invsqrt); // so=1 for pure Sylvester
      return;
    }
    const int m = h.m, p2 = h.p2, I = m * p2, R = I > 0 ? (int)(xf.size() / I) : 1; // SQUARE kron(O_m, S_p2)
    std::vector<IntFp> y((size_t)R * I), T((size_t)m * p2);
    std::vector<int64_t> r, Ti;
    if (hr) { r.resize((size_t)R * I); Ti.resize((size_t)m * p2); }
    for (int rr = 0; rr < R; ++rr) {
      for (int a = 0; a < m; ++a) { // block-WHT each of the m groups (integer +/-1 butterfly)
        for (int b = 0; b < p2; ++b) T[(size_t)a * p2 + b] = xf[(size_t)rr * I + a * p2 + b];
        wht_ip(&T[(size_t)a * p2], p2);
        if (hr) { for (int b = 0; b < p2; ++b) Ti[(size_t)a * p2 + b] = (*xr)[(size_t)rr * I + a * p2 + b];
                  wht_ip_i(&Ti[(size_t)a * p2], p2); }
      }
      for (int ap = 0; ap < m; ++ap) // O_m mix: y[a'][b'] = sum_a Oq[a][a'] T[a][b']
        for (int b = 0; b < p2; ++b) {
          IntFp acc = T[(size_t)0 * p2 + b] * pub(h.Oq[(size_t)0 * m + ap]);
          int64_t ra = hr ? Ti[(size_t)0 * p2 + b] * h.Oq[(size_t)0 * m + ap] : 0;
          for (int a = 1; a < m; ++a) {
            int64_t o = h.Oq[(size_t)a * m + ap];
            acc = acc + T[(size_t)a * p2 + b] * pub(o);
            if (hr) ra += Ti[(size_t)a * p2 + b] * o;
          }
          y[(size_t)rr * I + ap * p2 + b] = acc;
          if (hr) r[(size_t)rr * I + ap * p2 + b] = ra;
        }
    }
    finish(std::move(y), std::move(r), I, R, h.so * h.invsqrt);
  }

  // Hadamard-wrapped requant for MASSIVE-ACTIVATION residual streams (e.g. Gemma-3, ~1e5:1 range that
  // per-tensor int quant cannot represent). Quantize in a padded-Sylvester-rotated basis so per-channel
  // outliers spread across all N channels, then rotate back:  r_q = truncate_d(WHT_N(symquant(WHT_N(pad_N r)))).
  // Pure Sylvester (symmetric, WHT_N.WHT_N = N.I), committed*public (fast butterfly). Transparent to the
  // norms (no basis change) -- sidesteps QuaRot's sandwich-norm problem. Scales carry the 1/sqrt(N) factors.
  void hadamard_requant(std::vector<IntFp> &yq, std::vector<int64_t> *rf, const std::vector<double> &sin,
                        int d, std::vector<double> &sout, bool pt_or_stat, double fixed) {
    int N = 1; while (N < d) N <<= 1;
    const int rows = d > 0 ? (int)(yq.size() / d) : 1;
    const double invs = 1.0 / std::sqrt((double)N);
    const bool hr = rf && (int)rf->size() == rows * d;
    std::vector<IntFp> yr((size_t)rows * N);
    std::vector<int64_t> rr;
    if (hr) rr.assign((size_t)rows * N, 0);
    for (int r = 0; r < rows; ++r) { // zero-pad each row to N, then integer +/-1 WHT
      for (int i = 0; i < N; ++i) yr[(size_t)r * N + i] = (i < d) ? yq[(size_t)r * d + i] : pub(0);
      wht_ip(&yr[(size_t)r * N], N);
      if (hr) { for (int i = 0; i < d; ++i) rr[(size_t)r * N + i] = (*rf)[(size_t)r * d + i];
                wht_ip_i(&rr[(size_t)r * N], N); }
    }
    std::vector<double> sin_rot(sin.empty() ? (size_t)1 : sin.size());
    for (size_t i = 0; i < sin_rot.size(); ++i) sin_rot[i] = (sin.empty() ? 1.0 : sin[i]) * invs;
    std::vector<double> sout_rot;
    sym_quant_act(yr, hr ? &rr : nullptr, sin_rot, N, sout_rot, pt_or_stat, fixed);
    for (int r = 0; r < rows; ++r) { // WHT back, truncate to d
      wht_ip(&yr[(size_t)r * N], N);
      for (int i = 0; i < d; ++i) yq[(size_t)r * d + i] = yr[(size_t)r * N + i];
      if (hr) { wht_ip_i(&rr[(size_t)r * N], N);
                for (int i = 0; i < d; ++i) (*rf)[(size_t)r * d + i] = rr[(size_t)r * N + i]; }
    }
    sout.assign(sout_rot.size(), 0.0);
    for (size_t i = 0; i < sout_rot.size(); ++i) sout[i] = sout_rot[i] * invs;
  }

  // Constant-memory LM head + argmax.  The vocabulary projection normally materializes the whole
  // seq*vocab logit tensor AND commits the entire vocab*hidden weight -- both O(model) transients
  // that dominate peak RSS at scale.  Here the argmax (a revealed output) uses a PUBLIC winner:
  //   (0) ALICE finds a[pos]=argmax_v <hidden[pos], W[:,v]> in plaintext and reveals it.
  //   (1) commit m[pos] = ALICE's plaintext winner accumulator.
  //   (2) stream the vocabulary in blocks: commit the block's weight columns, IP-matmul the block's
  //       committed logits, range-check m[pos] - logit[pos,v] >= 0 for every v, then DISCARD the
  //       block.  In the block that contains a[pos], additionally open logit[pos,a[pos]] - m[pos] == 0
  //       so m is bound to the real committed computation (else a bogus-large m would pass trivially).
  // Soundness: m == the true logit at the public index a[pos] (winner-equality), and m >= every logit
  // (range checks) => a[pos] is a genuine argmax.  Peak memory is O(block*seq + hidden*block).
  void lm_head_stream_argmax(const OnnxNode &n) {
    const std::string &xin = n.input[0];
    if (!cref_.count(n.input[1])) { // symmetric-int quantize the weight (twin of matmul())
      resolve_ext(const_cast<OnnxTensor &>(g_->initializer.at(n.input[1]))); // LAZY_INIT: decode now
      const auto &Wt = g_->initializer.at(n.input[1]);
      double mxw = 1e-12;
      for (float v : Wt.data) { double a = v < 0 ? -(double)v : (double)v; if (a > mxw) mxw = a; }
      const int64_t WM = weight_max; const double swq = mxw / (double)WM;
      std::vector<int64_t> q(Wt.data.size());
      for (size_t i = 0; i < Wt.data.size(); ++i)
        q[i] = std::max(-WM, std::min(WM, (int64_t)llround((double)Wt.data[i] / swq)));
      cref_[n.input[1]] = std::move(q);
      scale_[n.input[1]] = swq;
      shp_[n.input[1]] = std::vector<int>(Wt.dims.begin(), Wt.dims.end());
    }
    const auto &xf = act_.at(xin);          // committed hidden, [npos*D] row-major
    const auto &W = cref_.at(n.input[1]);   // int weight, layout [D, V]: W[i*V + v]
    const auto &wsh = shp_.at(n.input[1]);
    const int D = wsh[0], V = wsh[1];
    const int npos = D > 0 ? (int)(xf.size() / D) : 1;
    const bool hr = party == ALICE && ref_.count(xin);
    const std::vector<int64_t> *xr = hr ? &ref_.at(xin) : nullptr;
    const int NT = qi->mat_threads;
    const std::size_t argbits = getenv("STREAM_ARGBITS") ? (std::size_t)atoi(getenv("STREAM_ARGBITS")) : 44;

    // (0) plaintext argmax + winner accumulator (ALICE), reveal a[] to both parties.
    std::vector<int> a(npos, 0);
    std::vector<int64_t> mval(npos, 0);
    if (party == ALICE)
      volefier::parallel_for((std::size_t)npos, NT, [&](std::size_t p) {
        int64_t best = INT64_MIN; int bi = 0;
        const int64_t *xp = &(*xr)[(std::size_t)p * D];
        for (int v = 0; v < V; ++v) {
          int64_t acc = 0;
          for (int i = 0; i < D; ++i) acc += xp[i] * W[(std::size_t)i * V + v];
          if (acc > best) { best = acc; bi = v; }
        }
        a[p] = bi; mval[p] = best;
      });
    { IOChannel *io = qi->io;
      if (party == ALICE) { io->send_data(a.data(), (int)(npos * sizeof(int))); io->flush(); }
      else { io->recv_data(a.data(), (int)(npos * sizeof(int))); } }
    // SOUNDNESS: a[pos] is PUBLIC (revealed), and the winner-equality below only binds m[pos] when
    // a[pos] lands in some vocab block.  An out-of-range a[pos] would land in NO block, leaving m[pos]
    // unbound -- a cheating prover could then set m huge, pass every dominance check, and "prove" a
    // garbage token is the argmax.  select_max_rows avoids this by deriving the index from a committed
    // one-hot (sum==1), which is in-range by construction; here we must range-check the revealed index
    // explicitly.  It is a public value, so both parties check it identically in plaintext.
    for (int p = 0; p < npos; ++p)
      if (a[p] < 0 || a[p] >= V)
        error("lm_head_stream_argmax: revealed argmax index out of range [0,V) -- rejecting\n");

    // (1) commit m[pos] (ALICE's plaintext winner accumulator; BOB feeds 0 -- the value rides its MAC).
    std::vector<IntFp> m(npos);
    { std::vector<uint64_t> mraw((std::size_t)npos, 0);
      if (party == ALICE)
        for (int p = 0; p < npos; ++p) mraw[p] = mval[p] >= 0 ? (uint64_t)mval[p] : PR - (uint64_t)(-mval[p]);
      batch_feed(m.data(), mraw.data(), (int64_t)npos); }

    // (2) stream the vocabulary in blocks.
    std::size_t VB = 8192;
    if (const char *e = getenv("LMHEAD_VOCAB_BLK")) VB = (std::size_t)atoll(e);
    if (VB == 0 || VB > (std::size_t)V) VB = (std::size_t)V;
    std::vector<IntFp> Wb, logit; std::vector<uint64_t> wraw;
    for (int v0 = 0; v0 < V; v0 += (int)VB) {
      const int vb = std::min((int)VB, V - v0);
      // commit the block's weight columns as vb rows of length D: Wb[j*D + i] = W[i*V + (v0+j)]
      Wb.assign((std::size_t)vb * D, IntFp());
      wraw.assign((std::size_t)vb * D, 0);
      if (party == ALICE)
        volefier::parallel_for((std::size_t)vb, NT, [&](std::size_t j) {
          for (int i = 0; i < D; ++i) { int64_t v = W[(std::size_t)i * V + (v0 + (int)j)];
            wraw[j * D + i] = v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v); }
        });
      batch_feed(Wb.data(), wraw.data(), (int64_t)((std::size_t)vb * D));
      // committed block logits: logit[p*vb + j] = <Wb[j], hidden[p]>  (Freivalds IP proof)
      qi->ipp->matmul_mm_flat_freivalds(logit, Wb, (std::size_t)vb, xf, (std::size_t)npos, (std::size_t)D, NT, nullptr);
      // dominance: diffs[p*vb + j] = m[p] - logit[p][j] >= 0  (one batched range check per block)
      std::vector<IntFp> diffs((std::size_t)npos * vb);
      volefier::parallel_for((std::size_t)npos, NT, [&](std::size_t p) {
        for (int j = 0; j < vb; ++j) diffs[p * vb + j] = m[p] + logit[p * (std::size_t)vb + j].negate();
      });
      rcmp->check_batch(diffs.data(), (std::size_t)npos * vb, argbits);
      // winner-equality for any position whose argmax falls in this block: logit[p][a-v0] - m[p] == 0
      std::vector<IntFp> zeq; zeq.reserve(npos);
      for (int p = 0; p < npos; ++p)
        if (a[p] >= v0 && a[p] < v0 + vb)
          zeq.push_back(logit[(std::size_t)p * vb + (a[p] - v0)] + m[p].negate());
      if (!zeq.empty()) batch_reveal_check_zero(zeq.data(), (int64_t)zeq.size());
      // Wb / logit / diffs freed at next iteration -> peak stays O(block).
    }
    lmhead_pred_ = std::move(a);
    streamed_lmhead_ = true;
    if (getenv("DBG_OPS")) fprintf(stderr, "[lmhead] streamed argmax over V=%d in blocks of %zu\n", V, VB);
  }

  void matmul(const OnnxNode &n) {
    const std::string &xin = n.input[0];
    if (n.input.size() >= 2 && n.input[1].find("hadamard") != std::string::npos &&
        g_->initializer.count(n.input[1])) { // structured online Hadamard -> fast WHT path
      const auto &W = g_->initializer.at(n.input[1]);
      if (W.dims.size() == 2 && !W.data.empty()) {
        const int di = (int)W.dims[0], dj = (int)W.dims[1];
        const bool square = di == dj;                             // kron(O_m, S_p2)  (Qwen/Llama)
        const bool rect = dj > di && (dj & (dj - 1)) == 0;        // padded pure Sylvester (GPT-2, dj=2^k)
        if (square || rect) { fast_hadamard(n); return; }
      }
    }
    if (wteT_src_.count(n.input[1])) {
      matmul_wte(n);
      return;
    } // LM head over the shared committed wte table
    if (!cref_.count(n.input[1])) {
      if (act_.count(n.input[1])) {
        matmul_aa(n);
        return;
      } // two committed activations (attention)
      // fp32 weight initializer -> PER-TENSOR SYMMETRIC INT8 quantization (zkGPT/DeepProve style):
      // s_w = max|W| / 127; matmul requant by (s_x*s_w)/s_out = c*2^e (scale_search), NOT uniform.
      resolve_ext(const_cast<OnnxTensor &>(g_->initializer.at(n.input[1]))); // LAZY_INIT: decode now
      const auto &Wt = g_->initializer.at(n.input[1]);
      double mxw = 1e-12;
      for (float v : Wt.data) {
        double a = v < 0 ? -(double)v : (double)v;
        if (a > mxw) mxw = a;
      }
      const int64_t WM = weight_max;
      const double swq = mxw / (double)WM;
      std::vector<int64_t> q(Wt.data.size());
      for (size_t i = 0; i < Wt.data.size(); ++i)
        q[i] = std::max(-WM, std::min(WM, (int64_t)llround((double)Wt.data[i] / swq)));
      cref_[n.input[1]] = std::move(q);
      scale_[n.input[1]] = swq;
      shp_[n.input[1]] = std::vector<int>(Wt.dims.begin(), Wt.dims.end());
    }
    const auto &xf = act_.at(xin);
    const int xz = zp_.at(xin);
    const double sx = scale_.at(xin);
    const auto &W = cref_.at(n.input[1]);
    const double sw = scale_.at(n.input[1]);
    const auto &wsh = shp_.at(n.input[1]);
    const bool gemm = n.op_type == "Gemm";
    const int tb = gemm && n.attr.count("transB") ? (int)n.attr.at("transB").i : 0;
    int out = tb ? wsh[0] : wsh[1], in = tb ? wsh[1] : wsh[0];
    std::vector<int64_t> beff(out, 0);
    std::vector<double> breal(out, 0.0); // breal = real bias (for per-row re-scaling)
    if (gemm && n.input.size() > 2 && g_->initializer.count(n.input[2])) {
      const auto &bt = g_->initializer.at(n.input[2]);
      for (int o = 0; o < out; ++o)
        breal[o] = !bt.data.empty() ? (double)bt.data[o]
                                    : (o < (int)bt.i64.size() ? (double)bt.i64[o] * sx * sw : 0.0);
    } else if (gemm && n.input.size() > 2 &&
               cref_.count(
                   n.input[2])) { // QDQ DEQUANTIZED int32 bias lives in cref_ (not g_->initializer)
      auto Bv = param_real(n.input[2]);
      for (int o = 0; o < out && o < (int)Bv.size(); ++o) breal[o] = Bv[o];
    }
    for (int o = 0; o < out; ++o) beff[o] = (int64_t)llround(breal[o] / (sx * sw));
    for (int o = 0; o < out; ++o) {
      int64_t ws = 0;
      for (int i = 0; i < in; ++i) ws += W[tb ? (int64_t)o * in + i : (int64_t)i * out + o];
      beff[o] -= (int64_t)xz * ws;
    }
    bool hr = party == ALICE && ref_.count(xin);
    const auto *xr = hr ? &ref_.at(xin) : nullptr;
    const int R = in > 0 ? (int)(xf.size() / in) : 1; // number of rows ([...,in] @ W)
    // PER-TOKEN: per-row input scales -> per-row bias int = round(breal/(s_row*s_w)) (symmetric =>
    // xz=0).
    const std::vector<double> *sxr =
        (sym_act_bits > 0 && rscale_.count(xin)) ? &rscale_.at(xin) : nullptr;
    auto srow = [&](int rr) { return sxr ? (*sxr)[std::min((int)sxr->size() - 1, rr)] : sx; };
    auto brow = [&](int rr, int o) -> int64_t {
      return sxr ? (int64_t)llround(breal[o] / (srow(rr) * sw)) : beff[o];
    };
    auto widx = [&](int o, int i) { return tb ? (int64_t)o * in + i : (int64_t)i * out + o; };
    std::vector<IntFp> y((size_t)R * out);
    std::vector<int64_t> r;
    if (hr) r.resize((size_t)R * out);
    // QuaRot online Hadamard: a PUBLIC orthonormal matrix (known to the verifier), so
    // committed*public (free scalar-linear) -- NOT committed witness (would be billions of VOLE
    // mults at 4096x4096).
    const bool pub_w = n.input[1].find("hadamard") != std::string::npos;
    if (commit_weights && !pub_w) { // PRIVATE model: committed weights, QuickSilver IP-proof matmul
      static double t_commit = 0, t_ipp = 0;
      static const bool MP = getenv("MATMUL_PROF");
      std::chrono::steady_clock::time_point _t;
      if (MP) _t = std::chrono::steady_clock::now();
      // Commit the weights BATCHED: one batch_feed per row (out feeds of `in`) instead of
      // out*in per-element commit_a feeds.  Encoding matches commit_a (signed -> field image).
      // Under W_PRECOMMIT the rows were committed up front (precommit_weights) -- reuse them.
      std::vector<std::vector<IntFp>> Wrows;
      { auto pit = wpre_.find(n.name);
        if (pit != wpre_.end()) { Wrows = std::move(pit->second); wpre_.erase(pit); }
      }
      if (Wrows.empty()) {
        Wrows.assign(out, std::vector<IntFp>(in));
        std::vector<uint64_t> wraw((size_t)in, 0);
        for (int o = 0; o < out; ++o) {
          if (party == ALICE)
            for (int i = 0; i < in; ++i) { int64_t v = W[widx(o, i)];
              wraw[(size_t)i] = v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v); }
          batch_feed(Wrows[o].data(), wraw.data(), (int64_t)in);
        }
      }
      std::vector<IntFp> bc;
      std::vector<std::vector<IntFp>>
          bcr; // per-output (default) or PER-ROW (per-token) bias commits
      if (sxr) {
        bcr.assign(R, std::vector<IntFp>(out));
        std::vector<uint64_t> braw((size_t)out, 0);
        for (int rr = 0; rr < R; ++rr) {
          if (party == ALICE)
            for (int o = 0; o < out; ++o) { int64_t v = brow(rr, o);
              braw[(size_t)o] = v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v); }
          batch_feed(bcr[rr].data(), braw.data(), (int64_t)out);
        }
      } else {
        bc.resize(out);
        std::vector<uint64_t> braw((size_t)out, 0);
        if (party == ALICE)
          for (int o = 0; o < out; ++o) { int64_t v = beff[o];
            braw[(size_t)o] = v >= 0 ? (uint64_t)v : PR - (uint64_t)(-v); }
        batch_feed(bc.data(), braw.data(), (int64_t)out);
      }
      if (MP) {
        t_commit += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t).count();
        _t = std::chrono::steady_clock::now();
      }
      static const bool FV = !getenv("FREIVALDS") || atoi(getenv("FREIVALDS")) != 0;   // DEFAULT ON (measured -4.2% e2e @500Mbps); FREIVALDS=0 disables
      // The matmul output outm[out][R] is a SECOND full out*R tensor alongside y (the LM head at
      // out=262144, R=512 duplicates 2.1 GB).  For a large-output matmul, process the R rows in
      // blocks: outm is then only out*rblk, freed after each block is folded into y.  Each block is
      // an independent Freivalds/IP proof (one extra batched round per block).  Small matmuls run whole.
      size_t rblk = 64;
      if (const char *e = getenv("MM_ROW_BLK")) rblk = (size_t)atoll(e);
      const bool mm_chunk = FV && rblk > 0 && rblk < (size_t)R && (size_t)R * out > ((size_t)16 << 20);
      const size_t step = mm_chunk ? rblk : (size_t)R;
      std::vector<std::vector<IntFp>> cols, outm;
      for (size_t r0 = 0; r0 < (size_t)R; r0 += step) {
        const size_t rb = std::min(step, (size_t)R - r0);
        cols.assign(rb, std::vector<IntFp>((size_t)in));
        for (size_t rr = 0; rr < rb; ++rr)
          for (int i = 0; i < in; ++i) cols[rr][i] = xf[(int64_t)(r0 + rr) * in + i];
        if (FV) // Mystique Freivalds: one length-M IP per matmul instead of out*R per-entry IPs
          qi->ipp->matmul_mm_freivalds(outm, cols, Wrows, qi->mat_threads); // outm[out][rb]
        else
          qi->ipp->matmul_mm(outm, cols, Wrows, qi->mat_threads); // outm[out][rb]
        for (size_t rr = 0; rr < rb; ++rr)
          for (int o = 0; o < out; ++o) {
            const size_t gr = r0 + rr;
            y[(int64_t)gr * out + o] = outm[o][rr] + (sxr ? bcr[gr][o] : bc[o]);
            if (hr) // reuse the committed matmul accumulator (VAL) -- no R*out*in plaintext recompute
              r[(int64_t)gr * out + o] =
                  brow((int)gr, o) + QuantInfer::decode_signed((uint64_t)VAL(outm[o][rr].value));
          }
      }
      if (MP) {
        t_ipp += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t).count();
        fprintf(stderr, "[MATMUL_PROF] weight-commit=%.1fs  IP-proof=%.1fs\n", t_commit, t_ipp);
      }
    } else { // PUBLIC model: committed*public free linear (scalar)
      for (int rr = 0; rr < R; ++rr)
        for (int o = 0; o < out; ++o) {
          IntFp acc = pub(brow(rr, o));
          int64_t ra = brow(rr, o);
          for (int i = 0; i < in; ++i) {
            acc = acc + xf[(int64_t)rr * in + i] * pub(W[widx(o, i)]);
            if (hr) ra += (*xr)[(int64_t)rr * in + i] * W[widx(o, i)];
          }
          y[(int64_t)rr * out + o] = acc;
          if (hr) r[(int64_t)rr * out + o] = ra;
        }
    }
    double sout = (sym_act_bits > 0 || !uniform_requant)
                      ? sx * sw
                      : 1.0 / (double)(1LL << P); // sym mode: keep accumulator scale; the run()
                                                  // hook quantizes per-token
    if (uniform_requant && sym_act_bits == 0 &&
        sout != sx * sw) { // requant accumulator (2^-2P) back to 2^-P (fp32 path)
      auto ce = QuantInfer::scale_search((sx * sw) / sout);
      std::vector<IntFp> yq;
      qi->requant_cm(yq, y, (uint64_t)ce.first, (std::size_t)(-ce.second));
      y = std::move(yq);
      if (hr) {
        int sh = -ce.second;
        int64_t c = ce.first, D = (int64_t)1 << sh;
        for (auto &v : r) {
          int64_t num = c * v;
          v = (num + D / 2) >> sh /* round-half-up: matches requant_cm_one */;
        }
      }
    }
    std::vector<int> ns = shp_.at(xin);
    if (!ns.empty())
      ns.back() = out;
    else
      ns = {out};
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = ns;
    scale_[n.output[0]] = sout;
    zp_[n.output[0]] = 0;
    if (hr) ref_[n.output[0]] = std::move(r);
    if (sym_act_bits > 0) {
      std::vector<double> ro(R);
      for (int rr = 0; rr < R; ++rr) ro[rr] = srow(rr) * sw;
      rscale_[n.output[0]] = std::move(ro);
    } // per-row output scale
  }

  // ---- Relu: max(zp, x_q)  (dequant >= 0 <=> x_q >= zp) ----
  void relu(const OnnxNode &n) {
    const std::string &in = n.input[0];
    const auto &x = act_.at(in);
    const int z = zp_.at(in);
    std::vector<IntFp> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      IntFp ge = cmp->is_le(pub(z), x[i], 30);
      y[i] = sel(ge, x[i], pub(z));
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(in);
    scale_[n.output[0]] = scale_.at(in);
    zp_[n.output[0]] = z;
    if (party == ALICE && ref_.count(in)) {
      const auto &xr = ref_.at(in);
      std::vector<int64_t> r(xr.size());
      for (size_t i = 0; i < xr.size(); ++i) r[i] = xr[i] > z ? xr[i] : z;
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- MaxPool: max x_q over window (monotone in dequant) ----
  void maxpool(const OnnxNode &n) {
    const std::string &in = n.input[0];
    const auto &xf = act_.at(in);
    const auto &xs = shp_.at(in);
    const int C = xs[0], H = xs[1], Wd = xs[2];
    const int K = n.attr.count("kernel_shape") ? (int)n.attr.at("kernel_shape").ints[0] : 2;
    const int stride = n.attr.count("strides") ? (int)n.attr.at("strides").ints[0] : K,
              pad = n.attr.count("pads") ? (int)n.attr.at("pads").ints[0] : 0;
    const int Ho = conv_out_dim(H, K, stride, pad), Wo = conv_out_dim(Wd, K, stride, pad);
    bool hr = party == ALICE && ref_.count(in);
    const auto *xr = hr ? &ref_.at(in) : nullptr;
    std::vector<IntFp> out((size_t)C * Ho * Wo);
    std::vector<int64_t> r;
    if (hr) r.resize((size_t)C * Ho * Wo);
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow) {
          std::vector<IntFp> win;
          int64_t mx = INT64_MIN;
          for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj) {
              int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
              if (ih >= 0 && ih < H && iw >= 0 && iw < Wd) {
                int64_t idx = (int64_t)c * H * Wd + ih * Wd + iw;
                win.push_back(xf[idx]);
                if (hr) mx = std::max(mx, (*xr)[idx]);
              }
            }
          out[(int64_t)c * Ho * Wo + oh * Wo + ow] = cmp->max_value(win, 30);
          if (hr) r[(int64_t)c * Ho * Wo + oh * Wo + ow] = mx;
        }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {C, Ho, Wo};
    scale_[n.output[0]] = scale_.at(in);
    zp_[n.output[0]] = zp_.at(in);
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- GlobalAveragePool: round(mean(x_q)) ----
  void gap(const OnnxNode &n) {
    const std::string &in = n.input[0];
    const auto &xf = act_.at(in);
    const auto &xs = shp_.at(in);
    const int C = xs[0], HW = prod(xs) / C;
    bool hr = party == ALICE && ref_.count(in);
    const auto *xr = hr ? &ref_.at(in) : nullptr;
    std::vector<IntFp> out(C);
    std::vector<int64_t> r;
    if (hr) r.resize(C);
    for (int c = 0; c < C; ++c) {
      IntFp s = pub(0);
      int64_t sm = 0;
      for (int p = 0; p < HW; ++p) {
        s = s + xf[(int64_t)c * HW + p];
        if (hr) sm += (*xr)[(int64_t)c * HW + p];
      }
      out[c] = divr->round_div(s, pub(HW));
      if (hr) r[c] = sm >= 0 ? (sm + HW / 2) / HW : -(((-sm) + HW / 2) / HW);
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = {C, 1, 1};
    scale_[n.output[0]] = scale_.at(in);
    zp_[n.output[0]] = zp_.at(in);
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- Add: FC bias (one operand an initializer) OR residual (two activations) ----
  // A DequantizeLinear of a FOLDED constant (e.g. BERT's token_type embedding, whose ids are all 0
  // -> Gather+Quantize const-folded) lands in cref_ (int table + scale), but elementwise ops read
  // act_. Materialize it as PUBLIC committed IntFps at its own scale so Add/Mul/Sub can consume it.
  void materialize_cref(const std::string &nm) {
    if (act_.count(nm) || !cref_.count(nm)) return;
    const auto &v = cref_.at(nm);
    std::vector<IntFp> a(v.size());
    std::vector<int64_t> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      a[i] = pub(v[i]);
      r[i] = v[i];
    }
    act_[nm] = std::move(a);
    ref_[nm] = std::move(r); // scale_/zp_ were set by dequant
  }
  // Read a weight/param as fp32 reals whether it is a RAW initializer (fp32/int) or a QDQ
  // DequantizeLinear constant in cref_ (int table * scale).  BERT QDQ wraps LN gamma/beta in DQ.
  std::vector<double> param_real(const std::string &nm) {
    std::vector<double> v;
    if (g_->initializer.count(nm)) {
      const auto &t = g_->initializer.at(nm);
      size_t m = std::max(t.data.size(), t.i64.size());
      v.resize(m);
      for (size_t i = 0; i < m; ++i) v[i] = (t.dtype == 1 ? (double)t.data[i] : (double)t.i64[i]);
    } else if (cref_.count(nm)) {
      const auto &c = cref_.at(nm);
      double s = scale_.at(nm);
      int64_t z = zp_.count(nm) ? zp_.at(nm) : 0;
      v.resize(c.size());
      for (size_t i = 0; i < c.size(); ++i) v[i] = ((double)c[i] - (double)z) * s;
    } else
      throw std::runtime_error("param_real: not found " + nm);
    return v;
  }
  void add(const OnnxNode &n) {
    materialize_cref(n.input[0]);
    materialize_cref(n.input[1]);
    const std::string &ain = n.input[0], &bin = n.input[1];
    if (g_->initializer.count(ain) || g_->initializer.count(bin)) { // bias add: act + constant
      const std::string actin = g_->initializer.count(ain) ? bin : ain,
                        cst = g_->initializer.count(ain) ? ain : bin;
      const auto &A = act_.at(actin);
      const double sa = scale_.at(actin);
      const int za = zp_.at(actin);
      const auto &bt = g_->initializer.at(cst);
      const int nb = (int)std::max(bt.data.size(), bt.i64.size());
      bool hr = party == ALICE && ref_.count(actin);
      const auto *ar = hr ? &ref_.at(actin) : nullptr;
      std::vector<IntFp> y(A.size());
      std::vector<int64_t> r;
      if (hr) r.resize(A.size());
      // GPT-2 additive mask is -inf -> clamp to ~-256 REAL so the softmax exp-key stays in [0,512)
      // at ANY P (else the mask weakens at high P).  The constant is quantized at the activation's
      // scale sa, so the clamp is 256/sa in those units: 2^(P+8) when sa = 2^-P, but the SYM_BITS
      // per-tensor scales are not 2^-P and a fixed 2^(P+8) there pushed the exp key out of range.
      const int64_t MCLAMP = (int64_t)std::min(std::llround(256.0 / sa), (long long)1 << 40);
      // Quantize the constant + build its public labels ONCE per (constant @ scale), reused across
      // the broadcast (heads) AND across layers -- the GPT-2 causal mask is identical every layer,
      // so this turns 38M per-element float-clamp + pub_label into one nb-sized precompute.
      const std::string ck = cst + "@" + std::to_string(sa);
      auto cit = add_const_cache_.find(ck);
      if (cit == add_const_cache_.end()) {
        const int M = std::max(1, nb);
        std::vector<int64_t> bqv((size_t)M);
        std::vector<IntFp> pubv((size_t)M);
        for (int j = 0; j < M; ++j) {
          int64_t bq;
          if (bt.dtype == 1) {
            double bf = (j < (int)bt.data.size() ? (double)bt.data[j] : 0.0) / sa;
            bq = std::isfinite(bf) ? (int64_t)std::max(-(double)MCLAMP,
                                                       std::min((double)MCLAMP, (double)llround(bf)))
                                   : (bf < 0 ? -MCLAMP : MCLAMP);
          } else
            bq = j < (int)bt.i64.size() ? std::max(-MCLAMP, std::min(MCLAMP, bt.i64[j])) : 0;
          bqv[(size_t)j] = bq;
          pubv[(size_t)j] = pub(bq);
        }
        cit = add_const_cache_.emplace(ck, std::make_pair(std::move(bqv), std::move(pubv))).first;
      }
      const auto &bqv = cit->second.first;
      const auto &pubv = cit->second.second;
      for (size_t i = 0; i < A.size(); ++i) {
        const size_t j = nb ? (i % (size_t)nb) : 0;
        y[i] = A[i] + pubv[j];
        if (hr) r[i] = (*ar)[i] + bqv[j];
      }
      act_[n.output[0]] = std::move(y);
      shp_[n.output[0]] = shp_.at(actin);
      scale_[n.output[0]] = sa;
      zp_[n.output[0]] = za;
      if (hr) ref_[n.output[0]] = std::move(r);
      return;
    }
    const auto &A = act_.at(ain), &B = act_.at(bin);
    const double sa = scale_.at(ain), sb = scale_.at(bin);
    const int za = zp_.at(ain), zb = zp_.at(bin);
    bool hr = party == ALICE && ref_.count(ain) && ref_.count(bin);
    const auto *ar = hr ? &ref_.at(ain) : nullptr, *br = hr ? &ref_.at(bin) : nullptr;
    const size_t na = A.size(), nb = B.size(),
                 no = std::max(
                     na, nb); // numpy broadcast (attn scores [.,H,S,S] + causal mask [.,1,S,S])
    std::vector<IntFp> y(no);
    std::vector<int64_t> r;
    if (hr) r.resize(no);
    const std::vector<double> *rsa =
        (sym_act_bits > 0 && rscale_.count(ain)) ? &rscale_.at(ain) : nullptr;
    const std::vector<double> *rsb =
        (sym_act_bits > 0 && rscale_.count(bin)) ? &rscale_.at(bin) : nullptr;
    if (rsa && rsb && na == nb) { // PER-TOKEN residual add: per-row rescale B -> A's per-row scale
      int d = (shp_.count(ain) && !shp_.at(ain).empty()) ? shp_.at(ain).back() : (int)na;
      int rows = d > 0 ? (int)(na / d) : 1;
      std::vector<double> ro(rows);
      for (int rr = 0; rr < rows; ++rr) {
        double s_a = (*rsa)[std::min((int)rsa->size() - 1, rr)],
               s_b = (*rsb)[std::min((int)rsb->size() - 1, rr)];
        ro[rr] = s_a;
        auto ce = QuantInfer::scale_search(s_b / s_a);
        int64_t c = ce.first;
        int e = ce.second;
        const size_t base = (size_t)rr * d;
        // BATCHED per-row requant of this row's d B-elements, then a local add.
        std::vector<IntFp> Bsh(d), bq(d);
        for (int j = 0; j < d; ++j) Bsh[j] = B[base + j] + pub(-zb);
        qi->requant_cm_batch(bq.data(), Bsh.data(), (uint64_t)c, (std::size_t)(-e), (size_t)d);
        for (int j = 0; j < d; ++j) {
          size_t i = base + j;
          y[i] = (A[i] + pub(-za)) + bq[j];
          if (hr) {
            int64_t N = c * ((*br)[i] - zb);
            int64_t rbi =
                (e > 0) ? (N << e) : (e == 0 ? N : ((N + ((int64_t)1 << (-e - 1))) >> (-e)));
            r[i] = ((*ar)[i] - za) + rbi;
          }
        }
      }
      act_[n.output[0]] = std::move(y);
      shp_[n.output[0]] = shp_.at(ain);
      scale_[n.output[0]] = ro.empty() ? 1.0 : ro[0];
      zp_[n.output[0]] = 0;
      rscale_[n.output[0]] = std::move(ro);
      if (hr) ref_[n.output[0]] = std::move(r);
      return;
    }
    const bool do_rq = (sb != sa); // rescale B to A's scale (equal scales -> no-op)
    std::pair<int, int> ce;
    int64_t Dd = 1; // scale_search ONCE per node, NOT per element
    if (do_rq) {
      ce = QuantInfer::scale_search(sb / sa);
      Dd = (int64_t)1 << (-ce.second);
    }
    // BATCHED requant of B (rescaled to A's scale) once over the nb unique elements,
    // then a local add loop -- vs a per-element requant_cm_one inside the add loop.
    std::vector<IntFp> bq(nb);
    std::vector<int64_t> rbq;
    if (hr) rbq.resize(nb);
    { std::vector<IntFp> Bsh(nb);
      for (size_t ib = 0; ib < nb; ++ib) Bsh[ib] = B[ib] + pub(-zb);
      if (do_rq) qi->requant_cm_batch(bq.data(), Bsh.data(), (uint64_t)ce.first, (std::size_t)(-ce.second), nb);
      else bq = std::move(Bsh);
      if (hr) for (size_t ib = 0; ib < nb; ++ib) {
        int64_t rbi = (*br)[ib] - zb;
        if (do_rq) { int e = ce.second; int64_t N = (int64_t)ce.first * rbi;
          rbi = (e >= 0) ? (N << e) : ((N + ((int64_t)1 << (-e - 1))) >> (-e)); }
        rbq[ib] = rbi;
      }
    }
    for (size_t i = 0; i < no; ++i) {
      size_t ia = i % na, ib = i % nb;
      y[i] = (A[ia] + pub(-za)) + bq[ib];
      if (hr) r[i] = ((*ar)[ia] - za) + rbq[ib];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = na >= nb ? shp_.at(ain) : shp_.at(bin);
    scale_[n.output[0]] = sa;
    zp_[n.output[0]] = 0;
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- two-activation MatMul (attention Q.K^T / attn.V), batched.  PER-TENSOR (DeepProve form):
  // keep both committed operands at their own scales; output the raw accumulator at s_A*s_B (the
  // hook / the next op's rescale requantizes it).  No 2^-P rescale.  Overflow-safe: int12 inputs ->
  // acc ~2^28 << field, and there is NO requant here so the c*acc multiply the old comment warned
  // about never happens.
  void matmul_aa(const OnnxNode &n) {
    const auto &A = act_.at(n.input[0]);
    const auto &B = act_.at(n.input[1]);
    const double sA = scale_.at(n.input[0]), sB = scale_.at(n.input[1]);
    const auto &sAs = shp_.at(n.input[0]);
    const auto &sBs = shp_.at(n.input[1]);
    const bool hr = party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1]);
    const std::vector<int64_t> *ar = hr ? &ref_.at(n.input[0]) : nullptr,
                               *br = hr ? &ref_.at(n.input[1]) : nullptr;
    // Dims from B (the full-shaped operand): contraction K = B's penultimate dim, batch G = B's
    // leading dims; rows Mm = A's element count / (G*K).  Robust to a flattened A shape (e.g. the
    // softmax scores arriving as [G*M*K] instead of [G,M,K]) -- which otherwise set K=A.back() and
    // drove an out-of-bounds read past the operands (benign on arm64, garbage products on x86).
    const int N = sBs.back();
    const int K = (int)(sBs.size() >= 2 ? sBs[sBs.size() - 2] : 1);
    int G = 1;
    for (size_t i = 0; i + 2 < sBs.size(); ++i) G *= sBs[i];
    const int Mm = (G > 0 && K > 0) ? (int)(A.size() / ((size_t)G * (size_t)K)) : 1;
    const bool Bb = sBs.size() >= 3;
    // Threaded QuickSilver IP matmul (per-head Q.K^T / attn.V): ONE batched inner-product proof
    // per score instead of K naive committed mul-gates -> the seq^2 attention is no longer a
    // single-threaded loop (matches ml-fixed).  Numerically identical to the Sum_k A*B raw
    // accumulator; per-tensor form keeps the raw acc at scale sA*sB (no rescale here).
    std::vector<IntFp> out;
    qi->ipp->matmul_attn(out, A.data(), B.data(), G, Mm, N, K, Bb, qi->mat_threads);
    std::vector<int64_t> r;
    if (hr) { // reuse the committed accumulator (VAL) -- no G*Mm*N*K plaintext recompute
      r.resize((size_t)G * Mm * N);
      for (size_t i = 0; i < r.size(); ++i)
        r[i] = QuantInfer::decode_signed((uint64_t)VAL(out[i].value));
      (void)ar;
      (void)br;
    }
    std::vector<int> ns = sBs;  // output [.., Mm, N] = B's shape with the contraction dim -> Mm rows
    if (ns.size() >= 2) ns[ns.size() - 2] = Mm;
    else ns = (Mm > 1 ? std::vector<int>{Mm, N} : std::vector<int>{N});
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = ns;
    scale_[n.output[0]] = sA * sB;
    zp_[n.output[0]] = 0;
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- Softmax (per last-dim row) via QuantSoftmax ----
  void softmax(const OnnxNode &n) {
    if (!exp_tab) throw std::runtime_error("Softmax needs exp table");
    const std::string &in = n.input[0];
    std::vector<int64_t> xr;
    auto x = rescale_to_P(in, &xr);
    if (getenv("DBG_SM") && party == ALICE && !xr.empty()) {
      int64_t mn = INT64_MAX, mx = INT64_MIN;
      for (auto v : xr) { mn = std::min(mn, v); mx = std::max(mx, v); }
      printf("[softmax] in=%s scale_in=%.3e range@2^-P=[%lld,%lld] keydomain=[0,%d)\n",
             in.c_str(), scale_.count(in) ? scale_.at(in) : -1.0, (long long)mn, (long long)mx, 1 << (P + 9));
      fflush(stdout);
    }
    const auto &xs = shp_.at(in);
    const int d = xs.back(), rows = prod(xs) / d; // QDQ scale -> 2^-P
    const bool lse =
        getenv("FUSED_NL") != nullptr; // DeepProve LogSumExp (no division); else zkGPT round_div
    // sum-to-one tolerance DERIVED from the error analysis (no-clamp exp table -> masked terms = 0,
    // no leak): each unmasked out_i rounds +-1/2 (<= d/2 total) + the log-sum-exp shift rounds
    // +-1/2 (-> +-1 on the sum). (At the 2^-P output the L1 error stays <0.01 for d <=
    // ~0.02*2^P; longer rows need a finer output scale.)
    const uint64_t err = (uint64_t)d / 2 + 2;
    std::size_t wfb = 0;
    for (uint64_t v = 2 * err; v; v >>= 1) ++wfb;
    QuantSoftmax sm(qi, cmp, exp_tab, divr, P, score_bits);
    // CAUSAL_SM: square self-attention scores [..,seq_q,seq_k] (seq_q==seq_k==d) with a lower-triangular
    // causal mask -> row r's query index is r%seq_q, so keys [0,query+1) are unmasked.  Pass that PUBLIC
    // valid_len so the division softmax drops the redundant per-element clamp on the masked/valid split.
    const bool causal_sm = getenv("CAUSAL_SM") != nullptr && !lse && (int)xs.size() >= 2 &&
                           (int)xs[xs.size() - 2] == d;
    const int seq_q = ((int)xs.size() >= 2) ? (int)xs[xs.size() - 2] : rows;
    std::vector<IntFp> y(x.size());
    for (int r = 0; r < rows; ++r) {
      std::vector<IntFp> row(x.begin() + (int64_t)r * d, x.begin() + (int64_t)(r + 1) * d);
      std::vector<IntFp> o;
      if (lse) {
        const int64_t *rl =
            (party == ALICE && (int)xr.size() == rows * d) ? &xr[(int64_t)r * d] : nullptr;
        o = sm.forward_lse(row, rl, rcmp, err, wfb);
      } else
        o = sm.forward(row, causal_sm ? (r % seq_q) + 1 : -1);
      for (int j = 0; j < d; ++j) y[(int64_t)r * d + j] = o[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = xs;
    scale_[n.output[0]] = 1.0 / (double)(1LL << P);
    zp_[n.output[0]] = 0; // probs @ 2^-P
    // exact-integer plaintext ref (matches QuantSoftmax) + OVERFLOW probe: the exp-LUT key must
    // stay in [0,T).
    if (party == ALICE && (int)xr.size() == rows * d) {
      std::vector<int64_t> r((size_t)rows * d);
      const double S = (double)(1LL << P);
      const int64_t T = (int64_t)1 << (P + 9);
      int64_t maxt = 0;
      for (int rr = 0; rr < rows; ++rr) {
        int64_t m = INT64_MIN;
        for (int i = 0; i < d; ++i) m = std::max(m, xr[(int64_t)rr * d + i]);
        if (lse) { // DeepProve LSE ref: out_i = exp_table[(m - logit_i) + shift], shift =
                   // round(log(sum_exp) * 2^P)
          double sum = 0;
          for (int i = 0; i < d; ++i) sum += std::exp(-(double)(m - xr[(int64_t)rr * d + i]) / S);
          int64_t shift = (int64_t)llround(std::log(sum) * S);
          if (shift < 0) shift = 0;
          for (int i = 0; i < d; ++i) {
            int64_t idx = (m - xr[(int64_t)rr * d + i]) + shift;
            if (idx > maxt) maxt = idx;
            int64_t ev = (int64_t)llround(std::exp(-(double)idx / S) * S);
            if (ev < 0) ev = 0;
            r[(int64_t)rr * d + i] = ev;
          } // NO clamp (no-clamp exp table): masked -> 0
        } else {
          std::vector<int64_t> e(d);
          int64_t sum = 0;
          for (int i = 0; i < d; ++i) {
            int64_t t = m - xr[(int64_t)rr * d + i];
            if (t > maxt) maxt = t;
            int64_t ev = (int64_t)llround(std::exp(-(double)t / S) * S);
            if (ev < 1) ev = 1;
            e[i] = ev;
            sum += ev;
          }
          for (int i = 0; i < d; ++i) {
            int64_t N = e[i] * ((int64_t)1 << P);
            r[(int64_t)rr * d + i] = (N + sum / 2) / sum;
          }
        }
      }
      ref_[n.output[0]] = std::move(r);
      if (getenv("DEBUG_REVEAL"))
        fprintf(stderr, "[SOFTMAX] %s %s max(idx)=%lld  exp_domain T=%lld -> %s\n",
                n.output[0].substr(0, 36).c_str(), lse ? "LSE" : "div", (long long)maxt,
                (long long)T, maxt >= T ? "*** OVERFLOW ***" : "in-range");
    }
  }
  // ---- LayerNormalization (per last-dim row) via QuantLayerNorm ----
  void layernorm(const OnnxNode &n) {
    if (!rcmp) throw std::runtime_error("LayerNorm needs RangeCheckLU");
    // LayerNorm is SCALE-INVARIANT: (x-mu)/sigma is unchanged by the input scale.  Compute on the
    // NATIVE (bounded int8) act_ rather than rescale_to_P -- in a PRE-LN model (GPT-2) the residual
    // grows unboundedly, so rescaling it to 2^-P makes var huge and dinv=2^2P/sqrt(var) collapse to
    // a coarse 1-bit integer (the layer-by-layer collapse).  The int8 input keeps var bounded ->
    // dinv precise.  dinv formula + requant are UNCHANGED (the scale cancels).  (BERT post-LN
    // already bounded.)
    const std::string &in = n.input[0];
    const auto &x = act_.at(in);
    std::vector<int64_t> xr;
    if (party == ALICE && ref_.count(in)) xr = ref_.at(in);
    const auto &xs = shp_.at(in);
    const int d = xs.back(), rows = prod(xs) / d;
    std::vector<int64_t> qg(d), qb(d);
    auto G = param_real(n.input[1]), B = param_real(n.input[2]);
    const double sc = (double)(1LL << P);
    for (int i = 0; i < d; ++i) {
      qg[i] = (int64_t)llround(G[i] * sc);
      qb[i] = (int64_t)llround(B[i] * sc);
    }
    std::vector<IntFp> cg(d), cb(d);
    for (int i = 0; i < d; ++i) {
      cg[i] = commit_a(qg[i]);
      cb[i] = commit_a(qb[i]);
    }
    QuantLayerNorm ln(qi, rcmp, divr, P, d);
    std::vector<IntFp> y(x.size());
    for (int r = 0; r < rows; ++r) {
      std::vector<IntFp> row(x.begin() + (int64_t)r * d, x.begin() + (int64_t)(r + 1) * d), o;
      ln.forward(o, row, cg, cb);
      for (int j = 0; j < d; ++j) y[(int64_t)r * d + j] = o[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = xs;
    scale_[n.output[0]] = 1.0 / sc;
    zp_[n.output[0]] = 0;
    // exact-integer plaintext ref (matches QuantLayerNorm step-by-step) so the inline mimic
    // survives LN
    if (party == ALICE && (int)xr.size() == rows * d) {
      std::vector<int64_t> r((size_t)rows * d);
      const int64_t HP = (int64_t)1 << (P - 1), DP = (int64_t)1 << P;
      int64_t minvq = INT64_MAX, maxdinv = 0, maxd2v = 0;
      auto rdiv = [](int64_t N, int64_t D) {
        return N >= 0 ? (N + D / 2) / D : -(((-N) + D / 2) / D);
      };
      auto rq = [&](int64_t N) {
        return (N + HP) >> P /* round-half-up: matches requant_cm_one */;
      };
      for (int rr = 0; rr < rows; ++rr) {
        int64_t sumx = 0;
        for (int i = 0; i < d; ++i) sumx += xr[(int64_t)rr * d + i];
        int64_t mu = rdiv(sumx, d), ssq = 0;
        std::vector<int64_t> c(d);
        for (int i = 0; i < d; ++i) {
          c[i] = xr[(int64_t)rr * d + i] - mu;
          ssq += c[i] * c[i];
        }
        int64_t vq = rdiv(ssq, d) + 1;
        int64_t dinv = llround((double)((uint64_t)1 << (2 * P)) /
                               std::sqrt((double)vq)); // +1 eps matches QuantLayerNorm
        if (vq < minvq) minvq = vq;
        if (dinv > maxdinv) maxdinv = dinv;
        if (std::llabs(dinv) * std::llabs(dinv) > maxd2v)
          maxd2v = std::llabs(dinv) * std::llabs(dinv) * (vq > 0 ? 1 : 0);
        for (int i = 0; i < d; ++i) r[(int64_t)rr * d + i] = rq(rq(c[i] * dinv) * qg[i]) + qb[i];
      }
      ref_[n.output[0]] = std::move(r);
      if (getenv("DEBUG_REVEAL"))
        fprintf(stderr,
                "[LN] %-44s min(var_q)=%lld  max(dinv)=%lld  dinv^2*vq~2^%.0f (field PR~2^61)\n",
                n.output[0].substr(0, 42).c_str(), (long long)minvq, (long long)maxdinv,
                maxd2v > 0 ? std::log2((double)maxd2v * (double)std::max((int64_t)1, minvq)) : 0.0);
    }
  }
  // ---- RMSNormalization (Llama/Qwen/Gemma) via QuantRMSNorm.  Like LayerNorm but no mean and no
  // beta; scale-invariant, so compute on the native committed act_ and emit at 2^-P.  (Gemma's
  // (1+gamma): pre-add 2^P to qg -- not needed for Qwen/Llama.) ----
  void rmsnorm(const OnnxNode &n) {
    if (!rcmp) throw std::runtime_error("RMSNorm needs RangeCheckLU");
    const std::string &in = n.input[0];
    const auto &x = act_.at(in);
    std::vector<int64_t> xr;
    if (party == ALICE && ref_.count(in)) xr = ref_.at(in);
    const auto &xs = shp_.at(in);
    const int d = xs.back(), rows = prod(xs) / d;
    auto G = param_real(n.input[1]);
    const double sc = (double)(1LL << P);
    std::vector<int64_t> qg(d);
    for (int i = 0; i < d; ++i) qg[i] = (int64_t)llround(G[i] * sc);
    std::vector<IntFp> cg(d);
    for (int i = 0; i < d; ++i) cg[i] = commit_a(qg[i]);
    QuantRMSNorm rms(qi, rcmp, divr, P, (uint64_t)d);
    std::vector<IntFp> y(x.size());
    for (int r = 0; r < rows; ++r) {
      std::vector<IntFp> row(x.begin() + (int64_t)r * d, x.begin() + (int64_t)(r + 1) * d), o;
      rms.forward(o, row, cg);
      for (int j = 0; j < d; ++j) y[(int64_t)r * d + j] = o[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = xs;
    scale_[n.output[0]] = 1.0 / sc;
    zp_[n.output[0]] = 0;
    // exact-integer plaintext ref (matches QuantRMSNorm step-by-step): ms=mean(x^2), dinv=round(
    // 2^2P/sqrt(ms)), y=rq(x*dinv), out=rq(y*gamma) -- no eps, no mean, no beta.
    if (party == ALICE && (int)xr.size() == rows * d) {
      std::vector<int64_t> r((size_t)rows * d);
      const int64_t HP = (int64_t)1 << (P - 1);
      auto rdiv = [](int64_t N, int64_t D) { return N >= 0 ? (N + D / 2) / D : -(((-N) + D / 2) / D); };
      auto rq = [&](int64_t N) { return (N + HP) >> P; };
      for (int rr = 0; rr < rows; ++rr) {
        int64_t ssq = 0;
        for (int i = 0; i < d; ++i) { int64_t v = xr[(int64_t)rr * d + i]; ssq += v * v; }
        int64_t ms = rdiv(ssq, d);
        int64_t dinv = llround((double)((uint64_t)1 << (2 * P)) / std::sqrt((double)(ms > 0 ? ms : 1)));
        for (int i = 0; i < d; ++i)
          r[(int64_t)rr * d + i] = rq(rq(xr[(int64_t)rr * d + i] * dinv) * qg[i]);
      }
      ref_[n.output[0]] = std::move(r);
    }
  }
  // ---- RotaryEmbedding (Llama/Qwen/Gemma), fused from the HF rotate_half decomposition. ----
  // x is committed [B,H,S,dh] at a per-tensor scale s_x; cos/sin are PUBLIC constants [.,.,S,dh] at
  // 2^-P with the two halves duplicated (cos[s,d]==cos[s,d+half]).  HF (NeoX) convention pairs
  // (d, d+half), NOT the interleaved (2i,2i+1) of apply_rope:
  //   out[d]      = x[d]*cos      - x[d+half]*sin
  //   out[d+half] = x[d+half]*cos + x[d]*sin        (cos,sin at index (s,d), d<half)
  // Each output is committed*public (FREE, no IP); one requant 2^P removes the cos/sin 2^-P factor,
  // so the output keeps the input scale (RoPE is a rotation -- norm/scale preserving).
  void rope(const OnnxNode &n) {
    const std::string &in = n.input[0];
    const auto &x = act_.at(in);
    const auto &xs = shp_.at(in);
    const int dh = xs.back(), half = dh / 2, S = xs[xs.size() - 2];
    const int64_t outer = (int64_t)prod(xs) / ((int64_t)S * dh); // B*H
    auto COS = param_real(n.input[1]), SIN = param_real(n.input[2]);
    const double sc = (double)(1LL << P);
    const int64_t coff = (int64_t)COS.size() - (int64_t)S * dh; // drop leading broadcast dims
    std::vector<int64_t> c((size_t)S * half), sn((size_t)S * half);
    for (int s = 0; s < S; ++s)
      for (int d = 0; d < half; ++d) {
        c[(size_t)s * half + d] = llround(COS[coff + (int64_t)s * dh + d] * sc);
        sn[(size_t)s * half + d] = llround(SIN[coff + (int64_t)s * dh + d] * sc);
      }
    std::vector<IntFp> pr(x.size());
    for (int64_t o = 0; o < outer; ++o)
      for (int s = 0; s < S; ++s) {
        const int64_t base = (o * S + s) * dh;
        for (int d = 0; d < half; ++d) {
          const IntFp a = x[base + d], b = x[base + d + half];
          const int64_t cc = c[(size_t)s * half + d], ss = sn[(size_t)s * half + d];
          pr[base + d] = a * pub_signed(cc) + b * pub_signed(-ss);
          pr[base + d + half] = b * pub_signed(cc) + a * pub_signed(ss);
        }
      }
    std::vector<IntFp> out;
    qi->requant_cm(out, pr, 1, (std::size_t)P);
    if (getenv("DBG_ROPE") && party == ALICE) {
      int64_t im = 0, om = 0;
      for (auto &v : x) { int64_t q = std::llabs((int64_t)QuantInfer::decode_signed(VAL(v.value))); if (q > im) im = q; }
      for (auto &v : out) { int64_t q = std::llabs((int64_t)QuantInfer::decode_signed(VAL(v.value))); if (q > om) om = q; }
      fprintf(stderr, "[rope] in=%s scale_in=%.4e dh=%d S=%d outer=%lld in|max|=%lld out|max|=%lld c[0]=%lld sn[0]=%lld\n",
              in.c_str(), scale_.at(in), dh, S, (long long)outer, (long long)im, (long long)om, (long long)c[0], (long long)sn[0]);
    }
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = xs;
    scale_[n.output[0]] = scale_.at(in);
    zp_[n.output[0]] = 0;
    if (party == ALICE && ref_.count(in)) {
      const auto &xr = ref_.at(in);
      std::vector<int64_t> r(xr.size());
      const int64_t HP = (int64_t)1 << (P - 1);
      auto rq = [&](int64_t N) { return (N + HP) >> P; };
      for (int64_t o = 0; o < outer; ++o)
        for (int s = 0; s < S; ++s) {
          const int64_t base = (o * S + s) * dh;
          for (int d = 0; d < half; ++d) {
            const int64_t a = xr[base + d], b = xr[base + d + half],
                          cc = c[(size_t)s * half + d], ss = sn[(size_t)s * half + d];
            r[base + d] = rq(a * cc - b * ss);
            r[base + d + half] = rq(b * cc + a * ss);
          }
        }
      ref_[n.output[0]] = std::move(r);
    }
  }
  // ---- Gelu via QuantActivation (lookup table) ----
  void gelu(const OnnxNode &n) {
    // BERT ASYMMETRIC-uint8 fused: look up the committed uint8 q directly (offset 0); the table
    // bakes (s_in,zp_in,s_out,zp_out) -> ONE lookup = dequant + gelu + requant + clamp.  No 2^-P
    // rescale.
    if (gelu_asym_tabs_.count(n.output[0])) {
      const std::string &in = n.input[0];
      const auto &x = act_.at(in);
      QuantActivation a(gelu_asym_tabs_.at(n.output[0]), 0);
      std::vector<IntFp> y;
      a.forward(y, x);
      const auto &sc = gelu_asym_scales_.at(n.output[0]); // {s_in, zp_in, s_out, zp_out}
      act_[n.output[0]] = std::move(y);
      shp_[n.output[0]] = shp_.at(in);
      scale_[n.output[0]] = sc[2];
      zp_[n.output[0]] = (int)llround(sc[3]);
      if (party == ALICE && ref_.count(in)) {
        const auto &xr = ref_.at(in);
        std::vector<int64_t> r(xr.size());
        const double k = std::sqrt(2.0 / M_PI);
        for (size_t i = 0; i < xr.size(); ++i) {
          double xv = ((double)xr[i] - sc[1]) * sc[0];
          double gv = 0.5 * xv * (1.0 + std::tanh(k * (xv + 0.044715 * xv * xv * xv)));
          long long q = llround(gv / sc[2] + sc[3]);
          r[i] = q < 0 ? 0 : (q > 255 ? 255 : q);
        }
        ref_[n.output[0]] = std::move(r);
      }
      return;
    }
    // DeepProve scale-baked FUSED path: look up the PER-TENSOR int directly (NO 2^-P rescale); the
    // per-op table bakes s_in and outputs gelu requanted to s_out + clamped -- ONE lookup = gelu +
    // requant + clamp.
    if (gelu_tabs_.count(n.output[0])) {
      const std::string &in = n.input[0];
      const auto &x = act_.at(in);
      QuantActivation a(gelu_tabs_.at(n.output[0]), gelu_fused_off);
      std::vector<IntFp> y;
      a.forward(y, x);
      double s_in = gelu_scales_.at(n.output[0]).first, s_out = gelu_scales_.at(n.output[0]).second;
      act_[n.output[0]] = std::move(y);
      shp_[n.output[0]] = shp_.at(in);
      scale_[n.output[0]] = s_out;
      zp_[n.output[0]] = 0;
      if (party == ALICE && ref_.count(in)) {
        const auto &xr = ref_.at(in);
        std::vector<int64_t> r(xr.size());
        const double k = std::sqrt(2.0 / M_PI);
        const int64_t LIM = ((int64_t)1 << (sym_act_bits - 1)) - 1;
        for (size_t i = 0; i < xr.size(); ++i) {
          double xv = (double)xr[i] * s_in;
          double gv = 0.5 * xv * (1.0 + std::tanh(k * (xv + 0.044715 * xv * xv * xv)));
          int64_t q = llround(gv / s_out);
          r[i] = q < -LIM ? -LIM : (q > LIM ? LIM : q);
        }
        ref_[n.output[0]] = std::move(r);
      }
      return;
    }
    if (!gelu_tab) throw std::runtime_error("Gelu needs table");
    const std::string &in = n.input[0];
    std::vector<int64_t> xr;
    auto xin = rescale_to_P(in, &xr); // QDQ scale -> gadget's 2^-P
    QuantActivation a(gelu_tab, gelu_off);
    std::vector<IntFp> y;
    a.forward(y, xin);
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(in);
    scale_[n.output[0]] = 1.0 / (double)(1LL << P);
    zp_[n.output[0]] = 0;
    if (party == ALICE && !xr.empty()) { // exact-int ref (true gelu) + overflow probe: |x_int| <
                                         // gelu_off or the LUT key is out of range
      std::vector<int64_t> r(xr.size());
      const double S = (double)(1LL << P), kk = std::sqrt(2.0 / M_PI);
      int64_t mx = 0;
      for (size_t i = 0; i < xr.size(); ++i) {
        if (std::llabs(xr[i]) > mx) mx = std::llabs(xr[i]);
        double x = (double)xr[i] / S;
        r[i] = (int64_t)llround(0.5 * x * (1.0 + std::tanh(kk * (x + 0.044715 * x * x * x))) * S);
      }
      ref_[n.output[0]] = std::move(r);
      if (getenv("DEBUG_REVEAL"))
        fprintf(stderr, "[GELU] max|x_int|=%lld  domain=+-%lld -> %s\n", (long long)mx,
                (long long)gelu_off,
                mx >= gelu_off ? "*** OVERFLOW (LUT key out of range) ***" : "in-range");
    }
  }
  // ---- SiLU (Llama/Qwen SwiGLU activation) via QuantActivation lookup.  Same gadget as Gelu:
  // bridge the per-tensor input to 2^-P, one LogUp lookup into build_silu_table, emit at 2^-P. ----
  void silu(const OnnxNode &n) {
    if (!silu_tab) throw std::runtime_error("SiLU needs table");
    const std::string &in = n.input[0];
    std::vector<int64_t> xr;
    auto xin = rescale_to_P(in, &xr); // per-tensor scale -> gadget 2^-P
    QuantActivation a(silu_tab, silu_off);
    std::vector<IntFp> y;
    a.forward(y, xin);
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(in);
    scale_[n.output[0]] = 1.0 / (double)(1LL << P);
    zp_[n.output[0]] = 0;
    if (party == ALICE && !xr.empty()) { // exact-int ref: silu(x)=x*sigmoid(x)
      std::vector<int64_t> r(xr.size());
      const double S = (double)(1LL << P);
      for (size_t i = 0; i < xr.size(); ++i) {
        double x = (double)xr[i] / S;
        r[i] = (int64_t)llround(x / (1.0 + std::exp(-x)) * S);
      }
      ref_[n.output[0]] = std::move(r);
    }
  }
  // ---- Tanh via QuantActivation (LogUp lookup) -- BERT pooler.  Same gadget as Gelu; the table
  // (build_tanh_table) bakes the input scale and the bounded (-1,1) output at 2^-P. ----
  void tanh_act(const OnnxNode &n) {
    // DeepProve scale-baked FUSED path: look up the committed uint8 q DIRECTLY (no 2^-P rescale);
    // the per-op table bakes (s_in,zp_in,s_out,zp_out) -> ONE lookup = dequant + tanh + requant +
    // clamp (asymmetric-aware).
    if (tanh_tabs_.count(n.output[0])) {
      const std::string &in = n.input[0];
      const auto &x = act_.at(in);
      QuantActivation a(tanh_tabs_.at(n.output[0]), tanh_fused_off);
      std::vector<IntFp> y;
      a.forward(y, x);
      const auto &sc = tanh_scales_.at(n.output[0]); // {s_in, zp_in, s_out, zp_out}
      act_[n.output[0]] = std::move(y);
      shp_[n.output[0]] = shp_.at(in);
      scale_[n.output[0]] = sc[2];
      zp_[n.output[0]] = (int)llround(sc[3]);
      if (party == ALICE && ref_.count(in)) {
        const auto &xr = ref_.at(in);
        std::vector<int64_t> r(xr.size());
        for (size_t i = 0; i < xr.size(); ++i) {
          double xv = ((double)xr[i] - sc[1]) * sc[0];
          long long q = llround(std::tanh(xv) / sc[2] + sc[3]);
          r[i] = q < 0 ? 0 : (q > 255 ? 255 : q);
        }
        ref_[n.output[0]] = std::move(r);
      }
      return;
    }
    if (!tanh_tab) throw std::runtime_error("Tanh needs table");
    const std::string &in = n.input[0];
    auto xin = rescale_to_P(in); // QDQ per-tensor scale -> gadget's 2^-P
    QuantActivation a(tanh_tab, tanh_off);
    std::vector<IntFp> y;
    a.forward(y, xin);
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(in);
    scale_[n.output[0]] = 1.0 / (double)(1LL << P);
    zp_[n.output[0]] = 0;
  }
  // ---- Transpose (general N-D), scale/zp pass through ----
  void transpose(const OnnxNode &n) {
    const std::string &in = n.input[0];
    if (!act_.count(in) &&
        g_->initializer.count(in)) { // transpose of a 2-D weight initializer (LM head: wte^T)
      const auto &t = g_->initializer.at(in);
      if (t.dims.size() ==
          2) { // NO-OP: don't materialize wte^T. Mark it as "wte consumed as rows";
        wteT_src_[n.output[0]] =
            in; // the LM-head matmul reads the shared committed table in place.
        shp_[n.output[0]] = {(int)t.dims[1], (int)t.dims[0]}; // logical transposed shape [D, V]
        scale_[n.output[0]] = 1.0 / (double)(1LL << P);
        return;
      }
      std::vector<int> xs(t.dims.begin(), t.dims.end());
      int nd = (int)xs.size();
      std::vector<int> perm;
      auto it = n.attr.find("perm");
      if (it != n.attr.end() && !it->second.ints.empty())
        for (auto p : it->second.ints) perm.push_back((int)p);
      else
        for (int i = nd - 1; i >= 0; --i) perm.push_back(i);
      std::vector<int64_t> ns(nd), ist(nd, 1), ost(nd, 1);
      for (int i = 0; i < nd; ++i) ns[i] = xs[perm[i]];
      for (int i = nd - 2; i >= 0; --i) {
        ist[i] = ist[i + 1] * xs[i + 1];
        ost[i] = ost[i + 1] * ns[i + 1];
      }
      int64_t total = 1;
      for (int d : xs) total *= d;
      std::vector<int64_t> q(total);
      std::vector<int> id(nd);
      for (int64_t o = 0; o < total; ++o) {
        int64_t rem = o, src = 0;
        for (int i = 0; i < nd; ++i) {
          id[i] = (int)(rem / ost[i]);
          rem %= ost[i];
        }
        for (int i = 0; i < nd; ++i) src += (int64_t)id[i] * ist[perm[i]];
        q[o] = (int64_t)llround((double)t.data[src] * (double)(1LL << P));
      }
      cref_[n.output[0]] = std::move(q);
      scale_[n.output[0]] = 1.0 / (double)(1LL << P);
      shp_[n.output[0]] = std::vector<int>(ns.begin(), ns.end());
      return;
    }
    const auto &x = act_.at(in);
    const auto &xs = shp_.at(in);
    const int nd = (int)xs.size();
    std::vector<int> perm;
    auto it = n.attr.find("perm");
    if (it != n.attr.end() && !it->second.ints.empty())
      for (auto p : it->second.ints) perm.push_back((int)p);
    else
      for (int i = nd - 1; i >= 0; --i) perm.push_back(i);
    std::vector<int64_t> ns(nd), ist(nd, 1), ost(nd, 1);
    for (int i = 0; i < nd; ++i) ns[i] = xs[perm[i]];
    for (int i = nd - 2; i >= 0; --i) {
      ist[i] = ist[i + 1] * xs[i + 1];
      ost[i] = ost[i + 1] * ns[i + 1];
    }
    int total = prod(xs);
    std::vector<IntFp> y(total);
    bool hr = party == ALICE && ref_.count(in);
    std::vector<int64_t> r;
    if (hr) r.resize(total);
    const auto *xr = hr ? &ref_.at(in) : nullptr;
    std::vector<int> id(nd);
    for (int o = 0; o < total; ++o) {
      int rem = o, src = 0;
      for (int i = 0; i < nd; ++i) {
        id[i] = rem / ost[i];
        rem %= ost[i];
      }
      for (int i = 0; i < nd; ++i) src += id[i] * ist[perm[i]];
      y[o] = x[src];
      if (hr) r[o] = (*xr)[src];
    }
    std::vector<int> nsi(ns.begin(), ns.end());
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = nsi;
    scale_[n.output[0]] = scale_.at(in);
    zp_[n.output[0]] = zp_.at(in);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  // ---- Split (equal last-axis), scale/zp pass through ----
  void split(const OnnxNode &n) {
    const std::string &in = n.input[0];
    const auto &x = act_.at(in);
    const auto &xs = shp_.at(in);
    const int nout = (int)n.output.size(), last = xs.back(), part = last / nout,
              outer = prod(xs) / last;
    bool hr = party == ALICE && ref_.count(in);
    const auto *xr = hr ? &ref_.at(in) : nullptr;
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
      scale_[n.output[o]] = scale_.at(in);
      zp_[n.output[o]] = zp_.at(in);
      if (hr) ref_[n.output[o]] = std::move(r);
    }
  }
  // ---- Gather rows at PUBLIC indices; data from initializer (commit rows) OR act_ (select rows)
  // ----
  void gather(const OnnxNode &n) {
    // index values are public (constant-derived): from initializer.i64 or the committed-constant
    // ref_
    std::vector<int64_t> idx;
    if (g_->initializer.count(n.input[1]))
      idx = g_->initializer.at(n.input[1]).i64;
    else if (ref_.count(n.input[1]))
      idx = ref_.at(n.input[1]);
    else
      throw std::runtime_error("Gather: non-constant (committed-ROM) indices unsupported");
    const int S = (int)idx.size();
    if (g_->initializer.count(n.input[0])) { // data is a constant table (embeddings)
      const std::string &tname = n.input[0];
      // UNTIED input embedding (no tied LM head reuses this table): commit ONLY the S selected rows
      // (seq*D), never the whole vocab*D table.  Committing the full table just to index seq rows is
      // an O(vocab*hidden) transient -- Llama's 128k*2048 embedding was a 6.3 GB commit at op#0, the
      // model's PEAK.  Selected-row commitment is the prover's same witness (the model is private
      // either way); only the tied case needs the whole table, to bind Gather cells == LM-head cells.
      // "tied" = the embedding is reused beyond this input Gather -- by the LM head, either as a
      // distinct wte^T handle (wteT_src_, set by our pre-pass) OR via a Transpose/MatMul that consumes
      // the SAME initializer (GPT-2/Gemma: wte feeds both the input Gather and a Transpose->LM head).
      // The Transpose that sets wteT_src_ runs LATE, so checking wteT_src_ alone misses it at Gather
      // time; scan the graph for any non-Gather consumer instead (order-independent).  Untied QuaRot
      // LLMs consume embed_tokens ONLY here -> selected-row commit; tied models keep the whole table.
      bool tied_reuse = false;
      for (const auto &kv : wteT_src_) if (kv.second == tname) { tied_reuse = true; break; }
      if (!tied_reuse)
        for (const auto &nn : g_->nodes) {
          if (nn.op_type == "Gather") continue;
          for (const auto &in : nn.input) if (in == tname) { tied_reuse = true; break; }
          if (tied_reuse) break;
        }
      if (!tied_reuse) {
        resolve_ext(const_cast<OnnxTensor &>(g_->initializer.at(tname))); // LAZY_INIT: decode embed now
        const auto &t = g_->initializer.at(tname);
        const int D = (int)(t.dims.size() > 1 ? t.dims[1] : 1);
        std::vector<IntFp> out((std::size_t)S * D);
        std::vector<int64_t> q((std::size_t)S * D, 0);
        std::vector<uint64_t> raw((std::size_t)S * D, 0);
        if (party == ALICE)
          for (int s = 0; s < S; ++s)
            for (int dd = 0; dd < D; ++dd) {
              double val = t.dtype == 1 ? (double)t.data[(std::size_t)idx[s] * D + dd]
                                        : (double)t.i64[(std::size_t)idx[s] * D + dd];
              int64_t qv = (int64_t)llround(val * (double)(1LL << P));
              q[(std::size_t)s * D + dd] = qv;
              raw[(std::size_t)s * D + dd] = qv >= 0 ? (uint64_t)qv : PR - (uint64_t)(-qv);
            }
        batch_feed(out.data(), raw.data(), (int64_t)((std::size_t)S * D));
        act_[n.output[0]] = std::move(out);
        shp_[n.output[0]] = {S, D};
        scale_[n.output[0]] = 1.0 / (double)(1LL << P);
        zp_[n.output[0]] = 0;
        if (party == ALICE) ref_[n.output[0]] = std::move(q);
        return;
      }
      // TIED: commit the embedding table ONCE as addressable cells, then INDEX it at the public idx --
      // the Gather output cells ARE the committed table cells (bound, no fresh commitment).  The same
      // committed table is reused (transpose-free) by the LM head, enforcing the tied weights.
      commit_table(tname);
      const int D = ctab_dim_.at(tname).second;
      const auto &c = ctab_.at(tname);
      const auto &cr = ctab_ref_.at(tname);
      std::vector<IntFp> out((size_t)S * D);
      std::vector<int64_t> q;
      if (party == ALICE) q.resize((size_t)S * D);
      for (int s = 0; s < S; ++s)
        for (int dd = 0; dd < D; ++dd) {
          out[(int64_t)s * D + dd] = c[(int64_t)idx[s] * D + dd];
          if (party == ALICE) q[(int64_t)s * D + dd] = cr[(int64_t)idx[s] * D + dd];
        }
      act_[n.output[0]] = std::move(out);
      shp_[n.output[0]] = {S, D};
      scale_[n.output[0]] = 1.0 / (double)(1LL << P);
      zp_[n.output[0]] = 0;
      if (party == ALICE) ref_[n.output[0]] = std::move(q);
    } else if (cref_.count(n.input[0])) { // QDQ: data is a DEQUANTIZED weight table (BERT word/pos
                                          // embeddings:
      // emb_weight(int8) -> DequantizeLinear -> Gather).  Index the int table, commit ONLY the
      // selected rows (seq rows, not the whole 30k-vocab table) dequantized to the gadget's 2^-P.
      const auto &tab = cref_.at(n.input[0]);
      const auto &ts = shp_.at(n.input[0]);
      int D = 1;
      for (size_t i = 1; i < ts.size(); ++i) D *= ts[i];
      const double sc = scale_.at(n.input[0]);
      const int64_t z = zp_.at(n.input[0]);
      std::vector<int64_t> q((size_t)S * D, 0);
      if (party == ALICE)
        for (int s = 0; s < S; ++s)
          for (int dd = 0; dd < D; ++dd)
            q[(int64_t)s * D + dd] = llround(((double)tab[(int64_t)idx[s] * D + dd] - (double)z) *
                                             sc * (double)(1LL << P));
      std::vector<IntFp> out;
      qi->commit_vec(out, q); // committed witness (private embedding) at 2^-P
      act_[n.output[0]] = std::move(out);
      shp_[n.output[0]] = {S, D};
      scale_[n.output[0]] = 1.0 / (double)(1LL << P);
      zp_[n.output[0]] = 0;
      if (party == ALICE) ref_[n.output[0]] = std::move(q);
    } else { // data is a committed activation -> select rows at public idx
      const auto &data = act_.at(n.input[0]);
      const auto &ds = shp_.at(n.input[0]);
      int D = 1;
      for (size_t i = 1; i < ds.size(); ++i) D *= ds[i];
      std::vector<IntFp> out((size_t)S * D);
      bool hr = party == ALICE && ref_.count(n.input[0]);
      const auto *dr = hr ? &ref_.at(n.input[0]) : nullptr;
      std::vector<int64_t> r;
      if (hr) r.resize((size_t)S * D);
      for (int s = 0; s < S; ++s)
        for (int dd = 0; dd < D; ++dd) {
          out[(int64_t)s * D + dd] = data[(int64_t)idx[s] * D + dd];
          if (hr) r[(int64_t)s * D + dd] = (*dr)[(int64_t)idx[s] * D + dd];
        }
      act_[n.output[0]] = std::move(out);
      shp_[n.output[0]] = {S, D};
      scale_[n.output[0]] = scale_.at(n.input[0]);
      zp_[n.output[0]] = zp_.at(n.input[0]);
      if (hr) ref_[n.output[0]] = std::move(r);
    }
  }
  // ---- Mul by a constant (scale) ----
  // committed activation / PUBLIC constant (BERT attention score / sqrt(d_head)) -- = multiply by
  // 1/const.
  void div_(const OnnxNode &n) {
    const std::string &a = n.input[0], &b = n.input[1];
    if (!(g_->initializer.count(b) || cref_.count(b)) || !act_.count(a))
      throw std::runtime_error("Div by an activation not supported");
    const auto &x = act_.at(a);
    auto M = param_real(b);
    const int nb = (int)M.size();
    std::vector<int64_t> qm(nb);
    for (int i = 0; i < nb; ++i)
      qm[i] = (int64_t)llround((1.0 / M[i]) * (double)(1LL << P)); // reciprocal @ 2^-P
    std::vector<IntFp> raw(x.size());
    for (size_t i = 0; i < x.size(); ++i) raw[i] = x[i] * pub(qm[nb == 1 ? 0 : i % nb]);
    std::vector<IntFp> out;
    qi->requant_cm(out, raw, 1, P);
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = shp_.at(a);
    scale_[n.output[0]] = scale_.at(a);
    zp_[n.output[0]] = zp_.at(a);
    if (party == ALICE && ref_.count(a)) {
      const auto &xr = ref_.at(a);
      std::vector<int64_t> r(xr.size());
      const int64_t HP = (int64_t)1 << (P - 1);
      for (size_t i = 0; i < xr.size(); ++i) {
        int64_t N = xr[i] * qm[nb == 1 ? 0 : (int)(i % nb)];
        r[i] = (N + HP) >> P;
      }
      ref_[n.output[0]] = std::move(r);
    }
  }
  void mul(const OnnxNode &n) {
    // committed activation x PUBLIC constant.  The constant may be a raw initializer OR a QDQ
    // DequantizeLinear constant in cref_ (BERT's 1/sqrt(d_head) attention scale folds to a Q/DQ'd
    // scalar), and may be on either input.  Two genuine activations remain unsupported.
    std::string a = n.input[0], b = n.input[1];
    auto isC = [&](const std::string &s) { return g_->initializer.count(s) || cref_.count(s); };
    if (isC(a) && !isC(b)) std::swap(a, b); // ensure b is the constant operand
    if (!isC(b) || !act_.count(a)) {
      // committed x committed (SwiGLU gate: SiLU(gate) (.) up).  Rescale both to 2^-P; the product is
      // at 2^-2P (one committed mult-gate per element), then requant back to 2^-P.
      if (act_.count(n.input[0]) && act_.count(n.input[1])) {
        std::vector<int64_t> ar, br;
        auto xa = rescale_to_P(n.input[0], &ar);
        auto xb = rescale_to_P(n.input[1], &br);
        std::vector<IntFp> prod(xa.size());
        for (size_t i = 0; i < xa.size(); ++i) prod[i] = xa[i] * xb[i];
        std::vector<IntFp> out;
        qi->requant_cm(out, prod, 1, P);
        act_[n.output[0]] = std::move(out);
        shp_[n.output[0]] = shp_.at(n.input[0]);
        scale_[n.output[0]] = 1.0 / (double)(1LL << P);
        zp_[n.output[0]] = 0;
        if (party == ALICE && ar.size() == xa.size() && br.size() == xa.size()) {
          std::vector<int64_t> r(ar.size());
          const int64_t HP = (int64_t)1 << (P - 1);
          for (size_t i = 0; i < ar.size(); ++i) { int64_t N = ar[i] * br[i]; r[i] = (N + HP) >> P; }
          ref_[n.output[0]] = std::move(r);
        }
        return;
      }
      throw std::runtime_error("Mul of two activations not supported");
    }
    const auto &x = act_.at(a);
    auto M = param_real(b);
    const int nb = (int)M.size();
    std::vector<int64_t> qm(nb);
    for (int i = 0; i < nb; ++i) qm[i] = (int64_t)llround(M[i] * (double)(1LL << P));
    std::vector<IntFp> raw(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      raw[i] = x[i] * pub(qm[nb == 1 ? 0 : i % nb]); // scale is public -> linear
    std::vector<IntFp> out;
    qi->requant_cm(out, raw, 1, P); // product @ scale*2^-P -> back by 2^P
    act_[n.output[0]] = std::move(out);
    shp_[n.output[0]] = shp_.at(a);
    scale_[n.output[0]] = scale_.at(a);
    zp_[n.output[0]] = zp_.at(a);
    if (party == ALICE && ref_.count(a)) {
      const auto &xr = ref_.at(a);
      std::vector<int64_t> r(xr.size());
      const int64_t HP = (int64_t)1 << (P - 1), DP = (int64_t)1 << P;
      for (size_t i = 0; i < xr.size(); ++i) {
        int64_t N = xr[i] * qm[nb == 1 ? 0 : (int)(i % nb)];
        r[i] = (N + HP) >> P /* round-half-up: matches requant_cm_one */;
      }
      ref_[n.output[0]] = std::move(r);
    }
  }

  // ---- Concat along an axis (activations at a shared scale) ----
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
    scale_[n.output[0]] = scale_.at(n.input[0]);
    zp_[n.output[0]] = zp_.at(n.input[0]);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  // ---- Slice (starts/ends/axes/steps as inputs, 1-D common case) ----
  void slice(const OnnxNode &n) {
    // Large folded constants (e.g. the seq^2 causal mask at seq>=256) exceed the
    // pre-commit threshold (ne<=65536), so commit on demand here before slicing.
    if (!act_.count(n.input[0]) && g_->initializer.count(n.input[0])) commit_const(n.input[0]);
    const auto &x = act_.at(n.input[0]);
    const auto &xs = shp_.at(n.input[0]);
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
    scale_[n.output[0]] = scale_.at(n.input[0]);
    zp_[n.output[0]] = zp_.at(n.input[0]);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  // ---- Expand: broadcast data to target shape ----
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
    scale_[n.output[0]] = scale_.at(n.input[0]);
    zp_[n.output[0]] = zp_.at(n.input[0]);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  // ---- Sub: elementwise a - b (b rescaled to a's scale) ----
  void sub(const OnnxNode &n) {
    const auto &A = act_.at(n.input[0]);
    const double sa = scale_.at(n.input[0]);
    std::vector<IntFp> B;
    std::vector<int64_t> Br;
    bool hr = party == ALICE && ref_.count(n.input[0]);
    if (g_->initializer.count(n.input[1])) {
      const auto &T = g_->initializer.at(n.input[1]);
      int nb = (int)std::max(T.data.size(), T.i64.size());
      for (int i = 0; i < nb; ++i) {
        int64_t q = (int64_t)llround((T.dtype == 1 ? T.data[i] : (double)T.i64[i]) / sa);
        Br.push_back(q);
        B.push_back(commit_a(q));
      }
    } else {
      B = act_.at(n.input[1]);
      if (hr && ref_.count(n.input[1]))
        Br = ref_.at(n.input[1]);
      else
        hr = false;
    }
    size_t nb = B.size();
    std::vector<IntFp> y(A.size());
    std::vector<int64_t> r;
    if (hr) r.resize(A.size());
    const auto *ar = hr ? &ref_.at(n.input[0]) : nullptr;
    for (size_t i = 0; i < A.size(); ++i) {
      size_t j = nb == A.size() ? i : (nb == 1 ? 0 : i % nb);
      y[i] = A[i] + B[j] * pub(-1);
      if (hr) r[i] = (*ar)[i] - Br[j];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    scale_[n.output[0]] = sa;
    zp_[n.output[0]] = zp_.at(n.input[0]);
    if (hr) ref_[n.output[0]] = std::move(r);
  }
  // commit a small folded-constant initializer into act_ (float @2^-P, int @scale-1)
  void commit_const(const std::string &nm) {
    const auto &t = g_->initializer.at(nm);
    size_t n = std::max(t.data.size(), t.i64.size());
    std::vector<int64_t> q(n);
    double sc;
    // clamp magnitude so huge mask constants (GPT-2 uses ~-1e9/-inf for masked
    // positions) stay inside the LUT / range-check domain: a clamped -CLAMP still
    // yields exp(max - (-CLAMP)) ~ 0 in softmax, but never overflows the tables.
    const int64_t CLAMP = (int64_t)1 << (P + 8); // ~-256 (real) at ANY P -> exp(max-(-CLAMP)) ~ 0
                                                 // (strong mask) yet within the [0,512) exp domain
    if (t.dtype == 1) {
      sc = 1.0 / (double)(1LL << P);
      for (size_t i = 0; i < n; ++i) {
        double v = (double)t.data[i] * (double)(1LL << P);
        q[i] = (int64_t)std::max(
            -(double)CLAMP,
            std::min((double)CLAMP,
                     std::isfinite(v) ? llround(v) : (v < 0 ? -(double)CLAMP : (double)CLAMP)));
      }
    } else {
      sc = 1.0;
      for (size_t i = 0; i < n; ++i) q[i] = std::max(-CLAMP, std::min(CLAMP, t.i64[i]));
    }
    std::vector<IntFp> c(n);
    for (size_t i = 0; i < n; ++i) c[i] = commit_a(q[i]);
    act_[nm] = std::move(c);
    shp_[nm] = std::vector<int>(t.dims.begin(), t.dims.end());
    scale_[nm] = sc;
    zp_[nm] = 0;
    if (party == ALICE) ref_[nm] = std::move(q);
  }
  // ---- IsNaN: identically false in the integer/fixed-point domain (no NaNs) ----
  void isnan(const OnnxNode &n) {
    const auto &x = act_.at(n.input[0]);
    std::vector<IntFp> y(x.size(), pub(0));
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    scale_[n.output[0]] = 1.0;
    zp_[n.output[0]] = 0;
    if (party == ALICE) ref_[n.output[0]] = std::vector<int64_t>(x.size(), 0);
  }
  // ---- And: elementwise logical-and of committed 0/1 tensors (a*b) ----
  void and_(const OnnxNode &n) {
    const auto &A = act_.at(n.input[0]);
    const auto &B = act_.at(n.input[1]);
    size_t nb = B.size();
    std::vector<IntFp> y(A.size());
    for (size_t i = 0; i < A.size(); ++i)
      y[i] = A[i] * B[nb == A.size() ? i : (nb == 1 ? 0 : i % nb)];
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(n.input[0]);
    scale_[n.output[0]] = 1.0;
    zp_[n.output[0]] = 0;
    if (party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1])) {
      const auto &ar = ref_.at(n.input[0]), &br = ref_.at(n.input[1]);
      std::vector<int64_t> r(ar.size());
      for (size_t i = 0; i < ar.size(); ++i)
        r[i] = ar[i] && br[nb == ar.size() ? i : i % nb] ? 1 : 0;
      ref_[n.output[0]] = std::move(r);
    }
  }
  // ---- Where(cond, a, b) = cond ? a : b  (cond a committed 0/1) ----
  void where(const OnnxNode &n) {
    const auto &C = act_.at(n.input[0]);
    const auto &A = act_.at(n.input[1]);
    const auto &B = act_.at(n.input[2]);
    size_t na = A.size(), nb = B.size(), nc = C.size();
    size_t no = std::max(std::max(na, nb), nc); // numpy broadcast
    const std::string &shapeop = na == no ? n.input[1] : (nb == no ? n.input[2] : n.input[0]);
    const std::string &dataop = nb >= na ? n.input[2] : n.input[1];
    bool hr = party == ALICE && ref_.count(n.input[0]) && ref_.count(n.input[1]) &&
              ref_.count(n.input[2]);
    std::vector<int64_t> r;
    if (hr) r.resize(no);
    const auto *cr = hr ? &ref_.at(n.input[0]) : nullptr, *ar = hr ? &ref_.at(n.input[1]) : nullptr,
               *br = hr ? &ref_.at(n.input[2]) : nullptr;
    std::vector<IntFp> y(no);
    for (size_t i = 0; i < no; ++i) {
      size_t ic = nc == no ? i : (nc == 1 ? 0 : i % nc), ia = na == no ? i : (na == 1 ? 0 : i % na),
             ib = nb == no ? i : (nb == 1 ? 0 : i % nb);
      y[i] = B[ib] + C[ic] * (A[ia] + B[ib] * pub(-1));
      if (hr) r[i] = (*cr)[ic] ? (*ar)[ia] : (*br)[ib];
    }
    act_[n.output[0]] = std::move(y);
    shp_[n.output[0]] = shp_.at(shapeop);
    scale_[n.output[0]] = scale_.at(dataop);
    zp_[n.output[0]] = zp_.at(dataop);
    if (hr) ref_[n.output[0]] = std::move(r);
  }

  // ---- Reshape: data order unchanged, recompute shape from the target-shape constant ----
  void reshape(const OnnxNode &n) {
    const std::string &in = n.input[0];
    act_[n.output[0]] = act_.at(in);
    if (ref_.count(in)) ref_[n.output[0]] = ref_.at(in);
    scale_[n.output[0]] = scale_.at(in);
    zp_[n.output[0]] = zp_.at(in);
    std::vector<int64_t> tgt;
    if (n.input.size() > 1 && g_->initializer.count(n.input[1]))
      for (auto z : g_->initializer.at(n.input[1]).i64) tgt.push_back(z);
    else if (n.input.size() > 1 && ref_.count(n.input[1]))
      for (auto z : ref_.at(n.input[1])) tgt.push_back(z);
    if (tgt.empty()) {
      shp_[n.output[0]] = shp_.at(in);
      return;
    }
    int64_t total = (int64_t)act_.at(in).size();
    int64_t known = 1;
    int neg = -1;
    const auto &is = shp_.at(in);
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
  void passthru(const OnnxNode &n) {
    const std::string &in = n.input[0];
    act_[n.output[0]] = act_.at(in);
    if (ref_.count(in)) ref_[n.output[0]] = ref_.at(in);
    scale_[n.output[0]] = scale_.count(in) ? scale_.at(in) : 1.0;
    zp_[n.output[0]] = zp_.count(in) ? zp_.at(in) : 0;
    int total = prod(shp_.at(in));
    shp_[n.output[0]] = {total};
  }
  // Unsqueeze/Squeeze: data is unchanged (a reindex), only the shape changes by inserting/removing
  // size-1 dims.  Flattening (passthru) breaks the GQA Unsqueeze->Expand broadcast alignment.
  void unsqueeze_op(const OnnxNode &n) {
    const std::string &in = n.input[0];
    act_[n.output[0]] = act_.at(in);
    if (ref_.count(in)) ref_[n.output[0]] = ref_.at(in);
    scale_[n.output[0]] = scale_.count(in) ? scale_.at(in) : 1.0;
    zp_[n.output[0]] = zp_.count(in) ? zp_.at(in) : 0;
    if (rscale_.count(in)) rscale_[n.output[0]] = rscale_.at(in);
    const std::vector<int> &xs = shp_.at(in);
    std::vector<int64_t> axes;
    if (n.input.size() > 1 && g_->initializer.count(n.input[1]))
      for (auto a : g_->initializer.at(n.input[1]).i64) axes.push_back(a);
    else if (n.attr.count("axes"))
      for (auto a : n.attr.at("axes").ints) axes.push_back(a);
    const int outrank = (int)xs.size() + (int)axes.size();
    std::set<int> axset;
    for (auto a : axes) axset.insert((int)(a < 0 ? a + outrank : a));
    std::vector<int> os;
    for (int i = 0, xi = 0; i < outrank; ++i) os.push_back(axset.count(i) ? 1 : xs[xi++]);
    shp_[n.output[0]] = os;
  }
  void squeeze_op(const OnnxNode &n) {
    const std::string &in = n.input[0];
    act_[n.output[0]] = act_.at(in);
    if (ref_.count(in)) ref_[n.output[0]] = ref_.at(in);
    scale_[n.output[0]] = scale_.count(in) ? scale_.at(in) : 1.0;
    zp_[n.output[0]] = zp_.count(in) ? zp_.at(in) : 0;
    if (rscale_.count(in)) rscale_[n.output[0]] = rscale_.at(in);
    const std::vector<int> &xs = shp_.at(in);
    std::vector<int64_t> axes;
    if (n.input.size() > 1 && g_->initializer.count(n.input[1]))
      for (auto a : g_->initializer.at(n.input[1]).i64) axes.push_back(a);
    else if (n.attr.count("axes"))
      for (auto a : n.attr.at("axes").ints) axes.push_back(a);
    std::set<int> axset;
    for (auto a : axes) axset.insert((int)(a < 0 ? a + (int)xs.size() : a));
    std::vector<int> os;
    for (int i = 0; i < (int)xs.size(); ++i)
      if (axes.empty() ? xs[i] != 1 : !axset.count(i)) os.push_back(xs[i]);
    if (os.empty()) os.push_back(1);
    shp_[n.output[0]] = os;
  }

  std::map<std::string, std::vector<int64_t>> cref_; // dequantized public int weights/bias by name

  // Shared committed table (e.g. wte): committed ONCE as addressable cells, indexed by the input
  // embedding Gather (public index -> reference a cell, free) AND dot-producted by the LM head
  // (committed*committed IP matmul over the same cells). One commitment binds both phases and
  // enforces the tied-weights property; no wte^T is ever materialized.
  std::map<std::string, std::vector<IntFp>>
      ctab_; // init name -> committed cells, row-major [rows*D] @ 2^-P
  std::map<std::string, std::vector<int64_t>> ctab_ref_; // ALICE plaintext ints (parallel to ctab_)
  std::map<std::string, std::pair<int, int>> ctab_dim_;  // name -> (rows, D)
  std::map<std::string, std::string>
      wteT_src_; // transposed-handle name -> underlying committed-table name

  void commit_table(const std::string &name) {
    if (ctab_.count(name)) return; // commit-once, idempotent
    auto _ct = std::chrono::steady_clock::now();
    resolve_ext(const_cast<OnnxTensor &>(g_->initializer.at(name))); // LAZY_INIT: decode table now
    const auto &t = g_->initializer.at(name);
    const int rows = (int)t.dims[0], D = (int)(t.dims.size() > 1 ? t.dims[1] : 1);
    auto &c = ctab_[name];
    c.resize((size_t)rows * D);
    auto &rr = ctab_ref_[name];
    rr.resize((size_t)rows * D);
    for (size_t k = 0; k < c.size(); ++k) {
      int64_t q = (int64_t)llround((t.dtype == 1 ? (double)t.data[k] : (double)t.i64[k]) *
                                   (double)(1LL << P));
      rr[k] = q;
      c[k] = commit_a(q);
    }
    ctab_dim_[name] = {rows, D};
    if (getenv("WTE_PROF"))
      fprintf(stderr, "[TABLE_COMMIT] %s: %zu entries (%dx%d) in %.2fs\n", name.c_str(),
              c.size(), rows, D,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - _ct).count());
  }
};

} // namespace volefier
#endif // ML_QUANT_GRAPH_INFER_H
