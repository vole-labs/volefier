#ifndef ML_IO_ONNX_FUSE_H
#define ML_IO_ONNX_FUSE_H

// Pattern-fusion pass: detect the decomposed LayerNorm / GELU subgraphs that
// transformer ONNX exports emit and rewrite each into the single fused op our
// prover handles efficiently (FixedLayerNorm / Gelu lookup table).
//
// LayerNorm:  ReduceMean(x) -> Sub(x,mean) -> Pow(.,2) -> ReduceMean ->
//             Add(.,eps) -> Sqrt -> Div(diff,std) -> Mul(.,gamma) -> Add(.,beta)
// GELU(tanh): ... -> Tanh -> Add(.,1) -> Mul(0.5*x, .)
//
// Each match is identified by walking the dataflow, then the whole subgraph is
// collected by backward reachability from the output (stopping at the op's input
// `x` and at initializers) and replaced. Run AFTER fold_constants (so eps/gamma/
// beta are initializers).

#include "volefier/io/onnx.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace volefier {

struct FuseStats {
  int layernorm = 0, gelu = 0;
};

inline OnnxGraph fuse_patterns(const OnnxGraph &g, FuseStats &st) {
  const int N = (int)g.nodes.size();
  std::map<std::string, int> producer;
  std::map<std::string, std::vector<int>> consumers;
  for (int i = 0; i < N; ++i) {
    for (const auto &o : g.nodes[i].output)
      if (!o.empty()) producer[o] = i;
    for (const auto &in : g.nodes[i].input)
      if (!in.empty()) consumers[in].push_back(i);
  }
  auto isinit = [&](const std::string &t) { return g.initializer.count(t) > 0; };
  auto prod = [&](const std::string &t) {
    auto it = producer.find(t);
    return it == producer.end() ? -1 : it->second;
  };
  auto opof = [&](int i) { return i < 0 ? std::string() : g.nodes[i].op_type; };

  std::vector<bool> removed(N, false);
  std::vector<OnnxNode> inserts;
  std::vector<int> insert_pos;

  // backward reachability from `out`, stopping at `stop` tensors + initializers
  auto collect = [&](const std::string &out, const std::set<std::string> &stop) {
    std::set<int> nodes;
    std::set<std::string> seen;
    std::vector<std::string> wl = {out};
    while (!wl.empty()) {
      std::string t = wl.back();
      wl.pop_back();
      if (seen.count(t)) continue;
      seen.insert(t);
      if (stop.count(t) || isinit(t)) continue;
      int pi = prod(t);
      if (pi < 0) continue;
      nodes.insert(pi);
      for (const auto &in : g.nodes[pi].input)
        if (!in.empty()) wl.push_back(in);
    }
    return nodes;
  };
  auto emit = [&](OnnxNode fused, const std::set<int> &sub) {
    int mn = N;
    for (int x : sub) {
      removed[x] = true;
      if (x < mn) mn = x;
    }
    inserts.push_back(fused);
    insert_pos.push_back(mn);
  };

  // ---- LayerNorm (anchor: Div) ----
  for (int di = 0; di < N; ++di) {
    if (removed[di] || g.nodes[di].op_type != "Div" || g.nodes[di].input.size() < 2) continue;
    const std::string diff = g.nodes[di].input[0], stdv = g.nodes[di].input[1];
    int subi = prod(diff);
    if (opof(subi) != "Sub") continue;
    const std::string x = g.nodes[subi].input[0], mean = g.nodes[subi].input[1];
    int meani = prod(mean);
    if (opof(meani) != "ReduceMean" || g.nodes[meani].input[0] != x) continue;
    int sqi = prod(stdv);
    if (opof(sqi) != "Sqrt") continue;
    int aei = prod(g.nodes[sqi].input[0]);
    if (opof(aei) != "Add") continue;
    std::string var, eps;
    for (const auto &t : g.nodes[aei].input) {
      if (isinit(t))
        eps = t;
      else
        var = t;
    }
    if (var.empty() || eps.empty()) continue;
    int vari = prod(var);
    if (opof(vari) != "ReduceMean") continue;
    int spi = prod(g.nodes[vari].input[0]);
    std::string spo = opof(spi);
    bool sqok = (spo == "Pow" && g.nodes[spi].input[0] == diff) ||
                (spo == "Mul" && g.nodes[spi].input.size() == 2 && g.nodes[spi].input[0] == diff &&
                 g.nodes[spi].input[1] == diff);
    if (!sqok) continue;
    const std::string norm = g.nodes[di].output[0];
    int muli = -1;
    for (int ci : consumers[norm])
      if (g.nodes[ci].op_type == "Mul") {
        muli = ci;
        break;
      }
    if (muli < 0) continue;
    std::string gamma;
    for (const auto &t : g.nodes[muli].input)
      if (isinit(t)) gamma = t;
    if (gamma.empty()) continue;
    int adi = -1;
    for (int ci : consumers[g.nodes[muli].output[0]])
      if (g.nodes[ci].op_type == "Add") {
        adi = ci;
        break;
      }
    if (adi < 0) continue;
    std::string beta;
    for (const auto &t : g.nodes[adi].input)
      if (isinit(t)) beta = t;
    if (beta.empty()) continue;
    const std::string out = g.nodes[adi].output[0];
    OnnxNode ln;
    ln.op_type = "LayerNormalization";
    ln.name = "fused_ln_" + std::to_string(st.layernorm);
    ln.input = {x, gamma, beta};
    ln.output = {out};
    OnnxAttr ea;
    ea.name = "epsilon";
    ea.f = (g.initializer.count(eps) && !g.initializer.at(eps).data.empty())
               ? g.initializer.at(eps).data[0]
               : 1e-5f;
    ln.attr["epsilon"] = ea;
    emit(ln, collect(out, {x}));
    st.layernorm++;
  }

  // ---- GELU tanh-approx (anchor: Tanh) ----
  for (int ti = 0; ti < N; ++ti) {
    if (removed[ti] || g.nodes[ti].op_type != "Tanh" || g.nodes[ti].output.empty()) continue;
    int ai = -1;
    for (int ci : consumers[g.nodes[ti].output[0]])
      if (g.nodes[ci].op_type == "Add") {
        ai = ci;
        break;
      }
    if (ai < 0) continue;
    int mi = -1;
    for (int ci : consumers[g.nodes[ai].output[0]])
      if (g.nodes[ci].op_type == "Mul") {
        mi = ci;
        break;
      }
    if (mi < 0) continue;
    std::string halfx;
    for (const auto &in : g.nodes[mi].input)
      if (in != g.nodes[ai].output[0]) halfx = in;
    int hi = prod(halfx);
    std::string x;
    if (opof(hi) == "Mul") {
      for (const auto &in : g.nodes[hi].input)
        if (!isinit(in)) x = in;
    } else
      x = halfx;
    if (x.empty()) continue;
    const std::string out = g.nodes[mi].output[0];
    OnnxNode ge;
    ge.op_type = "Gelu";
    ge.name = "fused_gelu_" + std::to_string(st.gelu);
    ge.input = {x};
    ge.output = {out};
    emit(ge, collect(out, {x}));
    st.gelu++;
  }

  // ---- GELU erf form (anchor: Erf):  0.5 * x * (1 + erf(x / sqrt(2)))  [BERT] ----
  for (int ei = 0; ei < N; ++ei) {
    if (removed[ei] || g.nodes[ei].op_type != "Erf" || g.nodes[ei].output.empty() ||
        g.nodes[ei].input.empty())
      continue;
    int dvi = prod(g.nodes[ei].input[0]);
    if (opof(dvi) != "Div") continue; // Erf <- Div(x, sqrt2)
    const std::string x = g.nodes[dvi].input[0];
    int ai = -1;
    for (int ci : consumers[g.nodes[ei].output[0]])
      if (g.nodes[ci].op_type == "Add") {
        ai = ci;
        break;
      }
    if (ai < 0) continue; // Erf -> Add(erf, 1)
    int mi = -1;
    for (int ci : consumers[g.nodes[ai].output[0]])
      if (g.nodes[ci].op_type == "Mul") {
        mi = ci;
        break;
      }
    if (mi < 0) continue; // Add -> Mul(x, .)
    int m2 = -1;
    for (int ci : consumers[g.nodes[mi].output[0]])
      if (g.nodes[ci].op_type == "Mul") {
        m2 = ci;
        break;
      } // Mul -> Mul(., 0.5)
    const std::string out = (m2 >= 0) ? g.nodes[m2].output[0] : g.nodes[mi].output[0];
    OnnxNode ge;
    ge.op_type = "Gelu";
    ge.name = "fused_gelu_" + std::to_string(st.gelu);
    ge.input = {x};
    ge.output = {out};
    emit(ge, collect(out, {x}));
    st.gelu++;
  }

  // ---- rebuild graph (fused node inserted where its earliest removed node was) ----
  OnnxGraph o;
  o.inputs = g.inputs;
  o.outputs = g.outputs;
  o.input_shapes = g.input_shapes;
  o.initializer = g.initializer;
  std::multimap<int, int> at;
  for (int k = 0; k < (int)inserts.size(); ++k) at.insert({insert_pos[k], k});
  for (int i = 0; i < N; ++i) {
    auto r = at.equal_range(i);
    for (auto it = r.first; it != r.second; ++it) o.nodes.push_back(inserts[it->second]);
    if (!removed[i]) o.nodes.push_back(g.nodes[i]);
  }
  return o;
}

} // namespace volefier
#endif // ML_IO_ONNX_FUSE_H
