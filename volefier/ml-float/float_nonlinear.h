#ifndef ML_FLOAT_NONLINEAR_H
#define ML_FLOAT_NONLINEAR_H

#include "emp-zk/emp-zk.h"
#include "emp-tool/emp-tool.h" // emp::Float (IEEE-754 single precision)
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// ACCURATE non-linear ML function proofs over IEEE-754 single-precision
// floating-point Boolean circuits -- the Mystique approach (USENIX'21, Sec 6.2).
//
// Counterpart to ml-quant / ml-fixed, which prove the SAME non-linear functions
// (ReLU, GeLU, Softmax, LayerNorm, Sigmoid) with LogUp lookup tables over a
// field.  Here every value is an emp::Float committed in the BOOLEAN ZK proof
// (setup_zk_bool), and the function is evaluated with exact IEEE-754 circuits:
//   - add / sub / mul / div / sqrt : IEEE-754 correct
//   - exp                          : float-accurate circuit approximation
// so there is NO quantization or table-resolution error -- the cost is paying
// for Boolean circuits (and arith<->bool conversion when fed by a field matmul)
// instead of a handful of lookups.
// ---------------------------------------------------------------------------

using namespace emp::rec;  // Integer/Float/Bit moved to emp::rec in new emp
namespace volefier {

using emp::Bit;
using emp::Float;

class FloatNonlinear {
public:
  // ---- ReLU: max(0, x) ----  (x<0)? 0 : x
  static Float relu(const Float &x) {
    Float zero(0.0f, PUBLIC), xm = x;
    return xm.If(xm.less_than(zero), zero); // If is not const-qualified
  }

  // ---- Sigmoid: 1 / (1 + e^{-x}) ----
  static Float sigmoid(const Float &x) {
    Float one(1.0f, PUBLIC);
    return one / (one + (-x).exp());
  }

  // ---- tanh(z) = (e^{2z} - 1) / (e^{2z} + 1) ----
  static Float tanh_(const Float &z) {
    Float one(1.0f, PUBLIC);
    Float e2 = (z + z).exp();
    return (e2 - one) / (e2 + one);
  }

  // ---- GeLU (tanh approximation, identical formula to our lookup gelu):
  //      0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) )) ----
  static Float gelu(const Float &x) {
    Float half(0.5f, PUBLIC), one(1.0f, PUBLIC);
    Float k(static_cast<float>(std::sqrt(2.0 / M_PI)), PUBLIC), c(0.044715f, PUBLIC);
    Float inner = k * (x + c * (x * x * x));
    return half * x * (one + tanh_(inner));
  }

  // ---- Softmax over a vector: exp(x_i - max) / sum ----
  static std::vector<Float> softmax(const std::vector<Float> &x) {
    Float m = x[0];
    for (std::size_t i = 1; i < x.size(); ++i) m = m.If(m.less_than(x[i]), x[i]); // running max
    std::vector<Float> e;
    e.reserve(x.size());
    Float sum(0.0f, PUBLIC);
    for (std::size_t i = 0; i < x.size(); ++i) {
      Float ei = (x[i] - m).exp();
      e.push_back(ei);
      sum = sum + ei;
    }
    std::vector<Float> out;
    out.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out.push_back(e[i] / sum);
    return out;
  }

  // ---- LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta ----
  static std::vector<Float> layernorm(const std::vector<Float> &x, const std::vector<Float> &gamma,
                                      const std::vector<Float> &beta, float eps = 1e-5f) {
    const std::size_t n = x.size();
    Float ninv(static_cast<float>(1.0 / n), PUBLIC), epsf(eps, PUBLIC);
    Float mean(0.0f, PUBLIC);
    for (const auto &xi : x) mean = mean + xi;
    mean = mean * ninv;
    Float var(0.0f, PUBLIC);
    for (const auto &xi : x) {
      Float dd = xi - mean;
      var = var + dd * dd;
    }
    var = var * ninv;
    Float denom = (var + epsf).sqrt(); // sqrt(var + eps)
    std::vector<Float> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back((x[i] - mean) / denom * gamma[i] + beta[i]);
    return out;
  }
};

} // namespace volefier

#endif // ML_FLOAT_NONLINEAR_H
