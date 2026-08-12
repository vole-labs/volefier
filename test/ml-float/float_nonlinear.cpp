// ml-float: prove accurate non-linear ML functions over IEEE-754 single-
// precision Boolean circuits (Mystique approach), the float counterpart of the
// LogUp-lookup versions in ml-quant / ml-fixed. Each function is evaluated on
// committed emp::Float values in the BOOLEAN ZK proof, revealed, and checked
// against a plaintext (double) reference.
//
//   test_ml-float_float_nonlinear 1 12345
//   test_ml-float_float_nonlinear 2 12345

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/ml-float/float_nonlinear.h"

using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = 1;
static int fails = 0;

static void chk(const char *name, double got, double want, double tol) {
  double err = fabs(got - want), rel = err / (fabs(want) + 1e-9);
  if (!(err < tol || rel < tol)) { if (++fails <= 20) cout << "  MISMATCH " << name << ": got " << got << " want " << want << " (abs " << err << ")" << endl; }
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO<NetIO> *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "----- ml-float: accurate non-linear via IEEE-754 Boolean circuits -----" << endl << endl;

  setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
  auto C   = [&](float v) { return Float(v, ALICE); };          // prover commits
  auto rev = [&](const Float &f) { return f.reveal<double>(PUBLIC); };
  ios[0]->flush();
  auto t_start = clock_start();

  // ---- ReLU : max(0, x) ----
  for (float x : {-3.5f, -0.2f, 0.0f, 0.7f, 2.9f}) {
    double g = rev(FloatNonlinear::relu(C(x)));
    if (party == ALICE) chk("relu", g, x > 0 ? x : 0.0, 1e-5);
  }
  // ---- Sigmoid : 1/(1+e^-x) ----
  for (float x : {-4.f, -1.f, 0.f, 1.5f, 3.f}) {
    double g = rev(FloatNonlinear::sigmoid(C(x)));
    if (party == ALICE) chk("sigmoid", g, 1.0 / (1.0 + exp(-(double)x)), 2e-2);
  }
  // ---- GeLU (tanh approx) ----
  {
    auto gref = [](double x) { double k = sqrt(2.0 / M_PI); return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x))); };
    for (float x : {-3.f, -0.5f, 0.f, 0.5f, 2.f}) {
      double g = rev(FloatNonlinear::gelu(C(x)));
      if (party == ALICE) chk("gelu", g, gref(x), 2e-2);
    }
  }
  // ---- Softmax over a vector ----
  {
    vector<float> xs = {1.0f, 2.0f, -0.5f, 3.0f};
    vector<Float> cx; for (float x : xs) cx.push_back(C(x));
    auto out = FloatNonlinear::softmax(cx);
    if (party == ALICE) {
      double mx = *max_element(xs.begin(), xs.end()), s = 0; for (float x : xs) s += exp(x - mx);
      for (size_t i = 0; i < xs.size(); ++i) chk("softmax", rev(out[i]), exp(xs[i] - mx) / s, 2e-2);
    } else for (auto &o : out) rev(o);
  }
  // ---- LayerNorm over a vector ----
  {
    vector<float> xs = {2.f, -1.f, 0.5f, 3.f, -2.f, 1.f};
    vector<Float> cx, cg, cb;
    for (float x : xs) cx.push_back(C(x));
    for (int i = 0; i < 6; ++i) { cg.push_back(C(1.2f)); cb.push_back(C(0.1f)); }
    auto out = FloatNonlinear::layernorm(cx, cg, cb);
    if (party == ALICE) {
      double m = 0; for (float x : xs) m += x; m /= 6;
      double v = 0; for (float x : xs) v += (x - m) * (x - m); v /= 6;
      double den = sqrt(v + 1e-5);
      for (size_t i = 0; i < 6; ++i) chk("layernorm", rev(out[i]), (xs[i] - m) / den * 1.2 + 0.1, 1e-3);
    } else for (auto &o : out) rev(o);
  }

  ios[0]->flush();
  double secs = time_from(t_start) / 1e6;
  bool cheated = finalize_zk_bool<BoolIO<NetIO>>();   // true => proof rejected
  if (party == ALICE) {
    if (fails) { cout << fails << " mismatches" << endl; error("ml-float nonlinear failed\n"); }
    // workload: 15 scalar (relu/sigmoid/gelu) + softmax(4) + layernorm(6) = 25 float-circuit evals
    cout << "proving (25 float-circuit non-linear evals): " << secs << " s  ("
         << (secs * 1000.0 / 25.0) << " ms / op avg)" << endl;
    cout << "pass ml-float non-linear (ReLU, Sigmoid, GeLU, Softmax, LayerNorm) over IEEE-754 float circuits" << endl;
  } else {
    cout << "verifier proof check: " << (cheated ? "REJECT" : "OK") << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
