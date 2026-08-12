// Demonstrates ml-quant doing the zkGPT / DeepProve PER-TENSOR quantization
// method (as opposed to uniform 2^-P fixed-point activations): a 3-layer MLP
// where EVERY activation tensor has its OWN scale S, and each dense layer
// requants by the per-tensor ratio S_in*S_w/S_out -> a distinct non-power-of-two
// c*2^e multiplier. ALICE checks the committed output against a scale-aware
// integer reference.
//
//   test_ml-quant_quant_mlp_pertensor 1 12345   # prover
//   test_ml-quant_quant_mlp_pertensor 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_nonlinear.h"   // QuantInfer + CompareLU + RangeCheckLU
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;

static int64_t rcm(int64_t acc, int64_t c, int M) {   // matches requant_cm: round(acc*c/2^M)
  int64_t N = acc * c, h = (int64_t)1 << (M - 1), D = (int64_t)1 << M;
  return (N >= 0) ? (N + h) / D : -(((-N) + h) / D);
}

static PRG *gprg;
static int64_t rnd(int64_t lo, int64_t hi) { if (party != ALICE) return 0; uint64_t r; gprg->random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); }

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------- per-tensor quantized MLP (zkGPT/DeepProve method) -------" << endl << endl;

  PRG prg; gprg = &prg;
  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);
  QuantInfer qi(party, P, ios[0], /*out_bits=*/20, /*wc=*/1, /*mat_threads=*/threads);
  RangeCheckLU rclu(party, ios[0]->io, 16);
  CompareLU cmp(&rclu);

  // dims and PER-TENSOR scales (each activation tensor has its own S; all distinct
  // -> non-power-of-two requant multipliers, the quantization method)
  const int d0 = 16, d1 = 24, d2 = 20, d3 = 8;
  const double sx = 1.0/256, sw1 = 1.0/200, sh1 = 1.0/300,
               sw2 = 1.0/180, sh2 = 1.0/260, sw3 = 1.0/210, sout = 1.0/128;

  // random integer (quantized) activations + weights
  auto rmat = [&](int r, int c, int64_t lo, int64_t hi){ vector<vector<int64_t>> m(r, vector<int64_t>(c)); for(auto&row:m)for(auto&v:row)v=rnd(lo,hi); return m; };
  auto rvec = [&](int n, int64_t lo, int64_t hi){ vector<int64_t> v(n); for(auto&e:v)e=rnd(lo,hi); return v; };
  vector<int64_t> x = rvec(d0, -100, 100);
  auto W1 = rmat(d1, d0, -40, 40); auto b1 = rvec(d1, -50, 50);
  auto W2 = rmat(d2, d1, -40, 40); auto b2 = rvec(d2, -50, 50);
  auto W3 = rmat(d3, d2, -40, 40); auto b3 = rvec(d3, -50, 50);

  // commit
  vector<IntFp> cx; qi.commit_vec(cx, x);
  vector<vector<IntFp>> cW1, cW2, cW3; qi.commit_mat(cW1, W1); qi.commit_mat(cW2, W2); qi.commit_mat(cW3, W3);
  vector<IntFp> cb1, cb2, cb3; qi.commit_vec(cb1, b1); qi.commit_vec(cb2, b2); qi.commit_vec(cb3, b3);

  auto relu = [&](vector<IntFp> &v){ for (auto &h : v) { IntFp ge = cmp.is_le(IntFp((uint64_t)0, PUBLIC), h, 18); h = ge * h; } };

  // committed forward with PER-TENSOR scales
  vector<IntFp> h1, h2, out;
  qi.dense_layer_q(h1, cW1, cb1, cx, sx, sw1, sh1);   relu(h1);
  qi.dense_layer_q(h2, cW2, cb2, h1, sh1, sw2, sh2);  relu(h2);
  qi.dense_layer_q(out, cW3, cb3, h2, sh2, sw3, sout);

  // scale-aware integer reference (same per-tensor multipliers)
  auto mult = [](double a, double b, double c){ auto m = QuantInfer::requant_mult(a, b, c); return pair<int64_t,int>{(int64_t)m.first, (int)m.second}; };
  size_t fails = 0;
  if (party == ALICE) {
    auto m1 = mult(sx, sw1, sh1), m2 = mult(sh1, sw2, sh2), m3 = mult(sh2, sw3, sout);
    cout << "per-tensor requant multipliers (c, shift):  L1=(" << m1.first << "," << m1.second
         << ")  L2=(" << m2.first << "," << m2.second << ")  L3=(" << m3.first << "," << m3.second << ")" << endl;
    vector<int64_t> r1(d1), r2(d2), ro(d3);
    for (int j = 0; j < d1; ++j) { int64_t a = b1[j]*0; for (int i=0;i<d0;++i) a += W1[j][i]*x[i]; r1[j] = max((int64_t)0, rcm(a, m1.first, m1.second) + b1[j]); }
    for (int j = 0; j < d2; ++j) { int64_t a = 0; for (int i=0;i<d1;++i) a += W2[j][i]*r1[i]; r2[j] = max((int64_t)0, rcm(a, m2.first, m2.second) + b2[j]); }
    for (int j = 0; j < d3; ++j) { int64_t a = 0; for (int i=0;i<d2;++i) a += W3[j][i]*r2[i]; ro[j] = rcm(a, m3.first, m3.second) + b3[j]; }
    for (int j = 0; j < d3; ++j) {
      int64_t got = QuantInfer::decode_signed(out[j].reveal());
      if (got != ro[j]) { if (fails < 5) cout << "out[" << j << "] " << got << " vs " << ro[j] << endl; fails++; }
    }
  } else {
    for (int j = 0; j < d3; ++j) out[j].reveal();
  }

  rclu.Teardown(); qi.destroy();
  finalize_zk_arith();
  _bench.report("mlp-pertensor (ml-quant)");
  if (party == ALICE) {
    if (fails) { cout << fails << " mismatches" << endl; error("per-tensor MLP failed\n"); }
    cout << "pass per-tensor quantized MLP (3 layers, each its own scale, c != 1)" << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
