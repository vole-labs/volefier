// Unit tests for the non-linear transformer gadgets (DeepProve-style):
//   softmax (max-subtract + exp lookup + normalize), GeLU (table), LayerNorm
//   (sqrt-free advice).  ALICE checks each against an exact integer reference.
//
//   test_ml-quant_quant_nonlinear 1 12345   # prover
//   test_ml-quant_quant_nonlinear 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_nonlinear.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;

const int P = 12;
const uint64_t TBL = 1u << 14;     // table size for exp / gelu
const int64_t GELU_OFF = 1 << 13;  // gelu index offset (x in [-2,2) * 2^P)

// reference integer ops matching the gadgets
static int64_t floordiv(int64_t a, int64_t b) {     // reference floor division
  int64_t q = a / b, r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) q--;
  return q;
}
static int64_t arshift(int64_t x, int p) { return x >> p; }  // matches truncate
static int64_t rounddiv(int64_t N, int64_t D) {              // matches Div::round_div
  return (N >= 0) ? (N + D / 2) / D : -(((-N) + D / 2) / D);
}
static int64_t rcm(int64_t acc, int64_t c, int M) {          // matches requant_cm: round(acc*c/2^M)
  int64_t N = acc * c, h = (int64_t)1 << (M - 1), D = (int64_t)1 << M;
  return (N >= 0) ? (N + h) / D : -(((-N) + h) / D);
}

static int64_t rng(PRG &prg, int64_t lo, int64_t hi) {
  uint64_t r; prg.random_data(&r, sizeof(r));
  return lo + static_cast<int64_t>(r % static_cast<uint64_t>(hi - lo + 1));
}

void run(BoolIO *ios[threads], int party) {
  PRG prg;
  setup_zk_arith(ios, threads, party);

  QuantInfer qi(party, P, ios[0], 15);
  RangeCheckLU rcmp(party, ios[0]->io, 16); CompareLU cmp(&rcmp);
  Div divr(party, ios[0]->io, /*range_bits=*/16);   // covers 2*sum and 2*d

  QuantTable expt(TBL, party, ios[0]->io);
  expt.Setup(build_exp_table(TBL, P));
  QuantTable gtab(TBL, party, ios[0]->io);
  gtab.Setup(build_gelu_table(TBL, P, GELU_OFF));

  vector<uint64_t> exp_ref = build_exp_table(TBL, P);
  vector<uint64_t> gelu_ref = build_gelu_table(TBL, P, GELU_OFF);

  // ---------------- LayerNorm (d = 8) ----------------
  {
    const size_t d = 8;
    vector<int64_t> x(d), g(d), b(d);
    if (party == ALICE)
      for (size_t i = 0; i < d; ++i) {
        x[i] = rng(prg, -3000, 3000);
        g[i] = (1 << P) + rng(prg, -200, 200);   // gamma ~ 1.0
        b[i] = rng(prg, -100, 100);              // beta ~ 0
      }
    vector<IntFp> cx, cg, cb;
    qi.commit_vec(cx, x); qi.commit_vec(cg, g); qi.commit_vec(cb, b);

    QuantLayerNorm ln(&qi, &rcmp, &divr, P, d);
    vector<IntFp> out;
    ln.forward(out, cx, cg, cb);

    vector<int64_t> ref(d);
    if (party == ALICE) {
      int64_t sx = 0; for (auto v : x) sx += v;
      int64_t mu = rounddiv(sx, d);
      int64_t ssq = 0; vector<int64_t> c(d);
      for (size_t i = 0; i < d; ++i) { c[i] = x[i] - mu; ssq += c[i] * c[i]; }
      int64_t vq = rounddiv(ssq, d);
      int64_t dinv = llround((double)(1ull << (2 * P)) / sqrt((double)vq));
      for (size_t i = 0; i < d; ++i) {
        int64_t y = rcm(c[i] * dinv, 1, P);
        ref[i] = rcm(y * g[i], 1, P) + b[i];
      }
    }
    for (size_t i = 0; i < d; ++i) {
      int64_t got = QuantInfer::decode_signed(out[i].reveal());
      if (party == ALICE && got != ref[i]) {
        cout << "LN[" << i << "] got " << got << " exp " << ref[i] << endl;
        error("layernorm mismatch\n");
      }
    }
    if (party == ALICE) cout << "pass LayerNorm" << endl;
  }

  // ---------------- GeLU (vector of 8) ----------------
  {
    const size_t n = 8;
    vector<int64_t> x(n);
    if (party == ALICE) for (size_t i = 0; i < n; ++i) x[i] = rng(prg, -4000, 4000);
    vector<IntFp> cx; qi.commit_vec(cx, x);

    QuantActivation gelu(&gtab, GELU_OFF);
    vector<IntFp> out; gelu.forward(out, cx);

    for (size_t i = 0; i < n; ++i) {
      int64_t got = QuantInfer::decode_signed(out[i].reveal());
      if (party == ALICE) {
        int64_t ref = QuantInfer::decode_signed(gelu_ref[x[i] + GELU_OFF]);
        if (got != ref) { cout << "GeLU[" << i << "] got " << got << " exp " << ref << endl; error("gelu mismatch\n"); }
      }
    }
    if (party == ALICE) cout << "pass GeLU" << endl;
  }

  // ---------------- Softmax (L = 6) ----------------
  {
    const size_t L = 6;
    vector<int64_t> x(L);
    if (party == ALICE) for (size_t i = 0; i < L; ++i) x[i] = rng(prg, -2000, 2000);
    vector<IntFp> cx; qi.commit_vec(cx, x);

    QuantSoftmax sm(&qi, &cmp, &expt, &divr, P, /*score_bits=*/14);
    vector<IntFp> out = sm.forward(cx);

    vector<int64_t> ref(L);
    if (party == ALICE) {
      int64_t m = x[0]; for (auto v : x) m = max(m, v);
      int64_t sum = 0; vector<int64_t> e(L);
      for (size_t i = 0; i < L; ++i) { e[i] = (int64_t)exp_ref[m - x[i]]; sum += e[i]; }
      for (size_t i = 0; i < L; ++i) ref[i] = rounddiv(e[i] * (1ll << P), sum);
    }
    for (size_t i = 0; i < L; ++i) {
      int64_t got = QuantInfer::decode_signed(out[i].reveal());
      if (party == ALICE && got != ref[i]) {
        cout << "softmax[" << i << "] got " << got << " exp " << ref[i] << endl;
        error("softmax mismatch\n");
      }
    }
    if (party == ALICE) cout << "pass Softmax" << endl;
  }

  rcmp.Teardown();

  divr.destroy();
  expt.Teardown();
  gtab.Teardown();
  qi.destroy();
  finalize_zk_arith();
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ ZK proof of transformer non-linear ops ------------" << endl << endl;
  run(ios, party);
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
