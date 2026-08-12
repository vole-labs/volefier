// Unit tests for the CNN / ResNet building blocks: Conv2D (im2col + IP matmul),
// BatchNorm (per-channel affine), MaxPool (CompareLU), AvgPool (Div). ALICE
// checks each against an exact integer reference.
//
//   test_ml-quant_quant_cnn 1 12345   # prover
//   test_ml-quant_quant_cnn 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_cnn.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;

static int64_t rcm(int64_t acc, int64_t c, int M) {   // matches requant_cm (round)
  int64_t N = acc * c, h = (int64_t)1 << (M - 1), D = (int64_t)1 << M;
  return (N >= 0) ? (N + h) / D : -(((-N) + h) / D);
}
static int64_t rounddiv(int64_t N, int64_t D) {       // matches Div::round_div
  return (N >= 0) ? (N + D / 2) / D : -(((-N) + D / 2) / D);
}

static PRG *gprg;
static int64_t rnd(int64_t lo, int64_t hi) { if (party != ALICE) return 0; uint64_t r; gprg->random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); }
static int64_t dec(uint64_t v) { return QuantInfer::decode_signed(v); }

void run(BoolIO *ios[threads], int party) {
  PRG prg; gprg = &prg;
  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);
  QuantInfer qi(party, P, ios[0], /*out_bits=*/20, /*wc=*/1, /*mat_threads=*/threads);
  RangeCheckLU rclu(party, ios[0]->io, 16);
  CompareLU cmp(&rclu);
  Div divr(party, ios[0]->io, /*range_bits=*/20);

  auto commitT = [&](Tensor3 &out, const vector<vector<vector<int64_t>>> &in) {
    out.assign(in.size(), {});
    for (size_t c = 0; c < in.size(); ++c) { out[c].assign(in[c].size(), {}); for (size_t h = 0; h < in[c].size(); ++h) qi.commit_vec(out[c][h], in[c][h]); }
  };

  size_t fails = 0;

  // ---------------- Conv2D (C_in=2, 5x5 -> C_out=3, K=3, s=1, p=1) ----------------
  {
    const int Ci = 2, Hh = 5, Ww = 5, Co = 3, K = 3, s = 1, pd = 1;
    const int M = Ci * K * K, Ho = 5, Wo = 5, Pn = Ho * Wo;
    vector<vector<vector<int64_t>>> in(Ci, vector<vector<int64_t>>(Hh, vector<int64_t>(Ww)));
    for (auto &c : in) for (auto &r : c) for (auto &v : r) v = rnd(-2000, 2000);
    vector<vector<int64_t>> ker(Co, vector<int64_t>(M));
    for (auto &r : ker) for (auto &v : r) v = rnd(-3, 3);
    vector<int64_t> bias(Co); for (auto &v : bias) v = rnd(-100, 100);

    Tensor3 cin; commitT(cin, in);
    vector<vector<IntFp>> cker(Co); for (int co = 0; co < Co; ++co) qi.commit_vec(cker[co], ker[co]);
    vector<IntFp> cbias; qi.commit_vec(cbias, bias);

    QuantConv2D conv(&qi, Ci, Hh, Ww, Co, K, s, pd);
    FMap out; conv.forward(out, cin, cker, cbias);

    for (int co = 0; co < Co; ++co)
      for (int oh = 0; oh < Ho; ++oh) for (int ow = 0; ow < Wo; ++ow) {
        int64_t raw = 0;
        for (int ci = 0; ci < Ci; ++ci) for (int ki = 0; ki < K; ++ki) for (int kj = 0; kj < K; ++kj) {
          int ih = oh * s - pd + ki, iw = ow * s - pd + kj;
          if (ih >= 0 && ih < Hh && iw >= 0 && iw < Ww) raw += in[ci][ih][iw] * ker[co][ci * K * K + ki * K + kj];
        }
        int64_t ref = rcm(raw, 1, P) + bias[co];
        int64_t got = dec(out[co][oh * Wo + ow].reveal());
        if (party == ALICE && got != ref) { if (fails < 5) cout << "conv[" << co << "][" << oh << "," << ow << "] " << got << " vs " << ref << endl; fails++; }
      }
    if (party == ALICE && fails == 0) cout << "pass Conv2D" << endl;
  }

  // ---------------- BatchNorm (C=3, P=9, per-channel affine) ----------------
  {
    const int C = 3, Pn = 9;
    vector<vector<int64_t>> x(C, vector<int64_t>(Pn)), g(C, vector<int64_t>(1)), b(C, vector<int64_t>(1));
    vector<int64_t> gamma(C), beta(C);
    for (int c = 0; c < C; ++c) { for (int p = 0; p < Pn; ++p) x[c][p] = rnd(-3000, 3000); gamma[c] = (1 << P) + rnd(-200, 200); beta[c] = rnd(-100, 100); }
    FMap cx(C); for (int c = 0; c < C; ++c) qi.commit_vec(cx[c], x[c]);
    vector<IntFp> cg, cb; qi.commit_vec(cg, gamma); qi.commit_vec(cb, beta);

    QuantBatchNorm bn(&qi, C, Pn);
    FMap out; bn.forward(out, cx, cg, cb);
    size_t bf = 0;
    for (int c = 0; c < C; ++c) for (int p = 0; p < Pn; ++p) {
      int64_t ref = rcm(x[c][p] * gamma[c], 1, P) + beta[c];
      int64_t got = dec(out[c][p].reveal());
      if (party == ALICE && got != ref) { if (bf < 5) cout << "bn[" << c << "][" << p << "] " << got << " vs " << ref << endl; bf++; }
    }
    fails += bf;
    if (party == ALICE && bf == 0) cout << "pass BatchNorm" << endl;
  }

  // ---------------- MaxPool / AvgPool (C=2, 4x4, K=2, s=2) ----------------
  {
    const int C = 2, Hh = 4, Ww = 4, K = 2, s = 2;
    vector<vector<vector<int64_t>>> in(C, vector<vector<int64_t>>(Hh, vector<int64_t>(Ww)));
    for (auto &c : in) for (auto &r : c) for (auto &v : r) v = rnd(-2000, 2000);
    Tensor3 cin; commitT(cin, in);
    const int Ho = 2, Wo = 2;

    QuantMaxPool mp(&qi, &cmp, C, Hh, Ww, K, s, 0, /*val_bits=*/14);
    FMap mo; mp.forward(mo, cin);
    QuantAvgPool ap(&qi, &divr, C, Hh, Ww, K, s, 0);
    FMap ao; ap.forward(ao, cin);

    size_t pf = 0;
    for (int c = 0; c < C; ++c) for (int oh = 0; oh < Ho; ++oh) for (int ow = 0; ow < Wo; ++ow) {
      int64_t mx = INT64_MIN, sm = 0; int cnt = 0;
      for (int ki = 0; ki < K; ++ki) for (int kj = 0; kj < K; ++kj) { int64_t v = in[c][oh * s + ki][ow * s + kj]; mx = max(mx, v); sm += v; ++cnt; }
      int64_t gm = dec(mo[c][oh * Wo + ow].reveal()), ga = dec(ao[c][oh * Wo + ow].reveal());
      if (party == ALICE) {
        if (gm != mx) { if (pf < 5) cout << "maxpool " << gm << " vs " << mx << endl; pf++; }
        if (ga != rounddiv(sm, cnt)) { if (pf < 5) cout << "avgpool " << ga << " vs " << rounddiv(sm, cnt) << endl; pf++; }
      }
    }
    fails += pf;
    if (party == ALICE && pf == 0) cout << "pass MaxPool + AvgPool" << endl;
  }

  rclu.Teardown();
  divr.destroy();
  qi.destroy();
  finalize_zk_arith();
  _bench.report("cnn (ml-quant)");
  if (party == ALICE) {
    if (fails) { cout << fails << " mismatches" << endl; error("CNN blocks failed\n"); }
    cout << "pass ALL CNN building blocks (Conv2D, BatchNorm, MaxPool, AvgPool)" << endl;
  }
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ ZK proof of CNN building blocks ------------" << endl << endl;
  run(ios, party);
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
