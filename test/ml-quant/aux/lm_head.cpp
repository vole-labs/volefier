// Test the LM head (hidden -> vocab logits) + argmax certification.
// ALICE checks the logits match a plaintext reference, commits the true argmax
// as the "claimed next token", and certify() proves token = argmax(logits) at
// every position (BOB aborts on a wrong token).
//
//   test_ml-quant_lm_head 1 12345   # prover
//   test_ml-quant_lm_head 2 12345   # verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_transformer.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;
const size_t S = 4, D = 16, VOCAB = 16;

static int64_t rcm(int64_t acc, int64_t c, int M) {   // matches requant_cm (wc=1 here)
  int64_t N = acc * c, h = (int64_t)1 << (M - 1), Dv = (int64_t)1 << M;
  return (N >= 0) ? (N + h) / Dv : -(((-N) + h) / Dv);
}

static PRG *gprg;
static int64_t rnd(int64_t lo, int64_t hi) { if (party != ALICE) return 0; uint64_t r; gprg->random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); }
static vector<int64_t> rvec(size_t n, int64_t lo, int64_t hi) { vector<int64_t> v(n); for (auto &e : v) e = rnd(lo, hi); return v; }
static vector<vector<int64_t>> rmat(size_t r, size_t c, int64_t lo, int64_t hi) { vector<vector<int64_t>> m(r); for (auto &row : m) row = rvec(c, lo, hi); return m; }

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "------------ ZK proof of LM head + argmax certification ------------" << endl << endl;

  PRG prg; gprg = &prg;
  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);
  QuantInfer qi(party, P, ios[0], /*out_bits=*/18, /*wc=*/1, /*mat_threads=*/1);
  RangeCheckLU rclu(party, ios[0]->io, /*table_bits=*/16);
  CompareLU cmp(&rclu);
  QuantLMHead lm(&qi, &cmp, D, VOCAB, /*logit_bits=*/14);

  // committed inputs (meaningful on ALICE)
  vector<vector<int64_t>> H = rmat(S, D, -2000, 2000), We = rmat(VOCAB, D, -20, 20);
  vector<int64_t> be = rvec(VOCAB, -100, 100);

  Mat cH, cWe; qi.commit_mat(cH, H); qi.commit_mat(cWe, We);
  vector<IntFp> cbe; qi.commit_vec(cbe, be);

  Mat logits; lm.forward(logits, cH, cWe, cbe);

  // reference logits + argmax (first max), committed as the claimed tokens
  vector<vector<int64_t>> ref(S, vector<int64_t>(VOCAB));
  vector<int64_t> argmax(S, 0);
  if (party == ALICE)
    for (size_t t = 0; t < S; ++t) {
      for (size_t v = 0; v < VOCAB; ++v) {
        int64_t acc = 0; for (size_t k = 0; k < D; ++k) acc += H[t][k] * We[v][k];
        ref[t][v] = rcm(acc, 1, P) + be[v];
      }
      argmax[t] = max_element(ref[t].begin(), ref[t].end()) - ref[t].begin();
    }
  vector<IntFp> ctoken(S);
  for (size_t t = 0; t < S; ++t) ctoken[t] = qi.commit(argmax[t]);

  // PROVE: claimed token = argmax(logits[t]) for every position
  lm.certify(logits, ctoken);

  // check logits match reference (argmax is proven cryptographically by certify)
  size_t fails = 0;
  for (size_t t = 0; t < S; ++t)
    for (size_t v = 0; v < VOCAB; ++v) {
      int64_t got = QuantInfer::decode_signed(logits[t][v].reveal());
      if (party == ALICE && got != ref[t][v]) { if (fails < 5) cout << "logit[" << t << "][" << v << "] " << got << " vs " << ref[t][v] << endl; fails++; }
    }

  rclu.Teardown();
  qi.destroy();
  finalize_zk_arith();
  _bench.report("lm_head (ml-quant)");
  if (party == ALICE) {
    if (fails) { cout << fails << " logit mismatches" << endl; error("lm head failed\n"); }
    cout << "pass LM head + argmax certification (" << S << " tokens, vocab " << VOCAB << ", argmax = ";
    for (size_t t = 0; t < S; ++t) cout << argmax[t] << (t + 1 < S ? "," : "");
    cout << ")" << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
