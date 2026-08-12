// Milestone 1 for a VOLE-ZK proof of quantized inference: a 2-layer quantized
// MLP proven end-to-end with emp-zk arithmetic + LogUp gadgets.
//
//   layer 1:  in=8  -> hidden=16,  requant + ReLU
//   layer 2:  hidden=16 -> out=4,  requant, no activation (logits)
//
// ALICE (prover) generates small Q-bit weights/activations, runs the protocol,
// and checks the revealed outputs against a plaintext fixed-point reference
// computed with exactly the same integer arithmetic the circuit proves
// (matmul -> arithmetic right shift by `precision` -> +bias -> ReLU).
//
//   Usage (two terminals):
//     test_ml-quant_quant_mlp 1 12345     # ALICE / prover
//     test_ml-quant_quant_mlp 2 12345     # BOB   / verifier

#include <iostream>
#include <vector>
#include <cstdint>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_nonlinear.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;

// Network shape (public).
const size_t IN_DIM  = 8;
const size_t HID_DIM = 16;
const size_t OUT_DIM = 4;
const size_t PRECISION = 4;          // fractional bits == requant shift
const size_t OUT_BITS  = 10;         // requant output clamp width (values are small)
const int64_t WC = 3;                // weight-scale mantissa (zkGPT c*2^e)
static int64_t rcm(int64_t acc, int64_t c, int M) { int64_t N = acc*c, h=(int64_t)1<<(M-1), D=(int64_t)1<<M; return (N>=0)?(N+h)/D:-(((-N)+h)/D); }
const uint64_t RELU_T  = 1024;       // ReLU clamp table size
const int64_t  RELU_OFF = 512;       // ReLU table index offset

// Small signed quantized weight in [-2^(b-1), 2^(b-1)-1]; here b=4 -> [-8,7].
static int64_t rand_q(PRG &prg) {
  uint64_t r;
  prg.random_data(&r, sizeof(r));
  return static_cast<int64_t>(r & 0xF) - 8;   // [-8, 7]
}

// Plaintext fixed-point dense layer, byte-for-byte identical to the circuit:
//   acc = W x ; rq = acc >> precision (arithmetic floor) ; rq += bias ; [ReLU]
static vector<int64_t> ref_dense(const vector<vector<int64_t>> &W,
                                 const vector<int64_t> &bias,
                                 const vector<int64_t> &x,
                                 bool relu) {
  vector<int64_t> y(W.size());
  for (size_t j = 0; j < W.size(); ++j) {
    int64_t acc = 0;
    for (size_t i = 0; i < x.size(); ++i) acc += W[j][i] * x[i];
    int64_t rq = rcm(acc, WC, (int)PRECISION) + bias[j];
    y[j] = (relu && rq < 0) ? 0 : rq;
  }
  return y;
}

void test_quant_mlp(BoolIO *ios[threads], int party) {
  PRG prg;   // independent per party; only ALICE's witness matters.

  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);

  QuantInfer qi(party, PRECISION, ios[0], OUT_BITS, WC);
  QuantTable rtab(RELU_T, party, ios[0]->io);
  rtab.Setup(build_relu_table(RELU_T, RELU_OFF));
  QuantActivation relu(&rtab, RELU_OFF);

  // ---- prover's plaintext quantized model + input (meaningful on ALICE) ----
  vector<vector<int64_t>> W1(HID_DIM, vector<int64_t>(IN_DIM, 0));
  vector<int64_t>         b1(HID_DIM, 0);
  vector<vector<int64_t>> W2(OUT_DIM, vector<int64_t>(HID_DIM, 0));
  vector<int64_t>         b2(OUT_DIM, 0);
  vector<int64_t>         x(IN_DIM, 0);

  if (party == ALICE) {
    for (auto &row : W1) for (auto &w : row) w = rand_q(prg);
    for (auto &v : b1) v = rand_q(prg);
    for (auto &row : W2) for (auto &w : row) w = rand_q(prg);
    for (auto &v : b2) v = rand_q(prg);
    for (auto &v : x)  v = rand_q(prg);
  }

  // ---- commit ----
  vector<vector<IntFp>> cW1, cW2;
  vector<IntFp> cb1, cb2, cx;
  qi.commit_mat(cW1, W1);  qi.commit_vec(cb1, b1);
  qi.commit_mat(cW2, W2);  qi.commit_vec(cb2, b2);
  qi.commit_vec(cx, x);

  // ---- prove inference (ReLU is now a table-lookup activation) ----
  vector<IntFp> h, h_act, logits;
  qi.dense_layer(h, cW1, cb1, cx);   // linear + requant + bias
  relu.forward(h_act, h);            // ReLU via clamp table
  qi.dense_layer(logits, cW2, cb2, h_act);

  // ---- reveal + check against plaintext reference (ALICE side) ----
  vector<int64_t> ref_h, ref_logits;
  if (party == ALICE) {
    ref_h      = ref_dense(W1, b1, x,     /*relu=*/true);
    ref_logits = ref_dense(W2, b2, ref_h, /*relu=*/false);
  }

  for (size_t j = 0; j < HID_DIM; ++j) {
    int64_t got = QuantInfer::decode_signed(h_act[j].reveal());
    if (party == ALICE && got != ref_h[j]) {
      std::cout << "hidden[" << j << "] = " << got << " expected " << ref_h[j] << std::endl;
      error("layer-1 mismatch\n");
    }
  }
  for (size_t j = 0; j < OUT_DIM; ++j) {
    int64_t got = QuantInfer::decode_signed(logits[j].reveal());
    if (party == ALICE && got != ref_logits[j]) {
      std::cout << "logit[" << j << "] = " << got << " expected " << ref_logits[j] << std::endl;
      error("layer-2 mismatch\n");
    }
  }

  if (party == ALICE) {
    std::cout << "pass quantized MLP inference (";
    for (size_t j = 0; j < OUT_DIM; ++j) std::cout << ref_logits[j] << (j + 1 < OUT_DIM ? ", " : "");
    std::cout << ")" << std::endl;
  }

  rtab.Teardown();
  qi.destroy();
  finalize_zk_arith();
  _bench.report("mlp (ml-quant)");
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);

  std::cout << std::endl << "------------ ZK proof of quantized MLP inference ------------" << std::endl << std::endl;

  test_quant_mlp(ios, party);

  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
