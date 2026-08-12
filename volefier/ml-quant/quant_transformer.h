#ifndef ML_QUANT_TRANSFORMER_H
#define ML_QUANT_TRANSFORMER_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-quant/quant_infer.h"
#include "volefier/ml-quant/quant_nonlinear.h"
#include "volefier/nonlinear/comparison.h" // RangeCheckLU + CompareLU (argmax)
#include "volefier/zk-ram/logup_keyvalue.h"
#include <cstdint>
#include <vector>
using namespace emp;

// Transformer-level building blocks: token embedding (LogUp key-value lookup),
// multi-head causal self-attention, and a GPT-2 style transformer block, all
// assembled from the quant_infer / quant_nonlinear gadgets.  Where DeepProve
// and zkGPT differ we follow DeepProve: lookup-based activations, sqrt-free
// LayerNorm, batched single-head attention, causal masking by prefix.

using Mat = std::vector<std::vector<IntFp>>; // row-major committed matrix

// ---------------------------------------------------------------------------
// Token embedding: row id -> d-dim committed row, plus a positional embedding.
class QuantEmbedding {
public:
  LogUpKeyValueLookupBatch *tok = nullptr;
  uint64_t vocab, d;
  int party;

  QuantEmbedding(uint64_t vocab_, uint64_t d_, int party_, IOChannel *io)
      : vocab(vocab_), d(d_), party(party_) {
    tok = new LogUpKeyValueLookupBatch(vocab_, d_, party_, io);
  }
  ~QuantEmbedding() {
    delete tok;
  }

  // table_raw: vocab*d row-major token-embedding matrix (field-encoded).
  void Setup(const std::vector<uint64_t> &table_raw) {
    tok->Setup(table_raw);
  }

  // H[i] = tok_emb[token_ids[i]] + pos_emb[i].
  void embed(Mat &H, std::vector<IntFp> &token_ids, const Mat &pos_emb) {
    const std::size_t s = token_ids.size();
    H.assign(s, std::vector<IntFp>(d));
    std::vector<IntFp> row;
    for (std::size_t i = 0; i < s; ++i) {
      tok->Access(row, token_ids[i]);
      for (std::size_t k = 0; k < d; ++k) H[i][k] = row[k] + pos_emb[i][k];
    }
  }

  void Teardown() {
    tok->Teardown();
  }
};

// ---------------------------------------------------------------------------
// PER-TENSOR quantization scale schedule (zkGPT/DeepProve method). Each named
// activation tensor has its own scale; LN/softmax/GeLU outputs anchor at 2^-P.
struct TScale {
  double w = 1.0 / 200, q = 1.0 / 300, k = 1.0 / 280, v = 1.0 / 320, score = 1.0 / 256,
         ctx = 1.0 / 290, hid = 1.0 / 310, res = 1.0 / 256;
};

// Multi-head causal self-attention with PER-TENSOR activation scales: Q/K/V,
// scores, context each carry their own scale; the LN output that feeds the
// attention is the 2^-P anchor, the output projection requants back to S_res.
class QuantAttention {
public:
  QuantInfer *qi;
  QuantSoftmax *sm;
  int P;
  std::size_t d, h, dh;
  TScale ts;
  double S_a; // 2^-P anchor (LN / softmax output scale, feeds Q/K/V)
  uint64_t cQ, cK, cV, cQK, cAV;
  std::size_t mQ, mK, mV, mQK, mAV;

  QuantAttention(QuantInfer *qi_, QuantSoftmax *sm_, int P_, std::size_t d_, std::size_t h_,
                 TScale ts_ = TScale())
      : qi(qi_), sm(sm_), P(P_), d(d_), h(h_), dh(d_ / h_), ts(ts_) {
    S_a = 1.0 / static_cast<double>(1ULL << P);
    auto mul = [](double r) {
      auto ce = QuantInfer::scale_search(r);
      return std::pair<uint64_t, std::size_t>{(uint64_t)ce.first, (std::size_t)(-ce.second)};
    };
    auto a = mul(S_a * ts.w / ts.q);
    cQ = a.first;
    mQ = a.second;
    auto b = mul(S_a * ts.w / ts.k);
    cK = b.first;
    mK = b.second;
    auto c = mul(S_a * ts.w / ts.v);
    cV = c.first;
    mV = c.second;
    auto e = mul(ts.q * ts.k / (std::sqrt((double)dh) * ts.score));
    cQK = e.first;
    mQK = e.second; // Q.K^T*1/sqrt(dh)
    auto f = mul(S_a * ts.v / ts.ctx);
    cAV = f.first;
    mAV = f.second; // probs(2^-P).V
  }

  // out: s x d (scale S_res). H = LN output: s x d (scale 2^-P anchor).
  void forward(Mat &out, const Mat &H, const Mat &Wq, const std::vector<IntFp> &bq, const Mat &Wk,
               const std::vector<IntFp> &bk, const Mat &Wv, const std::vector<IntFp> &bv,
               const Mat &Wo, const std::vector<IntFp> &bo) {
    const std::size_t s = H.size();
    Mat Q, K, V; // Q/K/V at their own per-tensor scales (batched matmul over all tokens)
    qi->dense_layer_q_mm(Q, Wq, bq, H, S_a, ts.w, ts.q);
    qi->dense_layer_q_mm(K, Wk, bk, H, S_a, ts.w, ts.k);
    qi->dense_layer_q_mm(V, Wv, bv, H, S_a, ts.w, ts.v);
    Mat ctx(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      for (std::size_t hh = 0; hh < h; ++hh) {
        const std::size_t off = hh * dh;
        std::vector<IntFp> qh(Q[i].begin() + off, Q[i].begin() + off + dh);
        std::vector<IntFp> scaled(i + 1);
        for (std::size_t j = 0; j <= i; ++j) { // Q.K^T * 1/sqrt(dh) -> S_score
          std::vector<IntFp> kh(K[j].begin() + off, K[j].begin() + off + dh);
          scaled[j] = qi->requant_cm_one(qi->dot_ip(qh, kh), cQK, mQK);
        }
        std::vector<IntFp> probs = sm->forward(scaled); // -> 2^-P anchor
        for (std::size_t k = 0; k < dh; ++k) {          // A.V -> S_ctx
          std::vector<IntFp> vk(i + 1);
          for (std::size_t j = 0; j <= i; ++j) vk[j] = V[j][off + k];
          ctx[i][off + k] = qi->requant_cm_one(qi->dot_ip(probs, vk), cAV, mAV);
        }
      }
    }
    qi->dense_layer_q_mm(out, Wo, bo, ctx, ts.ctx, ts.w, ts.res); // O proj -> S_res (residual scale)
  }
};

// ---------------------------------------------------------------------------
// GPT-2 transformer block:
//   a = x + Attn(LN1(x)); y = a + FFN(LN2(a)),  FFN = GeLU(x W1+b1) W2+b2.
class QuantTransformerBlock {
public:
  QuantInfer *qi;
  QuantLayerNorm *ln1, *ln2;
  QuantAttention *attn;
  QuantActivation *gelu;
  std::size_t d, ffn;
  TScale ts;
  double S_a; // 2^-P anchor (LN2 output, GeLU output)

  QuantTransformerBlock(QuantInfer *qi_, QuantLayerNorm *ln1_, QuantLayerNorm *ln2_,
                        QuantAttention *attn_, QuantActivation *gelu_, std::size_t d_,
                        std::size_t ffn_, TScale ts_ = TScale())
      : qi(qi_), ln1(ln1_), ln2(ln2_), attn(attn_), gelu(gelu_), d(d_), ffn(ffn_), ts(ts_),
        S_a(1.0 / static_cast<double>(1ULL << qi_->precision)) {}

  void forward(Mat &out, const Mat &x, const std::vector<IntFp> &g1, const std::vector<IntFp> &b1n,
               const std::vector<IntFp> &g2, const std::vector<IntFp> &b2n, const Mat &Wq,
               const std::vector<IntFp> &bq, const Mat &Wk, const std::vector<IntFp> &bk,
               const Mat &Wv, const std::vector<IntFp> &bv, const Mat &Wo,
               const std::vector<IntFp> &bo, const Mat &W1, const std::vector<IntFp> &b1,
               const Mat &W2, const std::vector<IntFp> &b2) {
    const std::size_t s = x.size();

    Mat normed(s), attn_out;
    for (std::size_t i = 0; i < s; ++i) ln1->forward(normed[i], x[i], g1, b1n);
    attn->forward(attn_out, normed, Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    Mat a(s);
    for (std::size_t i = 0; i < s; ++i) qi->add(a[i], x[i], attn_out[i]);

    // FFN with batched matmuls: LN2 -> W1 (d->ffn, S_h) -> GeLU (2^-P) -> W2 (ffn->d, S_res).
    Mat n2(s), hidden, geluout(s), ff;
    for (std::size_t i = 0; i < s; ++i) ln2->forward(n2[i], a[i], g2, b2n);
    qi->dense_layer_q_mm(hidden, W1, b1, n2, S_a, ts.w, ts.hid);
    for (std::size_t i = 0; i < s; ++i) gelu->forward(geluout[i], hidden[i]);
    qi->dense_layer_q_mm(ff, W2, b2, geluout, S_a, ts.w, ts.res);
    out.assign(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) qi->add(out[i], a[i], ff[i]);
  }
};

// ---------------------------------------------------------------------------
// LM head (unembedding) + argmax certification (DeepProve's output side):
//   logits[t] = requant(H[t] . We^T) + be       (We: vocab x d, tied to embed)
//   certify  : prove the claimed next token = argmax(logits[t]) at each position.
// This is the certification trick: the proven sequence is correct iff every
// position's argmax matches the claimed token, so proving inference reduces to
// one forward pass + an argmax check per position (linear, not quadratic).
class QuantLMHead {
public:
  QuantInfer *qi;
  CompareLU *cmp; // lookup-table argmax (DeepProve-style, no MSB)
  std::size_t d, vocab, logit_bits;

  // logit_bits bounds the logit magnitude so each argmax comparison is a
  // RangeCheckLU of ceil((logit_bits+1)/table_bits) lookups (cf. DeepProve).
  QuantLMHead(QuantInfer *qi_, CompareLU *cmp_, std::size_t d_, std::size_t vocab_,
              std::size_t logit_bits_ = 24)
      : qi(qi_), cmp(cmp_), d(d_), vocab(vocab_), logit_bits(logit_bits_) {}

  // logits: s x vocab.  H: s x d.  We: vocab x d.  be: vocab.
  void forward(Mat &logits, const Mat &H, const Mat &We, const std::vector<IntFp> &be) {
    const std::size_t s = H.size();
    logits.assign(s, std::vector<IntFp>(vocab));
    for (std::size_t t = 0; t < s; ++t) qi->dense_layer(logits[t], We, be, H[t]);
  }

  // Prove token_id[t] = argmax(logits[t]) for every position: argmax() returns
  // the committed (constrained-valid) argmax index; assert it equals the claimed
  // token. Each comparison is a range-check on the difference, not an MSB.
  void certify(const Mat &logits, std::vector<IntFp> &token_id) {
    for (std::size_t t = 0; t < logits.size(); ++t) {
      IntFp idx = cmp->argmax(logits[t], logit_bits);
      (idx + token_id[t].negate()).reveal_zero(); // claimed token == proven argmax
    }
  }
};

#endif
