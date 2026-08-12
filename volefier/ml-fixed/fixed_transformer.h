#ifndef ML_FIXED_TRANSFORMER_H
#define ML_FIXED_TRANSFORMER_H

#include "emp-zk/emp-zk.h"
#include "volefier/ml-fixed/fixed_infer.h"
#include "volefier/ml-fixed/fixed_nonlinear.h"
#include "volefier/nonlinear/comparison.h" // RangeCheckLU + CompareLU (argmax)
#include "volefier/zk-ram/logup_keyvalue.h"
#include <cstdint>
#include <map>
#include <string>
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
class FixedEmbedding {
public:
  LogUpKeyValueLookupBatch *tok = nullptr;
  uint64_t vocab, d;
  int party;

  FixedEmbedding(uint64_t vocab_, uint64_t d_, int party_, IOChannel *io)
      : vocab(vocab_), d(d_), party(party_) {
    tok = new LogUpKeyValueLookupBatch(vocab_, d_, party_, io);
  }
  ~FixedEmbedding() {
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
// Multi-head causal self-attention.  qinv = round((1/sqrt(dh)) * 2^P).
class FixedAttention {
public:
  FixedInfer *qi;
  FixedSoftmax *sm;
  int P;
  std::size_t d, h, dh;
  uint64_t qinv;

  // Opt-in cache of forward intermediates for a backward/training pass, keyed by name (mirrors
  // FixedGraphInfer::act_).  When save_intermediates is set, forward() records the committed
  // handles: "Q","K","V","ctx" as [s][d]; "scores","probs" as s*h ragged causal rows -- row
  // i*h+hh holds the length-(i+1) vector for query i, head hh.  Fetch after forward(); clear()
  // between steps.  Off by default, so the inference path pays nothing.
  bool save_intermediates = false;
  std::map<std::string, Mat> cache;
  void clear_cache() { cache.clear(); }

  FixedAttention(FixedInfer *qi_, FixedSoftmax *sm_, int P_, std::size_t d_, std::size_t h_,
                 uint64_t qinv_)
      : qi(qi_), sm(sm_), P(P_), d(d_), h(h_), dh(d_ / h_), qinv(qinv_) {}

  // Project x (length d) by W (d x d) + b -> length d, requant to scale 2^-P.
  std::vector<IntFp> proj(const Mat &W, const std::vector<IntFp> &b, const std::vector<IntFp> &x) {
    std::vector<IntFp> y;
    qi->dense_layer(y, W, b, x);
    return y;
  }

  // out: s x d.  H: s x d.  Wq/Wk/Wv/Wo: d x d.  bq/bk/bv/bo: d.
  void forward(Mat &out, const Mat &H, const Mat &Wq, const std::vector<IntFp> &bq, const Mat &Wk,
               const std::vector<IntFp> &bk, const Mat &Wv, const std::vector<IntFp> &bv,
               const Mat &Wo, const std::vector<IntFp> &bo) {
    const std::size_t s = H.size();
    Mat Q(s), K(s), V(s);
    for (std::size_t i = 0; i < s; ++i) {
      Q[i] = proj(Wq, bq, H[i]);
      K[i] = proj(Wk, bk, H[i]);
      V[i] = proj(Wv, bv, H[i]);
    }
    // scores/probs are per-(query,head) transients below; pre-size the ragged caches so the
    // backward pass can index them by row i*h+hh.
    if (save_intermediates) { cache["scores"].assign(s * h, {}); cache["probs"].assign(s * h, {}); }

    // ctx is kept at scale 2^-2P (the A.V requant is DEFERRED into the output
    // projection, which then truncates 2*P at once -- one requant instead of two).
    Mat ctx(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      for (std::size_t hh = 0; hh < h; ++hh) {
        const std::size_t off = hh * dh;
        // causal scores over j = 0..i. Q.K^T truncation and the 1/sqrt(dh)
        // scale are FUSED into one requant (shift = 2P).
        std::vector<IntFp> qh(Q[i].begin() + off, Q[i].begin() + off + dh);
        std::vector<IntFp> scaled(i + 1);
        for (std::size_t j = 0; j <= i; ++j) {
          std::vector<IntFp> kh(K[j].begin() + off, K[j].begin() + off + dh);
          scaled[j] = qi->requant_fp_shift(qi->dot_ip(qh, kh), qinv, 2 * (std::size_t)P);
        }
        std::vector<IntFp> probs = sm->forward(scaled);
        if (save_intermediates) {
          cache["probs"][i * h + hh] = probs;              // still needed for A.V below -> copy
          cache["scores"][i * h + hh] = std::move(scaled); // consumed by softmax already -> move
        }

        for (std::size_t k = 0; k < dh; ++k) { // A.V -- NO requant (scale 2^-2P)
          std::vector<IntFp> vk(i + 1);
          for (std::size_t j = 0; j <= i; ++j) vk[j] = V[j][off + k];
          ctx[i][off + k] = qi->dot_ip(probs, vk);
        }
      }
    }

    // Output projection on the 2^-2P ctx: Wo.ctx is at 2^-3P, so truncate 2P to
    // land back at 2^-P; the bias (2^-P) is added after.
    out.assign(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      std::vector<IntFp> acc;
      qi->linear_ip(acc, Wo, ctx[i]);
      std::vector<IntFp> tr;
      qi->requant_cm(tr, acc, qi->wc, 2 * (std::size_t)P); // weight matmul, deferred 2P
      out[i].resize(d);
      for (std::size_t j = 0; j < d; ++j) out[i][j] = tr[j] + bo[j];
    }
    // Stash the full matrices for a backward pass (moved -- dead after the output projection).
    if (save_intermediates) {
      cache["Q"] = std::move(Q);
      cache["K"] = std::move(K);
      cache["V"] = std::move(V);
      cache["ctx"] = std::move(ctx);
    }
  }

  // Same computation as forward(), but RETAINS the committed intermediates the
  // backward pass (AttentionBackward) needs: the Q/K/V projections, the per
  // (token,head) softmax probabilities, and ctx BEFORE the output projection
  // (so ctx_saved stays at scale 2^-2P, the deferred-requant scale).  forward()
  // is left untouched; this is an additive, logic-identical twin.
  //   probs_saved[i][hh] is the length-(i+1) causal probability vector.
  void forward_save(Mat &out, Mat &Q_saved, Mat &K_saved, Mat &V_saved,
                    std::vector<std::vector<std::vector<IntFp>>> &probs_saved, Mat &ctx_saved,
                    const Mat &H, const Mat &Wq, const std::vector<IntFp> &bq, const Mat &Wk,
                    const std::vector<IntFp> &bk, const Mat &Wv, const std::vector<IntFp> &bv,
                    const Mat &Wo, const std::vector<IntFp> &bo) {
    const std::size_t s = H.size();
    Q_saved.assign(s, {});
    K_saved.assign(s, {});
    V_saved.assign(s, {});
    for (std::size_t i = 0; i < s; ++i) {
      Q_saved[i] = proj(Wq, bq, H[i]);
      K_saved[i] = proj(Wk, bk, H[i]);
      V_saved[i] = proj(Wv, bv, H[i]);
    }

    probs_saved.assign(s, std::vector<std::vector<IntFp>>(h));
    ctx_saved.assign(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      for (std::size_t hh = 0; hh < h; ++hh) {
        const std::size_t off = hh * dh;
        std::vector<IntFp> qh(Q_saved[i].begin() + off, Q_saved[i].begin() + off + dh);
        std::vector<IntFp> scaled(i + 1);
        for (std::size_t j = 0; j <= i; ++j) {
          std::vector<IntFp> kh(K_saved[j].begin() + off, K_saved[j].begin() + off + dh);
          scaled[j] = qi->requant_fp_shift(qi->dot_ip(qh, kh), qinv, 2 * (std::size_t)P);
        }
        std::vector<IntFp> probs = sm->forward(scaled);
        probs_saved[i][hh] = probs; // committed probs, length i+1 (causal)

        for (std::size_t k = 0; k < dh; ++k) {
          std::vector<IntFp> vk(i + 1);
          for (std::size_t j = 0; j <= i; ++j) vk[j] = V_saved[j][off + k];
          // ctx_saved is at 2^-2P (deferred requant). AttentionBackward must
          // requant dCtx by P before feeding into Attn@V matmul_backward.
          ctx_saved[i][off + k] = qi->dot_ip(probs, vk);
        }
      }
    }

    out.assign(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      std::vector<IntFp> acc;
      qi->linear_ip(acc, Wo, ctx_saved[i]);
      std::vector<IntFp> tr;
      qi->requant_cm(tr, acc, qi->wc, 2 * (std::size_t)P);
      for (std::size_t j = 0; j < d; ++j) out[i][j] = tr[j] + bo[j];
    }
  }
};

// ---------------------------------------------------------------------------
// GPT-2 transformer block:
//   a = x + Attn(LN1(x)); y = a + FFN(LN2(a)),  FFN = GeLU(x W1+b1) W2+b2.
class FixedTransformerBlock {
public:
  FixedInfer *qi;
  FixedLayerNorm *ln1, *ln2;
  FixedAttention *attn;
  FixedActivation *gelu;
  std::size_t d, ffn;

  // Backward-pass intermediate cache (opt-in): retain the pre-GELU activations
  // (hidden) that GELUBackward needs, plus the LN2 output (normed2) that the FFN
  // input backward (dLdW1) needs.  Same save_intermediates flag + map pattern.
  bool save_intermediates = false;
  std::map<std::string, Mat> cache;
  void clear_cache() { cache.clear(); }

  FixedTransformerBlock(FixedInfer *qi_, FixedLayerNorm *ln1_, FixedLayerNorm *ln2_,
                        FixedAttention *attn_, FixedActivation *gelu_, std::size_t d_,
                        std::size_t ffn_)
      : qi(qi_), ln1(ln1_), ln2(ln2_), attn(attn_), gelu(gelu_), d(d_), ffn(ffn_) {}

  void forward(Mat &out, const Mat &x, const std::vector<IntFp> &g1, const std::vector<IntFp> &b1n,
               const std::vector<IntFp> &g2, const std::vector<IntFp> &b2n, const Mat &Wq,
               const std::vector<IntFp> &bq, const Mat &Wk, const std::vector<IntFp> &bk,
               const Mat &Wv, const std::vector<IntFp> &bv, const Mat &Wo,
               const std::vector<IntFp> &bo, const Mat &W1, const std::vector<IntFp> &b1,
               const Mat &W2, const std::vector<IntFp> &b2) {
    const std::size_t s = x.size();

    Mat normed(s), attn_out;
    for (std::size_t i = 0; i < s; ++i) ln1->forward(normed[i], x[i], g1, b1n);
    if (save_intermediates) cache["normed"] = normed; // LN1 output = attention input, for dLdWq/k/v
    attn->forward(attn_out, normed, Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    Mat a(s);
    for (std::size_t i = 0; i < s; ++i) qi->add(a[i], x[i], attn_out[i]);

    out.assign(s, std::vector<IntFp>(d));
    for (std::size_t i = 0; i < s; ++i) {
      std::vector<IntFp> n2;
      ln2->forward(n2, a[i], g2, b2n);
      if (save_intermediates) { // LN2 output -> FFN input backward (dLdW1)
        if (cache["normed2"].empty()) cache["normed2"].assign(s, std::vector<IntFp>());
        cache["normed2"][i] = n2;
      }
      std::vector<IntFp> hidden, geluout, ff;
      qi->dense_layer(hidden, W1, b1, n2); // d -> ffn
      if (save_intermediates) { // pre-GELU activations -> GELUBackward
        if (cache["hidden"].empty()) cache["hidden"].assign(s, std::vector<IntFp>());
        cache["hidden"][i] = hidden;
      }
      gelu->forward(geluout, hidden);
      if (save_intermediates) { // post-GELU activations -> FFN dLdW2
        if (cache["geluout"].empty()) cache["geluout"].assign(s, std::vector<IntFp>());
        cache["geluout"][i] = geluout;
      }
      qi->dense_layer(ff, W2, b2, geluout); // ffn -> d
      qi->add(out[i], a[i], ff);
    }
  }
};

// ---------------------------------------------------------------------------
// LM head (unembedding) + argmax certification (DeepProve's output side):
//   logits[t] = requant(H[t] . We^T) + be       (We: vocab x d, tied to embed)
//   certify  : prove the claimed next token = argmax(logits[t]) at each position.
// This is the certification trick: the proven sequence is correct iff every
// position's argmax matches the claimed token, so proving inference reduces to
// one forward pass + an argmax check per position (linear, not quadratic).
class FixedLMHead {
public:
  FixedInfer *qi;
  CompareLU *cmp; // lookup-table argmax (DeepProve-style, no MSB)
  std::size_t d, vocab, logit_bits;

  // logit_bits bounds the logit magnitude so each argmax comparison is a
  // RangeCheckLU of ceil((logit_bits+1)/table_bits) lookups (cf. DeepProve).
  FixedLMHead(FixedInfer *qi_, CompareLU *cmp_, std::size_t d_, std::size_t vocab_,
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
