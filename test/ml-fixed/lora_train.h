// LoRA fine-tuning proof for the ml-fixed track, shared by lora_train_step (one toy block) and
// multiblock_lora_train_step (GPT-2-small shape).  STEPS SGD steps run in ONE ZK session:
//   public:   the base model (every frozen weight incl. the LM head), the LoRA init
//             (A Kaiming-uniform, B = 0) and the hyperparameters -- the statement names the model
//   private:  the training tokens / input embeddings, the next-token labels (committed one-hot),
//             every activation, gradient and adapter value after the init
//   per step: W_eff = W + (alpha/r) B@A for Q and V of every block, proven from the committed
//             adapters -> proven forward chain (whole-sequence matmuls, forward_seq) -> softmax
//             cross-entropy over the last LOSS_TOKENS positions -> batched frozen-weight backward
//             (LoRABlockBackward) -> SGD on the adapters only, eta = LR / LOSS_TOKENS (mean loss).
//             The updated adapters feed the next step's W_eff.
// Model: random weights of the given shape by default.  GPT2_BIN=<gpt2_lora.bin> TOKENS=<ids>
// (tools/quantize_export/export_gpt2_lora.py) loads real GPT-2 and SEQ+1 real tokens instead: the
// tokens are committed one-hot, the input embeddings are proven as onehot @ wte + wpe, the labels
// are the same one-hot rows shifted by one, and the final LayerNorm ln_f runs before the tied head.
// Nothing is revealed to the verifier.  With CHECK=1 (default) the PROVER checks its own witness
// values (read locally from the commitments, never revealed) against plaintext replays: exact for
// W_eff, dL/dlogits, the head's dL/dH and the adapter updates; float tolerance for each block's
// backward, driven by that block's own forward intermediates and upstream gradient.  It also
// reports how many adapter entries each step changed and the cosine between the proven update and
// the float-reference update.
//
// Shape / run knobs (env): NB SEQ D HEADS FFN VOCAB RANK STEPS LOSS_TOKENS, P (default 12),
// MASTER_BITS (adapter extra precision, default 8), GRAD_BITS (extra precision of dL/dW_eff,
// default 8, must be < P), LR (default 1e-2), ALPHA (default 2*RANK), LSE (LogSumExp attention
// softmax, default 1; 0 = the division softmax), HEAD_EXP_BITS (extra output bits of the head's
// exp table, default 12), ATTN_EXP_BITS (extra output bits of the LSE attention softmax, default
// 8), GPT2_BIN, TOKENS, CHECK, ZK_THREADS, PEER_IP.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/div.h"
#include "volefier/ml-fixed/fixed_transformer.h"
#include "volefier/ml-fixed/fixed_backward.h"
using namespace emp;
using namespace volefier;
using namespace std;

struct LoraCfg { int NB, S, D, H, FFN, V, R, STEPS; };

// GPT2_BIN reader (layout in tools/quantize_export/export_gpt2_lora.py); values are quantized to
// round(v * 2^P) as they are read.
struct Gpt2Bin {
  int NB = 0, D = 0, H = 0, FFN = 0, V = 0, NPOS = 0;
  FILE *f = nullptr;
  bool open(const char *path) {
    char magic[8];
    int32_t hdr[6];
    f = fopen(path, "rb");
    if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "GPT2LORA", 8) || fread(hdr, 4, 6, f) != 6) return false;
    NB = hdr[0]; D = hdr[1]; H = hdr[2]; FFN = hdr[3]; V = hdr[4]; NPOS = hdr[5];
    return true;
  }
  void read(vector<int64_t> &dst, size_t n, double scale) {
    vector<float> buf(n);
    if (fread(buf.data(), 4, n, f) != n) { fprintf(stderr, "GPT2_BIN: short read\n"); exit(1); }
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) dst[i] = llround(buf[i] * scale);
  }
  void skip_blocks(int n) { fseek(f, (long)n * (4L * D * D + 2L * FFN * D + 9L * D + FFN) * 4, SEEK_CUR); }
};

static int env_i(const char *k, int def) { const char *v = getenv(k); return v ? atoi(v) : def; }
static double env_d(const char *k, double def) { const char *v = getenv(k); return v ? atof(v) : def; }

// round-half-up  N / 2^M, the rounding of FixedInfer::requant_cm
static int64_t rhu128(__int128 N, size_t M) {
  return (int64_t)((N + ((__int128)1 << (M - 1))) >> M);
}

static int lora_train_main(int argc, char **argv, LoraCfg c, const char *name) {
  int port, party;
  parse_party_and_port(argv, &party, &port);
  setvbuf(stdout, NULL, _IONBF, 0);
  const int threads = env_i("ZK_THREADS", 4);
  const int P = env_i("P", 12);
  const size_t E = env_i("MASTER_BITS", 8), G = env_i("GRAD_BITS", 8);
  const char *bin_path = getenv("GPT2_BIN");
  const bool real = bin_path != nullptr;
  Gpt2Bin gbin;
  if (real && !gbin.open(bin_path)) { fprintf(stderr, "cannot read GPT2_BIN %s\n", bin_path); return 1; }
  const int NB = env_i("NB", real ? gbin.NB : c.NB), S = env_i("SEQ", c.S);
  const int D = real ? gbin.D : env_i("D", c.D), H = real ? gbin.H : env_i("HEADS", c.H);
  const int FFN = real ? gbin.FFN : env_i("FFN", c.FFN), V = real ? gbin.V : env_i("VOCAB", c.V), R = env_i("RANK", c.R);
  const int STEPS = env_i("STEPS", c.STEPS), T = env_i("LOSS_TOKENS", S);
  const bool lse = env_i("LSE", 1) != 0;
  const int HX = env_i("HEAD_EXP_BITS", 12);
  const int AX = lse ? env_i("ATTN_EXP_BITS", 8) : 0; // extra output bits of the LSE attention softmax
  vector<int> ids; // real mode: SEQ+1 token ids; position t predicts ids[t+1]
  if (real) {
    const char *tp = getenv("TOKENS");
    FILE *tf = tp ? fopen(tp, "r") : nullptr;
    for (int v; tf && (int)ids.size() < S + 1 && fscanf(tf, "%d", &v) == 1;) ids.push_back(v);
    if (tf) fclose(tf);
    if ((int)ids.size() < S + 1 || NB > gbin.NB || S > gbin.NPOS) { fprintf(stderr, "need TOKENS with SEQ+1 ids, NB <= %d, SEQ <= %d\n", gbin.NB, gbin.NPOS); return 1; }
  }
  const int DH = D / H;
  const double LR = env_d("LR", 1e-2), ALPHA = env_d("ALPHA", 2.0 * R);
  const bool CHECK = env_i("CHECK", 1) != 0;
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE")) ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
  const int64_t _ev = getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL;
  const int _vt = getenv("VOLE_THREADS") ? atoi(getenv("VOLE_THREADS")) : -1;
  if (party == ALICE)
    printf("[cfg] %s: model=%s NB=%d SEQ=%d D=%d HEADS=%d FFN=%d VOCAB=%d RANK=%d STEPS=%d LOSS_TOKENS=%d P=%d MASTER_BITS=%zu GRAD_BITS=%zu LR=%g ALPHA=%g LSE=%d threads=%d CHECK=%d\n",
           name, real ? bin_path : "random", NB, S, D, H, FFN, V, R, STEPS, T, P, E, G, LR, ALPHA, (int)lse, threads, (int)CHECK);
  setup_zk_arith(ios, threads, party, _ev, ios[1], _vt);
  auto t0 = chrono::steady_clock::now();
  auto secs = [](chrono::steady_clock::time_point a) { return chrono::duration<double>(chrono::steady_clock::now() - a).count(); };
  double chk_s = 0;

  bool ok = true;
  {
    FixedInfer qi(party, P, ios[0], 32, threads);
    const double SCALE = (double)(1ULL << P);
    const uint64_t qinv = (uint64_t)llround((1.0 / std::sqrt((double)DH)) * SCALE);
    const double asc = (double)qinv / SCALE;
    // tables keep their real domain as P scales: exp [0,512) (as in gpt2_prove), gelu +-128.
    // gpt2_prove clamps into a +-64 GELU table with two comparisons per element; training keeps
    // the input unclamped (GELU' must see the same x), so the table is widened instead -- real
    // GPT-2 reaches |pre-GELU| ~63 on WikiText.  An input outside it only makes the proof fail.
    const size_t SB = P + 9;
    const uint64_t ET = 1ull << SB, GT = 1ull << (P + 8);
    const int64_t GOFF = 1LL << (P + 7);
    size_t sm_bits = P + 2; // softmax divisor sum(exp) < max(V,S) * 2^P
    for (int n = max(V, S); n > 1; n = (n + 1) / 2) ++sm_bits;

    RangeCheckLU rclu(party, ios[0]->io, 16);
    CompareLU cmp(&rclu);
    // the head divides over the whole vocabulary with exp at 2^-(P+HX): sum < V * 2^(P+HX)
    size_t hd_bits = P + HX + 2;
    for (int n = V; n > 1; n = (n + 1) / 2) ++hd_bits;
    Div divr_ln(party, ios[0]->io, 24), divr_sm(party, ios[0]->io, sm_bits), divr_head(party, ios[0]->io, hd_bits);
    FixedTable expt(ET, party, ios[0]->io); expt.Setup(build_exp_table(ET, P, !lse, P + AX)); // LSE needs the unclamped table
    FixedTable gtab(GT, party, ios[0]->io); gtab.Setup(build_gelu_table(GT, P, GOFF));
    FixedTable expt_head(ET, party, ios[0]->io); expt_head.Setup(build_exp_table(ET, P, false, P + HX));
    FixedSoftmax sm(&qi, &cmp, &expt_head, &divr_head, P, SB), sm_attn(&qi, &cmp, &expt, &divr_sm, P, SB);
    sm_attn.lse = lse;
    sm_attn.out_bits = P + AX; // LSE normalization slack ~ (L/2 + 2^AX) / 2^(P+AX) of the row sum
    sm_attn.max_slack = getenv("LSE_MAX_SLACK") ? atof(getenv("LSE_MAX_SLACK")) : 0.005;
    FixedActivation gelu_act(&gtab, GOFF);
    LayerNormBackward lnb(&qi, &divr_ln);
    SoftmaxBackward sbk(&qi);
    LMHeadBackward lmb(&qi);
    const double lscale = ALPHA / R, eta = LR / T;
    LoRAAdapter lora(&qi, D, D, R, E, G, lscale, eta); // Q and V adapters share the shape
    vector<FixedLayerNorm *> ln1(NB), ln2(NB);
    vector<FixedAttention *> attn(NB);
    vector<FixedTransformerBlock *> tblk(NB);
    GELUBackward gelu_bwd(&qi, GT, GOFF); // one GELU' table shared by every block
    vector<LoRABlockBackward *> bwd(NB);
    for (int b = 0; b < NB; ++b) {
      ln1[b] = new FixedLayerNorm(&qi, &rclu, &divr_ln, P, D);
      ln2[b] = new FixedLayerNorm(&qi, &rclu, &divr_ln, P, D);
      attn[b] = new FixedAttention(&qi, &sm_attn, P, D, H, qinv);
      tblk[b] = new FixedTransformerBlock(&qi, ln1[b], ln2[b], attn[b], &gelu_act, D, FFN);
      bwd[b] = new LoRABlockBackward(&qi, &lnb, &gelu_bwd, D, H, FFN, qinv, G);
    }
    FixedLayerNorm lnf(&qi, &rclu, &divr_ln, P, D); // final LayerNorm (real GPT-2 only)
    lnf.save_intermediates = true;

    // ---- plaintext model + data.  Both parties draw the PUBLIC part from the same seed; the
    //      private part (x, labels) is drawn last and only ALICE's copy is ever committed. ----
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto nx = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto ur = [&](double lo, double hi) { return lo + (hi - lo) * ((double)(nx() >> 11) / (double)(1ULL << 53)); };
    auto q1 = [&](double v) { return (int64_t)llround(v * SCALE); };
    auto fill = [&](vector<int64_t> &v, size_t n, double lo, double hi) { v.resize(n); for (auto &e : v) e = q1(ur(lo, hi)); };
    struct Blk { vector<int64_t> Wq, Wk, Wv, Wo, W1, W2, bq, bk, bv, bo, b1, b2, g1, b1n, g2, b2n; };
    vector<Blk> pb(NB);
    vector<int64_t> We, be, wpe, lnfg, lnfb;
    if (real) {
      gbin.read(We, (size_t)V * D, SCALE);
      gbin.read(wpe, (size_t)gbin.NPOS * D, SCALE);
      for (auto &B : pb) {
        gbin.read(B.g1, D, SCALE); gbin.read(B.b1n, D, SCALE);
        gbin.read(B.Wq, (size_t)D * D, SCALE); gbin.read(B.bq, D, SCALE);
        gbin.read(B.Wk, (size_t)D * D, SCALE); gbin.read(B.bk, D, SCALE);
        gbin.read(B.Wv, (size_t)D * D, SCALE); gbin.read(B.bv, D, SCALE);
        gbin.read(B.Wo, (size_t)D * D, SCALE); gbin.read(B.bo, D, SCALE);
        gbin.read(B.g2, D, SCALE); gbin.read(B.b2n, D, SCALE);
        gbin.read(B.W1, (size_t)FFN * D, SCALE); gbin.read(B.b1, FFN, SCALE);
        gbin.read(B.W2, (size_t)D * FFN, SCALE); gbin.read(B.b2, D, SCALE);
      }
      gbin.skip_blocks(gbin.NB - NB);
      gbin.read(lnfg, D, SCALE); gbin.read(lnfb, D, SCALE);
      fclose(gbin.f);
      be.assign(V, 0); // GPT-2's head is the tied wte, no bias
    } else for (auto &B : pb) {
      fill(B.Wq, (size_t)D * D, -0.02, 0.02); fill(B.Wk, (size_t)D * D, -0.02, 0.02);
      fill(B.Wv, (size_t)D * D, -0.02, 0.02); fill(B.Wo, (size_t)D * D, -0.02, 0.02);
      fill(B.W1, (size_t)FFN * D, -0.02, 0.02); fill(B.W2, (size_t)D * FFN, -0.02, 0.02);
      fill(B.bq, D, -0.3, 0.3); fill(B.bk, D, -0.3, 0.3); fill(B.bv, D, -0.3, 0.3); fill(B.bo, D, -0.3, 0.3);
      fill(B.b1, FFN, -0.2, 0.2); fill(B.b2, D, -0.2, 0.2);
      B.g1.assign(D, q1(1.0)); B.g2.assign(D, q1(1.0)); B.b1n.assign(D, 0); B.b2n.assign(D, 0);
    }
    if (!real) { fill(We, (size_t)V * D, -0.02, 0.02); fill(be, V, -0.3, 0.3); }
    const double MS = std::ldexp(1.0, (int)(P + E)); // adapter master scale
    const double abound = 1.0 / std::sqrt((double)D); // PEFT's kaiming_uniform(a=sqrt 5) bound
    vector<vector<int64_t>> A0q(NB), A0v(NB);
    for (int b = 0; b < NB; ++b) {
      A0q[b].resize((size_t)R * D); A0v[b].resize((size_t)R * D);
      for (auto &e : A0q[b]) e = llround(ur(-abound, abound) * MS);
      for (auto &e : A0v[b]) e = llround(ur(-abound, abound) * MS);
    }
    vector<int64_t> xq;
    vector<int> label(T);
    if (real) for (int t = 0; t < T; ++t) label[t] = ids[S - T + 1 + t];
    else {
      fill(xq, (size_t)S * D, -1, 1);
      for (auto &l : label) l = (int)(nx() % (uint64_t)V);
    }

    // ---- public constants + private commitments ----
    auto enc = [](int64_t v) { return v >= 0 ? (uint64_t)v : (uint64_t)(v + (int64_t)PR); };
    auto pubv = [&](const vector<int64_t> &src) { vector<IntFp> o(src.size()); for (size_t i = 0; i < src.size(); ++i) o[i] = IntFp(enc(src[i]), PUBLIC); return o; };
    auto pubm = [&](const vector<int64_t> &src, int rows, int cols) { Mat m(rows, vector<IntFp>(cols)); for (int r = 0; r < rows; ++r) for (int cc = 0; cc < cols; ++cc) m[r][cc] = IntFp(enc(src[(size_t)r * cols + cc]), PUBLIC); return m; };
    vector<Mat> cWq(NB), cWk(NB), cWv(NB), cWo(NB), cW1(NB), cW2(NB);
    vector<vector<IntFp>> cbq(NB), cbk(NB), cbv(NB), cbo(NB), cb1(NB), cb2(NB), cg1(NB), cb1n(NB), cg2(NB), cb2n(NB);
    for (int b = 0; b < NB; ++b) {
      const Blk &B = pb[b];
      cWq[b] = pubm(B.Wq, D, D); cWk[b] = pubm(B.Wk, D, D); cWv[b] = pubm(B.Wv, D, D); cWo[b] = pubm(B.Wo, D, D);
      cW1[b] = pubm(B.W1, FFN, D); cW2[b] = pubm(B.W2, D, FFN);
      cbq[b] = pubv(B.bq); cbk[b] = pubv(B.bk); cbv[b] = pubv(B.bv); cbo[b] = pubv(B.bo);
      cb1[b] = pubv(B.b1); cb2[b] = pubv(B.b2); cg1[b] = pubv(B.g1); cb1n[b] = pubv(B.b1n); cg2[b] = pubv(B.g2); cb2n[b] = pubv(B.b2n);
    }
    Mat cWe = pubm(We, V, D);
    vector<IntFp> cbe = pubv(be);
    const IntFp zero(0ULL, PUBLIC);
    vector<vector<IntFp>> Aq(NB), Av(NB), Bq(NB, vector<IntFp>((size_t)D * R, zero)), Bv(NB, vector<IntFp>((size_t)D * R, zero));
    for (int b = 0; b < NB; ++b) { Aq[b] = pubv(A0q[b]); Av[b] = pubv(A0v[b]); }
    Mat cx(S), onehot(T, vector<IntFp>(V));
    auto commit_onehot = [&](vector<IntFp> &row, int id) {
      vector<uint64_t> raw(V, 0);
      if (party == ALICE) raw[id] = 1;
      batch_feed(row.data(), raw.data(), V);
    };
    vector<IntFp> cnfg, cnfb;
    if (real) {
      // tokens 0..S committed one-hot; rows S-T+1..S double as the labels, whose one-hot form
      // cross_entropy_private proves each step, so rows 0..S-T are proven here
      Mat oh(S + 1, vector<IntFp>(V));
      for (int i = 0; i <= S; ++i) commit_onehot(oh[i], ids[i]);
      for (int i = 0; i <= S - T; ++i) {
        vector<IntFp> om1(V);
        IntFp sum(0ULL, PUBLIC);
        for (int v = 0; v < V; ++v) { om1[v] = oh[i][v] - 1ULL; sum = sum + oh[i][v]; }
        global_h2ip()->add_zero(oh[i].data(), om1.data(), V);
        (sum - 1ULL).reveal_zero();
      }
      // x = onehot @ wte + wpe: one Freivalds matmul against the public table, no requant
      vector<IntFp> raw, ohf;
      ohf.reserve((size_t)S * V);
      for (int i = 0; i < S; ++i) ohf.insert(ohf.end(), oh[i].begin(), oh[i].end());
      mm_seq(&qi, raw, ohf, pubv(We), S, D, V);
      for (int i = 0; i < S; ++i) {
        cx[i].resize(D);
        for (int k = 0; k < D; ++k) cx[i][k] = raw[(size_t)i * D + k] + IntFp(enc(wpe[(size_t)i * D + k]), PUBLIC);
      }
      for (int t = 0; t < T; ++t) onehot[t] = std::move(oh[S - T + 1 + t]);
      cnfg = pubv(lnfg); cnfb = pubv(lnfb);
    } else {
      for (int i = 0; i < S; ++i) qi.commit_vec(cx[i], vector<int64_t>(xq.begin() + (size_t)i * D, xq.begin() + (size_t)(i + 1) * D));
      for (int t = 0; t < T; ++t) commit_onehot(onehot[t], label[t]);
    }
    printf("[%d] committed (%.1fs)\n", party, secs(t0));

    // ---- prover-local checking helpers (ALICE reads her own witnesses; nothing is revealed) ----
    auto dq = [](const IntFp &z) { return FixedInfer::decode_signed((uint64_t)VAL(z.value)); };
    auto dqv = [&](const vector<IntFp> &v) { vector<int64_t> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = dq(v[i]); return o; };
    auto dqf = [&](const Mat &M) { vector<double> o; o.reserve(M.size() * (M.empty() ? 0 : M[0].size())); for (auto &r : M) for (auto &z : r) o.push_back(dq(z) / SCALE); return o; };
    const vector<uint64_t> ggt = build_gelu_grad_table(GT, P, GOFF);
    auto gg = [&](int64_t hq) { int64_t ix = std::min<int64_t>(std::max<int64_t>(hq + GOFF, 0), (int64_t)GT - 1); return FixedInfer::decode_signed(ggt[ix]) / SCALE; };
    auto ln_bwd = [&](const double *dy, const double *cc, double dinv, const double *g, double *dx) {
      vector<double> xh(D), g2v(D); double e1 = 0, e2 = 0;
      for (int k = 0; k < D; ++k) { xh[k] = cc[k] * dinv; g2v[k] = dy[k] * g[k]; e1 += g2v[k] * xh[k]; e2 += g2v[k]; }
      e1 /= D; e2 /= D;
      for (int k = 0; k < D; ++k) dx[k] = dinv * (g2v[k] - e2 - xh[k] * e1);
    };
    // float reference of one block's backward from its own forward caches + upstream dO
    auto block_ref = [&](int b, const vector<double> &dO, const vector<double> &WqE, const vector<double> &WvE,
                         vector<double> &gWq, vector<double> &gWv, vector<double> &gX) {
      const Blk &B = pb[b];
      vector<double> c1 = dqf(ln1[b]->cache_mat["c"]), c2 = dqf(ln2[b]->cache_mat["c"]);
      vector<double> d1(S), d2(S);
      for (int i = 0; i < S; ++i) { d1[i] = dq(ln1[b]->cache_scalar_vec["dinv"][i]) / SCALE; d2[i] = dq(ln2[b]->cache_scalar_vec["dinv"][i]) / SCALE; }
      vector<double> nrm = dqf(tblk[b]->cache["normed"]), Qr = dqf(attn[b]->cache["Q"]), Kr = dqf(attn[b]->cache["K"]), Vr = dqf(attn[b]->cache["V"]);
      Mat &hid = tblk[b]->cache["hidden"];
      Mat &pr = attn[b]->cache["probs"];
      vector<double> g1v(D), g2v(D);
      for (int k = 0; k < D; ++k) { g1v[k] = B.g1[k] / SCALE; g2v[k] = B.g2[k] / SCALE; }
      // FFN + LN2
      vector<double> gA((size_t)S * D);
      parallel_for(S, threads, [&](size_t i) {
        vector<double> dG(FFN, 0.0), gN2(D, 0.0), gDa(D);
        for (int k = 0; k < D; ++k) { const double o = dO[i * D + k]; const int64_t *w = &B.W2[(size_t)k * FFN]; for (int p = 0; p < FFN; ++p) dG[p] += w[p] / SCALE * o; }
        for (int p = 0; p < FFN; ++p) { const double gh = dG[p] * gg(dq(hid[i][p])); const int64_t *w = &B.W1[(size_t)p * D]; for (int cc = 0; cc < D; ++cc) gN2[cc] += w[cc] / SCALE * gh; }
        ln_bwd(gN2.data(), &c2[i * D], d2[i], g2v.data(), gDa.data());
        for (int k = 0; k < D; ++k) gA[i * D + k] = dO[i * D + k] + gDa[k];
      });
      // output projection
      vector<double> gctx((size_t)S * D, 0.0);
      parallel_for(S, threads, [&](size_t i) {
        for (int a = 0; a < D; ++a) { const double g = gA[i * D + a]; const int64_t *w = &B.Wo[(size_t)a * D]; for (int cc = 0; cc < D; ++cc) gctx[i * D + cc] += w[cc] / SCALE * g; }
      });
      // attention core, one head per task (heads touch disjoint columns)
      vector<double> gQ((size_t)S * D, 0.0), gK((size_t)S * D, 0.0), gV((size_t)S * D, 0.0);
      parallel_for(H, threads, [&](size_t hh) {
        const int off = (int)hh * DH;
        vector<double> pj(S), gp(S), gd(S);
        for (int i = 0; i < S; ++i) {
          const int L = i + 1;
          for (int j = 0; j < L; ++j) pj[j] = dq(pr[(size_t)i * H + hh][j]) / (SCALE * (1 << AX));
          double sd = 0;
          for (int j = 0; j < L; ++j) { double g = 0; for (int k = 0; k < DH; ++k) g += gctx[(size_t)i * D + off + k] * Vr[(size_t)j * D + off + k]; gp[j] = g; sd += pj[j] * g; }
          for (int j = 0; j < L; ++j) for (int k = 0; k < DH; ++k) gV[(size_t)j * D + off + k] += pj[j] * gctx[(size_t)i * D + off + k];
          for (int j = 0; j < L; ++j) gd[j] = pj[j] * (gp[j] - sd) * asc;
          for (int j = 0; j < L; ++j) for (int k = 0; k < DH; ++k) { gQ[(size_t)i * D + off + k] += gd[j] * Kr[(size_t)j * D + off + k]; gK[(size_t)j * D + off + k] += gd[j] * Qr[(size_t)i * D + off + k]; }
        }
      });
      // projections + LN1
      gX.assign((size_t)S * D, 0.0);
      parallel_for(S, threads, [&](size_t i) {
        vector<double> gN(D, 0.0), gDx(D);
        for (int a = 0; a < D; ++a) {
          const double q = gQ[i * D + a], kk = gK[i * D + a], v = gV[i * D + a];
          for (int cc = 0; cc < D; ++cc) gN[cc] += WqE[(size_t)a * D + cc] * q + B.Wk[(size_t)a * D + cc] / SCALE * kk + WvE[(size_t)a * D + cc] * v;
        }
        ln_bwd(gN.data(), &c1[i * D], d1[i], g1v.data(), gDx.data());
        for (int k = 0; k < D; ++k) gX[i * D + k] = gA[i * D + k] + gDx[k];
      });
      gWq.assign((size_t)D * D, 0.0); gWv.assign((size_t)D * D, 0.0);
      parallel_for(D, threads, [&](size_t a) {
        for (int i = 0; i < S; ++i) { const double q = gQ[(size_t)i * D + a], v = gV[(size_t)i * D + a]; for (int cc = 0; cc < D; ++cc) { gWq[a * D + cc] += q * nrm[(size_t)i * D + cc]; gWv[a * D + cc] += v * nrm[(size_t)i * D + cc]; } }
      });
    };
    // float comparison: ok if |got - ref| <= 0.06 + 1% of max|ref| everywhere, or if the whole
    // tensor points the same way (cosine >= 0.999).  Fixed-point rounding at 2^-P leaves a few-%
    // elementwise error on gradients summed over many tokens through sharp real attention; a
    // wrong gradient shows up as a low cosine.
    auto cosine = [](const vector<double> &a, const vector<double> &b) {
      double ab = 0, aa = 0, bb = 0;
      for (size_t i = 0; i < a.size(); ++i) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
      return (aa > 0 && bb > 0) ? ab / sqrt(aa * bb) : NAN;
    };
    auto cmpf = [&](const char *what, int b, const vector<double> &got, const vector<double> &ref) {
      double err = 0, mx = 0;
      for (size_t i = 0; i < ref.size(); ++i) { err = max(err, fabs(got[i] - ref[i])); mx = max(mx, fabs(ref[i])); }
      const double tol = 0.06 + 0.01 * mx, cs = cosine(got, ref);
      const bool good = err <= tol || cs >= 0.999;
      printf("[1]   blk%-2d %-9s max|err| %.5f  max|ref| %.4f  cos %.5f  %s\n", b, what, err, mx, cs, good ? "ok" : "FAIL");
      if (!good) ok = false;
    };

    for (int step = 0; step < STEPS; ++step) {
      auto ts = chrono::steady_clock::now();
      // ================= W_eff from the committed adapters =================
      vector<Mat> WQe(NB), WVe(NB);
      for (int b = 0; b < NB; ++b) { lora.effective(WQe[b], cWq[b], Bq[b], Aq[b]); lora.effective(WVe[b], cWv[b], Bv[b], Av[b]); }
      if (CHECK && party == ALICE) {
        auto tc = chrono::steady_clock::now();
        size_t bad = 0;
        for (int b = 0; b < NB; ++b)
          for (int which = 0; which < 2; ++which) {
            const vector<int64_t> Bp = dqv(which ? Bv[b] : Bq[b]), Ap = dqv(which ? Av[b] : Aq[b]);
            const vector<int64_t> &W = which ? pb[b].Wv : pb[b].Wq;
            const Mat &We_ = which ? WVe[b] : WQe[b];
            for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) {
              __int128 acc = 0;
              for (int k = 0; k < R; ++k) acc += (__int128)Bp[(size_t)i * R + k] * Ap[(size_t)k * D + j];
              if (dq(We_[i][j]) != W[(size_t)i * D + j] + rhu128(acc * (__int128)lora.s_c, lora.s_M)) ++bad;
            }
          }
        printf("[1] step %d  W_eff = W + (alpha/r)BA exact replay: %zu mismatches\n", step, bad);
        if (bad) ok = false;
        chk_s += secs(tc);
      }

      // ================= PROVEN FORWARD =================
      Mat Hc = cx;
      for (int b = 0; b < NB; ++b) {
        ln1[b]->save_intermediates = ln2[b]->save_intermediates = attn[b]->save_intermediates = tblk[b]->save_intermediates = true;
        ln1[b]->clear_cache(); ln2[b]->clear_cache(); attn[b]->clear_cache(); tblk[b]->clear_cache();
        Mat Ho;
        tblk[b]->forward_seq(Ho, Hc, cg1[b], cb1n[b], cg2[b], cb2n[b], WQe[b], cbq[b], cWk[b], cbk[b], WVe[b], cbv[b], cWo[b], cbo[b], cW1[b], cb1[b], cW2[b], cb2[b]);
        Hc = Ho;
      }
      const double t_fwd = secs(ts);
      if (CHECK && party == ALICE) { // the GELU table only covers |x| < GOFF (real 128)
        int64_t mh = 0;
        for (int b = 0; b < NB; ++b) for (auto &row : tblk[b]->cache["hidden"]) for (auto &z : row) mh = max(mh, std::abs(dq(z)));
        printf("[1] step %d  max |pre-GELU| %.2f (table covers %.0f)  %s\n", step, mh / SCALE, GOFF / SCALE, mh < GOFF ? "ok" : "OUT OF RANGE");
        if (mh >= GOFF) ok = false;
      }
      // LM head + cross-entropy over the last T positions (private labels)
      auto th = chrono::steady_clock::now();
      Mat Hl(Hc.end() - T, Hc.end()), Lg, pr(T), dl(T), dHl;
      if (real) { lnf.clear_cache(); for (auto &row : Hl) { vector<IntFp> o; lnf.forward(o, row, cnfg, cnfb); row = std::move(o); } }
      qi.dense_layer_mm(Lg, cWe, cbe, Hl);
      for (int t = 0; t < T; ++t) { pr[t] = sm.forward(Lg[t]); sbk.cross_entropy_private(dl[t], pr[t], onehot[t]); }
      lmb.input_grad(dHl, dl, cWe);
      Mat dLdH(S, vector<IntFp>(D, zero));
      for (int t = 0; t < T; ++t) {
        if (real) { vector<IntFp> dg, db; lnb.forward(dLdH[S - T + t], dg, db, dHl[t], lnf.cache_mat["c"][t], cnfg, lnf.cache_scalar_vec["dinv"][t], D); }
        else dLdH[S - T + t] = dHl[t];
      }
      const double t_head = secs(th);
      if (CHECK && party == ALICE) {
        auto tc = chrono::steady_clock::now();
        size_t bad_dl = 0, bad_dh = 0; double perr = 0, loss = 0;
        int wt = 0, wv = 0; double wgot = 0, wref = 0, wspread = 0;
        vector<vector<int64_t>> dlq(T);
        for (int t = 0; t < T; ++t) {
          const vector<int64_t> lq = dqv(Lg[t]), pq = dqv(pr[t]);
          dlq[t] = dqv(dl[t]);
          double mx = -1e300, sum = 0;
          for (int v = 0; v < V; ++v) mx = max(mx, lq[v] / SCALE);
          for (int v = 0; v < V; ++v) sum += exp(lq[v] / SCALE - mx);
          double mn = 1e300;
          for (int v = 0; v < V; ++v) mn = min(mn, lq[v] / SCALE);
          for (int v = 0; v < V; ++v) {
            const double ref = exp(lq[v] / SCALE - mx) / sum;
            if (fabs(pq[v] / SCALE - ref) > perr) { perr = fabs(pq[v] / SCALE - ref); wt = t; wv = v; wgot = pq[v] / SCALE; wref = ref; wspread = mx - mn; }
            if (dlq[t][v] != pq[v] - (v == label[t] ? (int64_t)SCALE : 0)) ++bad_dl;
          }
          loss += -(lq[label[t]] / SCALE - mx - log(sum));
        }
        vector<size_t> bad_t(T, 0);
        parallel_for(T, threads, [&](size_t t) {
          for (int k = 0; k < D; ++k) {
            __int128 acc = 0;
            for (int v = 0; v < V; ++v) acc += (__int128)dlq[t][v] * We[(size_t)v * D + k];
            if (dq(dHl[t][k]) != rhu128(acc, P)) ++bad_t[t];
          }
        });
        for (size_t x : bad_t) bad_dh += x;
        const bool hok = bad_dl == 0 && bad_dh == 0 && perr <= 0.01;
        printf("[1] step %d  loss head: mean CE %.4f | softmax max|err| %.5f | dL/dlogits exact: %zu mismatches | dL/dH exact: %zu mismatches  %s\n",
               step, loss / T, perr, bad_dl, bad_dh, hok ? "ok" : "FAIL");
        if (perr > 0.01) printf("[1]   softmax worst: pos %d token %d  proven %.5f  float %.5f  (row max-min logit %.1f)\n", wt, wv, wgot, wref, wspread);
        if (!hok) ok = false;
        if (real) { // ln_f backward against the float formula on its own cached c / dinv
          vector<double> got, ref, gf(D), dx(D);
          for (int k = 0; k < D; ++k) gf[k] = lnfg[k] / SCALE;
          const vector<double> cf = dqf(lnf.cache_mat["c"]), dyf = dqf(dHl);
          for (int t = 0; t < T; ++t) {
            ln_bwd(&dyf[(size_t)t * D], &cf[(size_t)t * D], dq(lnf.cache_scalar_vec["dinv"][t]) / SCALE, gf.data(), dx.data());
            for (int k = 0; k < D; ++k) { got.push_back(dq(dLdH[S - T + t][k]) / SCALE); ref.push_back(dx[k]); }
          }
          cmpf("ln_f dL/dH", NB, got, ref);
        }
        chk_s += secs(tc);
      }

      // ================= BACKWARD CHAIN + ADAPTER SGD =================
      auto tb = chrono::steady_clock::now();
      size_t nzB = 0, nzA = 0, nB = 0, nA = 0;
      for (int b = NB - 1; b >= 0; --b) {
        Mat dLdx;
        vector<IntFp> dWq, dWv;
        bwd[b]->forward(dLdx, dWq, dWv, dLdH, *ln1[b], *ln2[b], *attn[b], *tblk[b], WQe[b], cWk[b], WVe[b], cWo[b], cW1[b], cW2[b], cg1[b], cg2[b]);
        vector<IntFp> Bq2, Aq2, Bv2, Av2;
        lora.sgd_step(Bq2, Aq2, Bq[b], Aq[b], dWq);
        lora.sgd_step(Bv2, Av2, Bv[b], Av[b], dWv);
        if (CHECK && party == ALICE) {
          auto tc = chrono::steady_clock::now();
          vector<double> gWq, gWv, gX;
          block_ref(b, dqf(dLdH), dqf(WQe[b]), dqf(WVe[b]), gWq, gWv, gX);
          vector<double> fq(dWq.size()), fv(dWv.size());
          const double GS = SCALE * std::ldexp(1.0, (int)G); // dL/dW_eff scale 2^(P+G)
          for (size_t i = 0; i < fq.size(); ++i) { fq[i] = dq(dWq[i]) / GS; fv[i] = dq(dWv[i]) / GS; }
          cmpf("dL/dWq", b, fq, gWq); cmpf("dL/dWv", b, fv, gWv); cmpf("dL/dx", b, dqf(dLdx), gX);
          // adapter step: exact replay + effect
          for (int which = 0; which < 2; ++which) {
            const vector<int64_t> Bp = dqv(which ? Bv[b] : Bq[b]), Ap = dqv(which ? Av[b] : Aq[b]);
            const vector<int64_t> dW = dqv(which ? dWv : dWq), Bn = dqv(which ? Bv2 : Bq2), An = dqv(which ? Av2 : Aq2);
            const vector<double> &gW = which ? gWv : gWq;
            size_t bad = 0;
            vector<double> uB((size_t)D * R), iB((size_t)D * R, 0.0), uA((size_t)R * D), iA((size_t)R * D, 0.0);
            for (int i = 0; i < D; ++i) for (int k = 0; k < R; ++k) {
              __int128 acc = 0; double fl = 0;
              for (int j = 0; j < D; ++j) { acc += (__int128)dW[(size_t)i * D + j] * Ap[(size_t)k * D + j]; fl += gW[(size_t)i * D + j] * Ap[(size_t)k * D + j] / MS; }
              const int64_t u = rhu128(acc * (__int128)lora.g_c, lora.g_M), t = (size_t)i * R + k;
              if (Bn[t] != Bp[t] - u) ++bad;
              uB[t] = u / MS; iB[t] = eta * lscale * fl; nzB += u != 0;
            }
            for (int k = 0; k < R; ++k) for (int j = 0; j < D; ++j) {
              __int128 acc = 0; double fl = 0;
              for (int i = 0; i < D; ++i) { acc += (__int128)Bp[(size_t)i * R + k] * dW[(size_t)i * D + j]; fl += Bp[(size_t)i * R + k] / MS * gW[(size_t)i * D + j]; }
              const int64_t u = rhu128(acc * (__int128)lora.g_c, lora.g_M), t = (size_t)k * D + j;
              if (An[t] != Ap[t] - u) ++bad;
              uA[t] = u / MS; iA[t] = eta * lscale * fl; nzA += u != 0;
            }
            nB += uB.size(); nA += uA.size();
            printf("[1]   blk%-2d adapter %c: SGD exact replay %zu mismatches | cos(update, float update) B %.4f  A %.4f\n",
                   b, which ? 'V' : 'Q', bad, cosine(uB, iB), cosine(uA, iA));
            if (bad) ok = false;
          }
          chk_s += secs(tc);
        }
        Bq[b] = Bq2; Aq[b] = Aq2; Bv[b] = Bv2; Av[b] = Av2;
        dLdH = dLdx;
      }
      const double t_bwd = secs(tb);
      if (CHECK && party == ALICE)
        printf("[1] step %d  adapter entries changed: B %zu/%zu (%.1f%%)  A %zu/%zu (%.1f%%)\n", step, nzB, nB, 100.0 * nzB / nB, nzA, nA, 100.0 * nzA / nA);
      printf("[%d] step %d done: W_eff+forward %.1fs, head %.1fs, backward+SGD %.1fs (wall incl. prover checks)\n", party, step, t_fwd, t_head, t_bwd);
    }

    gelu_bwd.Teardown();
    gtab.Teardown(); expt.Teardown(); expt_head.Teardown(); rclu.Teardown(); divr_sm.destroy(); divr_head.destroy(); divr_ln.destroy(); qi.destroy();
  }
  finalize_zk_arith();
  const double dt = secs(t0);

  if (party == ALICE) {
    printf("[timing] ZK session %.1fs (of which prover-side checks %.1fs)\n", dt, chk_s);
    printf("%s %s (ml-fixed)\n", ok ? "pass" : "FAIL", name);
  } else
    printf("verifier: ml-fixed %s proof complete (%.1fs)\n", name, dt);
  cout.flush(); fflush(stdout);
  _exit(0);
}
