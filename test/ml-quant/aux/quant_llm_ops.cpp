// Unit test for the Llama/Gemma LLM operators added to ml-quant:
//   RMSNorm, SiLU (the SwiGLU activation), RoPE, and GQA repeat_kv.
// Each gadget runs on COMMITTED inputs and is checked BIT-EXACT against a plaintext
// reference computing the identical integer arithmetic.
#include <iostream>
#include <vector>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_infer.h"
#include "volefier/ml-quant/quant_nonlinear.h"
using namespace emp; using namespace volefier; using namespace std;
int port, party; const int threads = 4, P = 12;

static int64_t rdiv(int64_t N, int64_t D) { return N >= 0 ? (N + D / 2) / D : -(((-N) + D / 2) / D); }      // round_div
static int64_t rq1(int64_t N, int M) { int64_t D = (int64_t)1 << M, h = (int64_t)1 << (M - 1); return N >= 0 ? (N + h) / D : -(((-N) + h) / D); }  // requant_cm (c=1)

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port); setvbuf(stdout, NULL, _IONBF, 0);
  BoolIO *ios[threads]; for (int i = 0; i < threads; ++i) ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  block seed = makeBlock(0, 42); PRG prg(&seed);
  const double SC = (double)(1 << P);
  if (party == ALICE) cout << "\n----- ml-quant LLM ops: RMSNorm / SiLU / Tanh / RoPE / GQA -----\n";
  setup_zk_arith(ios, threads, party);
  int fails = 0;
  {
    QuantInfer qi(party, 16, ios[0], 24, 1, 4);
    RangeCheckLU rclu(party, ios[0]->io, 16); CompareLU cmp(&rclu); Div divr(party, ios[0]->io, 16);
    const uint64_t ST = 1u << 16; const int64_t SOFF = 1 << 15;
    QuantTable silut(ST, party, ios[0]->io); silut.Setup(build_silu_table(ST, P, SOFF));
    auto fill = [&](vector<int64_t> &v, int64_t lo, int64_t hi) { if (party == ALICE) for (auto &x : v) { uint64_t t; prg.random_data(&t, 8); x = lo + (int64_t)(t % (uint64_t)(hi - lo + 1)); } };
    // Collect (name, committed output, plaintext ref); compute ALL gadgets first, then reveal once
    // at the end (matches the working pattern -- no reveals interleaved with later computation).
    vector<string> names; vector<vector<IntFp>> outs; vector<vector<int64_t>> refs;
    auto chk = [&](const char *nm, vector<IntFp> out, vector<int64_t> ref) { names.push_back(nm); outs.push_back(std::move(out)); refs.push_back(std::move(ref)); };

    // ---------- 1) RMSNorm:  (x / sqrt(mean(x^2))) * gamma ----------
    { const int n = 8; vector<int64_t> xr(n, 0), gr(n, 0);
      fill(xr, -2000, 2000); fill(gr, (1 << P) - 500, (1 << P) + 500);   // gamma ~ 1.0 (+-0.12)
      vector<IntFp> cx, cg; qi.commit_vec(cx, xr); qi.commit_vec(cg, gr);
      QuantRMSNorm rms(&qi, &rclu, &divr, P, n); vector<IntFp> out; rms.forward(out, cx, cg);
      vector<int64_t> ref(n, 0);
      if (party == ALICE) { int64_t ssq = 0; for (auto v : xr) ssq += v * v; int64_t ms = rdiv(ssq, n);
        int64_t dinv = llround((double)((uint64_t)1 << (2 * P)) / sqrt((double)ms));
        for (int i = 0; i < n; ++i) ref[i] = rq1(rq1(xr[i] * dinv, P) * gr[i], P); }
      chk("RMSNorm:", out, ref); }

    // ---------- 2) SiLU (SwiGLU activation) via LUT ----------
    { const int n = 8; vector<int64_t> xr(n, 0); fill(xr, -3000, 3000);
      vector<IntFp> cx; qi.commit_vec(cx, xr);
      QuantActivation silu(&silut, SOFF); vector<IntFp> out; silu.forward(out, cx);
      vector<int64_t> ref(n, 0);
      if (party == ALICE) for (int i = 0; i < n; ++i) { double x = (double)xr[i] / SC; ref[i] = llround(x / (1.0 + exp(-x)) * SC); }
      chk("SiLU:   ", out, ref); }

    // ---------- 2b) Tanh (BERT pooler) via LUT -- UNIFORM 2^-P input (committed key already at 2^-P) ----------
    const uint64_t TT = 1u << 16; const int64_t TOFF = 1 << 15;               // domain real +-8 at P=12; tanh saturates within +-4
    QuantTable tanht(TT, party, ios[0]->io); tanht.Setup(build_tanh_table(TT, P, TOFF));
    { const int n = 8; vector<int64_t> xr(n, 0); fill(xr, -20000, 20000);     // real +-4.9: exercises both linear region AND saturation
      vector<IntFp> cx; qi.commit_vec(cx, xr);
      QuantActivation th(&tanht, TOFF); vector<IntFp> out; th.forward(out, cx);
      vector<int64_t> ref(n, 0);
      if (party == ALICE) for (int i = 0; i < n; ++i) ref[i] = llround(tanh((double)xr[i] / SC) * SC);   // tanh(x@2^-P) -> 2^-P
      chk("Tanh2^-P", out, ref); }

    // ---------- 2c) Tanh -- PER-TENSOR input scale s_x != 2^-P (the c*2^e scale BAKED into the LUT) ----------
    const double SX = 1.0 / (1 << 7);                                         // a per-tensor input scale s_x = 2^-7 (NOT the activation 2^-12)
    QuantTable tanhs(TT, party, ios[0]->io); tanhs.Setup(build_tanh_table_s(TT, P, TOFF, SX));
    { const int n = 8; vector<int64_t> xr(n, 0); fill(xr, -1500, 1500);       // committed ints AT SCALE s_x -> real +-11.7
      vector<IntFp> cx; qi.commit_vec(cx, xr);
      QuantActivation th(&tanhs, TOFF); vector<IntFp> out; th.forward(out, cx);
      vector<int64_t> ref(n, 0);
      if (party == ALICE) for (int i = 0; i < n; ++i) ref[i] = llround(tanh((double)xr[i] * SX) * SC);   // tanh(x@s_x) -> 2^-P
      chk("Tanh s_x", out, ref); }

    // ---------- 3) RoPE: rotate dim-pairs by PUBLIC angles ----------
    { const int seq = 2, H = 2, dh = 4, half = dh / 2; const int N = seq * H * dh;
      vector<int64_t> qr(N, 0); fill(qr, -3000, 3000);
      vector<int64_t> cosq(seq * half), sinq(seq * half);                  // PUBLIC (both parties compute identically)
      for (int s = 0; s < seq; ++s) for (int i = 0; i < half; ++i) { double th = (double)s * pow(10000.0, -2.0 * i / dh);
        cosq[s * half + i] = llround(cos(th) * SC); sinq[s * half + i] = llround(sin(th) * SC); }
      vector<IntFp> cq; qi.commit_vec(cq, qr);
      apply_rope(&qi, cq, seq, H, dh, cosq, sinq, P);
      vector<int64_t> ref(N, 0);
      if (party == ALICE) for (int s = 0; s < seq; ++s) for (int h = 0; h < H; ++h) { int64_t b = (int64_t)(s * H + h) * dh;
        for (int i = 0; i < half; ++i) { int64_t a = qr[b + 2 * i], d = qr[b + 2 * i + 1], c = cosq[s * half + i], sn = sinq[s * half + i];
          ref[b + 2 * i] = rq1(a * c - d * sn, P); ref[b + 2 * i + 1] = rq1(a * sn + d * c, P); } }
      chk("RoPE:   ", cq, ref); }

    // ---------- 4) GQA repeat_kv: expand KV heads (reuse committed values) ----------
    { const int seq = 2, Hkv = 2, dh = 4, rep = 3; vector<int64_t> kr(seq * Hkv * dh, 0); fill(kr, -3000, 3000);
      vector<IntFp> ck; qi.commit_vec(ck, kr);
      vector<IntFp> ek = repeat_kv(ck, seq, Hkv, dh, rep);
      vector<int64_t> ref((size_t)seq * Hkv * rep * dh, 0);
      if (party == ALICE) for (int s = 0; s < seq; ++s) for (int hk = 0; hk < Hkv; ++hk) for (int r = 0; r < rep; ++r) for (int d = 0; d < dh; ++d)
        ref[((int64_t)(s * Hkv * rep + hk * rep + r) * dh) + d] = kr[((int64_t)(s * Hkv + hk) * dh) + d];
      chk("GQA kv: ", ek, ref); }

    // reveal everything at once, then check (ALICE)
    for (size_t k = 0; k < outs.size(); ++k) { int64_t err = 0;
      for (size_t i = 0; i < outs[k].size(); ++i) { int64_t g = QuantInfer::decode_signed(outs[k][i].reveal()); if (party == ALICE) err = max(err, llabs(g - refs[k][i])); }
      if (party == ALICE) { cout << "  " << names[k] << (err == 0 ? "  PASS" : "  FAIL") << " (max-err " << err << ", " << outs[k].size() << " elems)\n"; if (err) fails++; } }
    // flush all LogUp instances BEFORE finalize (qi->rc_rem, divr->rc, rclu, silut) -- relying on
    // destructors (which run after finalize) deadlocks.
    rclu.Teardown(); silut.Teardown(); tanht.Teardown(); tanhs.Teardown(); qi.destroy(); divr.destroy();
    finalize_zk_arith();
  }
  if (party == ALICE) cout << (fails == 0 ? "pass: all ml-quant LLM ops verified bit-exact (incl. Tanh: uniform + per-tensor scale)\n" : "FAIL: some ops mismatched\n");
  else cout << "verifier: complete\n";
  for (int i = 0; i < threads; ++i) delete ios[i]; return fails == 0 ? 0 : 1;
}
