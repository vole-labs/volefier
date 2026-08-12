// Prove a real Qwen2.5 forward through ml-quant (RMSNorm + RoPE + GQA + SwiGLU).
// Input graph: FIXED-seq FUSED fp32 ONNX from  tools/export_llm_onnx.py (FIXED_SEQ=<n>) +
// tools/fuse_llm_onnx.py  (RMSNormalization / SiLU / RotaryEmbedding nodes; position_ids and
// attention_mask baked to constants, so only input_ids is a graph input).
// Uses the symmetric int-N path (SYM_BITS + static CALIB), like the GPT-2/BERT sym provers -- no QDQ.
//   pass 1:  CALIB_WRITE=<f> SYM_BITS=12 ... -> records per-tensor maxes
//   pass 2:  CALIB=<f>       SYM_BITS=12 ... -> proves with those fixed scales
#include <iostream>
#include <fstream>
#include <vector>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-quant/quant_graph_infer.h"
#include "test/utility.h"
using namespace emp;
using namespace volefier;
using namespace std;
int port, party;
const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4;
const int P = getenv("P") ? atoi(getenv("P")) : 12;
int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0); // unbuffered so DBG_OPS traces survive a crash under redirect
  int seq = argc > 3 ? atoi(argv[3]) : 16;
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";
  BoolIO *ios[1];
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  BoolIO *_vio = nullptr;
  // BENCH=1 (or PHASES=1): per-phase wall-time + COMMUNICATION, measured EXACTLY like the ml-quant
  // GPT-2 driver. Prover-side (ALICE) figures; the PROVING TOTAL excludes model load + fold/fuse.
  const bool BENCH = getenv("BENCH") || getenv("PHASES");
  auto comm = [&]() { return io_counter(ios[0]->io) + (_vio ? io_counter(_vio->io) : (uint64_t)0); };
  auto _ck = clock_start(); uint64_t _cb = comm(); auto _pk = _ck; uint64_t _pb = _cb;   // _ck/_cb: per-phase; _pk/_pb: proving window
  uint64_t _cr = 0;   // per-phase round counter (main socket)
  auto LAP = [&](const char *nm) { if (BENCH && party == ALICE) { double us = time_from(_ck); uint64_t b = comm() - _cb; fprintf(stderr, "[BENCH] %-18s %8.3f s  %8.2f MB  %5llu rounds  [rss cur %.0f peak %.0f MB]\n", nm, us / 1e6, b / 1e6, (unsigned long long)(ios[0]->io->rounds - _cr), getCurrentRSS() / 1e6, getPeakRSS() / 1e6); } _ck = clock_start(); _cb = comm(); _cr = ios[0]->io->rounds; };
  const char *mp = getenv("MODEL") ? getenv("MODEL") : "model/qwen2.5-0.5b-quarot-fused.onnx";
  OnnxGraph g = load_onnx(mp);
  auto inj = [&](const char *nm, vector<int64_t> iv) {
    OnnxTensor t; t.name = nm; t.dtype = 7; t.dims = {1, (int64_t)iv.size()}; t.i64 = iv;
    g.initializer[nm] = t; g.input_shapes[nm] = {1, (int64_t)iv.size()};
  };
  vector<int64_t> ids;
  if (getenv("IDS")) {
    string s = getenv("IDS"), c;
    for (char ch : s + ",") { if (ch == ',') { if (!c.empty()) ids.push_back(atoll(c.c_str())); c.clear(); } else c += ch; }
    seq = (int)ids.size();
  } else
    for (int i = 0; i < seq; ++i) ids.push_back(100 + i);
  inj("input_ids", ids); // position_ids / attention_mask are baked constants in the fused graph
  FoldStats fs; g = fold_constants(g, fs);
  FuseStats us; g = fuse_patterns(g, us);
  printf("[%d] loaded+folded: %zu nodes (seq=%d)\n", party, g.nodes.size(), seq);
  LAP("load+fold+fuse"); _pk = clock_start(); _pb = comm();   // proving window starts here (excl. model load)
  {
    map<string, int> bad;
    for (auto &n : g.nodes) if (!QuantGraphInfer::supported(n.op_type)) bad[n.op_type]++;
    if (!bad.empty()) { printf("UNSUPPORTED:"); for (auto &kv : bad) printf(" %s", kv.first.c_str()); printf("\n"); return 2; }
  }
  const int64_t _ev = getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL;
  _vio = getenv("BG_VOLE") ? new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE) : nullptr;
  const int _vt = getenv("VOLE_THREADS") ? atoi(getenv("VOLE_THREADS")) : -1;
  setup_zk_arith(ios, threads, party, _ev, _vio, _vt);
  LAP("setup_zk_arith");
  {
    QuantInfer qi(party, P, ios[0], 32, 1, threads);
    RangeCheckLU rclu(party, ios[0]->io, 16);
    CompareLU cmp(&rclu);
    // Div borrows QuantInfer's rc_rem [0,2^P) range table instead of allocating a second identical one.
    Div divr(party, ios[0]->io, 16, (size_t)P, qi.rc_rem);
    // Build only the activation lookup tables the graph actually dispatches to.  A model uses at most a
    // couple of these (Qwen/Llama: SiLU + Softmax, no Gelu; GPT-2: Gelu, no SiLU), so the absent ones are
    // dead -- skipping their Setup avoids a full-table commit and their end-of-run LogUp prove.
    auto uses = [&](const char *op) {
      for (auto &n : g.nodes) if (n.op_type == op) return true;
      return false;
    };
    const bool has_softmax = uses("Softmax");
    const bool has_gelu = uses("Gelu") || uses("Erf");
    const bool has_silu = uses("SiLU");
    const uint64_t ET = 1ull << (P + 9), GT = 1ull << (P + 8), ST = 1ull << (P + 8);
    const int64_t GOFF = 1LL << (P + 7), SOFF = 1LL << (P + 7);
    QuantTable expt(ET, party, ios[0]->io);  if (has_softmax) expt.Setup(build_exp_table(ET, P, !getenv("FUSED_NL")));
    QuantTable gtab(GT, party, ios[0]->io);  if (has_gelu)    gtab.Setup(build_gelu_table(GT, P, GOFF));
    QuantTable silut(ST, party, ios[0]->io); if (has_silu)    silut.Setup(build_silu_table(ST, P, SOFF));
    printf("[%d] activation tables built: exp=%d gelu=%d silu=%d\n", party, has_softmax, has_gelu, has_silu);
    QuantGraphInfer gi(&qi, &cmp, &divr, party);
    gi.P = P; gi.rcmp = &rclu; gi.exp_tab = &expt; gi.gelu_tab = &gtab; gi.gelu_off = GOFF;
    gi.silu_tab = &silut; gi.silu_off = SOFF; gi.score_bits = (size_t)(P + 9);
    gi.uniform_requant = !getenv("NO_UNIFORM");
    if (getenv("PUBLIC_WEIGHTS")) gi.commit_weights = false;
    if (getenv("SYM_BITS")) { gi.sym_act_bits = atoi(getenv("SYM_BITS")); gi.weight_max = (1 << (gi.sym_act_bits - 1)) - 1; gi.uniform_requant = false; }
    if (getenv("CALIB_WRITE")) gi.calib_write_ = true;
    else if (getenv("CALIB")) { std::ifstream f(getenv("CALIB")); std::string nm; double mx; int nc = 0; while (f >> nm >> mx) { gi.calib_[nm] = mx; ++nc; } printf("[%d] loaded %d calib scales\n", party, nc); }
    vector<int64_t> xr(seq, 0); vector<IntFp> cx; qi.commit_vec(cx, xr); vector<int64_t> oref;
    LAP("table-setup");
    printf("[%d] running %zu ops...\n", party, g.nodes.size());
    vector<IntFp> out = gi.run(g, cx, xr, 1.0 / (1 << P), 0, oref);
    LAP("forward-prove");
    printf("[%d] dispatched %zu ops, %zu logits\n", party, gi.ops_run.size(), out.size());
    if (getenv("TAP") && party == ALICE) gi.tap(getenv("TAP"), getenv("TAP_FILE") ? getenv("TAP_FILE") : "/tmp/tap.bin");
    if (getenv("OREF_DUMP") && party == ALICE && !oref.empty()) { // quantized logits @2^-P for accuracy check
      std::ofstream f(getenv("OREF_DUMP"), std::ios::binary); f.write((char *)oref.data(), (long)oref.size() * 8);
      printf("[%d] dumped %zu logits to %s\n", party, oref.size(), getenv("OREF_DUMP"));
    }
    if (getenv("CALIB_WRITE") && party == ALICE) { std::ofstream f(getenv("CALIB_WRITE")); for (auto &kv : gi.calib_max_) f << kv.first << " " << kv.second << "\n"; printf("[%d] wrote %zu calib scales to %s\n", party, gi.calib_max_.size(), getenv("CALIB_WRITE")); }
    gi.release_forward_state(); // free committed weights + LM-head table before argmax (argmax uses only `out`)
    const int Vsz = getenv("VOCAB") ? atoi(getenv("VOCAB")) : 151936;
    int npos; vector<int> pred;
    if (gi.streamed_lmhead_) { pred = gi.lmhead_pred_; npos = (int)pred.size(); } // STREAM_LMHEAD: argmax done inside run()
    else {
      npos = (out.size() % Vsz == 0) ? (int)(out.size() / Vsz) : 0;
      pred.assign(npos, 0);
      if (npos > 0) { vector<IntFp> selv, idxv; cmp.select_max_rows(out, (size_t)npos, (size_t)Vsz, 30, selv, idxv); for (int t = 0; t < npos; ++t) pred[t] = (int)idxv[t].reveal(); }
      else { for (auto &v : out) v.reveal(); }
    }
    LAP("argmax");
    rclu.Teardown();
    if (has_softmax) expt.Teardown();
    if (has_gelu) gtab.Teardown();
    if (has_silu) silut.Teardown();
    divr.destroy(); qi.destroy(); LAP("lookup-teardown"); finalize_zk_arith(); LAP("finalize_zk_arith");
    if (BENCH && party == ALICE) { double us = time_from(_pk); uint64_t b = comm() - _pb;
      fprintf(stderr, "[BENCH] === PROVING TOTAL   %8.3f s  %8.2f MB  (setup_zk -> finalize, excl. model load) ===\n", us / 1e6, b / 1e6);
      fprintf(stderr, "[BENCH] peak RSS  %.0f MB\n", getPeakRSS() / 1e6); }
    if (getenv("NETIO_STATS")) { auto *m = ios[0]->io;
      fprintf(stderr, "[NETIO] party=%d threads=%d rounds_total=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n", party, threads, (unsigned long long)m->rounds, (unsigned long long)m->flushes_count, m->send_counter / 1e6, m->recv_counter / 1e6);
      if (_vio) { auto *v = _vio->io; fprintf(stderr, "[NETIO] vole-socket: rounds=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n", (unsigned long long)v->rounds, (unsigned long long)v->flushes_count, v->send_counter / 1e6, v->recv_counter / 1e6); } }
    if (party == ALICE) {
      printf("predicted next-token ids: "); for (int t = 0; t < npos; ++t) printf("%d ", pred[t]); printf("\n");
      printf("pass: ml-quant proved Qwen forward [%s] (verifier accepted)\n", mp);
    } else printf("verifier: Qwen proof complete [%s]\n", mp);
  }
  delete ios[0]; if (_vio) delete _vio; return 0;
}
