// Fixed-point (ml-fixed) proof of a real GPT-2 forward pass.
// Uniform 2^-P fixed-point throughout (DeepProve "Case 1"); committed model weights.
// Public: prompt (input_ids), attention_mask(=1), position_ids. Mirrors test/ml/quant_gpt2_prove.cpp
// but on the ml-fixed backend (FixedGraphInfer + FixedInfer + Fixed{Softmax,LayerNorm} gadgets).
#include <iostream>
#include <vector>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-fixed/graph_infer.h"
#include "test/bench_e2e.h"
using namespace emp; using namespace volefier; using namespace std;
int port, party; const int threads = getenv("ZK_THREADS") ? atoi(getenv("ZK_THREADS")) : 4; const int P = getenv("P") ? atoi(getenv("P")) : 7;
int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port); setvbuf(stdout, NULL, _IONBF, 0);
  int seq = argc > 3 ? atoi(argv[3]) : 4;
  const char *mp = getenv("MODEL") ? getenv("MODEL") : "model/gpt2-12_nocache/model.onnx";
  const char *peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";   // BOB(verifier) connects to ALICE(prover); set PEER_IP=<prover IP> on BOB
  // Single-io engine: ONE main socket (port) carries ALL ZK protocol I/O; `threads` drives the
  // internal compute/VOLE pools, not sockets.  BG_VOLE adds one more socket (vole_io on port+1).
  BoolIO *ios[2] = {nullptr, nullptr};
  ios[0] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port), party == ALICE);
  if (getenv("BG_VOLE")) ios[1] = new BoolIO(new NetIO(party == ALICE ? nullptr : peer_ip, port + 1), party == ALICE);
  OnnxGraph g = load_onnx(mp);
  auto inj = [&](const char *nm, vector<int64_t> iv) { OnnxTensor t; t.name = nm; t.dtype = 7; t.dims = {1, (int64_t)iv.size()}; t.i64 = iv; g.initializer[nm] = t; g.input_shapes[nm] = {1, (int64_t)iv.size()}; };
  vector<int64_t> ids, mask, pos;
  if (getenv("IDS")) { string s = getenv("IDS"), cur; for (char c : s + ",") { if (c == ',') { if (!cur.empty()) ids.push_back(atoll(cur.c_str())); cur.clear(); } else cur += c; } seq = (int)ids.size(); }
  else for (int i = 0; i < seq; ++i) ids.push_back(100 + i);
  for (int i = 0; i < seq; ++i) { mask.push_back(1); pos.push_back(i); }
  inj("input_ids", ids); inj("attention_mask", mask); inj("position_ids", pos);
  FoldStats fs; g = fold_constants(g, fs); FuseStats us; g = fuse_patterns(g, us);
  printf("[%d] folded+fused: %zu nodes (seq=%d)\n", party, g.nodes.size(), seq);
  { map<string, int> bad; for (auto &n : g.nodes) if (!FixedGraphInfer::supported(n.op_type)) bad[n.op_type]++;
    if (!bad.empty()) { printf("UNSUPPORTED:"); for (auto &kv : bad) printf(" %s", kv.first.c_str()); printf("\n"); delete ios[0]; if (ios[1]) delete ios[1]; return 2; } }
  BenchE2E<BoolIO> _bench(party, ios, ios[1] ? 2 : 1);
  // EXP_VOLE>0 prepays the sVOLE (wire-free).  BG_VOLE=1 runs the sVOLE in a background producer
  // thread over a 2nd socket (port+threads) so it overlaps proving; the producer streams on demand
  // (consumption-driven finalize), so EXP_VOLE is ignored in background mode.
  const int64_t _ev = getenv("EXP_VOLE") ? atoll(getenv("EXP_VOLE")) : 0LL;
  const int _vt = getenv("VOLE_THREADS") ? atoi(getenv("VOLE_THREADS")) : -1;   // sVOLE pool size (independent of ostriple `threads`); <0 = same as threads
  setup_zk_arith(ios, threads, party, _ev, ios[1], _vt);   // ios[1] = vole socket (nullptr unless BG_VOLE)
  {
    FixedInfer qi(party, P, ios[0], 32, threads);   // mat_threads=threads -> threaded IP matmul
    RangeCheckLU rclu(party, ios[0]->io, 16); CompareLU cmp(&rclu); Div divr(party, ios[0]->io, 16);
    // Tables keep their REAL domain as P scales: exp [0,512) (score_bits=P+9), gelu +-64 (off=2^(P+6)).
    const std::size_t SB = getenv("SCORE_BITS") ? (std::size_t)atoi(getenv("SCORE_BITS")) : (std::size_t)(P + 9); const uint64_t ET = 1ull << SB;
    const uint64_t GT = 1u << (P + 7); const int64_t GOFF = 1LL << (P + 6);
    const bool lse = getenv("LSE") != nullptr;   // DeepProve LogSumExp softmax (no max_value/division)
    const int LX = lse ? (getenv("LSE_BITS") ? atoi(getenv("LSE_BITS")) : 8) : 0;   // extra LSE exp output bits
    const double LSE_MAX_SLACK = getenv("LSE_MAX_SLACK") ? atof(getenv("LSE_MAX_SLACK")) : 0.005;
    FixedTable expt(ET, party, ios[0]->io); expt.Setup(build_exp_table(ET, P, !lse, P + LX));  // LSE needs the non-clamped table
    // LSE: exp outputs at 2^-(P+LSE_BITS) keep the row-sum check (the only thing binding the prover's
    // shift) tight: slack ~ (L/2 + 2^LSE_BITS) / 2^(P+LSE_BITS).  LSE_MAX_SLACK refuses looser rows.
    FixedTable gtab(GT, party, ios[0]->io); gtab.Setup(build_gelu_table(GT, P, GOFF));
    FixedGraphInfer gi(&qi, &cmp, party, P); gi.divr = &divr; gi.rcmp = &rclu; gi.exp_tab = &expt; gi.gelu_tab = &gtab; gi.gelu_off = GOFF; gi.score_bits = SB; gi.lse_softmax = lse; gi.lse_extra_bits = LX; gi.lse_max_slack = LSE_MAX_SLACK;
    vector<int64_t> xr(seq, 0); vector<IntFp> cx; qi.commit_vec(cx, xr); vector<int64_t> oref;
    printf("[%d] running %zu ops...\n", party, g.nodes.size());
    vector<IntFp> out = gi.run(g, cx, xr, oref);
    printf("[%d] dispatched %zu ops, %zu logits\n", party, gi.ops_run.size(), out.size());
    const int V = 50257; const int npos = (out.size() % V == 0) ? (int)(out.size() / V) : 0; vector<int> pred(npos);
    if (npos > 0) {
      if (getenv("OLD_ARGMAX")) { for (int t = 0; t < npos; ++t) { vector<IntFp> row(out.begin() + (int64_t)t * V, out.begin() + (int64_t)(t + 1) * V); IntFp idx = cmp.argmax(row, 30); pred[t] = (int)idx.reveal(); } }
      else { vector<IntFp> selv, idxv;   // batched rows: bulk selector commit + RLC bitness, zero mult gates
        cmp.select_max_rows(out, (size_t)npos, (size_t)V, 30, selv, idxv);
        for (int t = 0; t < npos; ++t) pred[t] = (int)idxv[t].reveal(); } }
    else for (auto &v : out) v.reveal();
    rclu.Teardown(); expt.Teardown(); gtab.Teardown(); divr.destroy(); qi.destroy(); finalize_zk_arith();   // divr/qi BEFORE finalize: their range-check batches must be in the proof (else unverified + OOB on freed exec)
    _bench.report("gpt2 (ml-fixed)");
    if (getenv("NETIO_STATS")) { auto*m=ios[0]->io;
      fprintf(stderr,"[NETIO] party=%d threads=%d rounds_total=%llu rounds_max=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n",party,threads,(unsigned long long)m->rounds,(unsigned long long)m->rounds,(unsigned long long)m->flushes_count,m->send_counter/1e6,m->recv_counter/1e6);
      if (ios[1]) { auto*v=ios[1]->io; fprintf(stderr,"[NETIO] vole-socket: rounds=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n",(unsigned long long)v->rounds,(unsigned long long)v->flushes_count,v->send_counter/1e6,v->recv_counter/1e6); } }
    if (party == ALICE) { printf("predicted next-token ids (proven argmax): "); for (int t = 0; t < npos; ++t) printf("%d ", pred[t]); printf("\n");
      printf("pass: ml-fixed proved GPT-2 forward [%s] (verifier accepted)\n", mp); }
    else printf("verifier: GPT-2 fixed-point proof complete [%s]\n", mp);
  }
  cout.flush(); fflush(stdout); _exit(0);
}
