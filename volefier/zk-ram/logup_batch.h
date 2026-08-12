#ifndef ZK_LOGUP_BATCH_H
#define ZK_LOGUP_BATCH_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup.h"
#include <vector>

using namespace emp;

// Batched (multi-column) LogUp lookup argument.
//
// Witness and table rows each have K committed columns. Verifier samples a
// fresh row-linear-combination challenge c = (c_0, ..., c_{K-1}) inside
// Prove, the prover compresses every row to a single field element
//
//   RLC(row) = sum_{k=0}^{K-1} c_k * row[k]
//
// and the standard single-column LogUp identity is then proven on the
// compressed values:
//
//   sum_{i=0}^{W-1} 1/(x + RLC(witness_i))
//     = sum_{j=0}^{T-1} m[j]/(x + RLC(table_j)).
//
// Soundness. Two distinct K-tuples a != b collide under RLC with
// probability (K-1)/(PR-1) over random c (Schwartz-Zippel on the linear
// polynomial). Combined with the inner LogUp soundness ~(W+T)/PR, the
// total error is O((W+T+K)/PR), negligible for PR ~ 2^61.
//
// Completeness. For max_val == 0, the inner LogUp samples x from [0, PR)
// and may fail with probability (W+T)/PR if some x + RLC(...) = 0 mod PR.
// Callers with a tight bound on RLC values can pass max_val to keep x in
// [1, PR - max_val] and avoid the retry. With unbounded column values the
// RLC is essentially uniform in [0, PR), so 0 is the typical choice here.
class LogUpLookupBatch {
public:
    int party;
    IOChannel *io;

    // After Prove(), holds the committed multiplicities for all T table rows.
    std::vector<IntFp> multiplicities;

    LogUpLookupBatch(int party, IOChannel *io) : party(party), io(io) {}

    // COLUMN-major (one contiguous buffer per column, not a vector per row):
    //   wit_cols: K columns, wit_cols[k] has length W (column k of every witness row).
    //   tab_cols: K columns, tab_cols[k] has length T.
    // This avoids materializing W+T tiny per-row vectors -- at transformer scale W is tens of
    // millions, and a vector<vector<IntFp>> there is tens of millions of 2-element heap allocations
    // (~70 B/access of allocator overhead, and severe glibc fragmentation).  Callers already hold
    // their columns contiguously (access_keys / access_values), so they pass them with no copy.
    // mult_raw:  length T; ALICE provides histogram counts, BOB passes zeros.
    // max_val:   public upper bound on RLC values, or 0 for unbounded.
    void Prove(const std::vector<std::vector<IntFp>> &wit_cols,
               const std::vector<std::vector<IntFp>> &tab_cols,
               const std::vector<uint64_t> &mult_raw,
               uint64_t max_val = 0) {

        std::size_t K = wit_cols.empty() ? tab_cols.size() : wit_cols.size();
        if (K == 0) return;
        std::size_t W = wit_cols.empty() ? 0 : wit_cols[0].size();
        std::size_t T = tab_cols.empty() ? 0 : tab_cols[0].size();
        if (W == 0 && T == 0) return;

        LogUpSpan _sp_rlc(LP_RLC, io);
        // Step 0: verifier samples K row-linear-combination challenges.
        // Sampled here, after the witness/table commitments are fixed.
        std::vector<uint64_t> c_raw(K);
        if (party == ALICE) {
            io->recv_data(c_raw.data(), K * sizeof(uint64_t));
        } else {
            PRG prg;
            prg.random_data(c_raw.data(), K * sizeof(uint64_t));
            for (std::size_t k = 0; k < K; ++k) c_raw[k] = mod(c_raw[k]);
            io->send_data(c_raw.data(), K * sizeof(uint64_t)); io->flush();
        }
        // Step 1: compress each row to a single committed field element.
        // The challenge is a PUBLIC cleartext constant, so multiply with the uint64_t
        // overload (mul_const_gate: local lane math, no VOLE, no correction bytes).
        // The previous `* IntFp(c_raw[k], PUBLIC)` hit the IntFp*IntFp overload =
        // a REAL mul_gate per element: W*K gates -> ~8 B correction + 1 VOLE each
        // (measured ~1 GB / 128M gates on GPT-2 s512), for algebraically identical output.
        std::vector<IntFp> w_compressed(W);
        for (std::size_t i = 0; i < W; ++i) {
            IntFp acc(0ULL, PUBLIC);
            for (std::size_t k = 0; k < K; ++k)
                acc = acc + (wit_cols[k][i] * c_raw[k]);
            w_compressed[i] = acc;
        }
        std::vector<IntFp> t_compressed(T);
        for (std::size_t j = 0; j < T; ++j) {
            IntFp acc(0ULL, PUBLIC);
            for (std::size_t k = 0; k < K; ++k)
                acc = acc + (tab_cols[k][j] * c_raw[k]);
            t_compressed[j] = acc;
        }

        // Step 2: delegate to single-column LogUp on the compressed rows.
        // Multiplicities are indexed by table row, unchanged from raw form.
        _sp_rlc.stop();
        LogUpLookup inner(party, io);
        inner.Prove(w_compressed, t_compressed, mult_raw, max_val);
        multiplicities = std::move(inner.multiplicities);
    }
};

#endif
