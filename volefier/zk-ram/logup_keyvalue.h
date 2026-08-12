#ifndef ZK_LOGUP_KEYVALUE_H
#define ZK_LOGUP_KEYVALUE_H

#include "emp-zk/emp-zk.h"
#include "volefier/zk-ram/logup_batch.h"
#include <vector>

using namespace emp;

// LogUpKeyValueLookup: read-only ZKROM-style key-value table backed by
// the batched LogUp argument.  Keys are the implicit range [0, N); values
// are arbitrary committed (or public) field elements supplied at Setup.
//
// Each Access(id) returns a freshly committed IntFp holding table[id].
// The binding "(id, returned_value) is one of the rows {(j, table[j])}"
// is enforced collectively at Teardown by a single LogUpLookupBatch run
// over K=2 columns:
//
//   sum_i 1/(x + RLC(id_i, val_i))  =  sum_j m[j]/(x + RLC(j, table[j]))
//
// where the row-LC challenge is sampled inside the inner Prove and m[j]
// is the access count for key j.  A cheating prover that returns an
// incorrect value at some Access cannot match any table row under the
// random RLC, so the LogUp identity fails with overwhelming probability.
//
// Drop-in usage for the read-only patterns of ZKROM (Setup once + many
// Access + Teardown):
//
//   LogUpKeyValueLookup tab(N, party, io);
//   tab.Setup(table_values);
//   for each access: IntFp v = tab.Access(id);
//   tab.Teardown();
class LogUpKeyValueLookup {
public:
    int party;
    IOChannel *io;
    uint64_t N;

    // Committed values at keys 0..N-1.  Set by Setup().
    std::vector<IntFp> table_values;

    // Buffered access pairs and per-key multiplicities.
    std::vector<IntFp> access_keys;
    std::vector<IntFp> access_values;
    std::vector<uint64_t> mult_raw;

    uint64_t batch_sz; // incremental LogUp: auto-Teardown when the buffer hits this (mirrors
                       // LogUpRangeCheck), so accesses don't pile up across the whole forward.

    LogUpKeyValueLookup(uint64_t N_, int party_, IOChannel *io_, uint64_t batch_sz_ = 16 * 1024 * 1024)
        : party(party_), io(io_), N(N_), batch_sz(batch_sz_) {}

    // table_vals: length N, the value (committed or PUBLIC) at each key.
    void Setup(const std::vector<IntFp> &table_vals) {
        if (table_vals.size() != N)
            error("LogUpKeyValueLookup: Setup size mismatch\n");
        table_values = table_vals;
        mult_raw.assign(N, 0ULL);
        access_keys.clear();
        access_values.clear();
    }

    // Convenience: ALICE-commit raw uint64_t values; BOB commits zeros.
    void Setup(const std::vector<uint64_t> &table_vals_raw) {
        if (table_vals_raw.size() != N)
            error("LogUpKeyValueLookup: Setup size mismatch\n");
        // One batched feed for all N entries (the feed ignores the input array on the verifier,
        // so the prover's raw values vs the verifier's placeholders are handled correctly).
        table_values.resize(N);
        batch_feed(table_values.data(), const_cast<uint64_t *>(table_vals_raw.data()), (int64_t)N);
        mult_raw.assign(N, 0ULL);
        access_keys.clear();
        access_values.clear();
    }

    // Read the value at id.  ALICE looks it up locally and commits it;
    // BOB commits a placeholder zero.  The (id, val) pair is buffered
    // and bound to the table at Teardown.
    IntFp Access(IntFp &id) {
        IntFp val;
        if (party == ALICE) {
            uint64_t k = static_cast<uint64_t>(VAL(id.value));
            if (k >= N) { fprintf(stderr, "key out of range: k=%llu N=%llu\n", (unsigned long long)k, (unsigned long long)N); error("key out of range\n"); }
            uint64_t v = static_cast<uint64_t>(VAL(table_values[k].value));
            val = IntFp(v, ALICE);
            mult_raw[k]++;
        } else {
            val = IntFp(0ULL, ALICE);
        }
        access_keys.push_back(id);
        access_values.push_back(val);
        if (access_keys.size() >= batch_sz) Teardown(); // partial proof; table_values persists
        return val;
    }

    // Batched Access: commit ALL returned values with ONE batch_feed (bulk VOLE + one send)
    // instead of M per-element IntFp(.,ALICE) feeds.  Same buffered (id,val) pairs + LogUp proof.
    void Access(std::vector<IntFp> &out, const std::vector<IntFp> &ids) {
        const std::size_t M = ids.size();
        std::vector<uint64_t> vraw(M, 0);
        if (party == ALICE)
            for (std::size_t i = 0; i < M; ++i) {
                uint64_t k = static_cast<uint64_t>(VAL(ids[i].value));
                if (k >= N) error("key out of range\n");
                vraw[i] = static_cast<uint64_t>(VAL(table_values[k].value));
                mult_raw[k]++;
            }
        out.resize(M);
        batch_feed(out.data(), vraw.data(), (int64_t)M);
        access_keys.insert(access_keys.end(), ids.begin(), ids.end());
        access_values.insert(access_values.end(), out.begin(), out.end());
        if (access_keys.size() >= batch_sz) Teardown();
    }

    // Run one batched LogUp proof binding every buffered (id, val) to a
    // table row.  Clears the buffer so the object can be reused.
    void Teardown() {
        static const bool LOFF = getenv("LOGUP_OFF") != nullptr;
        if (LOFF) { access_keys.clear(); access_values.clear(); mult_raw.assign(N, 0ULL); return; }
        static const bool LPROF = getenv("LOGUP_PROF") != nullptr;
        auto _t0 = std::chrono::steady_clock::now();
        double _c0 = LPROF ? logup_cpu_self() : 0, _h0 = LPROF ? logup_cpu_thr() : 0;
        unsigned long long _r0 = LPROF ? static_cast<NetIO *>(io)->rounds : 0ULL;
        // Column-major: the access columns are already contiguous, so MOVE them in (no copy, no
        // per-access tiny-vector materialization).  The table is small (N rows), so its key column
        // is built and its value column copied.
        std::vector<IntFp> tab_keys(N);
        for (uint64_t j = 0; j < N; ++j) tab_keys[j] = IntFp(j, PUBLIC);
        std::vector<std::vector<IntFp>> wit_cols(2), tab_cols(2);
        wit_cols[0] = std::move(access_keys);
        wit_cols[1] = std::move(access_values);
        tab_cols[0] = std::move(tab_keys);
        tab_cols[1] = table_values;

        // Keys and table indices are in [0, N); values are unbounded committed elements, so the RLC
        // is essentially uniform in [0, PR).  max_val = 0 accepts the (W+N)/PR completeness gap.
        LogUpLookupBatch logup(party, io);
        logup.Prove(wit_cols, tab_cols, mult_raw, 0);

        access_keys.clear(); // already moved-from; reset for reuse
        access_values.clear();
        mult_raw.assign(N, 0ULL);
        if (LPROF) {
            auto &P = logup_prof();
            P.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
            P.cpu_self += logup_cpu_self() - _c0;
            P.cpu_thr += logup_cpu_thr() - _h0;
            P.rounds += static_cast<NetIO *>(io)->rounds - _r0;
            P.proves++;
            fprintf(stderr, "[LOGUP_PROF] cum wall %.3f s  cpu_self %.3f s  cpu_thr %.3f s  %llu rounds  %ld proves\n",
                    P.secs, P.cpu_self, P.cpu_thr, P.rounds, P.proves);
            for (int _k = 0; _k < LP_NCOMP; ++_k) {
                auto &C = logup_comp()[_k];
                if (C.wall > 0.001)
                    fprintf(stderr, "[LOGUP_COMP] %-16s wall %8.3f s  cpu %8.3f s  %9.1f MB\n",
                            logup_comp_name(_k), C.wall, C.cpu, C.bytes / 1e6);
            }
        }
    }
};


// LogUpKeyValueLookupBatch: K_val-column generalization of
// LogUpKeyValueLookup.  Each row is (key, val_0, ..., val_{K_val-1}).
// Access(out, id) writes the K_val committed values at row `id` to `out`.
// Teardown runs one LogUpLookupBatch over (1 + K_val)-column rows; the
// first column is the key (identity {0..N-1}), remaining columns are the
// values.
//
// Drop-in replacement for the read-only side of ZKROMBatch.  The Setup
// layout matches ZKROMBatch: table_vals is flat row-major,
// table_vals[j*K + k] is column k of row j.
class LogUpKeyValueLookupBatch {
public:
    int party;
    IOChannel *io;
    uint64_t N;       // number of rows
    uint64_t K;       // value columns per row

    // Committed values, row-major: table_values[j*K + k].
    std::vector<IntFp> table_values;

    // Buffered access tuples.  access_values is row-major:
    // access_values[i*K + k] is column k of access i.
    std::vector<IntFp> access_keys;
    std::vector<IntFp> access_values;
    std::vector<uint64_t> mult_raw;

    uint64_t batch_sz; // incremental LogUp auto-Teardown threshold (mirrors LogUpRangeCheck)

    LogUpKeyValueLookupBatch(uint64_t N_, uint64_t K_, int party_, IOChannel *io_,
                             uint64_t batch_sz_ = 16 * 1024 * 1024)
        : party(party_), io(io_), N(N_), K(K_), batch_sz(batch_sz_) {}

    // table_vals: length N*K, row-major.
    void Setup(const std::vector<IntFp> &table_vals) {
        if (table_vals.size() != N * K)
            error("LogUpKeyValueLookupBatch: Setup size mismatch\n");
        table_values = table_vals;
        mult_raw.assign(N, 0ULL);
        access_keys.clear();
        access_values.clear();
    }

    // Convenience: ALICE-commits raw uint64 values; BOB commits zeros.
    void Setup(const std::vector<uint64_t> &table_vals_raw) {
        if (table_vals_raw.size() != N * K)
            error("LogUpKeyValueLookupBatch: Setup size mismatch\n");
        // One batched feed for all N*K entries (the feed ignores the input array on the verifier).
        table_values.resize(N * K);
        batch_feed(table_values.data(), const_cast<uint64_t *>(table_vals_raw.data()), (int64_t)(N * K));
        mult_raw.assign(N, 0ULL);
        access_keys.clear();
        access_values.clear();
    }

    // Read the K values at row id into out.  ALICE looks them up locally
    // and commits; BOB commits placeholder zeros.  The whole tuple is
    // bound to a table row at Teardown.
    void Access(std::vector<IntFp> &out, IntFp &id) {
        if (out.size() != K) out.resize(K);
        if (party == ALICE) {
            uint64_t k = static_cast<uint64_t>(VAL(id.value));
            if (k >= N)
                error("LogUpKeyValueLookupBatch: key out of range\n");
            for (uint64_t c = 0; c < K; ++c) {
                uint64_t v = static_cast<uint64_t>(
                    VAL(table_values[k * K + c].value));
                out[c] = IntFp(v, ALICE);
            }
            mult_raw[k]++;
        } else {
            for (uint64_t c = 0; c < K; ++c)
                out[c] = IntFp(0ULL, ALICE);
        }
        access_keys.push_back(id);
        for (uint64_t c = 0; c < K; ++c)
            access_values.push_back(out[c]);
        if (access_keys.size() >= batch_sz) Teardown(); // partial proof; table_values persists
    }

    // Run one LogUpLookupBatch over (1 + K)-column rows.  Clears the
    // buffer; can be reused after another Setup().
    void Teardown() {
        static const bool LOFF = getenv("LOGUP_OFF") != nullptr;
        if (LOFF) { access_keys.clear(); access_values.clear(); mult_raw.assign(N, 0ULL); return; }
        static const bool LPROF = getenv("LOGUP_PROF") != nullptr;
        auto _t0 = std::chrono::steady_clock::now();
        double _c0 = LPROF ? logup_cpu_self() : 0, _h0 = LPROF ? logup_cpu_thr() : 0;
        unsigned long long _r0 = LPROF ? static_cast<NetIO *>(io)->rounds : 0ULL;
        std::size_t W = access_keys.size();
        // Column-major (1 + K columns).  Key column is contiguous (move); the K value columns are
        // de-interleaved from the row-major access_values -- K big buffers, not W tiny per-row vectors.
        std::vector<std::vector<IntFp>> wit_cols(1 + K), tab_cols(1 + K);
        wit_cols[0] = std::move(access_keys);
        for (uint64_t c = 0; c < K; ++c) {
            wit_cols[1 + c].resize(W);
            for (std::size_t i = 0; i < W; ++i)
                wit_cols[1 + c][i] = access_values[i * K + c];
        }
        std::vector<IntFp> tab_keys(N);
        for (uint64_t j = 0; j < N; ++j) tab_keys[j] = IntFp(j, PUBLIC);
        tab_cols[0] = std::move(tab_keys);
        for (uint64_t c = 0; c < K; ++c) {
            tab_cols[1 + c].resize(N);
            for (uint64_t j = 0; j < N; ++j)
                tab_cols[1 + c][j] = table_values[j * K + c];
        }

        // Keys are in [0, N); value columns are unbounded.  RLC is essentially uniform in [0, PR),
        // so max_val = 0 accepts the (W+N)/PR completeness gap.
        LogUpLookupBatch logup(party, io);
        logup.Prove(wit_cols, tab_cols, mult_raw, 0);

        access_keys.clear(); // wit_cols[0] moved-from; reset for reuse
        access_values.clear();
        mult_raw.assign(N, 0ULL);
        if (LPROF) {
            auto &P = logup_prof();
            P.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
            P.cpu_self += logup_cpu_self() - _c0;
            P.cpu_thr += logup_cpu_thr() - _h0;
            P.rounds += static_cast<NetIO *>(io)->rounds - _r0;
            P.proves++;
            fprintf(stderr, "[LOGUP_PROF] cum wall %.3f s  cpu_self %.3f s  cpu_thr %.3f s  %llu rounds  %ld proves\n",
                    P.secs, P.cpu_self, P.cpu_thr, P.rounds, P.proves);
            for (int _k = 0; _k < LP_NCOMP; ++_k) {
                auto &C = logup_comp()[_k];
                if (C.wall > 0.001)
                    fprintf(stderr, "[LOGUP_COMP] %-16s wall %8.3f s  cpu %8.3f s  %9.1f MB\n",
                            logup_comp_name(_k), C.wall, C.cpu, C.bytes / 1e6);
            }
        }
    }
};

#endif
