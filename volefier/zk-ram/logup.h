#ifndef ZK_LOGUP_H
#define ZK_LOGUP_H

#include "emp-zk/emp-zk.h"
#include "volefier/rlc_chi.h"
#include "volefier/linear/hadamard_ip.h"   // Hadamard2IP: shared RLC-then-IP batcher for (x+w)*inv==1
#include <vector>
#include <unordered_map>
#include <chrono>

using namespace emp;

// LOGUP_PROF=1: accumulate wall-time, CPU-time and NetIO rounds spent inside the LogUp
// Teardown/Prove machinery (cumulative [LOGUP_PROF] line per Teardown; last line = totals).
// cpu_self = process CPU (all threads; excludes blocked-on-recv waits) -> LOCAL COMPUTE.
// cpu_thr  = calling-thread CPU (lower bound if Prove threads internally).
// LOGUP_OFF=1: skip the LogUp proof machinery entirely (rc buffering + all Teardowns) --
// measurement-only (produces no valid proof); both parties must set it (symmetric) to stay in sync.
struct LogUpProfT { double secs = 0, cpu_self = 0, cpu_thr = 0; unsigned long long rounds = 0; long proves = 0; };
inline LogUpProfT &logup_prof() { static LogUpProfT p; return p; }
#ifdef __linux__
#include <sys/resource.h>
inline double logup_cpu_self() { rusage r; getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6; }
inline double logup_cpu_thr() { rusage r; getrusage(RUSAGE_THREAD, &r);
  return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6; }
#else
inline double logup_cpu_self() { return 0; }
inline double logup_cpu_thr() { return 0; }
#endif

// Per-component accounting inside the LogUp Prove: wall + process-CPU + NetIO bytes per slot.
// (fp_zkp_inner_prdt defers its wire traffic to FpPolyProof's buffer flush, so inner-prdt bytes
// land wherever the flush fires -- read the totals, not per-slot byte splits, for that one.)
enum { LP_MULT = 0, LP_CHAL, LP_AINV, LP_ICOMMIT, LP_FOLD, LP_IPRDT, LP_TABLE, LP_RLC, LP_NCOMP };
struct LogUpComp { double wall = 0, cpu = 0; unsigned long long bytes = 0; };
inline LogUpComp *logup_comp() { static LogUpComp c[LP_NCOMP]; return c; }
inline const char *logup_comp_name(int k) {
    static const char *n[LP_NCOMP] = {"mult-feed", "challenge/seeds", "alice-inverse", "inv-commit",
                                      "fold",      "inner-prdt",      "table+final",   "rlc-compress"};
    return n[k];
}
struct LogUpSpan {
    int k; IOChannel *io_; double t0 = 0, c0 = 0; unsigned long long b0 = 0; bool on, done = false;
    std::chrono::steady_clock::time_point w0;
    LogUpSpan(int k_, IOChannel *io__) : k(k_), io_(io__) {
        on = getenv("LOGUP_PROF") != nullptr;
        if (!on) return;
        w0 = std::chrono::steady_clock::now();
        c0 = logup_cpu_self();
        auto *n = static_cast<NetIO *>(io_);
        b0 = n->send_counter + n->recv_counter;
    }
    void stop() {
        if (!on || done) return;
        done = true;
        auto &C = logup_comp()[k];
        C.wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        C.cpu += logup_cpu_self() - c0;
        auto *n = static_cast<NetIO *>(io_);
        C.bytes += n->send_counter + n->recv_counter - b0;
    }
    ~LogUpSpan() { stop(); }
};

namespace logup_detail {
// base^e mod PR (square-and-multiply).  Used to let each worker thread compute
// the starting random weight seed^(lo+1) of its own segment, so the r_i =
// seed^(i+1) power chain can be built in parallel instead of one serial loop.
inline uint64_t pow_mod(uint64_t base, uint64_t e) {
    uint64_t r = 1; base = mod(base);
    while (e) { if (e & 1) r = mult_mod(r, base); base = mult_mod(base, base); e >>= 1; }
    return r;
}
// Σ_{k=1}^{n} seed^k  mod PR, closed form  seed*(seed^n - 1)/(seed - 1).
// This is exactly Σ_i r_i for the r_i = seed^(i+1) weights, computed in O(log n)
// so no serial accumulation of the constant is needed.
inline uint64_t geom_sum(uint64_t seed, uint64_t n) {
    if (n == 0) return 0;
    uint64_t s = mod(seed);
    if (s == 1) return mod(n);
    uint64_t num = mult_mod(s, add_mod(pow_mod(s, n), PR - 1));  // seed*(seed^n - 1)
    return mult_mod(num, pow_mod(mod(s + PR - 1), PR - 2));    // /(seed - 1), Fermat inverse
}
// Montgomery batch inversion: invs[i] = den[i]^{-1} mod PR with ONE mod_inv + ~3n mults
// (forward prefix products, single inversion, backward peel).  Serial chain -- at table
// scale (T <= 2^21) that is ~10 ms; callers with huge n should segment.  A zero denominator
// (only possible in the documented max_val==0 completeness gap, prob ~T/PR) is mapped to 1
// so it cannot poison the batch; the LogUp identity then fails exactly as it did before.
inline void batch_mod_inv(uint64_t *den, uint64_t *invs, std::size_t n) {
    if (n == 0) return;
    for (std::size_t i = 0; i < n; ++i)
        if (den[i] == 0) den[i] = 1;
    invs[0] = den[0];                              // invs doubles as the prefix buffer
    for (std::size_t i = 1; i < n; ++i) invs[i] = mult_mod(invs[i - 1], den[i]);
    uint64_t run = pow_mod(invs[n - 1], PR - 2);   // Fermat: ONE inversion per batch, cost immaterial
    for (std::size_t i = n; i-- > 1;) {
        invs[i] = mult_mod(run, invs[i - 1]);
        run = mult_mod(run, den[i]);
    }
    invs[0] = run;
}
}  // namespace logup_detail

// Direct LogUp lookup argument for VOLE-ZK.
//
// Proves the logarithmic derivative identity (Haboeck 2023, Lemma 5):
//
//   sum_{i=0}^{W-1} 1/(x + witness[i])  =  sum_{j=0}^{T-1} m[j]/(x + table[j])
//
// at a random challenge x sampled by the verifier.
//
// No padding, no sumcheck, no helper functions.
// The witness and table sides can have different sizes.
//
// Three batch-inverse variants live in this file; swap the active call in
// Prove() to benchmark them against each other:
//
//   batch_inverse (Montgomery): 1 ALICE commit + ~3(W+T) committed mults.
//     Fewest commits; observed fastest on emp-zk because ALICE commits each
//     cost a VOLE + per-call transmission while committed mults amortize
//     into a single batched end-of-protocol check.
//
//   batch_inverse_naive: (W+T) ALICE commits + (W+T) committed mults.
//     Commits every inverse directly and checks a_i * inv_i = 1 per slot.
//     Observed slower than Montgomery on emp-zk.
//
//   batch_inverse_v3: same commit/mult profile as naive, but on
//     ALICE's side the T table inverses are computed once in plaintext
//     and witness inverses are fetched from a local cache by witness
//     value. Saves W local mod_exp calls vs. naive; ZK communication is
//     identical.
//
// On top of batch inversion, Prove() adds T committed mults for the
// m[j] * inv[W+j] terms in Step 4 regardless of which variant is used.

// LOGUP_INVBATCH=1 routes every LogUp inverse check  (x+w)*inv == 1  into the shared
// process-global Hadamard2IP (volefier::global_h2ip()) so their RLC challenges batch across ALL
// teardowns -- one seed exchange per collector flush instead of one per teardown.  Opt-in: it
// trades ~dozens of seed round-trips (a WAN-latency win) for one buffered pass + held memory,
// matching the IP_DEFER / EXP_VOLE opt-in flags.  Off by default; the per-teardown inline path
// (own seed, own fp_zkp_inner_prdt) still runs when unset.
inline bool logup_invbatch_on() { static bool on = getenv("LOGUP_INVBATCH") != nullptr; return on; }
inline volefier::Hadamard2IP *logup_h2ip() {
    return logup_invbatch_on() ? volefier::global_h2ip() : nullptr;
}

class LogUpLookup {
public:
    int party;
    IOChannel *io;

    // After Prove(), holds the committed multiplicities for all T table entries.
    // These are the histogram counts.
    std::vector<IntFp> multiplicities;

    LogUpLookup(int party, IOChannel *io) : party(party), io(io) {}

    // Prove the LogUp identity.
    //
    // witnesses:  length W, committed field elements (the encoded lookup keys).
    // table:      length T, committed field elements (the encoded table entries).
    // mult_raw:   length T, ALICE provides histogram counts; BOB passes zeros.
    // max_val:    caller's public upper bound — every witness and table field
    //             value lies in [0, max_val). If max_val > 0 and max_val < PR,
    //             x is sampled from [1, PR - max_val], guaranteeing x + w is
    //             never 0 mod PR (neither as integers nor after wrap-around).
    //             If max_val == 0, caller provides no bound and x is sampled
    //             from the full field — completeness may fail with probability
    //             (W+T)/PR (negligible for PR ≈ 2^61).
    void Prove(const std::vector<IntFp> &witnesses,
               const std::vector<IntFp> &table,
               const std::vector<uint64_t> &mult_raw,
               uint64_t table_sz = 0) {

        std::size_t W = witnesses.size();
        std::size_t T = table.size();

        // Step 1: Prover commits multiplicities -- ONE batched feed (bulk VOLE + one send)
        // straight into the IntFp array, instead of T per-element IntFp(.,ALICE) feeds.
        multiplicities.resize(T);
        { LogUpSpan _sp(LP_MULT, io);
          batch_feed(multiplicities.data(), const_cast<uint64_t *>(mult_raw.data()), (int64_t)T); }

        // Step 2: Verifier samples random challenge x.
        //
        // Bounded case: sample x uniformly from [1, PR - max_val]. Then for
        // any w in [0, max_val): x + w >= 1 (no zero) and x + w <= PR - 1
        // (no wrap-around), so x + w is never 0 mod PR. Soundness range is
        // PR - max_val, essentially PR for max_val << PR.
        //
        // Unbounded case (max_val == 0): sample uniformly from [0, PR).
        uint64_t x_val = 0;
        LogUpSpan *_sp_chal = new LogUpSpan(LP_CHAL, io);
        if (party == ALICE) {
            io->recv_data(&x_val, sizeof(uint64_t));
        } else {
            PRG prg;
            prg.random_data_unaligned(&x_val, sizeof(uint64_t));   // &x_val 8-byte aligned; random_data writes a full 16-byte block -> stack overflow (fatal on some layouts)
            x_val = (x_val % (PR - table_sz)) + 1;  // in [1, PR - max_val]
            io->send_data(&x_val, sizeof(uint64_t)); io->flush();
        }
        delete _sp_chal;
        IntFp x(x_val, PUBLIC);

        // Step 3: Invert all W + T denominators.
        //
        // Witness denominators: x + witness[i]   for i = 0..W-1
        // Table denominators:   x + table[j]     for j = 0..T-1
        // (batch_inverse_v3 forms x+witness / x+table internally; the naive / Montgomery
        // variants would instead take a precomputed `denoms` = {x+witness, x+table}.)
        // Step 3+4 (CHUNKED): prove  sum_i 1/(x+w_i) - sum_j m_j/(x+t_j) == 0  WITHOUT ever holding
        // all W witness inverses at once.  The W-sized buffers (ivw / inv_w / denom_w) were the
        // teardown's peak memory (~5*W*16 B, several GB at transformer scale).  Here they are bounded
        // to one batch B; the running committed `sum` carries the witness contribution across batches.
        // Identical commitments and per-pair (x+w_i)*inv_i==1 checks as the old v3 path, just summed
        // incrementally -- soundness/output unchanged.
        IntFp sum(0ULL, PUBLIC);

        // ALICE pre-inverts the (small) table once; every witness looks up its inverse by value.
        // Montgomery-batched build: ONE mod_inv + ~3T mults instead of T extended-Euclid calls.
        // Dense-domain tables (identity range-check tables: values in [0, T)) additionally use a
        // FLAT ARRAY keyed by value, turning the per-witness unordered_map probe (the dominant
        // cost at tens of millions of witnesses) into an array load.  Sparse (RLC-compressed)
        // tables keep the hash map.  Pure ALICE-side plaintext prep -- nothing on the wire changes.
        std::unordered_map<uint64_t, uint64_t> inv_cache;
        std::vector<uint64_t> inv_flat;   // dense path; 0 = "not a table value" (real inverses are never 0)
        bool flat = false;
        LogUpSpan *_sp_ainv0 = new LogUpSpan(LP_AINV, io);
        if (party == ALICE && T > 0) {
            std::vector<uint64_t> tvals(T), den(T), invs(T);
            uint64_t max_tv = 0;
            for (std::size_t j = 0; j < T; ++j) {
                tvals[j] = static_cast<uint64_t>(VAL(table[j].value));
                if (tvals[j] > max_tv) max_tv = tvals[j];
                den[j] = mod(x_val + tvals[j]);
            }
            logup_detail::batch_mod_inv(den.data(), invs.data(), T);
            flat = (max_tv < (uint64_t)4 * T) && (max_tv < (uint64_t(1) << 27));
            if (flat) {
                inv_flat.assign((std::size_t)max_tv + 1, 0);
                for (std::size_t j = 0; j < T; ++j) inv_flat[tvals[j]] = invs[j];
            } else {
                inv_cache.reserve(T);
                for (std::size_t j = 0; j < T; ++j) inv_cache.emplace(tvals[j], invs[j]);
            }
        }

        delete _sp_ainv0;
        // Fresh post-commitment challenge -> per-pair random weights r_i.  The (x+a)*inv==1 checks are
        // proved by a QuickSilver inner product instead of committing+revealing each product; the random
        // r_i (picked by the verifier AFTER the inverses are committed) make the inner product bind EACH
        // pair, not just their sum.  Both sides expand the seed via the universal-hash coeff generator.
        auto fresh_seed = [&]() -> uint64_t {
            uint64_t sd = 0;
            if (party == ALICE) { io->recv_data(&sd, sizeof(uint64_t)); }
            else { PRG prg; prg.random_data_unaligned(&sd, sizeof(uint64_t)); sd = mod(sd); io->send_data(&sd, sizeof(uint64_t)); io->flush(); }
            return sd;
        };
        // Witness side, in batches of B: commit b inverses, prove (x+w)*inv==1 by inner product, fold into `sum`.
        const std::size_t B = std::size_t(16) << 20; // 16M witnesses / batch
        // Scratch buffers reused across chunks: allocating (and page-faulting) three
        // ~256 MB vectors PER chunk was memory-bound overhead; allocate once, reuse.
        FpOSTriple *ostr = FpPolyProof::fppolyproof->ostriple;
        const int NT = (ostr->threads_ > 1 && ostr->pool_) ? ostr->threads_ : 1;
        // When the global collector is set, the (x+w)*inv==1 checks are DEFERRED to it: this
        // fold then only forms (x+w) and Σ inv_b, and the chi weighting + inner product happen
        // once per collector flush (shared challenge) instead of once per batch here.
        volefier::Hadamard2IP *h2ip = logup_h2ip();
        const bool INVBATCH = h2ip != nullptr;
        std::vector<uint64_t> ivw(B);
        std::vector<IntFp> inv_b, denom_w;                 // reused scratch (non-INVBATCH path only)
        if (!INVBATCH) { inv_b.resize(B); denom_w.resize(B); }
        std::vector<uint64_t> pval((size_t)NT), pmac((size_t)NT), pchi((size_t)NT);
        for (std::size_t off = 0; off < W; off += B) {
            std::size_t b = (W - off < B) ? (W - off) : B;
            // INVBATCH: fresh per-batch columns the collector will OWN (moved in, no copy);
            // else reuse the once-allocated B-sized scratch.
            std::vector<IntFp> inv_own, denom_own;
            if (INVBATCH) { inv_own.resize(b); denom_own.resize(b); }
            IntFp *IB = INVBATCH ? inv_own.data() : inv_b.data();
            IntFp *DW = INVBATCH ? denom_own.data() : denom_w.data();
            { LogUpSpan _sp(LP_AINV, io);
            if (party == ALICE) {
                if (flat)
                    for (std::size_t i = 0; i < b; ++i) {
                        uint64_t wv = static_cast<uint64_t>(VAL(witnesses[off + i].value));
                        ivw[i] = (wv < inv_flat.size() && inv_flat[wv])
                                     ? inv_flat[wv]
                                     : mod_inv(mod(x_val + wv));   // miss = dishonest witness; identity fails anyway
                    }
                else
                    for (std::size_t i = 0; i < b; ++i) {
                        uint64_t wv = static_cast<uint64_t>(VAL(witnesses[off + i].value));
                        auto it = inv_cache.find(wv);
                        ivw[i] = (it != inv_cache.end()) ? it->second : mod_inv(mod(x_val + wv));
                    }
            }
            }
            // Commit the b inverses straight into the IntFp array (IntFp is layout-compatible
            // with __uint128_t).  Same batched feed as IntFpVec (bulk VOLE next_n + one send),
            // but skips the IntFpVec object AND the SoA->AoS decompose copy.
            { LogUpSpan _sp(LP_ICOMMIT, io); batch_feed(IB, ivw.data(), (int64_t)b); }
            // PARALLEL fold: build denom[i] = (x + w_i) [ * r_i when NOT batching], and reduce
            // the witness inverses sum += Σ inv_b[i] -- in ONE threaded pass over the worker
            // pool.  Computed with RAW inline SIMD lane arithmetic (add_mod/mult_mod on the
            // val|mac block) rather than the IntFp operators: those dispatch through ZKFpExec's
            // *virtual* add_gate / mul_const_gate, one non-inlinable indirect call per element --
            // the real cost.  The lane math is identical, all local (no gid/comm/shared state),
            // so it is race-free.  The r_i weights are INDEPENDENT uniforms (rlc_chi.h PRG
            // blocks; eps 1/PR): each worker regenerates the block covering its indices (block
            // layout is segmentation-independent) and accumulates its partial Σ r_i.  When
            // INVBATCH, r_i is skipped here -- the collector applies its own shared chi at flush.
            uint64_t seed = 0;
            if (!INVBATCH) { LogUpSpan _sp(LP_CHAL, io); seed = fresh_seed(); }
            LogUpSpan *_sp_fold = new LogUpSpan(LP_FOLD, io);
            const block xblk = (block)x.value;
            auto fold_work = [&](int tid, std::size_t lo, std::size_t hi) {
                uint64_t sv = 0, sm = 0, sc = 0;
                std::vector<uint64_t> w((size_t)volefier::kChiBlk);
                int64_t cur = -1;
                for (std::size_t i = lo; i < hi; ++i) {
                    block dv = add_mod(xblk, (block)witnesses[off + i].value);
                    if (!INVBATCH) {
                        const int64_t blk = (int64_t)(i / (std::size_t)volefier::kChiBlk);
                        if (blk != cur) { volefier::prg_chi_block(seed, (uint64_t)blk, w.data()); cur = blk; }
                        const uint64_t r = w[i % (std::size_t)volefier::kChiBlk];
                        dv = mult_mod(dv, r);
                        sc = add_mod(sc, r);
                    }
                    DW[i].value = (__uint128_t)dv;
                    __uint128_t v = IB[i].value;
                    sv = add_mod(sv, (uint64_t)(v & 0xFFFFFFFFFFFFFFFFULL));
                    sm = add_mod(sm, (uint64_t)(v >> 64));
                }
                pval[(size_t)tid] = sv; pmac[(size_t)tid] = sm; pchi[(size_t)tid] = sc;
            };
            std::fill(pval.begin(), pval.end(), 0);  // serial path writes only [0]; clear stale
            std::fill(pmac.begin(), pmac.end(), 0);
            std::fill(pchi.begin(), pchi.end(), 0);
            // Small batches: the pool dispatch (future alloc + wakeup ×NT) costs more
            // than the work, so run serial -- mirrors FpPolyProof::par_sum_'s kParMin guard.
            if (NT <= 1 || b < (std::size_t)(1 << 16)) {
                fold_work(0, 0, b);
            } else {
                const std::size_t per = b / (std::size_t)NT;
                std::vector<std::future<void>> fut;
                for (int t = 0; t < NT - 1; ++t) {
                    std::size_t lo = (std::size_t)t * per, hi = lo + per;
                    fut.push_back(ostr->pool_->enqueue([&, t, lo, hi]() { fold_work(t, lo, hi); }));
                }
                fold_work(NT - 1, (std::size_t)(NT - 1) * per, b);
                for (auto &f : fut) f.get();
            }
            _sp_fold->stop(); delete _sp_fold;
            if (INVBATCH) {
                // Hand the committed (x+w) and inverse columns to the shared Hadamard2IP; the
                // shared-challenge chi weighting + inner product fire at its next flush.
                h2ip->add_owned(std::move(denom_own), std::move(inv_own), 1);   // (x+w)*inv == 1
            } else {
                uint64_t cst = 0;                                            // Σ r_i (reduced partials)
                for (int t = 0; t < NT; ++t) cst = add_mod(cst, pchi[(size_t)t]);
                { LogUpSpan _sp(LP_IPRDT, io);
                  fp_zkp_inner_prdt(denom_w.data(), inv_b.data(), cst ? PR - cst : 0, (int64_t)b); }
            }
            // Combine the per-thread partials into the running committed sum (component-wise
            // field add on the val/mac lanes == Σ inv_b[i], then one IntFp add into `sum`).
            // Independent of chi, so it stays here even when the IP is deferred to the collector.
            uint64_t tv = 0, tm = 0;
            for (int t = 0; t < NT; ++t) { tv = add_mod(tv, pval[(size_t)t]); tm = add_mod(tm, pmac[(size_t)t]); }
            IntFp partial; partial.value = (((__uint128_t)tm) << 64) ^ (__uint128_t)tv;
            sum = sum + partial;
            // ivw / inv_own / denom_own freed here (non-INVBATCH) -> witness-side peak stays ~O(B)
        }

        // Table side (T = table size, small -- unchanged): commit T inverses, subtract m_j/(x+t_j).
        {
            LogUpSpan _sp_tab(LP_TABLE, io);
            std::vector<uint64_t> ivt(T, 0);
            if (party == ALICE)
                for (std::size_t j = 0; j < T; ++j) {
                    uint64_t tv = static_cast<uint64_t>(VAL(table[j].value));
                    ivt[j] = flat ? inv_flat[tv] : inv_cache[tv];
                }
            std::vector<IntFp> inv_t_d(T);
            batch_feed(inv_t_d.data(), ivt.data(), (int64_t)T);  // commit straight into IntFp array
            std::vector<IntFp> denom_t(T);
            for (std::size_t j = 0; j < T; ++j) denom_t[j] = x + table[j];
            if (!INVBATCH) {
              std::vector<uint64_t> s(T); volefier::prg_coeff_gen(s.data(), fresh_seed(), (int64_t)T);   // independent chi
              uint64_t cst = 0;
              for (std::size_t j = 0; j < T; ++j) { denom_t[j] = denom_t[j] * s[j]; cst = add_mod(cst, s[j]); }
              fp_zkp_inner_prdt(denom_t.data(), inv_t_d.data(), cst ? PR - cst : 0, (int64_t)T);
            }
            // (INVBATCH: the (x+t)*inv==1 table checks are deferred to the collector below,
            //  AFTER the aggregate copies inv_t_d, so it can then be moved in without a copy.)
            // Aggregate LogUp identity  Σ_i inv[i] = Σ_j m_j·inv_t_d[j]  as ONE committed
            // inner product instead of T committed mul_gates + a reveal.  Append the pair
            // (sum_w, -1): the product proves  Σ_j m_j·inv_t_d[j] + sum_w·(-1) = 0, i.e.
            // ⟨m, inv_t_d⟩ = Σ_i inv[i].  The -1 is a PUBLIC constant (verifier-pinned via
            // pub_label: key = pub_mac - (-1)·Δ), so its value can't be forged and nothing
            // is revealed.  No r-weighting: we bind the aggregate sum, not each term.
            std::vector<IntFp> xs(T + 1), ys(T + 1);
            for (std::size_t j = 0; j < T; ++j) { xs[j] = multiplicities[j]; ys[j] = inv_t_d[j]; }
            xs[T] = sum;                    // committed witness sum  Σ_i inv[i]
            ys[T] = IntFp(PR - 1, PUBLIC);  // public -1
            fp_zkp_inner_prdt(xs.data(), ys.data(), 0, (int64_t)(T + 1));  // asserts Σ = 0
            if (INVBATCH) {
                // ys already copied inv_t_d, so both columns can now be moved into the batcher;
                // its shared-challenge flush proves each (x+t_j)*inv_t_d[j] == 1.
                h2ip->add_owned(std::move(denom_t), std::move(inv_t_d), 1);
            }
        }
    }

private:
    // v3 variant: compute the T table inverses once in plaintext on
    // ALICE's side, cache them in a hashmap keyed on the table value,
    // then look up each witness's inverse by its value. Commits all W + T
    // inverses and checks each with a_i * inv_i == 1, same ZK structure as
    // the naive variant. The savings over naive are purely local compute:
    // for honest lookups, every witness value matches some table value, so
    // its inverse is a cache hit and no mod_inv is recomputed.
    //
    // Works for any table shape (identity, RLC-compressed, arbitrary), since
    // both witness and table inverses are computed against the actual
    // table[j].value rather than the index j.  Cache hit rate stays at ~100%
    // for honest provers even with RLC-compressed tables, because every
    // witness row's compressed value exactly matches the compressed value of
    // the table row it claims to lookup.
    //
    // Cost: (W + T) ALICE commits + W committed mults for the witness-side
    // checks (table-side checks are free — PUBLIC * ALICE). Same as
    // batch_inverse_naive; benchmark vs. Montgomery is expected to track the
    // naive variant since the ZK communication profile is identical.
    std::vector<IntFp> batch_inverse_v3(
            const std::vector<IntFp> &witnesses,
            const std::vector<IntFp> &table,
            const IntFp &x) {
        std::size_t W = witnesses.size();
        std::size_t T = table.size();
        uint64_t x_val = static_cast<uint64_t>(VAL(x.value));

        // Local plaintext cache keyed on table value: inv_cache[t] = 1/(x+t).
        // ALICE only; BOB doesn't see plaintext values.
        std::unordered_map<uint64_t, uint64_t> inv_cache;
        if (party == ALICE) {
            inv_cache.reserve(T);
            for (std::size_t j = 0; j < T; ++j) {
                uint64_t tv = static_cast<uint64_t>(VAL(table[j].value));
                if (inv_cache.find(tv) == inv_cache.end())
                    inv_cache[tv] = mod_inv(mod(x_val + tv));
            }
        }

        std::vector<IntFp> inv(W + T);

        // Witness inverses: cache hit when wv matches some table value
        // (the honest case). Fall back to mod_inv on miss — happens for
        // dishonest provers, or for legitimately distinct values that
        // weren't pre-cached.
        // VECTORIZED witness-inverse check: commit all W inverses in ONE batched feed, prove the
        // W products (x+w_i)*inv_i with ONE batched/threaded mul_gate, and ONE batched zero-check
        // -- replacing W serial feeds + W mul_gates + W reveals (the teardown's hot loop).
        {
            std::vector<uint64_t> ivw((size_t)W, 0);
            if (party == ALICE)
                for (std::size_t i = 0; i < W; ++i) {
                    uint64_t wv = static_cast<uint64_t>(VAL(witnesses[i].value));
                    auto it = inv_cache.find(wv);
                    ivw[i] = (it != inv_cache.end()) ? it->second : mod_inv(mod(x_val + wv));
                }
            IntFpVec inv_w(ivw.data(), (int64_t)W, ALICE);                          // one feed
            std::vector<IntFp> denom_w((size_t)W);
            // denom = x + witness via the IntFp `x` (public on BOTH parties). NOTE: do NOT use
            // IntFpVec::operator+(x_val) here -- x_val = VAL(x.value) is the challenge on the prover
            // but 0 on the verifier, so the broadcast would desync. operator+ itself is correct.
            for (std::size_t i = 0; i < W; ++i) denom_w[i] = x + witnesses[i];
            IntFpVec chk = IntFpVec::compose(denom_w) * inv_w - 1ULL;
            chk.reveal_check_zero();                                                // one batched zero-check
            inv_w.decompose(&inv[0]);                                               // inv[0..W) = committed inverses
        }

        // Table inverses: cache always hits since it was built from
        // the same table values. Uses table[j].value, NOT j, so non-
        // identity tables (RLC-compressed, irregular) are correct.
        // VECTORIZED table-inverse check: one batched feed for all T inverses, one batched/threaded
        // mul_gate for the (x+t_j)*inv check, one batched zero-check (T = table size, up to ~2^21).
        {
            std::vector<uint64_t> ivt((size_t)T, 0);
            if (party == ALICE)
                for (std::size_t j = 0; j < T; ++j)
                    ivt[j] = inv_cache[static_cast<uint64_t>(VAL(table[j].value))];
            IntFpVec inv_t(ivt.data(), (int64_t)T, ALICE);
            std::vector<IntFp> denom_t((size_t)T);
            for (std::size_t j = 0; j < T; ++j) denom_t[j] = x + table[j];
            IntFpVec chk = IntFpVec::compose(denom_t) * inv_t - 1ULL;   // NOTE: IntFpVec::operator+(uint64_t)
            chk.reveal_check_zero();                                    // desyncs the mult-gate order, so we
            inv_t.decompose(&inv[W]);                                   // form denom via scalar add, not + x.
        }

        return inv;
    }

    // Naive batch inversion: ALICE commits each 1/a_i directly, and each
    // pair is checked with a single committed mult a_i * inv_i == 1.
    //
    // Cost: n ALICE commitments + n committed mults. Benchmarks put this
    // noticeably slower than the Montgomery variant above on emp-zk: the
    // n VOLE-consuming commits dominate, whereas Montgomery's extra mults
    // amortize into emp-zk's batched end-of-protocol check. Kept here for
    // reference / future benchmarking if the cost ratio ever changes.
    std::vector<IntFp> batch_inverse_naive(const std::vector<IntFp> &vals) {
        std::size_t n = vals.size();
        std::vector<IntFp> inv(n);
        for (std::size_t i = 0; i < n; ++i) {
            uint64_t iv = 0;
            if (party == ALICE) {
                uint64_t v = static_cast<uint64_t>(VAL(vals[i].value));
                //iv = mod_exp(v, PR - 2);
                iv = mod_inv(v);
            }
            inv[i] = IntFp(iv, ALICE);
            IntFp check = vals[i] * inv[i] + IntFp(1ULL, PUBLIC).negate();
            check.reveal_zero();
        }
        return inv;
    }

    // Montgomery batch inversion of committed field elements.
    //
    // Given [a_0, ..., a_{n-1}], returns [1/a_0, ..., 1/a_{n-1}].
    //
    // 1. Forward pass: prefix[i] = a_0 * ... * a_{i-1}
    // 2. ALICE commits inverse of total product, verified via product * inv = 1.
    // 3. Backward pass: inv[i] = prefix[i] * running_inv,
    //    then running_inv *= a[i].
    //
    // Cost: 3*(n-1) committed multiplications + 1 ALICE commitment.
    std::vector<IntFp> batch_inverse(const std::vector<IntFp> &vals) {
        std::size_t n = vals.size();
        if (n == 0) return {};

        // Forward pass
        std::vector<IntFp> prefix(n + 1);
        prefix[0] = IntFp(1ULL, PUBLIC);
        for (std::size_t i = 0; i < n; ++i)
            prefix[i + 1] = prefix[i] * vals[i];

        // ALICE commits inverse of total product
        //
        // To benchmark the two modular-inverse methods, swap the two lines
        // below.  Extended Euclidean is typically ~2-3x faster than Fermat
        // for this field size, but on specific hardware / compiler the gap
        // may differ.
        IntFp total_inv;
        if (party == ALICE) {
            uint64_t pval = static_cast<uint64_t>(VAL(prefix[n].value));
            uint64_t ival = mod_inv(pval);              // extended Euclidean
            //uint64_t ival = mod_exp(pval, PR - 2);   // Fermat's little theorem
            total_inv = IntFp(ival, ALICE);
        } else {
            total_inv = IntFp(0ULL, ALICE);
        }

        // Verify: prefix[n] * total_inv = 1
        IntFp check = prefix[n] * total_inv + IntFp(1ULL, PUBLIC).negate();
        check.reveal_zero();

        // Backward pass
        std::vector<IntFp> inv(n);
        IntFp running = total_inv;
        for (std::size_t i = n; i > 0; --i) {
            inv[i - 1] = prefix[i - 1] * running;
            running = running * vals[i - 1];
        }
        return inv;
    }

    // Modular inverse via extended Euclidean algorithm (from SCL).
    // Computes a^{-1} mod PR.
    static uint64_t mod_inv(const uint64_t a) {
        if (a == 0) error("zero not invertible\n");

        int64_t k = 0;
        int64_t new_k = 1;
        int64_t r = (int64_t)PR;
        int64_t new_r = (int64_t)a;

        while (new_r != 0) {
            int64_t q = r / new_r;
            int64_t tmp = new_k;
            new_k = k - (__int128)q * tmp;
            k = tmp;
            tmp = new_r;
            new_r = r - (__int128)q * tmp;
            r = tmp;
        }

        if (k < 0) k = k + (int64_t)PR;
        return (uint64_t)k;
    }

    // Modular exponentiation: base^exp mod PR, via square-and-multiply.
    // Used for modular inverse via Fermat's little theorem: a^{-1} = a^{PR-2}.
    // Kept for benchmarking against mod_inv (extended Euclidean).
    static uint64_t mod_exp(uint64_t base, uint64_t exp) {
        uint64_t result = 1, b = mod(base);
        while (exp > 0) {
            if (exp & 1) result = mult_mod(result, b);   // Mersenne reduction, no 128-bit division
            b = mult_mod(b, b);
            exp >>= 1;
        }
        return result;
    }
};


// LogUpRangeCheck: drop-in replacement for a ZKROM with an identity table.
//
// The "table" is the public range [0, N).  All accesses are buffered
// during the Access() phase, and at Teardown() a single LogUp proof
// verifies that every accessed value lies in [0, N).
//
// Since the table is identity (table[i] = i), Access(id) returns id
// unchanged — the same IntFp object, so downstream committed arithmetic
// is unaffected.  This makes it a drop-in replacement for patterns like
//   reads[i] = rom->Access(com_indices[i]);
// where the ROM was only serving as a range check.
//
// Usage:
//   LogUpRangeCheck rc(N, party, io);
//   rc.Setup();
//   for each access: IntFp v = rc.Access(x);  // returns x
//   rc.Teardown();
//
// Automatic batching: when the witness buffer reaches `batch_sz`, Access()
// runs Teardown() inline, producing one self-contained LogUp proof for the
// batch and resetting state.  This mirrors ZKROM's behavior in rom.h and
// keeps the per-call work pipelined with the rest of the iteration's proof
// traffic, instead of leaving one giant proof for process exit.  The
// conjunction of independent batch proofs is itself a sound range check.
class LogUpRangeCheck {
public:
    int party;
    IOChannel *io;
    uint64_t N;
    uint64_t batch_sz;  // auto-teardown threshold

    // Buffered access values and per-table-entry counts.
    std::vector<IntFp> witnesses;
    std::vector<uint64_t> mult_raw;

    LogUpRangeCheck(uint64_t N_, int party_, IOChannel *io_,
                    uint64_t batch_sz_ = 16 * 1024 * 1024)
        : party(party_), io(io_), N(N_), batch_sz(batch_sz_) {}

    void Setup() {
        mult_raw.assign(N, 0ULL);
        witnesses.clear();
    }

    // Buffer id and return id itself (identity lookup).  Callers that
    // used to write `reads[i] = rom->Access(x)` can keep the same line.
    // When the buffer reaches batch_sz we run a partial Teardown so the
    // proof work spreads across the run rather than piling up at exit.
    IntFp Access(IntFp& id) {
        static const bool LOFF = getenv("LOGUP_OFF") != nullptr;
        if (LOFF) return id;   // measurement-only: skip buffering + histogram (symmetric)
        witnesses.push_back(id);
        if (party == ALICE) {
            uint64_t v = static_cast<uint64_t>(VAL(id.value));
            if (v < N) mult_raw[v]++;
        }
        if (witnesses.size() >= batch_sz) {
            Teardown();
        }
        return id;
    }

    // Run the LogUp proof that all currently-buffered accesses lie in
    // [0, N), then clear the buffer so the object can be reused.  Safe
    // to call when the buffer is empty (becomes a no-op) — this lets a
    // caller's final Teardown() still work even if the last Access()
    // already auto-tore down.
    void Teardown() {
        if (witnesses.empty()) return;
        static const bool LPROF = getenv("LOGUP_PROF") != nullptr;
        auto _t0 = std::chrono::steady_clock::now();
        double _c0 = LPROF ? logup_cpu_self() : 0, _h0 = LPROF ? logup_cpu_thr() : 0;
        unsigned long long _r0 = LPROF ? static_cast<NetIO *>(io)->rounds : 0ULL;

        std::vector<IntFp> table(N);
        for (uint64_t i = 0; i < N; ++i)
            table[i] = IntFp(i, PUBLIC);

        LogUpLookup logup(party, io);
        // All table and witness values are in [0, N) by construction (table
        // is {0, ..., N-1} and witnesses are range-checked against it), so
        // x is sampled from [1, PR - N] and zero denominators are impossible.
        logup.Prove(witnesses, table, mult_raw, N);

        witnesses.clear();
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
