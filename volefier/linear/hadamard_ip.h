#ifndef EMP_ZKML_HADAMARD_IP_H
#define EMP_ZKML_HADAMARD_IP_H

#include "emp-zk/emp-zk.h"
#include "volefier/rlc_chi.h"
#include <vector>
#include <cstdint>
#include <future>

using namespace emp;   // IntFp / IOChannel / PRG / mod / block / FpPolyProof / fp_zkp_inner_prdt

namespace volefier {

// Hadamard2IP: the ONE collector for every "prove a ∘ b == c element-wise" claim -- the
// RLC-then-IP conversions where N INDEPENDENT constraints (each a_i*b_i = c) must be batched
// with random coefficients or they cancel:
//   * bitness   g(g-1)   == 0   (comparison.h is_le / select_max)
//   * inverse   (x+w)*inv == 1   (logup.h LogUp fold + table -- the dominant population)
// Distinct from the AGGREGATE inner products (matmul, LayerNorm sum-of-squares) which need no
// per-element RLC and go straight to their IP.
//
// Each add() buffers one (a,b,target) column as a segment.  A flush draws ONE shared challenge
// for the WHOLE buffered batch (chi index continuous across segments), converts every element to
// a weighted term  chi_k * a_k, and hands each segment's inner product
//   sum_k chi_k a_k b_k  -  target * sum_k chi_k  = 0
// to fp_zkp_inner_prdt, so it rides FpPolyProof's single deferred batch_check.  So instead of one
// seed exchange PER conversion site (per LogUp teardown, per comparison), there is one seed
// exchange per FLUSH (buffer-cap bounded) -- a WAN-latency win.
//
// chi_k are PRG-independent (rlc_chi.h): a lying element survives only if the single random
// linear combination vanishes -- prob ~ 1/PR, independent of N.
//
// Two ingest paths keep the hot path cheap:
//   add_eq / add_zero : COPY borrowed columns (small claims: bitness, per-op).
//   add_owned         : MOVE owned columns (no copy: LogUp's 16M-element batches).
// The flush weights every segment with RAW SIMD lane arithmetic across the worker pool (mirrors
// LogUp's fold), so 100M+ elements are not paid as per-element virtual mul_const dispatches.
class Hadamard2IP {
public:
    int party;
    IOChannel *io;
    std::size_t buffer_sz;            // element cap before an auto-flush (bounds held memory)
    std::size_t held = 0;
    struct Seg { std::vector<IntFp> a, b; uint64_t tgt; };  // a_i*b_i == tgt for all i
    std::vector<Seg> segs_;

    Hadamard2IP(int party_, IOChannel *io_, std::size_t cap = (std::size_t)32 << 20)
        : party(party_), io(io_), buffer_sz(cap) {}
    ~Hadamard2IP() { flush(); }

    // Register  a[i]*b[i] == 0  (bitness g(g-1)=0, selector u(1-u)=0).  Copies the columns.
    void add_zero(const IntFp *a, const IntFp *b, std::size_t n) { add_eq(a, b, 0, n); }

    // Register  a[i]*b[i] == c  (public constant target).  Copies the columns.
    void add_eq(const IntFp *a, const IntFp *b, uint64_t c, std::size_t n) {
        if (n == 0) return;
        if (held && held + n > buffer_sz) flush();
        segs_.push_back(Seg{std::vector<IntFp>(a, a + n), std::vector<IntFp>(b, b + n), c});
        held += n;
        if (held >= buffer_sz) flush();
    }

    // Register  a[i]*b[i] == c  taking OWNERSHIP of the columns (no copy) -- the hot path.
    // c=1 is the LogUp inverse check  (x+w)*inv == 1.
    void add_owned(std::vector<IntFp> &&a, std::vector<IntFp> &&b, uint64_t c) {
        const std::size_t n = a.size();
        if (n == 0) return;
        if (held && held + n > buffer_sz) flush();
        segs_.push_back(Seg{std::move(a), std::move(b), c});
        held += n;
        if (held >= buffer_sz) flush();
    }

    // One shared challenge for the whole buffered batch; chi index runs CONTINUOUSLY across the
    // concatenated segments (block layout is segmentation-independent, so each segment weights
    // itself with a base offset).  Each segment -> fp_zkp_inner_prdt asserting
    //   sum chi (a b) - tgt*(sum chi) = 0.
    void flush() {
        if (held == 0) return;
        uint64_t seed = 0;                 // post-commitment: every buffered a/b is already committed
        io->flush();
        if (party == ALICE) {
            io->recv_data(&seed, sizeof(uint64_t));
        } else {
            PRG prg;
            prg.random_data_unaligned(&seed, sizeof(uint64_t));
            seed = mod(seed);
            io->send_data(&seed, sizeof(uint64_t));
            io->flush();
        }
        FpOSTriple *ostr = FpPolyProof::fppolyproof->ostriple;
        const int NT = (ostr->threads_ > 1 && ostr->pool_) ? ostr->threads_ : 1;
        std::size_t base = 0;              // global chi index at the current segment's start
        std::vector<uint64_t> pchi((size_t)NT);
        for (Seg &sg : segs_) {
            const std::size_t n = sg.a.size();
            IntFp *A = sg.a.data();
            IntFp *B = sg.b.data();
            // A[i] *= chi_{base+i}  (raw SIMD, mul_const on the val|mac block) and reduce sum chi.
            auto chi_work = [&](int tid, std::size_t lo, std::size_t hi) {
                uint64_t sc = 0;
                std::vector<uint64_t> w((size_t)kChiBlk);
                int64_t cur = -1;
                for (std::size_t i = lo; i < hi; ++i) {
                    const std::size_t g = base + i;
                    const int64_t blk = (int64_t)(g / (std::size_t)kChiBlk);
                    if (blk != cur) { prg_chi_block(seed, (uint64_t)blk, w.data()); cur = blk; }
                    const uint64_t r = w[g % (std::size_t)kChiBlk];
                    A[i].value = (__uint128_t)mult_mod((block)A[i].value, r);
                    sc = add_mod(sc, r);
                }
                pchi[(size_t)tid] = sc;
            };
            std::fill(pchi.begin(), pchi.end(), 0);
            if (NT <= 1 || n < (std::size_t)(1 << 16)) {
                chi_work(0, 0, n);
            } else {
                const std::size_t per = n / (std::size_t)NT;
                std::vector<std::future<void>> fut;
                for (int t = 0; t < NT - 1; ++t) {
                    std::size_t lo = (std::size_t)t * per, hi = lo + per;
                    fut.push_back(ostr->pool_->enqueue([&, t, lo, hi]() { chi_work(t, lo, hi); }));
                }
                chi_work(NT - 1, (std::size_t)(NT - 1) * per, n);
                for (auto &f : fut) f.get();
            }
            uint64_t schi = 0;
            for (int t = 0; t < NT; ++t) schi = add_mod(schi, pchi[(size_t)t]);
            uint64_t cst = sg.tgt ? mult_mod(mod(sg.tgt), schi) : 0;   // tgt * sum chi
            fp_zkp_inner_prdt(A, B, cst ? PR - cst : 0, (int64_t)n);   // sum chi a b - cst = 0
            base += n;
        }
        segs_.clear();
        held = 0;
    }
};

// Process-global batcher: created by the inference driver, shared by every RLC-then-IP site
// (comparison bitness + LogUp inverse checks), flushed once in the driver's destroy() before
// finalize_zk_arith() runs FpPolyProof's batch_check.
inline Hadamard2IP *&global_h2ip() { static Hadamard2IP *p = nullptr; return p; }

}  // namespace volefier

#endif
