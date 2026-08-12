#ifndef ML_IP_MATMUL_H
#define ML_IP_MATMUL_H

#include "emp-zk/emp-zk.h"
#include "volefier/rlc_chi.h"
#include <vector>
#include <cstdint>
#include <thread>
#include <functional>
#include <algorithm>
using namespace emp;

// Standalone QuickSilver inner-product proof + matmul, reimplementing emp-zk's
// FpPolyProof::zkp_inner_prdt WITHOUT the global singleton, so that multiple
// instances can run on separate threads / io channels (the emp-zk singleton is
// the one thing blocking a multi-threaded matmul).
//
// The inner-product trick: to prove committed z = sum_i a_i*b_i, prove the
// augmented inner product  [a..., -1] . [b..., z] = 0  (public constant 0).
// Each inner product is one batched-check entry (no per-product commitment),
// so matmul communication is O(#outputs), not O(#products).
//
// IntFp.value packs (value<<64 | MAC) for ALICE and (.. | key) for BOB; we read
// HIGH64/LOW64 exactly as FpPolyProof does, so the MAC algebra is identical.

namespace volefier {

static inline FpOSTriple *get_ostriple(int party) {
  if (party == ALICE)
    return ((ZKFpExecPrv *)ZKFpExec::zk_exec)->ostriple;
  return ((ZKFpExecVer *)ZKFpExec::zk_exec)->ostriple;
}

template <class Fn>
inline void parallel_for(std::size_t n, int nthreads, Fn &&fn) {
  if (nthreads <= 1) { for (std::size_t j = 0; j < n; ++j) fn(j); return; }
  std::vector<std::thread> ths;
  std::size_t chunk = (n + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    std::size_t lo = (std::size_t)t * chunk, hi = std::min(n, lo + chunk);
    if (lo >= hi) break;
    ths.emplace_back([&fn, lo, hi] { for (std::size_t j = lo; j < hi; ++j) fn(j); });
  }
  for (auto &th : ths) th.join();
}

template <typename IO> class IPProof {
public:
  int party;
  IO *io;
  FpOSTriple *ot;   // shared VOLE (ZK mask) + delta
  uint64_t delta = 0;
  int buffer_sz = 1 << 22, num = 0;   // 4194304
  std::vector<uint64_t> buf, buf1;   // ALICE: (A0, A1) per entry; BOB: B in buf

  // IP_DEFER=1: skip the per-matmul eager batch_check and let claims from MANY ops share one
  // challenge (flush at the buffer cap + one explicit batch_check at the end of the graph run).
  // Sound because every buffered entry snapshots the committed MACs, and chi is drawn AFTER all
  // of them -- the same rationale as the existing cap auto-flush, which already mixes ops.
  bool defer_ = false;

  // FV_DEFER (default ON; both parties must agree): defer the per-matmul Freivalds CHALLENGE.  Each
  // matmul_mm_freivalds call still commits its outputs immediately, but instead of exchanging a
  // seed right away it queues an owned snapshot of (kernel, cols, C); one seed exchange then
  // serves the whole queue, with per-job INDEPENDENT u/v expanded from (seed, job index).  Sound
  // for the same reason as IP_DEFER: every queued job's operands and outputs are committed before
  // the group seed exists.  Jobs own copies (the call-site operands are transients), so a byte
  // cap (FV_BUF_MB, default 256) bounds the extra memory by flushing early.
  struct FvJob {
    std::vector<std::vector<IntFp>> kernel, cols;
    std::vector<IntFp> C;
  };
  std::vector<FvJob> fv_q_;
  std::size_t fv_bytes_ = 0, fv_cap_ = 0;
  bool fv_defer_ = false, in_fv_flush_ = false;
  int fv_threads_ = 1;

  IPProof(int party_, IO *io_, FpOSTriple *ot_)
      : party(party_), io(io_), ot(ot_), buf(buffer_sz), buf1(buffer_sz) {
    if (party == BOB) delta = LOW64(ot->delta);
    defer_ = getenv("IP_DEFER") != nullptr;
    fv_defer_ = !getenv("FV_DEFER") || atoi(getenv("FV_DEFER")) != 0;   // DEFAULT ON (-7.3s @500Mbps/60ms); FV_DEFER=0 disables
    fv_cap_ = (getenv("FV_BUF_MB") ? (std::size_t)atoll(getenv("FV_BUF_MB")) : 256) << 20;
    if (const char *e = getenv("IP_BUF")) { buffer_sz = atoi(e); buf.resize(buffer_sz); buf1.resize(buffer_sz); }
  }
  ~IPProof() { fv_flush(); batch_check(); }

  // Ensure `need` free slots starting at `num`: flush the pending claims if they would not fit,
  // grow the buffer if `need` alone exceeds it.  Returns the base slot for the caller to fill.
  std::size_t reserve(std::size_t need) {
    if (num && (std::size_t)num + need > (std::size_t)buffer_sz) batch_check();
    if (need > (std::size_t)buffer_sz) buffer_sz = (int)need;
    if (buf.size() < (std::size_t)buffer_sz) { buf.resize(buffer_sz); buf1.resize(buffer_sz); }
    return (std::size_t)num;
  }

  // Prove and return committed z = sum_i a_i*b_i.
  IntFp inner_product(const std::vector<IntFp> &a,
                           const std::vector<IntFp> &b) {
    const std::size_t m = a.size();
    IntFp z;
    if (party == ALICE) {
      uint64_t acc = 0;
      for (std::size_t i = 0; i < m; ++i)
        acc = add_mod(acc, mult_mod(static_cast<uint64_t>(VAL(a[i].value)),
                                    static_cast<uint64_t>(VAL(b[i].value))));
      z = IntFp(acc, ALICE);
    } else {
      z = IntFp(static_cast<uint64_t>(0), ALICE);
    }
    std::vector<IntFp> x(m + 1), y(m + 1);
    for (std::size_t i = 0; i < m; ++i) { x[i] = a[i]; y[i] = b[i]; }
    x[m] = IntFp(PR - 1, PUBLIC);   // -1
    y[m] = z;
    accumulate(x.data(), y.data(), /*constant=*/0, (int)(m + 1));
    return z;
  }

  // Accumulate one inner-product claim sum_i px_i*py_i = -constant into the
  // batched check (mirrors FpPolyProof::zkp_inner_prdt).
  void accumulate(const IntFp *px, const IntFp *py, uint64_t constant, int len) {
    if (num >= buffer_sz) batch_check();
    if (party == ALICE) {
      uint64_t A0 = 0, A1 = 0;
      for (int i = 0; i < len; ++i) {
        uint64_t w0 = VAL(px[i].value), m0 = MAC(px[i].value);
        uint64_t w1 = VAL(py[i].value), m1 = MAC(py[i].value);
        A0 = add_mod(A0, mult_mod(m0, m1));
        A1 = add_mod(A1, add_mod(mult_mod(m0, w1), mult_mod(m1, w0)));
      }
      buf[num] = A0; buf1[num] = A1;
    } else {
      uint64_t B = 0;
      for (int i = 0; i < len; ++i)
        B = add_mod(B, mult_mod(MAC(px[i].value), MAC(py[i].value)));
      B = add_mod(B, mult_mod(constant, mult_mod(delta, delta)));
      buf[num] = B;
    }
    num++;
  }

  // Random-linear-combination check of all buffered claims (mirrors
  // FpPolyProof::batch_check). Uses one fresh VOLE correlation as the ZK mask.
  void batch_check() {
    if (!in_fv_flush_) fv_flush();   // pending FV jobs' claims must be covered by this challenge
    if (num == 0) return;
    uint64_t seed;
    io->flush();
    std::vector<uint64_t> chi(num);
    __uint128_t ope_data;
    uint64_t check_sum[2];
    if (party == ALICE) {
      io->recv_data(&seed, sizeof(uint64_t));
      prg_coeff_gen(chi.data(), seed, num);   // independent chi (rlc_chi.h): eps 1/PR vs num/PR
      check_sum[0] = vector_inn_prdt_sum_red(chi.data(), buf.data(), num);
      check_sum[1] = vector_inn_prdt_sum_red(chi.data(), buf1.data(), num);
      ot->draw_vole((AuthValueFp *)&ope_data, 1);   // unified draw: FIFO from bg pipe in background mode (was vole->next_n -> raced the producer)
      check_sum[0] = add_mod(check_sum[0], MAC(ope_data));
      check_sum[1] = add_mod(check_sum[1], VAL(ope_data));
      io->send_data(check_sum, 2 * sizeof(uint64_t));
    } else {
      PRG prg;
      prg.random_data_unaligned(&seed, sizeof(uint64_t));   // &seed is 8-byte aligned; random_data needs 16 (writes a full block -> overflows)
      seed = mod(seed);
      io->send_data(&seed, sizeof(uint64_t));
      prg_coeff_gen(chi.data(), seed, num);   // independent chi (rlc_chi.h): eps 1/PR vs num/PR
      uint64_t B = vector_inn_prdt_sum_red(chi.data(), buf.data(), num);
      ot->draw_vole((AuthValueFp *)&ope_data, 1);   // unified draw: FIFO from bg pipe in background mode (was vole->next_n -> raced the producer)
      B = add_mod(B, MAC(ope_data));
      io->recv_data(check_sum, 2 * sizeof(uint64_t));
      uint64_t tmp = add_mod(B, mult_mod(check_sum[1], delta));
      if (tmp != check_sum[0]) error("standalone IP zkp fails");
    }
    num = 0;
  }

  // FUSED phase-1: in ONE pass over the operands, compute the cleartext inner product `acc`
  // (returned for ALICE; used to commit z) AND the partial check coefficients over the m REAL
  // elements (A0,A1 for ALICE; B for BOB) into slot `slot`.  Bbase[i*stride] is the second
  // operand (stride=1 contiguous, stride=N for an attention column).  The augmented (-1, z) term
  // is deferred to add_z_term once z is committed, so each operand cell is read exactly once
  // (vs the old acc pass + compute_entry pass reading every cell twice).
  uint64_t compute_entry_fused(std::size_t slot, const IntFp *Arow, const IntFp *Bbase,
                               std::size_t stride, int m) {
    if (party == ALICE) {
      uint64_t acc = 0, A0 = 0, A1 = 0;
      for (int i = 0; i < m; ++i) {
        uint64_t w0 = VAL(Arow[i].value), m0 = MAC(Arow[i].value);
        const __uint128_t &bv = Bbase[(std::size_t)i * stride].value;
        uint64_t w1 = VAL(bv), m1 = MAC(bv);
        acc = add_mod(acc, mult_mod(w0, w1));
        A0 = add_mod(A0, mult_mod(m0, m1));
        A1 = add_mod(A1, add_mod(mult_mod(m0, w1), mult_mod(m1, w0)));
      }
      buf[slot] = A0; buf1[slot] = A1;
      return acc;
    }
    uint64_t B = 0;
    for (int i = 0; i < m; ++i)
      B = add_mod(B, mult_mod(MAC(Arow[i].value), MAC(Bbase[(std::size_t)i * stride].value)));
    buf[slot] = B;
    return 0;
  }

  // Add the augmented z-term  (px = -1, py = z)  -- the i=m element of  [W,-1].[x,z]  -- to slot's
  // partial coefficients.  O(1) per output, run after z is committed.  This is exactly the last
  // iteration of the original augmented loop, so splitting it off is value-identical (constant 0).
  void add_z_term(std::size_t slot, const IntFp &neg1, const IntFp &z) {
    if (party == ALICE) {
      uint64_t w0 = VAL(neg1.value), m0 = MAC(neg1.value);
      uint64_t w1 = VAL(z.value), m1 = MAC(z.value);
      buf[slot] = add_mod(buf[slot], mult_mod(m0, m1));
      buf1[slot] = add_mod(buf1[slot], add_mod(mult_mod(m0, w1), mult_mod(m1, w0)));
    } else {
      buf[slot] = add_mod(buf[slot], mult_mod(MAC(neg1.value), MAC(z.value)));
    }
  }

  // Threaded matmul: out[j] = W[j].x, proven by ONE batch-check at the end.
  //   phase 1: ALICE computes acc[j] locally (parallel); commit all z (one feed)
  //   phase 2: compute A0/A1/B for every output locally (parallel)  <-- the heavy part
  //   phase 3: a single batch_check over all entries (one network round)
  // Only phases that touch the network (the feed and the batch_check) are
  // serial / single-io; the O(n*m) field arithmetic is fully parallel.
  void matmul(std::vector<IntFp> &out, const std::vector<std::vector<IntFp>> &W,
              const std::vector<IntFp> &x, int nthreads = 1) {
    const std::size_t n = W.size(), m = x.size();
    std::vector<uint64_t> acc_raw(n, 0);
    const std::size_t base = reserve(n);
    parallel_for(n, nthreads, [&](std::size_t j) {           // fused: acc + partial coeffs, one pass
      acc_raw[j] = compute_entry_fused(base + j, W[j].data(), x.data(), 1, (int)m);
    });
    out.resize(n);
    batch_feed(out.data(), acc_raw.data(), (int)n);          // commit all z (one comm)
    IntFp neg1(PR - 1, PUBLIC);
    parallel_for(n, nthreads, [&](std::size_t j) { add_z_term(base + j, neg1, out[j]); });  // O(1) z-term
    num += (int)n;
    if (!defer_) batch_check();                              // one network round (deferred under IP_DEFER)
  }

  // Flat-input overload: W is row-major in `Wflat` (row j = Wflat[j*D .. j*D+D)).
  // Identical proof to the nested version -- reads the committed weight cells IN PLACE
  // (no array-of-rows copy), so a caller holding a flat committed table (e.g. wte) can
  // prove out[j] = <row_j, x> without materializing a std::vector<std::vector<IntFp>>.
  void matmul(std::vector<IntFp> &out, const std::vector<IntFp> &Wflat, std::size_t n,
              std::size_t D, const std::vector<IntFp> &x, int nthreads = 1) {
    const std::size_t m = x.size();
    std::vector<uint64_t> acc_raw(n, 0);
    const std::size_t base = reserve(n);
    parallel_for(n, nthreads, [&](std::size_t j) {           // fused, in place (no row copy)
      acc_raw[j] = compute_entry_fused(base + j, &Wflat[j * D], x.data(), 1, (int)m);
    });
    out.resize(n);
    batch_feed(out.data(), acc_raw.data(), (int)n);          // commit all z (one comm)
    IntFp neg1(PR - 1, PUBLIC);
    parallel_for(n, nthreads, [&](std::size_t j) { add_z_term(base + j, neg1, out[j]); });
    num += (int)n;
    if (!defer_) batch_check();                              // one network round (deferred under IP_DEFER)
  }

  // Batched matrix-matrix IP proof: out[co][p] = <kernel[co], cols[p]> for ALL
  // (co,p), with ONE batch_feed + ONE batch_check (vs one pair per output channel).
  // This is the whole conv layer as a single proof -- the O(Cout*P*M) field work is
  // fully threaded, and the two network rounds are paid ONCE per layer, not Cout times.
  void matmul_mm(std::vector<std::vector<IntFp>> &out,           // [Cout][P]
                 const std::vector<std::vector<IntFp>> &cols,    // [P][M]
                 const std::vector<std::vector<IntFp>> &kernel,  // [Cout][M]
                 int nthreads = 1) {
    const std::size_t Cout = kernel.size(), Pn = cols.size();
    const std::size_t m = Cout ? kernel[0].size() : 0;
    const std::size_t N = Cout * Pn;                            // flat idx = co*Pn + p
    std::vector<uint64_t> acc_raw(N, 0);
    const std::size_t base = reserve(N);
    parallel_for(N, nthreads, [&](std::size_t idx) {            // fused: acc + partial coeffs
      const std::size_t co = idx / Pn, p = idx % Pn;
      acc_raw[idx] = compute_entry_fused(base + idx, kernel[co].data(), cols[p].data(), 1, (int)m);
    });
    std::vector<IntFp> z(N);
    batch_feed(z.data(), acc_raw.data(), (int)N);              // commit all Cout*P z (one comm)
    IntFp neg1(PR - 1, PUBLIC);
    parallel_for(N, nthreads, [&](std::size_t idx) { add_z_term(base + idx, neg1, z[idx]); });
    num += (int)N;
    if (!defer_) batch_check();                               // one round per layer (deferred under IP_DEFER)
    out.assign(Cout, std::vector<IntFp>(Pn));
    for (std::size_t co = 0; co < Cout; ++co)
      for (std::size_t p = 0; p < Pn; ++p) out[co][p] = z[co * Pn + p];
  }

  // ---- Freivalds matmul (Mystique §3.3) ------------------------------------------------------
  // Verifier samples a 16-byte seed AFTER the matrices are committed and sends it; both parties
  // expand it (same PRG) into nu+nv uniform F_p challenges. Public-coin: the prover learning u,v
  // is fine because the inputs/outputs are already MAC-bound; soundness needs only seed-after-commit.
  void gen_challenge(std::vector<uint64_t> &u, int nu, std::vector<uint64_t> &v, int nv) {
    u.resize(nu);
    v.resize(nv);
    uint64_t seed;
    io->flush();
    if (party == BOB) {
      PRG prg0;
      prg0.random_data_unaligned(&seed, sizeof(uint64_t));   // &seed 8-byte aligned; avoid 16-byte-block overflow
      seed = mod(seed);
      io->send_data(&seed, sizeof(uint64_t));
    } else {
      io->recv_data(&seed, sizeof(uint64_t));
    }
    block sblk = makeBlock(seed, 0x9e3779b97f4a7c15ULL);
    PRG prg(&sblk);
    std::vector<uint64_t> r((std::size_t)(nu + nv));
    prg.random_data(r.data(), (std::size_t)(nu + nv) * sizeof(uint64_t));
    for (int i = 0; i < nu; ++i) u[(std::size_t)i] = mod(r[(std::size_t)i]);
    for (int i = 0; i < nv; ++i) v[(std::size_t)i] = mod(r[(std::size_t)(nu + i)]);
  }

  // Cleartext inner product only (the output value), NO QuickSilver coefficients: ALICE returns
  // sum_i VAL(A_i)*VAL(B_{i*stride}); BOB returns 0. Used to commit C before the Freivalds fold.
  uint64_t compute_acc_only(const IntFp *Arow, const IntFp *Bbase, std::size_t stride, int m) {
    if (party != ALICE) return 0;
    uint64_t acc = 0;
    for (int i = 0; i < m; ++i)
      acc = add_mod(acc, mult_mod((uint64_t)VAL(Arow[i].value),
                                  (uint64_t)VAL(Bbase[(std::size_t)i * stride].value)));
    return acc;
  }

  // Freivalds proof of out[co][p] = <kernel[co], cols[p]> for ALL (co,p): commit C, then prove
  // A.B = C with ONE length-M inner product  (u^T A)(B v) = u^T C v  instead of Cout*Pn per-entry
  // IPs. The QuickSilver coefficient work drops from O(Cout*Pn*M) to O(M(Cout+Pn) + Cout*Pn).
  // x_k = sum_co u[co]*kernel[co][k];  y_k = sum_p v[p]*cols[p][k];  z = sum_{co,p} u[co]v[p]*C[co][p].
  // Each fold is a PUBLIC linear combination of committed cells, so val and mac (high64) lanes scale
  // identically on both parties -- the result is a valid commitment (the MAC relation is linear).
  void matmul_mm_freivalds(std::vector<std::vector<IntFp>> &out,
                           const std::vector<std::vector<IntFp>> &cols,    // [Pn][M]
                           const std::vector<std::vector<IntFp>> &kernel,  // [Cout][M]
                           int nthreads = 1, bool flush = true, int tamper_idx = -1) {
    const std::size_t Cout = kernel.size(), Pn = cols.size();
    const std::size_t M = Cout ? kernel[0].size() : 0;
    const std::size_t N = Cout * Pn;
    if (buf.empty()) { buf.resize(1); buf1.resize(1); }
    // 1. cleartext matmul -> commit C  (only the value, no per-entry coefficients)
    std::vector<uint64_t> acc_raw(N, 0);
    parallel_for(N, nthreads, [&](std::size_t idx) {
      const std::size_t co = idx / Pn, p = idx % Pn;
      acc_raw[idx] = compute_acc_only(kernel[co].data(), cols[p].data(), 1, (int)M);
    });
    // test-only: corrupt one committed output to confirm the Freivalds check REJECTS it.
    if (tamper_idx >= 0 && party == ALICE && (std::size_t)tamper_idx < N)
      acc_raw[(std::size_t)tamper_idx] = add_mod(acc_raw[(std::size_t)tamper_idx], 1);
    std::vector<IntFp> C(N);
    batch_feed(C.data(), acc_raw.data(), (int)N);                    // commit all C[co][p]
    // output committed C (before the fold: identical in immediate and deferred modes)
    out.assign(Cout, std::vector<IntFp>(Pn));
    for (std::size_t co = 0; co < Cout; ++co)
      for (std::size_t p = 0; p < Pn; ++p) out[co][p] = C[co * Pn + p];
    if (fv_defer_ && tamper_idx < 0) {
      // 2'. DEFER the challenge: queue an owned snapshot; one seed serves the whole queue later.
      FvJob j; j.kernel = kernel; j.cols = cols; j.C = std::move(C);
      fv_bytes_ += (Cout * M + Pn * M + N) * sizeof(IntFp);
      fv_q_.push_back(std::move(j));
      if (nthreads > fv_threads_) fv_threads_ = nthreads;
      if (fv_bytes_ >= fv_cap_) fv_flush();
      return;
    }
    // 2. challenge AFTER the commit
    std::vector<uint64_t> u, v;
    gen_challenge(u, (int)Cout, v, (int)Pn);
    fv_fold(kernel, cols, C, u, v, nthreads);
    if (flush && !defer_) batch_check();
  }

  // 3+4 of the Freivalds proof: fold x = u^T kernel, y = v^T cols (length M each), z = u^T C v,
  // and accumulate the ONE inner-product claim  (sum_k x_k y_k) - z == 0.
  void fv_fold(const std::vector<std::vector<IntFp>> &kernel, const std::vector<std::vector<IntFp>> &cols,
               const std::vector<IntFp> &C, const std::vector<uint64_t> &u, const std::vector<uint64_t> &v,
               int nthreads) {
    const std::size_t Cout = kernel.size(), Pn = cols.size();
    const std::size_t M = Cout ? kernel[0].size() : 0;
    std::vector<IntFp> x(M + 1), y(M + 1);
    parallel_for(M, nthreads, [&](std::size_t k) {
      uint64_t xlo = 0, xhi = 0, ylo = 0, yhi = 0;
      for (std::size_t co = 0; co < Cout; ++co) {
        const __uint128_t &V = kernel[co][k].value;
        xlo = add_mod(xlo, mult_mod(u[co], (uint64_t)VAL(V)));
        xhi = add_mod(xhi, mult_mod(u[co], (uint64_t)MAC(V)));
      }
      for (std::size_t p = 0; p < Pn; ++p) {
        const __uint128_t &V = cols[p][k].value;
        ylo = add_mod(ylo, mult_mod(v[p], (uint64_t)VAL(V)));
        yhi = add_mod(yhi, mult_mod(v[p], (uint64_t)MAC(V)));
      }
      x[k].value = ((__uint128_t)xhi << 64) | xlo;
      y[k].value = ((__uint128_t)yhi << 64) | ylo;
    });
    uint64_t zlo = 0, zhi = 0;
    for (std::size_t co = 0; co < Cout; ++co)
      for (std::size_t p = 0; p < Pn; ++p) {
        const uint64_t c = mult_mod(u[co], v[p]);
        const __uint128_t &V = C[co * Pn + p].value;
        zlo = add_mod(zlo, mult_mod(c, (uint64_t)VAL(V)));
        zhi = add_mod(zhi, mult_mod(c, (uint64_t)MAC(V)));
      }
    x[M] = IntFp(PR - 1, PUBLIC);                                    // -1
    y[M].value = ((__uint128_t)zhi << 64) | zlo;                     // z = u^T C v
    accumulate(x.data(), y.data(), 0, (int)(M + 1));
  }

  // Flush the deferred-Freivalds queue: ONE seed exchange, then per-job INDEPENDENT u/v expanded
  // from (seed, job index) and the usual fold.  The claims enter the IP buffer and are covered by
  // the NEXT batch_check (the public batch_check calls this first, so nothing is left behind).
  void fv_flush() {
    if (fv_q_.empty()) return;
    in_fv_flush_ = true;
    uint64_t seed;
    io->flush();
    if (party == BOB) {
      PRG prg0;
      prg0.random_data_unaligned(&seed, sizeof(uint64_t));
      seed = mod(seed);
      io->send_data(&seed, sizeof(uint64_t));
      io->flush();
    } else {
      io->recv_data(&seed, sizeof(uint64_t));
    }
    for (std::size_t q = 0; q < fv_q_.size(); ++q) {
      const FvJob &j = fv_q_[q];
      const std::size_t Cout = j.kernel.size(), Pn = j.cols.size();
      std::vector<uint64_t> u(Cout), v(Pn);
      block sblk = makeBlock(seed ^ 0x9e3779b97f4a7c15ULL, (uint64_t)q + 1);   // per-job nonce
      PRG prg(&sblk);
      prg.random_data(u.data(), (int)(Cout * sizeof(uint64_t)));
      prg.random_data(v.data(), (int)(Pn * sizeof(uint64_t)));
      for (auto &e : u) e = mod(e);
      for (auto &e : v) e = mod(e);
      fv_fold(j.kernel, j.cols, j.C, u, v, fv_threads_);
    }
    fv_q_.clear();
    fv_bytes_ = 0;
    in_fv_flush_ = false;
  }

  // Flat matrix-matrix IP: out[p*n + j] = <Wflat[j*D..], Xflat[p*D..]> for ALL (p in [0,Pn),
  // j in [0,n)) in ONE batch_feed + ONE batch_check (vs Pn separate matmul calls -> Pn rounds).
  // Both operands read IN PLACE (no nested-vector copy) -- e.g. the LM head over the committed
  // [V,D] table x [R,D] hidden.  If acc_out != null (ALICE), it receives the cleartext field
  // accumulators so the caller can rebuild a plaintext ref without recomputing the product.
  void matmul_mm_flat(std::vector<IntFp> &out, const std::vector<IntFp> &Wflat, std::size_t n,
                      const std::vector<IntFp> &Xflat, std::size_t Pn, std::size_t D, int nthreads,
                      std::vector<uint64_t> *acc_out = nullptr) {
    const std::size_t N = n * Pn;
    std::vector<uint64_t> acc_raw(N, 0);
    const std::size_t sbase = reserve(N);
    parallel_for(N, nthreads, [&](std::size_t idx) {           // out[p*n + j]; j = weight/vocab row
      const std::size_t p = idx / n, j = idx % n;
      acc_raw[idx] = compute_entry_fused(sbase + idx, &Wflat[j * D], &Xflat[p * D], 1, (int)D);
    });
    out.resize(N);
    batch_feed(out.data(), acc_raw.data(), (std::int64_t)N);    // commit all n*Pn outputs (one comm)
    IntFp neg1(PR - 1, PUBLIC);
    parallel_for(N, nthreads, [&](std::size_t idx) { add_z_term(sbase + idx, neg1, out[idx]); });
    num += (int)N;
    if (!defer_) batch_check();                                // ONE network round (deferred under IP_DEFER)
    if (acc_out) *acc_out = std::move(acc_raw);
  }

  // Freivalds version of matmul_mm_flat (e.g. the LM head over [n,D] weights x [Pn,D] hidden).
  // C[p*n+j] = <Wflat[j], Xflat[p]>; prove with ONE length-D IP: u over Pn rows, v over n cols,
  //   x_k = sum_p u[p]*Xflat[p*D+k];  y_k = sum_j v[j]*Wflat[j*D+k];  z = sum_{p,j} u[p]v[j]*C[p*n+j].
  void matmul_mm_flat_freivalds(std::vector<IntFp> &out, const std::vector<IntFp> &Wflat, std::size_t n,
                                const std::vector<IntFp> &Xflat, std::size_t Pn, std::size_t D, int nthreads,
                                std::vector<uint64_t> *acc_out = nullptr, bool flush = true) {
    const std::size_t N = n * Pn;
    if (buf.empty()) { buf.resize(1); buf1.resize(1); }
    std::vector<uint64_t> acc_raw(N, 0);
    parallel_for(N, nthreads, [&](std::size_t idx) {
      const std::size_t p = idx / n, j = idx % n;
      acc_raw[idx] = compute_acc_only(&Wflat[j * D], &Xflat[p * D], 1, (int)D);
    });
    out.resize(N);
    batch_feed(out.data(), acc_raw.data(), (std::int64_t)N);    // commit C
    std::vector<uint64_t> u, v;
    gen_challenge(u, (int)Pn, v, (int)n);                       // u: rows p,  v: cols j
    std::vector<IntFp> x(D + 1), y(D + 1);
    parallel_for(D, nthreads, [&](std::size_t k) {
      uint64_t xlo = 0, xhi = 0, ylo = 0, yhi = 0;
      for (std::size_t p = 0; p < Pn; ++p) {
        const __uint128_t &V = Xflat[p * D + k].value;
        xlo = add_mod(xlo, mult_mod(u[p], (uint64_t)VAL(V)));
        xhi = add_mod(xhi, mult_mod(u[p], (uint64_t)MAC(V)));
      }
      for (std::size_t j = 0; j < n; ++j) {
        const __uint128_t &V = Wflat[j * D + k].value;
        ylo = add_mod(ylo, mult_mod(v[j], (uint64_t)VAL(V)));
        yhi = add_mod(yhi, mult_mod(v[j], (uint64_t)MAC(V)));
      }
      x[k].value = ((__uint128_t)xhi << 64) | xlo;
      y[k].value = ((__uint128_t)yhi << 64) | ylo;
    });
    std::vector<uint64_t> zplo((std::size_t)Pn, 0), zphi((std::size_t)Pn, 0);  // z-fold parallel over p
    parallel_for(Pn, nthreads, [&](std::size_t p) {
      uint64_t lo = 0, hi = 0;
      for (std::size_t j = 0; j < n; ++j) {
        const uint64_t c = mult_mod(u[p], v[j]);
        const __uint128_t &V = out[p * n + j].value;
        lo = add_mod(lo, mult_mod(c, (uint64_t)VAL(V)));
        hi = add_mod(hi, mult_mod(c, (uint64_t)MAC(V)));
      }
      zplo[p] = lo; zphi[p] = hi;
    });
    uint64_t zlo = 0, zhi = 0;
    for (std::size_t p = 0; p < Pn; ++p) { zlo = add_mod(zlo, zplo[p]); zhi = add_mod(zhi, zphi[p]); }
    x[D] = IntFp(PR - 1, PUBLIC);
    y[D].value = ((__uint128_t)zhi << 64) | zlo;
    accumulate(x.data(), y.data(), 0, (int)(D + 1));
    if (flush && !defer_) batch_check();
    if (acc_out) *acc_out = std::move(acc_raw);
  }

  // Attention matmul, in place on FLAT activations -- the missing interface for two-activation
  // (Q.K^T, A.V) products.  raw[g*M*N + m*N + j] = <A[g,m,:], B[g,:,j]> for all (g,m,j):
  //   A flat [G,M,K]  (row (g,m) at A[(g*M+m)*K .. +K], contiguous)
  //   B flat [G,K,N]  (col (g,j) is B[Boff + j + k*N] for k=0..K-1, stride N);  Boff = Bb? g*K*N : 0
  // Same QuickSilver IP proof as matmul_mm (one batched entry per output, not K multiplies) and
  // the same threaded local compute, but reads activations IN PLACE (no [M][K]/[N][K] copies) and
  // does ONE batch_check per group (head).  Reused scratch -> no per-seq^2 allocation.
  void matmul_attn(std::vector<IntFp> &raw, const IntFp *A, const IntFp *B,
                   int G, int M, int N, int K, bool Bb, int nthreads) {
    raw.resize((std::size_t)G * M * N);
    const IntFp neg1(PR - 1, PUBLIC);
    const std::size_t NO = (std::size_t)M * N;
    const std::size_t TOT = (std::size_t)G * NO;                      // all G heads share ONE batch_check
    const std::size_t sbase = reserve(TOT);
    std::vector<uint64_t> acc_raw(NO);
    for (int g = 0; g < G; ++g) {
      const std::int64_t Aoff = (std::int64_t)g * M * K;
      const std::int64_t Boff = Bb ? (std::int64_t)g * K * N : 0;
      IntFp *zg = &raw[(std::size_t)g * M * N];                       // commit z straight into raw (no copy)
      const std::size_t base = sbase + (std::size_t)g * NO;           // this head's slot range in buf/buf1
      parallel_for(NO, nthreads, [&](std::size_t idx) {              // fused: acc + partial coeffs, one pass
        const int m = (int)(idx / N), j = (int)(idx % N);
        acc_raw[idx] = compute_entry_fused(base + idx, &A[Aoff + (std::int64_t)m * K], &B[Boff + j],
                                           (std::size_t)N, K);
      });
      batch_feed(zg, acc_raw.data(), (int)NO);                        // commit z (one buffered feed per head)
      parallel_for(NO, nthreads, [&](std::size_t idx) { add_z_term(base + idx, neg1, zg[idx]); });
    }
    num += (int)TOT;
    if (!defer_) batch_check();                                      // ONE round for ALL G heads (deferred under IP_DEFER)
  }

  // Freivalds version of matmul_attn: per head g, prove C_g = A_g.B_g with ONE length-K IP
  //   (u_g^T A_g)(B_g v_g) = u_g^T C_g v_g  instead of M*N per-entry IPs. One IP slot per head,
  // all G batched into a single batch_check. u over G*M, v over G*N (independent per head).
  void matmul_attn_freivalds(std::vector<IntFp> &raw, const IntFp *A, const IntFp *B,
                             int G, int M, int N, int K, bool Bb, int nthreads, bool flush = true) {
    raw.resize((std::size_t)G * M * N);
    const std::size_t NO = (std::size_t)M * N;
    if (buf.size() < (std::size_t)G) { buf.resize((std::size_t)G); buf1.resize((std::size_t)G); }
    std::vector<uint64_t> acc_raw(NO);
    for (int g = 0; g < G; ++g) {                                  // 1. commit C per head
      const std::int64_t Aoff = (std::int64_t)g * M * K, Boff = Bb ? (std::int64_t)g * K * N : 0;
      IntFp *zg = &raw[(std::size_t)g * M * N];
      parallel_for(NO, nthreads, [&](std::size_t idx) {
        const int m = (int)(idx / N), j = (int)(idx % N);
        acc_raw[idx] = compute_acc_only(&A[Aoff + (std::int64_t)m * K], &B[Boff + j], (std::size_t)N, K);
      });
      batch_feed(zg, acc_raw.data(), (int)NO);
    }
    std::vector<uint64_t> u, v;
    gen_challenge(u, G * M, v, G * N);                             // 2. challenge after commit
    // NOTE: no `num = 0` reset here -- under IP_DEFER that would silently DISCARD pending
    // deferred claims from earlier ops (an unproven-matmul soundness hole).  accumulate()
    // below self-appends at `num`, so pending + per-head claims all reach the next check.
    for (int g = 0; g < G; ++g) {                                  // 3. per-head fold + one IP slot
      const std::int64_t Aoff = (std::int64_t)g * M * K, Boff = Bb ? (std::int64_t)g * K * N : 0;
      const uint64_t *ug = &u[(std::size_t)g * M], *vg = &v[(std::size_t)g * N];
      const IntFp *Cg = &raw[(std::size_t)g * M * N];
      std::vector<IntFp> x(K + 1), y(K + 1);
      parallel_for(K, nthreads, [&](std::size_t k) {
        uint64_t xlo = 0, xhi = 0, ylo = 0, yhi = 0;
        for (int m = 0; m < M; ++m) {
          const __uint128_t &W = A[Aoff + (std::int64_t)m * K + (std::int64_t)k].value;
          xlo = add_mod(xlo, mult_mod(ug[m], (uint64_t)VAL(W)));
          xhi = add_mod(xhi, mult_mod(ug[m], (uint64_t)MAC(W)));
        }
        for (int j = 0; j < N; ++j) {
          const __uint128_t &W = B[Boff + (std::int64_t)k * N + j].value;
          ylo = add_mod(ylo, mult_mod(vg[j], (uint64_t)VAL(W)));
          yhi = add_mod(yhi, mult_mod(vg[j], (uint64_t)MAC(W)));
        }
        x[k].value = ((__uint128_t)xhi << 64) | xlo;
        y[k].value = ((__uint128_t)yhi << 64) | ylo;
      });
      std::vector<uint64_t> zmlo((std::size_t)M, 0), zmhi((std::size_t)M, 0);  // z = u^T C v, parallel over m
      parallel_for(M, nthreads, [&](std::size_t m) {
        uint64_t lo = 0, hi = 0;
        for (int j = 0; j < N; ++j) {
          const uint64_t c = mult_mod(ug[m], vg[j]);
          const __uint128_t &W = Cg[(std::size_t)m * N + j].value;
          lo = add_mod(lo, mult_mod(c, (uint64_t)VAL(W)));
          hi = add_mod(hi, mult_mod(c, (uint64_t)MAC(W)));
        }
        zmlo[m] = lo; zmhi[m] = hi;
      });
      uint64_t zlo = 0, zhi = 0;
      for (int m = 0; m < M; ++m) { zlo = add_mod(zlo, zmlo[(std::size_t)m]); zhi = add_mod(zhi, zmhi[(std::size_t)m]); }
      x[K] = IntFp(PR - 1, PUBLIC);
      y[K].value = ((__uint128_t)zhi << 64) | zlo;
      accumulate(x.data(), y.data(), 0, K + 1);                   // slot g
    }
    if (flush && !defer_) batch_check();                                      // ONE round for all heads
  }

};

// Matrix x vector: out[j] = W[j] . x, each proven by the standalone IP proof.
template <typename IO>
static inline void ip_linear(IPProof<IO> &ip, std::vector<IntFp> &out,
                             const std::vector<std::vector<IntFp>> &W,
                             const std::vector<IntFp> &x) {
  out.resize(W.size());
  for (std::size_t j = 0; j < W.size(); ++j) out[j] = ip.inner_product(W[j], x);
}

}  // namespace volefier

#endif
