// bench_logup.cpp - Standalone benchmark of LogUpRangeCheck.
//
// Proves: every accessed index ∈ [0, N).  Apples-to-apples partner of
// bench_zkrom.cpp on the same workload (same N, W, access pattern).
//
// Auto-batching threshold matches ZKROM's ram_batch_sz default (1 M),
// so periodic teardowns are spread across the run rather than piling
// into one giant proof at the end.
//
// Usage:
//   bin/test_ram_bench_logup <party> <port> <logN> <W> [IP]
//     party: 1 (ALICE) or 2 (BOB)
//     port:  port for the NetIO connection
//     logN:  log2(table size N)
//     W:     number of accesses
//     IP:    ALICE's IP, BOB only (default localhost)
//
// Output ends with one parseable line:
//   [RESULT] variant=logup N=<N> W=<W> time_us=<t> bytes=<b>

#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "volefier/zk-ram/logup.h"
#if defined(__APPLE__)
    #include <sys/resource.h>
    #include <mach/mach.h>
#elif defined(__linux__)
    #include <sys/resource.h>
#endif

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;
std::size_t batch_sz_local = 1000000;

uint64_t comm_io(BoolIO *ios[threads]) {
    uint64_t c = 0;
    for (int i = 0; i < threads; ++i) c += io_counter(ios[i]->io);
    return c;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cout << "usage: " << argv[0] << " PARTY PORT logN W [IP]\n";
        return -1;
    }
    parse_party_and_port(argv, &party, &port);
    int logN = atoi(argv[3]);
    uint64_t N = 1ULL << logN;
    uint64_t W = (uint64_t)atoll(argv[4]);
    const char *ip = (argc >= 6) ? argv[5] : nullptr;

    BoolIO* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO(
            new NetIO(party == ALICE ? ip : nullptr, port + i),
            party == ALICE);

    setup_zk_arith(ios, threads, party);

    IOChannel *io = ios[party - 1]->io;

    uint64_t c0 = comm_io(ios);
    auto t0 = clock_start();

    LogUpRangeCheck rc(N, party, io, batch_sz_local);
    rc.Setup();

    for (uint64_t k = 0; k < W; ++k) {
        // ALICE-committed indices, matching bench_zkrom's pattern.
        IntFp idx = (party == ALICE) ? IntFp(k % N, ALICE) : IntFp(0ULL, ALICE);
        rc.Access(idx);
    }

    rc.Teardown();  // finalizes any remaining partial batch

    double t = time_from(t0);
    uint64_t bytes = comm_io(ios) - c0;

    finalize_zk_arith();

    cout << "[RESULT] variant=logup N=" << N << " W=" << W
         << " time_us=" << (uint64_t)t << " bytes=" << bytes << endl;

    for (int i = 0; i < threads; ++i) {
        auto *_io = ios[i]->io;
        delete ios[i];
        delete _io;
    }
    return 0;
}
