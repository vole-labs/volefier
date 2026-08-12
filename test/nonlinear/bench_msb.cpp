#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/nonlinear/msb_logup.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

int dim = 100000;

uint64_t comm(BoolIO *ios[threads]) {
    uint64_t c = 0;
    for (int i = 0; i < threads; ++i) c += io_counter(ios[i]->io);  // was ios[i]->counter
    return c;
}

int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);

    BoolIO *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO(
            new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
            party == ALICE);

    cout << endl
         << "------------ MSB_LOGUP / ReLU bench ------------" << endl
         << endl;

    setup_zk_arith(ios, threads, party);

    // Our LogUp-based MSB.  Internally uses one LogUpRangeCheck on [0, 4096).
    // The 2nd ctor arg is ignored (kept for API compat with the ZKROM-based MSB).
    MSB_LOGUP *msb = new MSB_LOGUP(party, /*ram_batch_sz=*/1000000, ios[0]->io);

    // Pre-generate `dim` random witnesses (ALICE only; BOB sees zeros).
    // Same pattern as the reference relu.cpp.
    uint64_t *witness = new uint64_t[dim];
    memset(witness, 0, dim * sizeof(uint64_t));
    __uint128_t *randomness = new __uint128_t[dim];
    PRG prg(fix_key);
    prg.random_block((block*)randomness, dim);
    for (int i = 0; i < dim; i++) {
        if (party == ALICE)
            witness[i] = ((uint64_t)randomness[i]) % PR;
    }

    // ====== timed region begins ======
    uint64_t com_start = comm(ios);
    auto t_start = clock_start();

    // Commit witnesses.
    vector<IntFp> x(dim), y(dim);
    for (int i = 0; i < dim; i++)
        x[i] = IntFp(witness[i], ALICE);

    // Batched MSB:  y[i] = MSB(x[i]).  Each MSB issues 5 range-check accesses
    // into the embedded LogUpRangeCheck on [0, 4096).
    msb->compute(y, x, dim);

    // ReLU = (1 - MSB(x)) * x.  One ALICE * ALICE committed mult per element
    // (matches what the reference relu.cpp does after its ZKcmpPositive).
    for (int i = 0; i < dim; i++)
        y[i] = (IntFp(1LL, PUBLIC) + y[i].negate()) * x[i];

    // Close out the LogUp range-check accumulated by MSB.
    msb->destroy();
    delete msb;

    // Final batched MAC verification kept inside the timer.
    finalize_zk_arith();

    double t_us = time_from(t_start);
    // ====== timed region ends ======

    uint64_t com_used = comm(ios) - com_start;

    cout << "time - MSB_LOGUP ReLU:   " << t_us / 1e6
         << " s    party=" << party << endl;
    cout << "communication (KB):      " << com_used / 1024.0 << endl;
    cout << "avg per ReLU (us):       " << t_us / dim << endl;
    cout << "avg per ReLU (bytes):    " << (double)com_used / dim << endl;

    // ===== correctness check =====
    // Same convention as relu.cpp:  output = x if x is "small" (< PR/2),
    // else 0.  In our two's-complement field the test "x < (PR+1)/2 = 2^60"
    // is equivalent to "MSB(x) == 0", which is what (witness >> 60) checks.
    if (party == ALICE) {
        int faults = 0;
        for (int i = 0; i < dim; i++) {
            uint64_t expect = ((witness[i] >> 60) == 0) ? witness[i] : 0;
            uint64_t got    = (uint64_t)HIGH64(y[i].value);
            if (expect != got) {
                if (faults < 3)
                    cout << "fault i=" << i
                         << "  witness=" << witness[i]
                         << "  expect="  << expect
                         << "  got="     << got << endl;
                faults++;
            }
        }
        cout << "faults: " << faults << " / " << dim << endl;
    }
    cout << "finish test" << endl;

    delete[] witness;
    delete[] randomness;
    for (int i = 0; i < threads; i++) {
        auto *_io = ios[i]->io;
        delete ios[i];
        delete _io;
    }
    return 0;
}
