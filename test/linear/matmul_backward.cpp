#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/linear/ip_matmul_backward.h"
#include <cstdio>
#include <vector>
using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = 1;

static int64_t decode(uint64_t v) {
    return (v > (PR >> 1)) ? (int64_t)v - (int64_t)PR : (int64_t)v;
}

int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO(
            new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
            party == ALICE);
    printf("[%d] connected\n", party); fflush(stdout);

    setup_zk_arith(ios, threads, party);
    printf("[%d] setup done\n", party); fflush(stdout);

    bool ok = true;

    {
        IPProof<BoolIO> ipp(party, ios[0], get_ostriple(party));
        printf("[%d] IPProof created\n", party); fflush(stdout);

        const int M=2, K=3, N=4;

        vector<int64_t> Ar  = {1,2,3, 4,5,6};
        vector<int64_t> Br  = {1,0,1,0, 0,1,0,1, 1,1,0,0};
        vector<int64_t> dCr(M*N, 1);

        vector<int64_t> dAr(M*K, 0);
        for (int m=0; m<M; ++m)
            for (int k=0; k<K; ++k)
                for (int n=0; n<N; ++n)
                    dAr[m*K+k] += dCr[m*N+n] * Br[k*N+n];

        vector<int64_t> dBr(K*N, 0);
        for (int k=0; k<K; ++k)
            for (int n=0; n<N; ++n)
                for (int m=0; m<M; ++m)
                    dBr[k*N+n] += Ar[m*K+k] * dCr[m*N+n];

        auto commit = [&](const vector<int64_t> &v) {
            vector<IntFp> c(v.size());
            for (size_t i=0; i<v.size(); ++i)
                c[i] = (party==ALICE) ? IntFp((uint64_t)v[i], ALICE)
                                      : IntFp(0ULL, ALICE);
            return c;
        };

        auto cA  = commit(Ar);
        auto cB  = commit(Br);
        auto cdC = commit(dCr);
        printf("[%d] inputs committed — A:%zu B:%zu dC:%zu\n",
               party, cA.size(), cB.size(), cdC.size()); fflush(stdout);

        printf("[%d] calling matmul_backward M=%d K=%d N=%d\n", party, M, K, N); fflush(stdout);

        vector<IntFp> dLdA, dLdB;
        matmul_backward(ipp, dLdA, dLdB, cA, cB, cdC, M, K, N, threads);
        printf("[%d] matmul_backward done — dLdA:%zu dLdB:%zu\n",
               party, dLdA.size(), dLdB.size()); fflush(stdout);

        for (int i=0; i<M*K; ++i) {
            int64_t got = decode(dLdA[i].reveal());
            if (party == ALICE && got != dAr[i]) {
                printf("dL/dA[%d]: got %ld expected %ld\n", i, got, dAr[i]);
                ok = false;
            }
        }
        printf("[%d] dLdA reveals done\n", party); fflush(stdout);

        for (int i=0; i<K*N; ++i) {
            int64_t got = decode(dLdB[i].reveal());
            if (party == ALICE && got != dBr[i]) {
                printf("dL/dB[%d]: got %ld expected %ld\n", i, got, dBr[i]);
                ok = false;
            }
        }
        printf("[%d] dLdB reveals done\n", party); fflush(stdout);

    } 
    printf("[%d] finalizing\n", party); fflush(stdout);
    finalize_zk_arith();
    printf("[%d] finalized\n", party); fflush(stdout);

    if (party == ALICE)
        printf("%s\n", ok ? "pass matmul_backward" : "FAIL matmul_backward");

    _exit(0);
}