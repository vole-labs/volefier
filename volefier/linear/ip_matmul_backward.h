#ifndef LINEAR_IP_MATMUL_BACKWARD_H
#define LINEAR_IP_MATMUL_BACKWARD_H

#include "volefier/linear/ip_matmul.h"

namespace volefier {

template <typename IO>
static inline void matmul_backward(
    IPProof<IO> &ipp,
    std::vector<IntFp> &dLdA,
    std::vector<IntFp> &dLdB,
    const std::vector<IntFp> &A,
    const std::vector<IntFp> &B,
    const std::vector<IntFp> &dLdC,
    int M, int K, int N,
    int nthreads = 1
) {
    std::vector<IntFp> BT((std::size_t)N * K);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            BT[(std::size_t)n * K + k] = B[(std::size_t)k * N + n];

    ipp.matmul_attn(dLdA, dLdC.data(), BT.data(),
                    1, M, K, N, false, nthreads);

    std::vector<IntFp> AT((std::size_t)K * M);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            AT[(std::size_t)k * M + m] = A[(std::size_t)m * K + k];

    ipp.matmul_attn(dLdB, AT.data(), dLdC.data(),
                    1, K, N, M, false, nthreads);
}

} 
#endif