
namespace M9 {

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;
int _prev_m=0, _prev_n=0, _prev_k=0;

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
};

template<int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<1, NUM_SM, BM, BN, TM, TN> {
    int block;
    int it;
    int total_blocks_m, total_blocks_n;

    __device__ __forceinline__ Schedule(int M, int N, int _block) {
        block = _block;
        it = 0;
        total_blocks_m = CEIL_DIV(M, BM);
        total_blocks_n = CEIL_DIV(N, BN);
        assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0);
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        int num = it*NUM_SM + block;
        if (num >= total_blocks_m*total_blocks_n) {return false;}
        
        int cur_tile = num / (TM*TN);
        int cur_tile_pos = num % (TM*TN);
        block_m = TM*(cur_tile / (total_blocks_n/TN));
        block_n = TN*(cur_tile % (total_blocks_n/TN));
        block_m += cur_tile_pos / TN;
        block_n += cur_tile_pos % TN;
        ++it;
        return true;
    }
};

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM, int CLUSTER_M, int CLUSTER_N>
__global__  __launch_bounds__(NUM_THREADS) void  __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1) matmulKernel9(int M, int N, int K, bf16* C, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N=BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
    assert((M / BM) % CLUSTER_M == 0);
    assert((N / BN) % CLUSTER_N == 0);

    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);
    bf16 *sA = s.A;
    bf16 *sB = s.B;
    // Declare barriers
    __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];
    uint32_t rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);

    const int num_blocks_k = K / BK;
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier(&full[i], 0, 1);
            init_barrier(&empty[i], 0, num_consumers*CLUSTERS);
        }
    }
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    Schedule<1, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN*CLUSTER_N, 16/CLUSTER_M, 8/CLUSTER_N> schedule(M, N, rank);
    
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rank_m = rank / CLUSTER_N;
    uint32_t rank_n = rank % CLUSTER_N;

    // Producer
    if (wg_idx == 0) {
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<num_regs>();
        if (tid == 0) {
            int p = 0;
            int qidx = 0;
            uint32_t col_mask = 0;
            for (int i = 0; i < CLUSTER_M; ++i) {
                col_mask |= (1 << (i * CLUSTER_N));
            }
            int num_block_m, num_block_n;
            while (schedule.next(num_block_m, num_block_n)) {
                num_block_n = num_block_n * CLUSTER_N + rank_n;
                num_block_m = num_block_m * CLUSTER_M + rank_m;
                
                for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                    if (qidx == QSIZE) { qidx = 0; p ^= 1;}
                    wait(&empty[qidx], p);
                    
                    expect_bytes(&full[qidx], (BK*BN+BK*BM)*sizeof(bf16));
                    if constexpr (CLUSTER_N > 1) {
                        uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                        if (rank_n == 0) {
                            load_async_multicast(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM, mask);
                        }
                    } else {
                        load_async(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM);
                    }

                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0) {
                            load_async_multicast(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], block_k_iter*BK, num_block_n*BN, col_mask << rank_n);
                        }
                    } else {
                        load_async(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], block_k_iter*BK, num_block_n*BN);
                    }
                }
            }
        }
    } else {
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc<num_regs>();
        float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];
        --wg_idx;
        for (int qidx = 0; qidx < QSIZE; ++qidx) {
            if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
        }
        int p = 0;
        int qidx = 0;
        int lane = tid % 32, warp = tid / 32, row = warp*16 + lane / 4;
        int num_block_m, num_block_n;
        while (schedule.next(num_block_m, num_block_n)) {
            num_block_n = num_block_n * CLUSTER_N + rank_n;
            num_block_m = num_block_m * CLUSTER_M + rank_m;
            {
                if (qidx == QSIZE) {qidx = 0; p ^= 1; };
                wait(&full[qidx], p);
                warpgroup_arrive();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                    bf16 *wgmma_sA = sA + qidx*BK*BM + 64*(m_it + wg_idx*B_WG_M/WGMMA_M)*WGMMA_M;
                    bf16 *wgmma_sB = sB + qidx*BK*BN;
                    {
                        wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);
                        #pragma unroll
                        for (int k_it = 1; k_it < 64/WGMMA_K; ++k_it) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                        }
                        wgmma_sA += 64*BM;
                        wgmma_sB += 64*BN;
                    }
                    #pragma unroll
                    for (int bk = 64; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int k_it = 0; k_it < 64/WGMMA_K; ++k_it) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                        }
                        wgmma_sA += 64*BM;
                        wgmma_sB += 64*BN;
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
                ++qidx;
            }
            for (int block_k_iter = 1; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                if (qidx == QSIZE) {qidx = 0; p ^= 1; };
                wait(&full[qidx], p);
                warpgroup_arrive();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                    bf16 *wgmma_sA = sA + qidx*BK*BM + 64*(m_it + wg_idx*B_WG_M/WGMMA_M)*WGMMA_M;
                    bf16 *wgmma_sB = sB + qidx*BK*BN;
                    #pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int k_it = 0; k_it < 64/WGMMA_K; ++k_it) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                        }
                        wgmma_sA += 64*BM;
                        wgmma_sB += 64*BN;
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
            }

            // __nv_bfloat162* block_C = reinterpret_cast<__nv_bfloat162*>(C + num_block_m*BM*N + num_block_n*BN);
            bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                int yo = m_it*WGMMA_M + wg_idx*B_WG_M;
                if (row + 8 + yo + num_block_m*BM >= M) continue;
                #pragma unroll
                for (int w = 0; w < WGMMA_N; w+=16) if (w+num_block_n*BN < N) {
                    // int col = 8*w + (tid % 4);
                    // #define IDX(i, j) ((((i) + yo)*N/2 + (j)))
                    // block_C[IDX(row, col)] = __halves2bfloat162(d[m_it][w][0], d[m_it][w][1]);
                    // block_C[IDX(row + 8, col)] = __halves2bfloat162(d[m_it][w][2], d[m_it][w][3]);
                    // block_C[IDX(row, col + 4)] = __halves2bfloat162(d[m_it][w][4], d[m_it][w][5]);
                    // block_C[IDX(row + 8, col + 4)] = __halves2bfloat162(d[m_it][w][6], d[m_it][w][7]);
                    // #undef IDX
                    int col = w + 2*(tid % 4);
                    #define IDX(i, j) ((j)*M + ((i) + yo))
                    #define ST(i, j, v) __stwt(&block_C[IDX(i, j)], v);
                    
                    ST(row+8, col, d[m_it][w/16][2]);
                    ST(row, col, d[m_it][w/16][0]);
                    
                    ST(row+8, col+1, d[m_it][w/16][3]);
                    ST(row, col+1, d[m_it][w/16][1]);
                    
                    ST(row+8, col+8, d[m_it][w/16][6]);
                    ST(row, col+8, d[m_it][w/16][4]);
                    
                    ST(row+8, col+9, d[m_it][w/16][7]);
                    ST(row, col+9, d[m_it][w/16][5]);
                    
                    #undef IDX
                    #undef ST
                }
            }
        }
    }
}

void runKernel9(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
    constexpr int BM = 64*2;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128*3;
    constexpr int QSIZE = 3;
    constexpr int CLUSTER_M = 2;
    constexpr int CLUSTER_N = 1;
    constexpr int NUM_SM = 128;
    static_assert(NUM_SM % (CLUSTER_M*CLUSTER_N) == 0);

    if (_prev_m != M) {
        d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
        d_tma_map_B = create_tensor_map<BN, BK>(B, N, K);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }
    // Assert cached values are of same size
    assert (M == _prev_m && N == _prev_n && K == _prev_k);
    auto* kernel = matmulKernel9<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M, CLUSTER_N>;
    size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<NUM_SM, NUM_THREADS, sMemSize>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
}
    
} // namespace M9

using M9::runKernel9;
