
namespace M11 {

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;
CUtensorMap d_tma_map_C;
int _prev_m=0, _prev_n=0, _prev_k=0;


template<int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

constexpr int SPACE_LEN = 128;
int *_dspace;

template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<2, NUM_SM, BM, BN, TM, TN> {
    int it;
    int *space;

    __device__ __forceinline__ Schedule(int M, int N, int block, int *_space) {
        it = 0;
        space = _space;
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        if (it >= SPACE_LEN) {
            return false;
        }
        int now = space[it];
        if (now == -1) {
            return false;
        }
        block_m = now >> 16;
        block_n = (now & ((1<<16)-1));
        ++it;
        return true;
    }
};

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
    alignas(128) bf16 C[BN*BM];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
    int space[SPACE_LEN];
};

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM, int CLUSTER_M, int CLUSTER_N>
__global__  __launch_bounds__(NUM_THREADS) void  __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1) matmulKernel11(int M, int N, int K, const __grid_constant__ CUtensorMap tensorMapC, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB, int* dspace) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N=BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
    assert((M / BM) % CLUSTER_M == 0);
    assert((N / BN) % CLUSTER_N == 0);

    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);
    bf16 *sA = s.A, *sB = s.B, *sC = s.C;
    uint64_t *full = s.full, *empty = s.empty;
    int *space = s.space;

    uint32_t rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);
    // Load schedule for this SM
    if (threadIdx.x < SPACE_LEN) space[threadIdx.x] = dspace[rank*SPACE_LEN+threadIdx.x];

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


    Schedule<2, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN*CLUSTER_N, 16/CLUSTER_M, 8/CLUSTER_N> schedule(M, N, rank, &space[0]);

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

            asm volatile("cp.async.bulk.wait_group 0;");

            int lane = tid % 32, warp = tid / 32;
            int row = warp*16 + lane / 4;

            bf16* block_sC = sC + wg_idx*B_WG_M*BN;
            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                int yo = m_it*WGMMA_M;
                #pragma unroll
                for (int w = 0; w < WGMMA_N; w+=16) {
                    int col = w + 2*(tid % 4);
                    #define ST(i, j, v) block_sC[(j)*B_WG_M + (i) + yo] = v
                    
                    ST(row, col, d[m_it][w/16][0]);
                    ST(row+8, col, d[m_it][w/16][2]);
                    
                    
                    ST(row, col+1, d[m_it][w/16][1]);
                    ST(row+8, col+1, d[m_it][w/16][3]);
                    
                    
                    ST(row, col+8, d[m_it][w/16][4]);
                    ST(row+8, col+8, d[m_it][w/16][6]);
                    
                    
                    ST(row, col+9, d[m_it][w/16][5]);
                    ST(row+8, col+9, d[m_it][w/16][7]);
                    
                    #undef ST
                }
            }
            // Wait for all 256 consumer threads to reach here
            asm volatile("bar.sync 1, 256;\n");

            if (threadIdx.x == 128) {
                store_async(&tensorMapC, (bf16*)&sC[0], num_block_m*BM, num_block_n*BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
    }
}

// Rotate/flip quadrant appropriately
void rot(int n, int& x, int& y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            x = n-1 - x;
            y = n-1 - y;
        }
        // Swap x and y
        int t = x;
        x = y;
        y = t;
    }
}

// Convert distance along curve to (x,y) point
void d2xy(int n, int d, int& x, int& y) {
    int rx, ry, s, t = d;
    x = y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        x += s * rx;
        y += s * ry;
        t /= 4;
    }
}

void createHilbert(int M, int N, int CORES, int *space) {
    int dim = (1 << (32 - __builtin_clz(max(M, N) - 1)));
    int core = 0;
    std::vector<std::string> v(dim, std::string(dim, '.'));
    memset(space, -1, sizeof(int)*CORES*SPACE_LEN);
    int FCORES = 64;
    int total = 0;
    std::vector<std::vector<int>> pos(CORES, std::vector<int>());
    for (int i = 0; i < dim*dim; ++i) {
        int x, y;
        d2xy(dim, i, x, y);
        if (x < M && y < N) {
            assert(loc < SPACE_LEN);
            assert(v[x][y] == '.');
            v[x][y] = '*';
            ++total;
            pos[core].push_back((x << 16) | y);
            ++core;
            if (core == FCORES) {core = 0;}
        }
    }
    core = FCORES;
    for (int i = 0; i < FCORES; ++i) {
        if (pos.back().size() >= pos[0].size()-1) break;
        pos[core].push_back(pos[i].back());
        pos[i].pop_back();
        ++core;
        if (core == CORES) {core = FCORES;}
    }
    for (int i = 0; i < CORES; ++i) {
        for (int j = 0; j < pos[i].size(); ++j) {
            space[i*SPACE_LEN + j] = pos[i][j];
        }
    }
    assert(total == M*N);
}

void runKernel11(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
    constexpr int BM = 128;
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
        d_tma_map_C = create_tensor_map<BN, BM, false>(C, N, M);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
        int *space;
        space = (int*)malloc(sizeof(int)*NUM_SM*SPACE_LEN);
        createHilbert(CEIL_DIV(M, BM*CLUSTER_M), CEIL_DIV(N, BN*CLUSTER_N), NUM_SM/CLUSTER_M/CLUSTER_N, space);
        cudaCheck(cudaMalloc((void **)&_dspace, sizeof(int)*NUM_SM*SPACE_LEN));
        cudaCheck(cudaMemcpy(_dspace, space, sizeof(int)*NUM_SM*SPACE_LEN, cudaMemcpyHostToDevice));
    }
    // Assert cached values are of same size
    assert (M == _prev_m && N == _prev_n && K == _prev_k);
    auto* kernel = matmulKernel11<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M, CLUSTER_N>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<NUM_SM, NUM_THREADS, sMemSize>>>(M, N, K, d_tma_map_C, d_tma_map_A, d_tma_map_B, _dspace);
}
    
} // namespace M11

using M11::runKernel11;
    