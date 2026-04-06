#include "../include/b200_utils.cuh"
namespace M1 {

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;
CUtensorMap d_tma_map_C;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

constexpr int SPACE_LEN = 128;
int *_dspace;

// Persistent kernel schedule (Hilbert curve tile assignment)
template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

template <int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<2, NUM_SM, BM, BN, TM, TN> {
  int it;
  int *space;

  __device__ __forceinline__ Schedule(int M, int N, int block, int *_space) {
    it = 0;
    space = _space;
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (it >= SPACE_LEN) {
      return false;
    }
    int now = space[it];
    if (now == -1) {
      return false;
    }
    block_m = now >> 16;
    block_n = (now & ((1 << 16) - 1));
    ++it;
    return true;
  }
};

// Shared memory layout for the B200 kernel
// With cta_group::2, each CTA loads FULL A but only HALF of B
template <int BM, int BN_PER_CTA, int BK, int QSIZE, int NUM_STAGES, int BN_STORE>
struct SMem {
  alignas(128) bf16 A[BM * BK * NUM_STAGES];
  alignas(128) bf16 B[BN_PER_CTA * BK * NUM_STAGES];
  alignas(128) bf16 C[BM * BN_STORE]; // epilogue: TMEM → sC → TMA store
  alignas(8) uint64_t tma_mbar[NUM_STAGES]; // TMA completion (data loaded)
  alignas(8) uint64_t mma_mbar[NUM_STAGES]; // MMA completion (buffer recyclable)
  alignas(8) uint64_t mainloop_mbar;        // mainloop done signal
  int tmem_addr;                             // tmem base address from alloc
  int space[SPACE_LEN];
};

// ---------------------------------------------------------------------------
// Kernel: warp-specialized GEMM using tcgen05 MMA
//
// Thread structure (128 threads = 4 warps):
//   Warp 0: TMA producer (1 elected thread loads data)
//   Warp 1: MMA (1 elected thread in CTA0 issues MMA)
//   All 4 warps: epilogue (read tmem, convert, store to global)
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_THREADS, int NUM_STAGES, int NUM_SM,
          int CLUSTER_M, int CLUSTER_N>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    matmulKernel1(int M, int N, int K,
                  const __grid_constant__ CUtensorMap tensorMapA,
                  const __grid_constant__ CUtensorMap tensorMapB,
                  const __grid_constant__ CUtensorMap tensorMapC,
                  bf16 *dC, int *dspace) {
  constexpr int CTA_GROUP = CLUSTER_M; // cta_group matches cluster M dim
  constexpr int BN_PER_CTA = BN / CTA_GROUP;
  constexpr int MMA_M = BM * CTA_GROUP; // total M across CTA group
  constexpr int MMA_K = 16;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  extern __shared__ __align__(128) uint8_t smem_raw[];
  auto &s = *reinterpret_cast<SMem<BM, BN_PER_CTA, BK, NUM_STAGES, NUM_STAGES, BN> *>(smem_raw);
  bf16 *sA = s.A;
  bf16 *sB = s.B;
  bf16 *sC = s.C;
  int *space = s.space;

  // Cluster rank
  uint32_t cluster_id = get_cluster_id();

  // Load schedule for this cluster
  if (tid < SPACE_LEN)
    space[tid] = dspace[cluster_id * SPACE_LEN + tid];

  int cta_rank = get_cluster_ctarank();

  const int num_blocks_k = K / BK;

  // smem addresses for mbarriers
  const int tma_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(s.tma_mbar));
  const int mma_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(s.mma_mbar));
  const int mainloop_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(&s.mainloop_mbar));
  const int tmem_smem_addr =
      static_cast<int>(__cvta_generic_to_shared(&s.tmem_addr));

  // Initialize barriers and allocate tmem
  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      // TMA mbar: both CTAs arrive (via expect_tx)
      init_barrier(&s.tma_mbar[i], CTA_GROUP, 0);
      // MMA mbar: 1 arrival from tcgen05.commit_mcast
      init_barrier(&s.mma_mbar[i], 1, 0);
    }
    // mainloop done: 1 arrival from tcgen05.commit_mcast
    init_barrier(&s.mainloop_mbar, 1, 0);
    fence_mbarrier_init_cluster();
  } else if (warp_id == 1) {
    // Allocate tmem (all 32 threads in warp must participate)
    tcgen05_alloc<CTA_GROUP>(tmem_smem_addr, BN);
  }

  // Cluster barrier to ensure init visible to all CTAs
  cluster_sync();

  const int taddr = s.tmem_addr; // tmem base (typically 0)

  // Schedule: each cluster processes BM × BN output tiles
  Schedule<2, NUM_SM / CLUSTERS, BM, BN, 16, 8> schedule(M, N, cluster_id,
                                                          space);

  // Smem sizes in bytes for A and B per stage
  constexpr int A_bytes = BM * BK * sizeof(bf16);
  constexpr int B_bytes = BN_PER_CTA * BK * sizeof(bf16);

  int num_block_m, num_block_n;
  while (schedule.next(num_block_m, num_block_n)) {
    int phase = 0;

    // ===== TMA warp =====
    if (warp_id == 0 && elect_sync()) {
      for (int iter_k = 0; iter_k < num_blocks_k; iter_k++) {
        int stage = iter_k % NUM_STAGES;

        // Wait for MMA to finish with this buffer (recycle)
        // Phase trick: mma_mbar starts at phase 0. Wait with phase^1
        // succeeds immediately on first use (parity mismatch = already past)
        wait(&s.mma_mbar[stage], phase ^ 1);

        // Flip phase when cycling through all stages
        if (stage == NUM_STAGES - 1)
          phase ^= 1;

        // Both CTAs arrive on CTA0's tma_mbar via cluster addressing
        // (& 0xFEFFFFFF maps any CTA's smem addr to CTA0's equivalent)
        int mbar_cta0 = (tma_mbar_addr + stage * 8) & 0xFEFFFFFF;

        // Arrive + set expected tx bytes (cluster-visible)
        mbarrier_arrive_expect_tx_cluster(mbar_cta0, A_bytes + B_bytes);

        int off_k = iter_k * BK;
        int a_smem = static_cast<int>(
            __cvta_generic_to_shared(&sA[stage * BM * BK]));
        int b_smem = static_cast<int>(
            __cvta_generic_to_shared(&sB[stage * BN_PER_CTA * BK]));

        // Load A — each CTA loads its OWN BM rows, targeting CTA0's mbar
        tma_load_3d_cta_group<CTA_GROUP>(
            a_smem, &tensorMapA, mbar_cta0,
            num_block_m * MMA_M + cta_rank * BM, off_k / 64);

        // Load B — each CTA loads its own HALF of B, targeting CTA0's mbar
        tma_load_3d_cta_group<CTA_GROUP>(
            b_smem, &tensorMapB, mbar_cta0,
            num_block_n * BN + cta_rank * BN_PER_CTA, off_k / 64);
      }
    }

    // ===== MMA warp =====
    if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
      for (int iter_k = 0; iter_k < num_blocks_k; iter_k++) {
        int stage = iter_k % NUM_STAGES;

        // Wait for TMA to load this stage
        wait(&s.tma_mbar[stage], phase);
        tcgen05_fence_after(); // needed after mbarrier wait, before MMA

        // Flip phase when cycling
        if (stage == NUM_STAGES - 1)
          phase ^= 1;

        // Issue MMA — 4 calls per stage (MMA_K=16, 64/16=4)
        int a_smem_base = static_cast<int>(
            __cvta_generic_to_shared(&sA[stage * BM * BK]));
        int b_smem_base = static_cast<int>(
            __cvta_generic_to_shared(&sB[stage * BN_PER_CTA * BK]));

        // First k2=0: enable_d = iter_k (0 to clear on first K iter)
        {
          tcgen05_mma<BN, MMA_M, CTA_GROUP>(
              taddr, make_smem_desc_int(a_smem_base),
              make_smem_desc_int(b_smem_base), iter_k);
          for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
            tcgen05_mma<BN, MMA_M, CTA_GROUP>(
                taddr, make_smem_desc_int(a_smem_base + k2 * 32),
                make_smem_desc_int(b_smem_base + k2 * 32), 1);
          }
        }
        // Remaining 64-column groups within BK
        for (int k1 = 1; k1 < BK / 64; k1++) {
          for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
            tcgen05_mma<BN, MMA_M, CTA_GROUP>(
                taddr,
                make_smem_desc_int(a_smem_base + k1 * BM * 128 + k2 * 32),
                make_smem_desc_int(b_smem_base + k1 * BN_PER_CTA * 128 + k2 * 32),
                1);
          }
        }

        // Commit MMA → arrive on mma_mbar in BOTH CTAs
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + stage * 8, cta_mask);
      }

      // Signal mainloop done to both CTAs
      constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
      tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr, cta_mask);
    }

    // ===== Epilogue: TMEM → shared memory → TMA store =====
    __syncthreads(); // ensure all warps reach here

    // Wait for mainloop to finish
    wait(&s.mainloop_mbar, 0);
    tcgen05_fence_after();

    // Each warp reads 32 dp-lanes from TMEM, 8 columns at a time
    // and writes to shared memory in TMA 3D layout:
    //   sC[(col/64)*BM*64 + row*64 + col%64]
    int dp_row = cta_rank * 128 + warp_id * 32;

    for (int n = 0; n < BN / 8; n++) {
      float tmp[8];
      int t_addr = taddr + (dp_row << 16) + (n * 8);

      tcgen05_ld_32x32b_x8(t_addr, tmp);
      tcgen05_wait_ld();

      // Convert f32 → bf16
      __nv_bfloat162 out[4];
      for (int i = 0; i < 4; i++)
        out[i] = __float22bfloat162_rn(make_float2(tmp[i * 2], tmp[i * 2 + 1]));

      // Store to shared memory in TMA 3D layout:
      // dim0 = col%64, dim1 = row, dim2 = col/64
      // smem_index = dim2 * (BM*64) + dim1 * 64 + dim0
      int col_group = n / 8;           // dim2
      int col_in_group = (n % 8) * 8;  // dim0 start
      int smem_idx = col_group * (64 * BM) + tid * 64 + col_in_group;
      reinterpret_cast<int4 *>(&sC[smem_idx])[0] =
          reinterpret_cast<int4 *>(out)[0];
    }

    __syncthreads(); // ensure all threads have written to smem

    // TMA store: bulk copy sC → global C
    if (warp_id == 0 && elect_sync()) {
      int global_row = num_block_m * MMA_M + cta_rank * BM;
      int global_col = num_block_n * BN;
      store_async(&tensorMapC, sC, global_col, global_row);
      cp_async_bulk_commit_group();
      cp_async_bulk_wait_group<0>();
    }

    __syncthreads(); // ensure store done before next tile

    // Re-init all barriers for next tile
    if (warp_id == 0 && elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        init_barrier(&s.tma_mbar[i], CTA_GROUP, 0);
        init_barrier(&s.mma_mbar[i], 1, 0);
      }
      init_barrier(&s.mainloop_mbar, 1, 0);
      fence_mbarrier_init_cluster();
    }
    cluster_sync();
  }

  // Deallocate tmem
  __syncthreads();
  if (warp_id == 0)
    tcgen05_dealloc<CTA_GROUP>(taddr, BN);
}

// ---------------------------------------------------------------------------
// Hilbert curve schedule
// ---------------------------------------------------------------------------
void rot(int n, int &x, int &y, int rx, int ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    int t = x;
    x = y;
    y = t;
  }
}

void d2xy(int n, int d, int &x, int &y) {
  int rx, ry, s, t = d;
  x = y = 0;
  for (s = 1; s < n; s *= 2) {
    rx = 1 & (t / 2);
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
  memset(space, -1, sizeof(int) * CORES * SPACE_LEN);
  int FCORES = 64;
  if (FCORES > CORES)
    FCORES = CORES;
  int total = 0;
  std::vector<std::vector<int>> pos(CORES, std::vector<int>());
  for (int i = 0; i < dim * dim; ++i) {
    int x, y;
    d2xy(dim, i, x, y);
    if (x < M && y < N) {
      assert((int)pos[core].size() < SPACE_LEN);
      assert(v[x][y] == '.');
      v[x][y] = '*';
      ++total;
      pos[core].push_back((x << 16) | y);
      ++core;
      if (core == FCORES) {
        core = 0;
      }
    }
  }
  core = FCORES;
  for (int i = 0; i < FCORES; ++i) {
    if (pos.back().size() >= pos[0].size() - 1)
      break;
    pos[core].push_back(pos[i].back());
    pos[i].pop_back();
    ++core;
    if (core == CORES) {
      core = FCORES;
    }
  }
  for (int i = 0; i < CORES; ++i) {
    for (int j = 0; j < (int)pos[i].size(); ++j) {
      space[i * SPACE_LEN + j] = pos[i][j];
    }
  }
  assert(total == M * N);
}

// ---------------------------------------------------------------------------
// Kernel launch wrapper
// ---------------------------------------------------------------------------
void runKernel1(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128; // 4 warps
  constexpr int NUM_STAGES = 3;
  constexpr int CLUSTER_M = 2;
  constexpr int CLUSTER_N = 1;
  constexpr int NUM_SM = 148;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
  constexpr int BN_PER_CTA = BN / CLUSTER_M;
  static_assert(NUM_SM % CLUSTERS == 0);

  if (_prev_m != M) {
    // A: each CTA loads BM rows × BK cols
    d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
    // B: each CTA loads BN/CTA_GROUP rows × BK cols (half of B)
    d_tma_map_B = create_tensor_map<BN_PER_CTA, BK>(B, N, K);
    // C: each CTA stores BM rows × BN cols via TMA (no swizzle)
    d_tma_map_C = create_tensor_map<BM, BN, false>(C, M, N);

    _prev_m = M;
    _prev_n = N;
    _prev_k = K;

    // Schedule: tiles are MMA_M × BN (256×256), assigned to clusters
    constexpr int MMA_M = BM * CLUSTER_M;
    int num_clusters = NUM_SM / CLUSTERS;
    int *space = (int *)malloc(sizeof(int) * num_clusters * SPACE_LEN);
    createHilbert(CEIL_DIV(M, MMA_M), CEIL_DIV(N, BN), num_clusters, space);
    cudaCheck(
        cudaMalloc((void **)&_dspace, sizeof(int) * num_clusters * SPACE_LEN));
    cudaCheck(cudaMemcpy(_dspace, space, sizeof(int) * num_clusters * SPACE_LEN,
                         cudaMemcpyHostToDevice));
    free(space);
  }
  assert(M == _prev_m && N == _prev_n && K == _prev_k);

  auto *kernel =
      matmulKernel1<BM, BN, BK, NUM_THREADS, NUM_STAGES, NUM_SM, CLUSTER_M,
                    CLUSTER_N>;
  constexpr size_t sMemSize =
      sizeof(SMem<BM, BN_PER_CTA, BK, NUM_STAGES, NUM_STAGES, BN>);
  static_assert(sMemSize < 256 * 1024);
  cudaCheck(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

  kernel<<<NUM_SM, NUM_THREADS, sMemSize>>>(M, N, K, d_tma_map_A, d_tma_map_B,
                                            d_tma_map_C, C, _dspace);
}

} // namespace M1

using M1::runKernel1;
