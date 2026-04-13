#include "../include/b200_utils.cuh"
#include "../include/hilbert.hpp"
namespace M1 {

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;
CUtensorMap d_tma_map_C;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

constexpr int SPACE_LEN = 128;
int *_dspace;

struct Kernel1Params {
  static constexpr int BM          = 128;
  static constexpr int BN          = 256;
  static constexpr int BK          = 64;
  static constexpr int NUM_THREADS = 128;
  static constexpr int NUM_STAGES  = 3;
  static constexpr int NUM_SM      = 148;
  static constexpr int CLUSTER_M   = 2;
  static constexpr int CLUSTER_N   = 1;
};

// ---------------------------------------------------------------------------
// Kernel configuration: derives all compile-time constants from a Params struct.
// Define a Params struct with named fields, then pass it to KernelConfig.
// ---------------------------------------------------------------------------
template <typename Params>
struct KernelConfig : Params {
  using Params::BM;
  using Params::BN;
  using Params::BK;
  using Params::NUM_THREADS;
  using Params::NUM_STAGES;
  using Params::NUM_SM;
  using Params::CLUSTER_M;
  using Params::CLUSTER_N;

  // Derived constants
  static constexpr int CTA_GROUP = CLUSTER_M;
  static constexpr int BN_PER_CTA = BN / CTA_GROUP;
  static constexpr int MMA_M = BM * CTA_GROUP;
  static constexpr int MMA_K = 16;
  static constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  static constexpr int A_BYTES = BM * BK * sizeof(bf16);
  static constexpr int B_BYTES = BN_PER_CTA * BK * sizeof(bf16);

  // Number of MMA_K-wide sub-tiles within a 64-column group
  static constexpr int K2_ITERS = 64 / MMA_K;
  // Number of 64-column groups within BK
  static constexpr int K1_ITERS = BK / 64;

  // Strides within shared memory for the 3D TMA layout (in bytes)
  static constexpr int A_K1_STRIDE = BM * 128;        // stride between 64-col groups for A
  static constexpr int B_K1_STRIDE = BN_PER_CTA * 128; // stride between 64-col groups for B
  static constexpr int K2_STRIDE = 32;                 // stride between MMA_K sub-tiles (16 bf16 = 32 bytes)

  static constexpr int16_t CTA_MASK = (1 << CTA_GROUP) - 1;
};

// ---------------------------------------------------------------------------
// Runtime thread-level info: computed once at kernel entry from tid/cluster
// ---------------------------------------------------------------------------
struct ThreadInfo {
  int tid;
  int warp_id;
  int lane_id;
  uint32_t cluster_id;
  uint32_t cta_rank;

  __device__ __forceinline__ ThreadInfo() {
    tid = threadIdx.x;
    warp_id = tid / 32;
    lane_id = tid % 32;
    cluster_id = get_cluster_id();
    cta_rank = get_cluster_ctarank();
  }

  __device__ __forceinline__ bool is_tma_warp() const { return warp_id == 0; }
  __device__ __forceinline__ bool is_mma_warp() const { return warp_id == 1; }
  __device__ __forceinline__ bool is_cta0() const { return cta_rank == 0; }
};

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
template <typename Cfg>
struct SMem {
  alignas(128) bf16 A[Cfg::BM * Cfg::BK * Cfg::NUM_STAGES];
  alignas(128) bf16 B[Cfg::BN_PER_CTA * Cfg::BK * Cfg::NUM_STAGES];
  alignas(128) bf16 C[Cfg::BM * Cfg::BN]; // epilogue: TMEM -> sC -> TMA store
  alignas(8) uint64_t tma_mbar[Cfg::NUM_STAGES]; // TMA completion (data loaded)
  alignas(8) uint64_t mma_mbar[Cfg::NUM_STAGES]; // MMA completion (buffer recyclable)
  alignas(8) uint64_t mainloop_mbar;        // mainloop done signal
  int tmem_addr;                             // tmem base address from alloc
  int space[SPACE_LEN];
};

// ---------------------------------------------------------------------------
// Device helper: initialize / re-init all barriers
// ---------------------------------------------------------------------------
template <typename Cfg, typename SMem_t>
__device__ __forceinline__ void init_barriers(SMem_t &s, const ThreadInfo &t) {
  if (t.is_tma_warp() && elect_sync()) {
    for (int i = 0; i < Cfg::NUM_STAGES; i++) {
      init_barrier(&s.tma_mbar[i], Cfg::CTA_GROUP, 0);
      init_barrier(&s.mma_mbar[i], 1, 0);
    }
    init_barrier(&s.mainloop_mbar, 1, 0);
    fence_mbarrier_init_cluster();
  }
}

// ---------------------------------------------------------------------------
// Device helper: TMA producer warp — loads A and B tiles for all K iterations
// ---------------------------------------------------------------------------
template <typename Cfg, typename SMem_t>
__device__ __forceinline__ void tma_warp(
    SMem_t &s, const ThreadInfo &t,
    const CUtensorMap &tensorMapA, const CUtensorMap &tensorMapB,
    int num_blocks_k, int num_block_m, int num_block_n) {

  if (!(t.is_tma_warp() && elect_sync())) return;

  int phase = 0;

  for (int iter_k = 0; iter_k < num_blocks_k; iter_k++) {
    int stage = iter_k % Cfg::NUM_STAGES;

    // Wait for MMA to finish with this buffer (recycle)
    // Phase trick: mma_mbar starts at phase 0. Wait with phase^1
    // succeeds immediately on first use (parity mismatch = already past)
    wait(&s.mma_mbar[stage], phase ^ 1);

    if (stage == Cfg::NUM_STAGES - 1)
      phase ^= 1;

    // Both CTAs arrive on CTA0's tma_mbar via cluster addressing
    int mbar_cta0 = mbar_cta0_addr(&s.tma_mbar[stage]);

    mbarrier_arrive_expect_tx_cluster(mbar_cta0, Cfg::A_BYTES + Cfg::B_BYTES);

    int off_k = iter_k * Cfg::BK;

    // Load A — each CTA loads its OWN BM rows, targeting CTA0's mbar
    tma_load_3d_cta_group<Cfg::CTA_GROUP>(
        &s.A[stage * Cfg::BM * Cfg::BK], &tensorMapA, mbar_cta0,
        num_block_m * Cfg::MMA_M + t.cta_rank * Cfg::BM, off_k / 64);

    // Load B — each CTA loads its own HALF of B, targeting CTA0's mbar
    tma_load_3d_cta_group<Cfg::CTA_GROUP>(
        &s.B[stage * Cfg::BN_PER_CTA * Cfg::BK], &tensorMapB, mbar_cta0,
        num_block_n * Cfg::BN + t.cta_rank * Cfg::BN_PER_CTA, off_k / 64);
  }
}

// ---------------------------------------------------------------------------
// Device helper: MMA warp — issues tcgen05 MMA for all K iterations
//
// The MMA loop has two levels because shared memory uses 3D TMA tiling:
//   k1 (outer): iterates over 64-column groups within BK
//   k2 (inner): iterates over MMA_K=16 sub-tiles within each 64-col group
//
// enable_d controls accumulation: 0 = clear accumulator, 1 = accumulate.
// The first MMA call (iter_k==0, k1==0, k2==0) passes iter_k=0 to clear.
// All subsequent calls pass 1 to accumulate.
// ---------------------------------------------------------------------------
template <typename Cfg, typename SMem_t>
__device__ __forceinline__ void mma_warp(
    SMem_t &s, const ThreadInfo &t,
    int num_blocks_k) {

  if (!(t.is_cta0() && t.is_mma_warp() && elect_sync())) return;

  int taddr = s.tmem_addr;
  int phase = 0;

  for (int iter_k = 0; iter_k < num_blocks_k; iter_k++) {
    int stage = iter_k % Cfg::NUM_STAGES;

    // Wait for TMA to load this stage
    wait(&s.tma_mbar[stage], phase);
    tcgen05_fence_after();

    if (stage == Cfg::NUM_STAGES - 1)
      phase ^= 1;

    int a_smem_base = cvta_generic_to_shared(&s.A[stage * Cfg::BM * Cfg::BK]);
    int b_smem_base = cvta_generic_to_shared(&s.B[stage * Cfg::BN_PER_CTA * Cfg::BK]);

    // First k2=0 of k1=0: enable_d = iter_k (0 to clear on first K iter)
    {
      tcgen05_mma<Cfg::BN, Cfg::MMA_M, Cfg::CTA_GROUP>(
          taddr, make_smem_desc_int(a_smem_base),
          make_smem_desc_int(b_smem_base), iter_k);
      for (int k2 = 1; k2 < Cfg::K2_ITERS; k2++) {
        tcgen05_mma<Cfg::BN, Cfg::MMA_M, Cfg::CTA_GROUP>(
            taddr,
            make_smem_desc_int(a_smem_base + k2 * Cfg::K2_STRIDE),
            make_smem_desc_int(b_smem_base + k2 * Cfg::K2_STRIDE), 1);
      }
    }
    // Remaining 64-column groups within BK
    for (int k1 = 1; k1 < Cfg::K1_ITERS; k1++) {
      for (int k2 = 0; k2 < Cfg::K2_ITERS; k2++) {
        tcgen05_mma<Cfg::BN, Cfg::MMA_M, Cfg::CTA_GROUP>(
            taddr,
            make_smem_desc_int(a_smem_base + k1 * Cfg::A_K1_STRIDE + k2 * Cfg::K2_STRIDE),
            make_smem_desc_int(b_smem_base + k1 * Cfg::B_K1_STRIDE + k2 * Cfg::K2_STRIDE),
            1);
      }
    }

    // Commit MMA -> arrive on mma_mbar in BOTH CTAs
    tcgen05_commit_mcast<Cfg::CTA_GROUP>(&s.mma_mbar[stage], Cfg::CTA_MASK);
  }

  // Signal mainloop done to both CTAs
  tcgen05_commit_mcast<Cfg::CTA_GROUP>(&s.mainloop_mbar, Cfg::CTA_MASK);
}

// ---------------------------------------------------------------------------
// Device helper: Epilogue — TMEM -> shared memory -> TMA store
//
// Each warp reads 32 dp-lanes from TMEM, 8 columns at a time,
// converts f32 -> bf16, writes to shared in TMA 3D layout, then
// one elected thread does TMA store to global.
// ---------------------------------------------------------------------------
template <typename Cfg, typename SMem_t>
__device__ __forceinline__ void epilogue(
    SMem_t &s, const ThreadInfo &t,
    const CUtensorMap &tensorMapC,
    int num_block_m, int num_block_n) {

  __syncthreads();

  // Wait for mainloop to finish
  wait(&s.mainloop_mbar, 0);
  tcgen05_fence_after();

  int taddr = s.tmem_addr;
  int dp_row = t.cta_rank * 128 + t.warp_id * 32;

  for (int n = 0; n < Cfg::BN / 8; n++) {
    float tmp[8];
    int t_addr = taddr + (dp_row << 16) + (n * 8);

    tcgen05_ld_32x32b_x8(t_addr, tmp);
    tcgen05_wait_ld();

    // Convert f32 -> bf16
    __nv_bfloat162 out[4];
    for (int i = 0; i < 4; i++)
      out[i] = __float22bfloat162_rn(make_float2(tmp[i * 2], tmp[i * 2 + 1]));

    // Store to shared memory in TMA 3D layout:
    // dim0 = col%64, dim1 = row, dim2 = col/64
    // smem_index = dim2 * (BM*64) + dim1 * 64 + dim0
    int col_group = n / 8;
    int col_in_group = (n % 8) * 8;
    int smem_idx = col_group * (64 * Cfg::BM) + t.tid * 64 + col_in_group;
    reinterpret_cast<int4 *>(&s.C[smem_idx])[0] =
        reinterpret_cast<int4 *>(out)[0];
  }

  __syncthreads();

  // TMA store: bulk copy sC -> global C
  if (t.is_tma_warp() && elect_sync()) {
    int global_row = num_block_m * Cfg::MMA_M + t.cta_rank * Cfg::BM;
    int global_col = num_block_n * Cfg::BN;
    store_async(&tensorMapC, s.C, global_col, global_row);
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }

  __syncthreads();
}

// ---------------------------------------------------------------------------
// Kernel: warp-specialized GEMM using tcgen05 MMA
//
// Thread structure (128 threads = 4 warps):
//   Warp 0: TMA producer (1 elected thread loads data)
//   Warp 1: MMA (1 elected thread in CTA0 issues MMA)
//   All 4 warps: epilogue (read tmem, convert, store to global)
// ---------------------------------------------------------------------------
template <typename Cfg>
__global__
__launch_bounds__(Cfg::NUM_THREADS) void __cluster_dims__(Cfg::CLUSTER_M * Cfg::CLUSTER_N, 1, 1)
    matmulKernel1(int M, int N, int K,
                  const __grid_constant__ CUtensorMap tensorMapA,
                  const __grid_constant__ CUtensorMap tensorMapB,
                  const __grid_constant__ CUtensorMap tensorMapC,
                  bf16 *dC, int *dspace) {

  using SMem_t = SMem<Cfg>;

  extern __shared__ __align__(128) uint8_t smem_raw[];
  auto &s = *reinterpret_cast<SMem_t *>(smem_raw);

  ThreadInfo t;

  // Load schedule for this cluster
  if (t.tid < SPACE_LEN)
    s.space[t.tid] = dspace[t.cluster_id * SPACE_LEN + t.tid];

  const int num_blocks_k = K / Cfg::BK;

  // Initialize barriers and allocate tmem
  init_barriers<Cfg>(s, t);
  if (t.is_mma_warp())
    tcgen05_alloc<Cfg::CTA_GROUP>(&s.tmem_addr, Cfg::BN);
  cluster_sync();

  // Schedule: each cluster processes BM x BN output tiles
  Schedule<2, Cfg::NUM_SM / Cfg::CLUSTERS, Cfg::BM, Cfg::BN, 16, 8> schedule(
      M, N, t.cluster_id, s.space);

  int num_block_m, num_block_n;
  while (schedule.next(num_block_m, num_block_n)) {
    // ===== TMA warp =====
    tma_warp<Cfg>(s, t, tensorMapA, tensorMapB, num_blocks_k, num_block_m, num_block_n);

    // ===== MMA warp =====
    mma_warp<Cfg>(s, t, num_blocks_k);

    // ===== Epilogue: TMEM -> shared memory -> TMA store =====
    epilogue<Cfg>(s, t, tensorMapC, num_block_m, num_block_n);

    // Re-init all barriers for next tile
    init_barriers<Cfg>(s, t);
    cluster_sync();
  }

  // Deallocate tmem
  __syncthreads();
  if (t.is_tma_warp())
    tcgen05_dealloc<Cfg::CTA_GROUP>(s.tmem_addr, Cfg::BN);
}

// ---------------------------------------------------------------------------
// Kernel launch wrapper
// ---------------------------------------------------------------------------
void runKernel1(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
  using Cfg = KernelConfig<Kernel1Params>;
  static_assert(Cfg::NUM_SM % Cfg::CLUSTERS == 0);

  if (_prev_m != M) {
    d_tma_map_A = create_tensor_map<Cfg::BM, Cfg::BK>(A, M, K);
    d_tma_map_B = create_tensor_map<Cfg::BN_PER_CTA, Cfg::BK>(B, N, K);
    d_tma_map_C = create_tensor_map<Cfg::BM, Cfg::BN, false>(C, M, N);

    _prev_m = M;
    _prev_n = N;
    _prev_k = K;

    int num_clusters = Cfg::NUM_SM / Cfg::CLUSTERS;
    int *space = (int *)malloc(sizeof(int) * num_clusters * SPACE_LEN);
    createHilbert(CEIL_DIV(M, Cfg::MMA_M), CEIL_DIV(N, Cfg::BN), num_clusters, SPACE_LEN, space);
    cudaCheck(
        cudaMalloc((void **)&_dspace, sizeof(int) * num_clusters * SPACE_LEN));
    cudaCheck(cudaMemcpy(_dspace, space, sizeof(int) * num_clusters * SPACE_LEN,
                         cudaMemcpyHostToDevice));
    free(space);
  }
  assert(M == _prev_m && N == _prev_n && K == _prev_k);

  auto *kernel = matmulKernel1<Cfg>;
  constexpr size_t sMemSize = sizeof(SMem<Cfg>);
  static_assert(sMemSize < 256 * 1024);
  cudaCheck(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

  kernel<<<Cfg::NUM_SM, Cfg::NUM_THREADS, sMemSize>>>(M, N, K, d_tma_map_A, d_tma_map_B,
                                                       d_tma_map_C, C, _dspace);
}

} // namespace M1

using M1::runKernel1;
