#pragma once
// B200 (SM100 / Blackwell) PTX helper wrappers.
// All TCGEN05, TMEM, cluster and mbarrier primitives used across matmul
// kernels.

#include "utils.cuh"
#include <cassert>
#include <cuda.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Cluster topology queries
// ---------------------------------------------------------------------------

// Returns the cluster ID of the current CTA within the grid.
__device__ static __forceinline__ uint32_t get_cluster_id() {
  uint32_t id;
  asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(id));
  return id;
}

// Returns the CTA rank within the current cluster (0..cluster_size-1).
__device__ static __forceinline__ uint32_t get_cluster_ctarank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank));
  return rank;
}

// ---------------------------------------------------------------------------
// Cluster-level barriers
// ---------------------------------------------------------------------------

// Ensures all mbarrier inits from this CTA are visible across the cluster.
__device__ static __forceinline__ void fence_mbarrier_init_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// Cluster barrier: signal arrival (release semantics).
__device__ static __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;");
}

// Cluster barrier: wait for all CTAs in the cluster to arrive (acquire semantics).
__device__ static __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;");
}

// Cluster barrier: arrive + wait (full sync across cluster).
__device__ static __forceinline__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// ---------------------------------------------------------------------------
// Mbarrier arrive with expected TX bytes (cluster-visible, raw smem address)
// ---------------------------------------------------------------------------

// Arrive on a cluster-visible mbarrier and set expected transaction bytes.
// mbar_addr: shared memory address of the mbarrier (cluster-visible int addr)
// bytes: expected number of bytes to be delivered by async operations.
__device__ static __forceinline__ void
mbarrier_arrive_expect_tx_cluster(int mbar_addr, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, "
      "[%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

// ---------------------------------------------------------------------------
// TMA 3D async load with cta_group (raw smem addresses)
// ---------------------------------------------------------------------------

// TMA bulk tensor 3D load with cta_group cooperation, using raw int addresses.
// dst_smem:   shared memory destination address (int)
// tensor_map: pointer to the CUtensorMap descriptor
// mbar_addr:  cluster-visible mbarrier address (int) for completion tracking
// dim1, dim2: tile coordinates (dim0 is always 0 for 3D TMA tiling)
template <int CTA_GROUP>
__device__ static __forceinline__ void
tma_load_3d_cta_group(int dst_smem, const CUtensorMap *tensor_map,
                      int mbar_addr, int dim1, int dim2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
      "complete_tx::bytes.cta_group::%4"
      " [%0], [%1, {%3, %5, %6}], [%2];"
      :
      : "r"(dst_smem), "l"((uint64_t)tensor_map), "r"(mbar_addr),
        "n"(0), "n"(CTA_GROUP), "r"(dim1), "r"(dim2)
      : "memory");
}

// ---------------------------------------------------------------------------
// TCGEN05 TMEM load
// ---------------------------------------------------------------------------

// Load 8 x 32-bit values from tmem (32x32b tile, x8 elements).
// tmem_addr: tmem address encoding (dp_row << 16 | col_offset)
// out[8]:    destination array for 8 float values
__device__ static __forceinline__ void tcgen05_ld_32x32b_x8(int tmem_addr,
                                                             float out[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
        "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
      : "r"(tmem_addr));
}

// ---------------------------------------------------------------------------
// Async bulk copy commit/wait
// ---------------------------------------------------------------------------

// Commit all pending async bulk copy operations in the current group.
__device__ static __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

// Wait until all async bulk copy groups have completed.
// N: number of groups allowed to remain in-flight (0 = wait for all).
template <int N = 0>
__device__ static __forceinline__ void cp_async_bulk_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N));
}

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------
__device__ static inline uint64_t desc_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 0x4);
}

// TCGEN05 shared memory descriptor (different from SM90 WGMMA descriptor!)
// Layout: addr[0:13] | SBO[32:45] | LBO[46] | swizzle[61:62]
__device__ uint64_t make_smem_desc(bf16 *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  constexpr uint64_t SBO = 8 * 128; // stride byte offset = 1024
  return desc_encode(addr) | (desc_encode(SBO) << 32) | (1ULL << 46) // LBO = 1
         | (2ULL << 61); // 128B swizzle (mode 2)
}

// Same but takes an int smem address directly (matching reference pattern)
__device__ uint64_t make_smem_desc_int(int addr) {
  constexpr int SBO = 8 * 128;
  return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

// ---------------------------------------------------------------------------
// Warp-level elect (returns 1 for exactly one thread in the warp)
// ---------------------------------------------------------------------------
__device__ static __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t"
               "}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

// ---------------------------------------------------------------------------
// Warpgroup register management (setmaxnreg)
// ---------------------------------------------------------------------------
template <uint32_t RegCount> __device__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> __device__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// ---------------------------------------------------------------------------
// TCGEN05 tmem alloc / dealloc
// ---------------------------------------------------------------------------
template <int CTA_GROUP = 1>
__device__ __forceinline__ void tcgen05_alloc(int smem_addr, int size) {
  asm volatile(
      "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          smem_addr),
      "r"(size), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void tcgen05_dealloc(int taddr, int size) {
  asm volatile(
      "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(size), "n"(CTA_GROUP));
}

// ---------------------------------------------------------------------------
// TCGEN05 MMA — proper instruction descriptor encoding
//
// i_desc bit layout (from PTX ISA):
//   [0]    TransA     [1]    TransB
//   [2]    NegateA    [3]    NegateB
//   [4:5]  D type (01=FP32)
//   [7:8]  A type (01=BF16)
//   [10:11] B type (01=BF16)
//   [17:23] MMA_N >> 3
//   [24:31] MMA_M >> 4
//
// enable_d: 0 = clear accumulator, 1 = accumulate
// ---------------------------------------------------------------------------
template <int BN, int MMA_M, int CTA_GROUP>
__device__ __forceinline__ void tcgen05_mma(uint32_t tmem_addr, uint64_t desc_a,
                                            uint64_t desc_b, int enable_d) {
  constexpr uint32_t i_desc =
      (1U << 4)                                   // D type = FP32
      | (1U << 7)                                  // A type = BF16
      | (1U << 10)                                 // B type = BF16
      | ((uint32_t)(BN) >> 3 << 17)                // MMA_N
      | ((uint32_t)(MMA_M) >> 4 << 24);            // MMA_M

  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
               "}"
               :
               : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(i_desc),
                 "r"(enable_d), "n"(CTA_GROUP));
}

// ---------------------------------------------------------------------------
// TCGEN05 commit / commit_mcast
// ---------------------------------------------------------------------------
template <int CTA_GROUP = 1>
__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"(mbar_addr),
      "n"(CTA_GROUP)
      : "memory");
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void tcgen05_commit_mcast(int mbar_addr,
                                                     int16_t cta_mask) {
  asm volatile(
      "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;" ::"r"(mbar_addr),
      "h"(cta_mask), "n"(CTA_GROUP)
      : "memory");
}

// ---------------------------------------------------------------------------
// TCGEN05 fences and waits
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tcgen05_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// ---------------------------------------------------------------------------
// TMA tensor map creation
//
// 2D variant (M2-M6): tiles up to 64 columns wide, descriptor written to
// *tma_map 3D variant (M8-M12): supports wide tiles via 3D tiling, returns by
// value
// ---------------------------------------------------------------------------
template <int BlockMajorSize, int BlockMinorSize>
__host__ inline void create_tensor_map(CUtensorMap *tma_map, bf16 *gmem_ptr,
                                       int blocks_height, int blocks_width) {
  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width,
                                 (uint64_t)BlockMajorSize * blocks_height, 1, 1,
                                 1};
  uint64_t gmem_prob_stride[5] = {
      sizeof(bf16), sizeof(bf16) * BlockMinorSize * blocks_width, 0, 0, 0};
  uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize),
                                uint32_t(BlockMajorSize), 1, 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  CUresult result = cuTensorMapEncodeTiled(
      tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
      gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  assert(result == CUDA_SUCCESS);
}

template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true,
          bool padding = false>
__host__ static inline CUtensorMap
create_tensor_map(bf16 *gmem_ptr, int global_height, int global_width) {
  CUtensorMap tma_map;
  void *gmem_address = (void *)gmem_ptr;
  static_assert(BlockMinorSize >= 64);
  assert(global_width % 64 == 0);
  uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height,
                                 (uint64_t)global_width / 64, 1, 1};
  uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width,
                                  64 * sizeof(bf16), 0, 0, 0};
  uint32_t smem_box_shape[5] = {padding ? 72u : 64u, uint32_t(BlockMajorSize),
                                uint32_t(BlockMinorSize / 64), 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  CUresult result = cuTensorMapEncodeTiled(
      &tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_address,
      gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  assert(result == CUDA_SUCCESS);
  return tma_map;
}

// ---------------------------------------------------------------------------
// PTX mbarrier primitives (SM90+)
// ---------------------------------------------------------------------------
__device__ static __forceinline__ void
init_barrier(uint64_t *bar, int thread_count, int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void expect_bytes(uint64_t *bar,
                                                    uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          bar_ptr),
      "r"(bytes));
}

// Arrive with expect_tx using cluster-visible address
__device__ static __forceinline__ void
expect_bytes_cluster(uint64_t *bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, "
               "[%0], %1;" ::"r"(bar_ptr),
               "r"(bytes)
               : "memory");
}

// 3D TMA async load
__device__ static inline void load_async(bf16 *dst, void const *src_tma_map,
                                         uint64_t *bar, int global_col_idx,
                                         int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  // 3D tiling: dim0=64 cols, dim1=row, dim2=col/64
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
                 "r"(global_row_idx), "r"(global_col_idx / 64)
               : "memory");
}

// 3D TMA async load with cta_group (for cta_group::2 TMA cooperation)
template <int CTA_GROUP = 1>
__device__ static inline void
load_async_cg(bf16 *dst, void const *src_tma_map, uint64_t *bar,
              int global_col_idx, int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
      "complete_tx::bytes.cta_group::%6"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
        "r"(global_row_idx), "r"(global_col_idx / 64), "n"(CTA_GROUP)
      : "memory");
}

__device__ static inline void store_async(void const *dst_tma_map, bf16 *src,
                                          int global_col_idx,
                                          int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
               " [%0, {%2, %3, %4}], [%1];"
               :
               : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(global_row_idx),
                 "r"(global_col_idx / 64)
               : "memory");
}

__device__ static __forceinline__ void wait(uint64_t *bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
               "%1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(kPhaseBit));
}

__device__ static __forceinline__ void arrive(uint64_t *bar,
                                              uint32_t count = 1) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

__device__ void arrive_cluster(uint64_t *bar, uint32_t cta_id,
                               uint32_t count = 1) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .b32 remAddr32;\n"
               "mapa.shared::cluster.u32  remAddr32, %0, %1;\n"
               "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n"
               "}" ::"r"(smem_addr),
               "r"(cta_id), "r"(count));
}

__device__ static __forceinline__ void wait_cluster(uint64_t *bar,
                                                    int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cluster.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(kPhaseBit));
}

__device__ static inline void
load_async_multicast(bf16 *dst, void const *src_tma_map, uint64_t *bar,
                     int global_col_idx, int global_row_idx,
                     uint16_t cluster_mask) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
               "complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
                 "r"(global_row_idx), "r"(global_col_idx / 64),
                 "h"(cluster_mask)
               : "memory");
}
