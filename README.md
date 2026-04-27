# B200 GEMM in raw CUDA

Hand-written Blackwell (B200, sm_100a) bf16 × bf16 → bf16 matmul with fp32 accumulate. No CUTLASS, no Mojo, no high-level template library — just CUDA C++ and inline PTX (`tcgen05.mma`, `cp.async.bulk.tensor`, `mbarrier`, cluster sync).

```
N=8192 (square, bf16, fp32 accum)
Kernel: ~1490 TFLOPS
cuBLAS: ~1570 TFLOPS
```

## Lineage

- Started from pranjalssh's H100 fast.cu walkthrough — persistent-kernel + Hilbert-curve scheduling, warp specialization, TMA / mbarrier idioms: <https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog>
- Architectural cues for Blackwell (cta_group::2, TMEM accumulators, async commit chains) from Modular's B200 matmul series: <https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-1-introduction>

## What's actually here

- **2-CTA cluster** with `tcgen05.mma.cta_group::2` — a pair of CTAs cooperatively executes a 256×256 MMA, accumulating into TMEM.
- **TMA loads** (`cp.async.bulk.tensor.3d`) for A/B using cta_group::2 cooperation, async completion tracked via `mbarrier::complete_tx`.
- **Persistent kernel + Hilbert-curve tile schedule** for L2 reuse across tiles.
- **Warp specialization**: warp 0 = TMA producer (1 elected thread), warp 1 = MMA leader on CTA0 (1 elected thread), all 4 warps drive the epilogue.
- **Epilogue**: TMEM → smem → gmem via `tcgen05.ld` → bf16 conversion → TMA bulk store.

## Optimizations on top of the H100-inspired baseline

1. **Blackwell port** — replaced wgmma with `tcgen05.mma`, allocator-managed TMEM for accumulators, `cta_group::2` for cluster-cooperative MMA + commit. (1150 TFLOPS at this point.)
2. **Deeper pipeline via A/B ∪ C union** — the C epilogue smem aliases the A+B stages region. C is only live between `mainloop_mbar` completion and the next `cluster_sync`, during which A/B aren't accessed, so the same shared-memory bytes serve both. Frees ~64 KB / SM and lets `NUM_STAGES` grow from 3 to 5, giving HBM/L2 latency more room to hide. Took 1150 → ~1490 TFLOPS (+30%).

## Build / run

```
make b200
```

Runs the build + benchmark on Modal against a B200 SXM (see `Makefile` and `main.py`).

## TODO / open ideas (not yet landed)

- **2x2 cluster + A multicast** — halves A HBM traffic by row-broadcasting A across paired CTAs. Sync layer is fiddly: cross-pair WAR on multicast A smem (rank 0's TMA overwriting rank 2's smem before pair 2's MMA finishes reading) and forward RAW visibility of partner's A — both fixable with helper warps + dedicated mbars, but the cross-CTA arrive_cluster latency on the critical path made the version we got working slower than the 2x1 baseline. Sketch and stash preserved in git history.
- **Pipelined epilogue** — overlap TMEM → smem with smem → gmem store via double-buffered C halves; alternatively overlap the final store with the next tile's mainloop.
- **TMEM circular buffer** so the next tile's MMA can begin while the previous tile's epilogue is still draining TMEM.
- **Split b200 kernel** - split the single b200 kernel into multiple kernels each with a new optimization introduced.
