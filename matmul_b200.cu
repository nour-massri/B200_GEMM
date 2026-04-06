#include <ctime>
#include <cublas_v2.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "include/b200_utils.cuh"
#include "include/utils.cuh"

#include "b200_kernels/matmul_1.cuh"

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // Our kernel computes C_row = A_row * B_row^T
  // cuBLAS col-major: C = B^T * A → in row-major: C_row[i][j] = A[i]·B[j]
  cublasStatus_t status =
      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, B,
                   CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C,
                int *DB = nullptr) {
  switch (kernel_num) {
  case 0:
    runCublasGemmBF16(M, N, K, A, B, C);
    break;
  case 1:
    runKernel1(M, N, K, A, B, C, nullptr);
    break;
  }
}
__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

int main() {
  warmupKernel<<<1024, 1024>>>();

  cublasCreate(&cublas_handle);
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  long max_size = 8192;
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr,
       *C_ref = nullptr; // host matrices
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr,
       *dC_ref = nullptr; // device matrices

  int *DB = nullptr;
  int *dDB = nullptr;

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 8;
  bool run_verif = true;
  for (int kernel_num : {0, 1}) {
    // for (int kernel_num : {0, 1}) {
    // Give the GPU some rest to avoid thermal throttling
    sleep(5);
    std::cout << "KERNEL " << kernel_num << std::endl;
    // Verify against cuBLAS. Also serves as a warmup step.
    if (run_verif) {
      memset(C, 0, sizeof(bf16) * max_size * max_size);
      cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size,
                           cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size,
                           cudaMemcpyHostToDevice));
      memset(DB, ~0, sizeof(int) * max_size * 128);
      cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128,
                           cudaMemcpyHostToDevice));
      run_kernel(0, m, n, k, dA, dB, dC_ref); // cuBLAS
      run_kernel(kernel_num, m, n, k, dA, dB, dC,
                 dDB); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size,
                 cudaMemcpyDeviceToHost);

      if (kernel_num >= 1 && !verify_matrix(C_ref, C, m * n)) {
        std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness "
                     "verification against cuBLAS. ~~~~~~~~~~~~~~~~"
                  << std::endl;
      }

      cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);

      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;
      while (DB[i] != ~0) {
        sumLoad += DB[i], cntLoad += DB[i + 1];
        sumCompute += DB[i + 2], cntCompute += DB[i + 3];
        sumStore += DB[i + 4], cntStore += DB[i + 5];
        i += 6;
        times++;
      }
      if (times > 0) {
        printf("Load: %f, Compute: %f,  Store: %f, Datapoints: %d\n",
               (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute,
               (sumStore + .0) / cntStore, times);
      }
    }

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    long flops = (2LL * m) * (n * k);
    printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. "
           "size: (%ld).\n\n",
           elapsed_time / 1000.0 / repeat_times,
           (repeat_times * flops * 1e-9) / elapsed_time, m);
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  return 0;
};
