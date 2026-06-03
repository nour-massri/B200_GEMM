#include "../include/b200_utils.cuh"

__global__ void Kernel2(int M, int N, int K, const bf16 *A, const bf16 *B,
                        bf16 *C, int *D = nullptr) {

  int i = blockIdx.x;
  int j = blockIdx.y;
  float sum = 0.0;
  for (int k = 0; k < K; k++)
    sum +=
        __bfloat162float(*(A + i * K + k)) * __bfloat162float(*(B + j * K + k));
  *(C + i * N + j) = __float2bfloat16(sum);
}

void runKernel2(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C,
                int *D = nullptr) {

  dim3 gridDim(M, N);
  dim3 ctaDim(1);
  Kernel2<<<gridDim, ctaDim>>>(M, N, K, A, B, C, D);
}
