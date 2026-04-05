NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w \
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math 

LD_FLAGS = -lcublas -lcuda

# Original H100 build for local H100 GPU 
matmul:
	nvcc $(NVCC_FLAGS) -arch=sm_90a $(LD_FLAGS) matmul_h100.cu -o out/matmul

h100: matmul_h100.cu
	.venv/bin/python3 main.py matmul_h100.cu \
		--gpu H100 \
		--image nvidia/cuda:12.4.1-devel-ubuntu22.04 \
		--yes \
		-- $(NVCC_FLAGS) -arch=sm_90a $(LD_FLAGS)

# B200 build + run via Modal
b200: matmul_b200.cu
	.venv/bin/python3 main.py matmul_b200.cu \
	    --gpu B200 \
	    --image nvidia/cuda:12.8.0-devel-ubuntu22.04 \
	    --yes \
	    -- $(NVCC_FLAGS) -arch=sm_100a $(LD_FLAGS)

.PHONY: matmul h100 b200
