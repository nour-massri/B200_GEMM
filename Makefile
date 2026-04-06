NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w \
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math 

LD_FLAGS = -lcublas -lcuda

setup:
	python3.14 -m venv .venv
	.venv/bin/pip3 install -r requirements.txt

# --- LOCAL COMPILE TARGETS (Dry Run / Syntax Check) ---
# Use -c to compile without linking, so no GPU/Driver is required locally.

compile_h100:
	nvcc $(NVCC_FLAGS) -arch=sm_90a -c matmul_h100.cu -o out/matmul_h100.o

compile_b200:
	@rm -f out/matmul_b200.ptx
	@nvcc $(NVCC_FLAGS) -arch=compute_100a --ptx matmul_b200.cu -o out/matmul_b200.ptx 2>&1 \
		| grep -v '^ptxas' || true
	@test -f out/matmul_b200.ptx \
		&& echo "compile_b200: OK (PTX generated, ptxas skipped — needs CUDA 13.2)" \
		|| { echo "compile_b200: FAILED"; exit 1; }

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
	    --image nvidia/cuda:13.2.0-devel-ubuntu22.04 \
	    --yes \
	    -- $(NVCC_FLAGS) -gencode arch=compute_100a,code=sm_100a -Xptxas -O0 $(LD_FLAGS)

.PHONY: setup matmul h100 b200
