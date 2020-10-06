#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define BLOCKSIZE 256
#define LOG_BLOCKSIZE 8
#define BANKSIZE 32
#define LOG_BANKSIZE 5
// memory addresses 0, BANKSIZE, 2 * BANKSIZE belong to the same bank.
// at certain points of upsweep and downsweep of scan(), we would work on each BANKSIZE-th address,
// resulting in memory conflicts. We add an offset so that address 0 belongs to bank 0,
// address BANKSIZE belongs to bank 1 etc.
// We define an OFFSET macro, and use it whenever we use shared memory.
#define OFFSET(n) ((n) >> LOG_BANKSIZE)
#define MAXOFFSET ((BLOCKSIZE) >> LOG_BANKSIZE)

// MUST BE ASSOCIATIVE
__device__ inline int f(int a, int b){
    return a + b;
}

/**
 * This build on top of parallel_reversed.cu. In that version of the prefix-scan algorithm,
 * only one bank was used for strides >= banksize. This variant fixes this issue, by adding an
 * offset to each shared memory access. If address #i is referenced, then #(i+i/BANKSIZE) will be
 * used instead. So address 0 belongs to bank 0, address 32 belongs to bank 1, etc.
**/
__global__ void scan(const int n, int *x, int *out){
    // Also, padding needs to be added to scan_v, so that the offset doesn't
    // result in out-of-bounds memory accesses.
    __shared__ int scan_v[2 * BLOCKSIZE + 2 * MAXOFFSET];
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x * 2;

    scan_v[2 * tid + OFFSET(2 * tid)] = x[i];
    scan_v[((2 * tid) ^ 1) + OFFSET((2 * tid) ^ 1)] = x[i ^ 1];

    scan_v[((2 * tid) ^ 1) + OFFSET((2 * tid) ^ 1)] = f(
        scan_v[((2 * tid) ^ 1) + OFFSET((2 * tid) ^ 1)],
        scan_v[2 * tid + OFFSET(2 * tid)]
    );
    __syncthreads();

    for(int j = LOG_BLOCKSIZE - 2; j >= 0; j--){
        if(tid < (1 << j)){
            int j_complement = LOG_BLOCKSIZE - j;
            int curr_tid = (tid << j_complement) | ((1 << j_complement) - 1);
            int oth = curr_tid - (1 << (j_complement - 1));
            scan_v[curr_tid + OFFSET(curr_tid)] = f(
                scan_v[curr_tid + OFFSET(curr_tid)],
                scan_v[oth + OFFSET(oth)]
            );
        }
        __syncthreads();
    }

    for(int j = 0; j < LOG_BLOCKSIZE; j++){
        if(tid < (1 << j)){
            int j_complement = (LOG_BLOCKSIZE - 1) - j;
            int curr_tid = ((tid + 1) << (j_complement + 1)) | ((1 << j_complement) - 1);
            int oth = curr_tid - (1 << j_complement);
            scan_v[curr_tid + OFFSET(curr_tid)] = f(
                scan_v[curr_tid + OFFSET(curr_tid)],
                scan_v[oth + OFFSET(oth)]
            );
        }
        __syncthreads();
    }

    x[i] = scan_v[2 * tid + OFFSET(2 * tid)];
    x[i ^ 1] = scan_v[((2 * tid) ^ 1) + OFFSET((2 * tid) ^ 1)];
    if(tid == blockDim.x - 1){
        out[blockIdx.x] = scan_v[((2 * tid) ^ 1) + OFFSET((2 * tid) ^ 1)];
    }
}

__global__ void propagate(const int n, int *in, int *out){
    int bid = blockIdx.x + 1;
    int i = bid * blockDim.x + threadIdx.x;

    out[i] = f(out[i], in[bid - 1]);
}

std::vector<int> get_levels(const int n, int block_size){
    std::vector<int> res;

    int x = n;
    while(x > 1){
        res.push_back(x);
        x = (x + block_size - 1) / block_size;
    }

    res.push_back(1);

    return res;
}

int main(){
    const int n = (1 << 28);
    const int block_size = BLOCKSIZE;
    assert(n % block_size == 0);

    std::vector<int> levels = get_levels(n, block_size);

    for(int i : levels){
        std::cout << i << ' ';
    }
    std::cout << std::endl;

    int *x = (int *) malloc(n * sizeof(int));
    assert(x != NULL);

    for(int i = 0; i < n; i++){
        x[i] = 1;
    }

    int *d_arrays[levels.size()];
    for(int i = 0; i < levels.size(); i++){
        cudaMalloc(&d_arrays[i], levels[i] * sizeof(int));
        assert(d_arrays[i] != NULL);
    }

    cudaMemcpy(d_arrays[0], x, levels[0] * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 1; i < levels.size(); i++){
        int block_count = levels[i];
        scan<<<block_count, block_size / 2>>>(levels[i - 1], d_arrays[i - 1], d_arrays[i]);
    }

    for(int i = levels.size() - 2; i >= 1; i--){
        int block_count = levels[i];
        propagate<<<block_count - 1, block_size>>>(levels[i - 1], d_arrays[i], d_arrays[i - 1]);
    }

    int *result = (int *) malloc(n * sizeof(int));
    cudaMemcpy(result, d_arrays[0], n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++){
        if(result[i] != i + 1){
            std::cerr << i << ' ' << i + 1 << ' ' << result[i] << '\n';
            return -1;
        }
    }

    std::cout << "memory usage: " << n * sizeof(int) << " bytes" << std::endl;
}