#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define BLOCKSIZE 128
#define LOG_BLOCKSIZE 7

// MUST BE ASSOCIATIVE
__device__ inline int f(int a, int b){
    return a + b;
}

/**
 * In this variant, several optimizations have been applied:
 *
 * - Since only half the threads were doing work in parallel_main.cu, only BLOCKSIZE / 2 threads
 *   now make up a block, reducing the number of warps scheduled.
 *
 * - In parallel_cu, a warp would always have divergence in that some threads would do work and
 *   others would idle. In this variant, if the current downsweep iteration needs 8 workers, then
 *   only the first 8 threads of the block will actually do work, reducing dependence.
**/
__global__ void scan(const int n, int *x, int *out){
    __shared__ int scan_v[2 * BLOCKSIZE];
    int tid = threadIdx.x;
    int tcount = blockDim.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x * 2;

    scan_v[2 * tid] = x[i];
    scan_v[(2 * tid) ^ 1] = x[i ^ 1];
    __syncthreads();

    for(int i = 1, j = 1; j <= LOG_BLOCKSIZE; i = (i << 1) + 1, j++){
        int curr_tid = (tid << j) | i;
        int oth = curr_tid - (1 << (j - 1));
        if(curr_tid < 2 * tcount){
            scan_v[curr_tid] = f(scan_v[curr_tid], scan_v[oth]);
        }
        __syncthreads();
    }

    for(int i = BLOCKSIZE / 2, j = LOG_BLOCKSIZE - 1; i >= 1; i /= 2, j--){
        int curr_tid = ((tid + 1) << (j + 1)) | (i - 1);
        int oth = curr_tid - i;
        if(curr_tid < BLOCKSIZE){
            scan_v[curr_tid] = f(scan_v[curr_tid], scan_v[oth]);
        }
        __syncthreads();
    }

    x[i] = scan_v[2 * tid];
    x[i ^ 1] = scan_v[(2 * tid) ^ 1];
    if(tid == blockDim.x - 1){
        out[blockIdx.x] = scan_v[(2 * tid) ^ 1];
    }
}

/**
 * One slight optimization was done here:
 * - The first block in parallel_main.cu doesn't do any work. We shift each block by +1, resulting
 *   in one less thread block scheduled, and no need for the condition anymore.
**/
__global__ void propagate(const int n, int *in, int *out){
    int bid = blockIdx.x + 1;
    int tcount = blockDim.x;
    int tid = threadIdx.x;
    int i = bid * tcount + tid;

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