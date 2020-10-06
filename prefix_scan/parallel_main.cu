#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define BLOCKSIZE 128

// MUST BE ASSOCIATIVE
__device__ inline int f(int a, int b){
    return a + b;
}

/**
 * Implements an inclusive prefix-scan algorithm ON EACH BLOCK using a recursive pattern modeled
 * after https://en.wikipedia.org/wiki/Prefix_sum#/media/File:Prefix_sum_16.svg.
 *
 * Arguments: x points to an array of size n, while out points to an array of size ceil(n / BLOCKSIZE).
 *
 * Description of stages:
 *
 * - First there's an downsweep step. In the first iteration, every second thread sums with
 *   its predecessor. In the second iteration, every fourth thread sums with its second predecessor.
 *   This continues until the last thread contains the sum of the whole block.
 *   For example, element #1 stores sum of #0 and #1. Element #7 stores sum of #0 through #7.
 *   Element #2 stores only itself, as it is an odd-ordered element (0-indexing!).
 *   For an 8-element array with all values = 1, the array will look like 1 2 1 4 1 2 1 8.
 *
 * - Then there's a upsweep step. This propagates changes from lower elements to upper elements.
 *   It is difficult to explain this algorithm without sounding like a robot, so I'd recommend
 *   the wikipedia link, as it shows visually what's happening for n=16.
 *   I'll just posit that the element #6 in the array 1 2 1 4 1 2 1 8 will get summed as
     elements #3 + #5 + #6.
**/

__global__ void scan(const int n, int *x, int *out){
    __shared__ int scan_v[BLOCKSIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // We use shared memory to reduce latency.
    scan_v[tid] = x[i];
    __syncthreads();

    // Downsweep step
    for(int i = 1, j = 1; j < BLOCKSIZE; i = (i << 1) + 1, j *= 2){
        if((tid & i) == i){
            scan_v[tid] = f(scan_v[tid], scan_v[tid - j]);
        }
        __syncthreads();
    }

    // Upsweep step
    for(int i = BLOCKSIZE / 2; i >= 1; i /= 2){
        int oth = tid - i;
        if(((tid + 1) & (2 * i - 1)) == i and 0 <= oth){
            scan_v[tid] = f(scan_v[tid], scan_v[oth]);
        }
        __syncthreads();
    }

    x[i] = scan_v[tid];
    if(tid == blockDim.x - 1){
        out[blockIdx.x] = scan_v[tid];
    }
}

/**
 * Given an array in of size n, and an array out of size n * BLOCKSIZE,
 * add the value in[i - 1] to out[BLOCKSIZE * i + k], for all k from [0, BLOCKSIZE).
**/
__global__ void propagate(const int n, int *in, int *out){
    int bid = blockIdx.x;
    int tcount = blockDim.x;
    int tid = threadIdx.x;
    int i = bid * tcount + tid;

    if(bid == 0){
        return;
    }

    out[i] = f(out[i], in[bid - 1]);
}

// Since this algorithm recursively calculates prefix scans on blocks, and blocks of blocks etc,
// we get the number of recursive calls we'll have to do, as well as the size of the array at each
// recursive step.
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
        scan<<<block_count, block_size>>>(levels[i - 1], d_arrays[i - 1], d_arrays[i]);
    }

    for(int i = levels.size() - 1; i >= 1; i--){
        int block_count = levels[i];
        propagate<<<block_count, block_size>>>(levels[i - 1], d_arrays[i], d_arrays[i - 1]);
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