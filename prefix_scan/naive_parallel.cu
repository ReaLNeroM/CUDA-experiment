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
 * Implements prefix-scan using a Hillis-Steele algorithm.
 * Since Hillis-Steele assumes as many concurrent processors as data lines, we need to use
 * "double-buffering" to simulate concurrent modifications.
 * Since this algorithm requires 2 DRAM accesses per thread, this is a slow algorithm.
 * In my results, it's still faster than a CPU algorithm. GPUs are so cool :)
**/
__global__ void prefix_scan(const int n, const int jump, int* old, int* nnew){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(0 <= i - jump){
        nnew[i] = f(old[i], old[i - jump]);
    } else {
        nnew[i] = old[i];
    }
}

int main(){
    const int n = (1 << 28);
    const int block_size = BLOCKSIZE;
    assert(n % block_size == 0);

    int *x = (int *) malloc(n * sizeof(int));
    assert(x != NULL);

    for(int i = 0; i < n; i++){
        x[i] = 1;
    }

    int *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int block_count = n / block_size;
    for(int i = 1, j = 0; i < n; i *= 2, j++){
        prefix_scan<<<block_count, block_size>>>(n, i, d_x, d_y);
        std::swap(d_x, d_y);
    }

    int *result = (int *) malloc(n * sizeof(int));
    cudaMemcpy(result, d_x, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Test to make sure prefix scan is correct.
    for(int i = 0; i < n; i++){
        if(result[i] != i + 1){
            std::cerr << i << ' ' << i + 1 << ' ' << result[i] << '\n';
            return -1;
        }
    }

    std::cout << "memory usage: " << n * sizeof(int) << " bytes" << std::endl;
}