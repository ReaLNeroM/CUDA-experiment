#include <iostream>
#include <chrono>
#include <cassert>
#include <cstdlib>

__global__ void add_1(int n, float *x, float *y, float *ans){
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for(int i = tid; i < n; i += stride){
        ans[i] = x[i] + y[i];
    }
}

__global__ void add_2(int n, float *x, float *y, float *ans){
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for(int i = tid; i < n; i += stride){
        ans[i] = x[i] + y[i];
        __syncthreads();
    }
}


__global__ void add_3(int n, float *x, float *y, float *ans){
    int tid = threadIdx.x;
    int stride = blockDim.x;

    int rowstart = ((long long) tid * n) / stride;
    int rowend = ((long long) (tid + 1) * n) / stride;

    for(int i = rowstart; i < rowend; i++){
        ans[i] = x[i] + y[i];
    }
}

__global__ void add_4(int n, float *x, float *y, float *ans){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    ans[i] = x[i] + y[i];
}

int main(int argc, char** argv){
    assert(argc == 2);
    int algo_type = std::stoi(argv[1]);

    int n = (1 << 25);

    float *x, *y, *ans;

    x = (float *) malloc(n * sizeof(float));
    assert(x != NULL);
    y = (float *) malloc(n * sizeof(float));
    assert(y != NULL);
    ans = (float *) malloc(n * sizeof(float));
    assert(ans != NULL);

    for(int i = 0; i < n; i++){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    float *d_x, *d_y, *d_ans;
    cudaMalloc(&d_x, n * sizeof(float));
    assert(d_x != NULL);
    cudaMalloc(&d_y, n * sizeof(float));
    assert(d_y != NULL);
    cudaMalloc(&d_ans, n * sizeof(float));
    assert(d_ans != NULL);

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int grid_size = (n + 256 - 1) / 256;

    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
    if(algo_type == 1){
        add_1<<<1, 256>>>(n, d_x, d_y, d_ans);
    } else if(algo_type == 2){
        add_2<<<1, 256>>>(n, d_x, d_y, d_ans);
    } else if(algo_type == 3){
        add_3<<<1, 256>>>(n, d_x, d_y, d_ans);
    } else if(algo_type == 4){
        add_4<<<grid_size, 256>>>(n, d_x, d_y, d_ans);
    }
    cudaDeviceSynchronize();
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "add_" << algo_type << ": " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << "us\n";

    cudaMemcpy(ans, d_ans, n * sizeof(float), cudaMemcpyDeviceToHost);

    float err = 0.0;
    for(int i = 0; i < n; i++){
        err += abs(ans[i] - 3.0);
    }

    std::cout << err << '\n';

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_ans);

    free(x);
    free(y);
    free(ans);
}