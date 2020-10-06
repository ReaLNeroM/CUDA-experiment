// Output of "python3 circuit2.py prefix_sum.crc" is pasted in as the kernel
// along with some boilerplate code to test this out.
// Expected output: "0 1 3 6 10 15 21 28\n".

#include <iostream>

__global__ void kogge_stone(int *x0, int *x1){
    int tid = threadIdx.x;
    x1[tid] = x0[tid];
    switch(tid){
        case 1:
            x1[1] += x0[0];
            break;
        case 2:
            x1[2] += x0[1];
            break;
        case 3:
            x1[3] += x0[2];
            break;
        case 4:
            x1[4] += x0[3];
            break;
        case 5:
            x1[5] += x0[4];
            break;
        case 6:
            x1[6] += x0[5];
            break;
        case 7:
            x1[7] += x0[6];
            break;
    }
    __syncthreads();
    x0[tid] = x1[tid];
    switch(tid){
        case 2:
            x0[2] += x1[0];
            break;
        case 3:
            x0[3] += x1[1];
            break;
        case 4:
            x0[4] += x1[2];
            break;
        case 5:
            x0[5] += x1[3];
            break;
        case 6:
            x0[6] += x1[4];
            break;
        case 7:
            x0[7] += x1[5];
            break;
    }
    __syncthreads();
    x1[tid] = x0[tid];
    switch(tid){
        case 4:
            x1[4] += x0[0];
            break;
        case 5:
            x1[5] += x0[1];
            break;
        case 6:
            x1[6] += x0[2];
            break;
        case 7:
            x1[7] += x0[3];
            break;
    }
    __syncthreads();
    x0[tid] = x1[tid];
    switch(tid){
    }
}

int main(){
    int *x;
    x = (int *) malloc(8 * sizeof(int));

    for(int i = 0; i < 8; i++){
        x[i] = i;
    }

    int *d_x0, *d_x1;

    cudaMalloc(&d_x0, 8 * sizeof(int));
    cudaMalloc(&d_x1, 8 * sizeof(int));

    cudaMemcpy(d_x0, x, 8 * sizeof(int), cudaMemcpyHostToDevice);

    kogge_stone<<<1, 8>>>(d_x0, d_x1);

    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x0, 8 * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 8; i++){
        std::cout << x[i] << ' ';
    }
    std::cout << '\n';
}
