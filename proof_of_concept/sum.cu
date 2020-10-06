// Output of "python3 circuit2.py sum.crc" is pasted in as the kernel
// along with some boilerplate code to test this out.
// Expected output: "28 1 5 3 22 5 13 7\n".

#include <iostream>

__global__ void sum(int *x0, int *x1){
    int tid = threadIdx.x;
    x1[tid] = x0[tid];
    switch(tid){
        case 0:
            x1[0] += x0[1];
            break;
        case 2:
            x1[2] += x0[3];
            break;
        case 4:
            x1[4] += x0[5];
            break;
        case 6:
            x1[6] += x0[7];
            break;
    }
    __syncthreads();
    x0[tid] = x1[tid];
    switch(tid){
        case 0:
            x0[0] += x1[2];
            break;
        case 4:
            x0[4] += x1[6];
            break;
    }
    __syncthreads();
    x1[tid] = x0[tid];
    switch(tid){
        case 0:
            x1[0] += x0[4];
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

    sum<<<1, 8>>>(d_x0, d_x1);

    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x0, 8 * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 8; i++){
        std::cout << x[i] << ' ';
    }
    std::cout << '\n';
}
