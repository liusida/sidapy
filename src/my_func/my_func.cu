#include <stdio.h>

#include <npp.h>

#include "my_func.h"

__global__ void kernel(int total_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < total_num && j < total_num) {
        printf("kernel %d %d\n", i,j);
    }
}

void my_func() {
    int N = 5;
    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    kernel<<<dimGrid, dimBlock>>>(N);
    cudaDeviceSynchronize();
}