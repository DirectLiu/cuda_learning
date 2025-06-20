#include <stdio.h>
#include <cuda_runtime.h>

__global__ void my_first_kernel()
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = gridDim.x;
    printf("Hello world from GPU thread(thread index:%d, block index:%d, grid dim: %d)\n", tid, bid, gid);
}

// total threads: block_size * grid_size
int main()
{
    printf("Hello World from CPU\n");

    int block_size = 3;
    int grid_size = 2;

    my_first_kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}