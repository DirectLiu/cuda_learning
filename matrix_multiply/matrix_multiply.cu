#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixMulBasic(float *C, float *A, float *B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMulTiled(float *C, float *A, float *B, int M, int N, int K) {
    // 为每个block分配共享内存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 计算线程对应的全局行列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // 遍历所以分块
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 将A和B的当前分块加载到共享内存
        int tiledK = t * BLOCK_SIZE;
        int aCol = tiledK + threadIdx.x;
        int bRow = tiledK + threadIdx.y;

        // 边界检查
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); //等待所有线程完成加载

        // 计算当前分块的乘积并累加
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads(); // 等待所以线程完成计算
    }

    // 写入结果
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matrixMultiply(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cudaMemcpyHostToDevice);

    // 计算网格和块维度
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE );

    // 启动核函数
    matrixMulTiled<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N, K);

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}