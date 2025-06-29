#include <iostream>
#include <cuda_runtime.h>

// daxpy kernel: y[i] = a * x[i] + y[i]
__global__ void daxpy(int n, double a, const double* x, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main()
{
    // Report device count
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        std::cout << "cudaGetDeviceCount error: "
                  << cudaGetErrorString(status) << std::endl;
        return 1;
    }
    std::cout << "Detected " << deviceCount 
              << " CUDA device(s)" << std::endl;
    if (deviceCount == 0) {
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name << std::endl;

    // Parameters for daxpy
    const int N = 1 << 20;     // 1 million elements
    const double a = 2.5;

    // Host allocations
    double *h_x = new double[N];
    double *h_y = new double[N];
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0;
        h_y[i] = 2.0;
    }

    // Device allocations
    double *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel: use 256 threads per block
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    daxpy<<<blocks, threadsPerBlock>>>(N, a, d_x, d_y);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back and check
    cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "y[0] = " << h_y[0]
              << "  (expected " << (a * 1.0 + 2.0) << ")" << std::endl;

    // Clean up
    delete[] h_x;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
