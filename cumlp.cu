#include <cuda_runtime.h>
#include <stdio.h>

// Kernel runs on GPU
__global__ void addArrays(float* a, float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

int main() {
    int size = 1000;
    int bytes = size * sizeof(float);
    
    // 1. Allocate memory on HOST (CPU)
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_result = new float[size];
    
    // Initialize with some values
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 2. Allocate memory on DEVICE (GPU)
    float* d_a;
    float* d_b;
    float* d_result;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, bytes);
    
    // 3. Copy data from HOST to DEVICE
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // 4. Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    // 1000 + 255 / 256 = 4 blocks needed
    
    // <<<blocks, threads>>> syntax
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);
    
    // 5. Copy result back from DEVICE to HOST
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);
    
    // 6. Verify
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_result[i]);
    }
    
    // 7. Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
    
    return 0;
}