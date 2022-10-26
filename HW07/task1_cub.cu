#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char *argv[]) {

    // Get Command Line Input
    int n = atoi(argv[1]);
    
    // Set up host arrays
    float h_in[n];
    float sum = 0;
    
    // Initialization for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    // Generate random float values and populate host vector
    std::uniform_real_distribution<float> RD(-1.0,1.0);
    for (int i = 0; i<n; i++){
        h_in[i] = RD(generator);
        }
    
    // Set up device arrays
    float* d_in = NULL;
    (g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n));
    // Initialize device input
    cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);
    // Setup device output array
    float* d_sum = NULL;
    g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1);
    
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Do and time the actual reduce operation
    cudaEventRecord(start);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    // Get elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Print outputs
    printf("%f \n%f \n",gpu_sum,ms);

    // Cleanup
    if (d_in) g_allocator.DeviceFree(d_in);
    if (d_sum) g_allocator.DeviceFree(d_sum);
    if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
    
    return 0;
}