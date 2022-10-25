#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>

int main(int argc, char *argv[]){

    // Get Command Line Input
    int n = atoi(argv[1]);

    // Allocate host vector
    thrust::host_vector<float> rand_vec(n);
    
    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    
    // Generate random float values and populate host vector
    std::uniform_real_distribution<float> RD(-1.0,1.0);
    for (int i = 0; i<n; i++){
        rand_vec[i] = RD(generator);
        }
    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device vector and copy host values
    thrust::device_vector<float> d_vec(n);
    thrust::copy(rand_vec.begin(), rand_vec.end(), d_vec.begin());
    
    // Reduce data on device vector
    cudaEventRecord(start);
    float sum = thrust::reduce(d_vec.begin(), d_vec.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get function elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Print Results
    printf("%f \n",sum);
    printf("%f \n",ms);

    return(0);
}

