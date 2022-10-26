#include "count.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

int main(int argc, char *argv[]){

    // Get Command Line Input
    int n = atoi(argv[1]);

    // Allocate host vectors
    thrust::host_vector<int> h_in(n);

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    // Generate random float values and populate host vector
    std::uniform_real_distribution<float> RD(0,500);
    for (int i = 0; i<n; i++){
        h_in[i] = static_cast<int>(RD(generator));
    }

    // Sort host vector
    thrust::sort(h_in.begin(),h_in.end());
    // Get count on unique values in host vector
    int unique = 1;
    for (int i = 1; i<n; i++){
        if (h_in[i] != h_in[i-1]) unique += 1;
    }
           
    // Allocate device vectors
    thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> values(unique); 
    thrust::device_vector<int> counts(unique);

    // Call count function
    cudaEventRecord(start);
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get function elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    // Copy Device values to host
    thrust::host_vector<int> hvalues = values;
    thrust::host_vector<int> hcounts = counts;
    // Print Results
    printf("%d\n%d\n%f\n",hvalues[unique-1],hcounts[unique-1],ms);
    
return(0);
}