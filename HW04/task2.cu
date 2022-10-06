#include "stencil.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>

using namespace std;
int main(int argc, char *argv[]){

    // Get Command Line Input
    int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);

    // Initialize Arrays on the Host
    float* image = new float[n];
    float* output = new float[n];
    float* mask = new float[2*R+1];

    // Initialize Arrays on the Device
    float *imaged = NULL;
    float *outputd = NULL;
    float *maskd = NULL;
    cudaMalloc(&imaged, n*sizeof(float));
    cudaMalloc(&outputd, n*sizeof(float));
    cudaMalloc(&maskd, (2*R+1)*sizeof(float));

    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());
    
    // Generate random float values and populate host arrays
    uniform_real_distribution<float> RD(-1.0,1.0);
    for (int i = 0; i<n; i++){
        image[i] = RD(generator);
    }
    for (int j = 0; j<(2*R+1);j++){
        mask[j] = RD(generator);
    }
    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy Randomized Arrays from Host to Device
    cudaMemcpy(imaged,image,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(maskd,mask,(2*R+1)*sizeof(float),cudaMemcpyHostToDevice);


    // Call and time stencil function
    cudaEventRecord(start);
    stencil(imaged,maskd,outputd,n,R,threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate time taken
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy Results from device to host memory
    cudaMemcpy(output,outputd,n*sizeof(float),cudaMemcpyDeviceToHost);

    // Print Results
    cout << output[n-1] << "\n";
    cout << ms << "\n";

    // Deallocate memory
    cudaFree(outputd);
    cudaFree(imaged);
    cudaFree(maskd);
    delete[] output;
    delete[] image;
    delete[] mask;
}
