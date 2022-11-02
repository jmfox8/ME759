#include "matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>

int main(int argc, char *argv[]){
    // Get Command Line Input
    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    if (threads < 1 || threads > 20){
        std::cout<<"Thread count out of bounds";
        exit(0);
    }
    
    // Intialize and Allocate Memory for Arrays
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];

    // Initialization for OpenMP
    omp_set_num_threads(threads); 
    
    // Initialization for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate arrays
    std::uniform_real_distribution<float> RD(-10.0,10.0);
    for (int i = 0; i<n*n; i++){
            A[i] = RD(generator);
            B[i] = RD(generator);
        }
    
    // Execute Function Call with timing
    start = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end-start);

    //Print results
    std::cout << C[0] << "\n";
    std::cout << C[n*n-1] << "\n";
    std::cout << ms.count() <<"\n";
   
    // Deallocate Memory for Arrays
    delete[] A;
    delete[] B;
    delete[] C;
}