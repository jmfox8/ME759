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
        exit();
    }
    
    // Intialize and Allocate Memory for Arrays
    double* A = new float[n*n];
    double* B = new float[n*n];
    double* C = new float[n*n];

    // Initialization for timing
    duration<double, std::milli> ms;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;

    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());

    // Generate random float values and populate arrays
    std::uniform_real_distribution<float> RD(-10.0,10.0);
    for (int i = 0; i<n*n; i++){
            A[i] = RD(generator);
            B[i] = RD(generator);
        }

    // Execute Function Call with timing
    start = high_resolution_clock::now();
    mmul(A, B, C, n);
    end = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);


    //Print results
    std::cout << C[0] << "\n";
    std::cout << C[n*n-1] << "\n";
    std::cout << ms.count() <<"\n";
   
    // Deallocate Memory for Arrays
    delete[] A;
    delete[] B;
    delete[] C;
}