#include "reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
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
    float* arr = new float[n];
    float res = 0.0;

    // Initialize Open MP
    omp_set_num_threads(threads);

    // Initialization for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate array arr
    std::uniform_real_distribution<float> RD(-1,1);
    for (int i = 0; i<n; i++){
            arr[i] = RD(generator);
        }

    // Call and time reduce function
    start = std::chrono::high_resolution_clock::now();
    res = reduce(arr,0,n);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end-start);
    
    // Print results
    std::cout<<res<<"\n"<<ms.count()<<"\n";

    // Deallocate Memory
    delete[] arr;

    return(0);
}
