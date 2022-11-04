#include "msort.h"
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
    int t = atoi(argv[2]);
    int ts = atoi(argv[3]);
    if (t < 1 || t > 20){
        std::cout<<"Thread count out of bounds";
        exit(0);
    }

    // Intialize and Allocate Memory for Arrays
    int* arr = new int[n];

    // Initialization for OpenMP
    omp_set_num_threads(t); 
    
    // Initialization for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate arrays
    std::uniform_real_distribution<float> RD(-1000,1000);
    for (int i = 0; i<n; i++){
            arr[i] = static_cast<int>(RD(generator));
        }
        
    // Execute Function Call with timing
    start = std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end-start);

    //Print results
    std::cout << arr[0] << "\n";
    std::cout << arr[n-1] << "\n";
    std::cout << ms.count() <<"\n";
   
    // Deallocate Memory for Arrays
    delete[] arr;
}