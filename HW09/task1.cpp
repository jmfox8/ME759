#include "cluster.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <algorithm> // std::sort
#include <chrono>
#include <omp.h>


int main(int argc, char *argv[]){
    // Get Command Line Input
    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    if (threads < 1 || threads > 10){
        std::cout<<"Thread count out of bounds";
        exit(0);
    }
    
    // Intialize and Allocate Memory for Arrays
    float* arr = new float[n];
    float* centers = new float[threads];
    float* dists = new float[threads];
    
    // Initialization for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate array arr
    std::uniform_real_distribution<float> RD(0,n);
    for (int i = 0; i<n; i++){
            arr[i] = RD(generator);
        }
    // Sort array arr
    std::sort(arr, arr+n);
    
    //Populate array dists
    for (int i = 1; i<=threads; i++){
        centers[i-1] = (2*i-1)*n/2/threads;
    }

    //Execute and time cluster function
    start = std::chrono::high_resolution_clock::now();
    cluster(n,threads,arr,centers,dists); 
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end-start);

    // Calculate maximum of dists array
    float dist_max = *std::max_element(dists,dists+threads);
    int dist_max_index = std::find(dists, dists+threads, dist_max) - dists;

    // Print results
    printf("%f \n%d \n",dist_max,dist_max_index);
    printf("%f \n",ms.count());

    // Deallocate Memory
    delete[] arr;
    delete[] centers;
    delete[] dists;
    return(0);
}
