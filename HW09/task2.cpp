#include "montecarlo.h"
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
    if (threads < 1 || threads > 10){
        std::cout<<"Thread count out of bounds";
        exit(0);
    }
    
    // Intialize and Allocate Memory for Arrays
    float* x = new float[n];
    float* y = new float[n];

    // Initialize Variables
    float r = 1.0;
    float incircle;

    // Initialization for OpenMP
    omp_set_num_threads(threads); 
    
    // Initialization for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate array arr
    std::uniform_real_distribution<float> RD(-r,r);
    for (int i = 0; i<n; i++){
            x[i] = RD(generator);
            y[i] = RD(generator);
        }

    //Execute and time cluster function
    start = std::chrono::high_resolution_clock::now();
    incircle = montecarlo(n,x,y,r); 
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end-start);

    // Calculate estimated value of pi
    float piest = 4*incircle/n;

    // Print outputs
    printf("%f \n%f \n",piest,ms.count());

    // Deallocate Memory
    delete[] x;
    delete[] y;
    return(0);
}