#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>

int main(int argc, char *argv[]){
    // Define input variables from command line
    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    if (threads < 1 || threads > 20){
        std::cout<<"Thread count out of bounds";
        exit(0);
    }
    int m = 3;

    // Allocate Memory for arrays
    float* image_arr = new float[n*n];
    float* mask_arr = new float[m*m];
    float* conv_arr = new float[n*n];

    // Initialization for OpenMP
    omp_set_num_threads(threads); 

    // Create test arrays
    //float test_image_arr[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
    //float* test_conv_arr = new float[16];
    //float test_mask_arr[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};

    // Initialization for timing
    std::chrono::duration<double, std::milli> duration_ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate and fill random values for the image matrix
    std::uniform_real_distribution<float> RD(-10.0,10.0);
    for(int i = 0; i<n*n; i++){
        image_arr[i]=RD(generator);
    }
    
    // Generate and fill random values for the mask matrix
    std::uniform_real_distribution<float> RD2(-1.0,1.0);
    for(int i = 0; i<m*m; i++){
        mask_arr[i]=RD2(generator);
    }
    
    //Execute and time Algorithm Execution
    start = std::chrono::high_resolution_clock::now();
    convolve(image_arr, conv_arr, n, mask_arr, m); 
    //convolve(test_image_arr, test_conv_arr, 4, test_mask_arr, 3);
    end = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end-start);

    //Print results
    std::cout << conv_arr[0] << "\n";
    std::cout << conv_arr[n*n-1] << "\n";
    std::cout << duration_ms.count() <<"\n";
   
    //Deallocate Preallocated Arrays
    delete[] image_arr;
    delete[] conv_arr;
    delete[] mask_arr;
    //delete[] test_conv_arr;
}