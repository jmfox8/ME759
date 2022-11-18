#include "optimize.h"
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
    // Define input variable from command line
    int n = atoi(argv[1]);
    
    // Create vector
    vector<float> v(n);
    
    // Initialization for timing
    duration<double, std::milli> ms;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    
    // Allocate Memory for arrays
    float* inputarray = new float[n];
    float* outputarray = new float[n];
}