#include "matmul.cuh"
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
    unsigned int threads_per_block = atoi(argv[2]);

    // Initialize Arrays on the Host
    float* Ah = new float[n*n];
    float* Bh = new float[n*n];
    float* Ch = new float[n*n];

    // Initialize Arrays on Device
    float *Ad = NULL;
    float *Bd = NULL;
    float *Cd = NULL;
    cudaMalloc(&Ad, n*n*sizeof(float));
    cudaMalloc(&Bd, n*n*sizeof(float));
    cudaMalloc(&Cd, n*n*sizeof(float));




}