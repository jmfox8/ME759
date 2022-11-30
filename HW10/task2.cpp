#include "reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include <mpi.h>


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
    float global_res;

    // Initialize Open MP
    omp_set_num_threads(threads);

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate array arr
    std::uniform_real_distribution<float> RD(-1,1);
    for (int i = 0; i<n; i++){
            arr[i] = RD(generator);
        }

    // Initialize MPI
    int my_rank;
    double timestart, timeend;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    float res;
    MPI_Barrier(MPI_COMM_WORLD);
    timestart = MPI_Wtime();
    res = reduce(arr,0,n);
    MPI_Reduce(&res, &global_res,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    if (my_rank == 0){
        timeend = MPI_Wtime();
        double time = timeend-timestart;
        std::cout<<global_res<<"\n"<<time*1000<<"\n";
    }
    MPI_Finalize();

    // Deallocate Arrays
    delete[] arr;

    return(0);
}