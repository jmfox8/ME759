#include "mpi.h"
#include <math.h>
#include <random>
#include <iostream>

int main(int argc, char *argv[]){
    // Get Command Line Input
    int n = atoi(argv[1]);

    // Intialize and Allocate Memory for Arrays
    float* arr1 = new float[n];
    float* arr2 = new float[n];

    // Initialize Randomization and Populate Arrays
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> RD(0,5);
    for (int i = 0; i<n; i++){
            arr1[i] = RD(generator);
            arr2[i] = RD(generator);
        }

    // Initialize Variables
    int my_rank,p;
    int tag1 = 0;
    int tag2 = 1;
    int tag3 = 2;
    MPI_Status status1, status2, status3;
    double time0_start, time0_end, time1_start, time1_end;

    // Initialize MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Call MPI Send, Receive and timing for master process
    if (my_rank == 0){    
        time0_start = MPI_Wtime();    
        MPI_Send(arr1, n, MPI_FLOAT, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(arr2, n, MPI_FLOAT, 1, tag2, MPI_COMM_WORLD, &status2);
        time0_end = MPI_Wtime();
        double time0 = time0_end - time0_start;
        MPI_Send(&time0, 1, MPI_DOUBLE, 1, tag3, MPI_COMM_WORLD);
    }
    
    // Call MPI Send, receive and timing for worker process
    else if (my_rank == 1){
        time1_start = MPI_Wtime();
        MPI_Recv(arr1, n, MPI_FLOAT, 0, tag1, MPI_COMM_WORLD, &status1);
        MPI_Send(arr2, n, MPI_FLOAT, 0, tag2, MPI_COMM_WORLD);
        time1_end = MPI_Wtime();
        double time0;
        double time1 = time1_end - time1_start;
        MPI_Recv(&time0, 1, MPI_DOUBLE, 1, tag3, MPI_COMM_WORLD, &status3);
        double timet = time0 + time1;
        printf("%f\n",timet);
    }

    MPI_Finalize();

}