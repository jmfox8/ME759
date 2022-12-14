#include "single_definitions.cuh"
#include "RK4.cuh"
#include <cstddef>
#include <math.h>
#include <iostream>
#include <chrono>
#include <cuda.h>

#define PI 3.14159


// CONFIGURED FOR SINGLE INVERTED PENDULUM
int main(int argc, char *argv[]){

    //Initialize Variables and structs
    float tmax_amp, tmin_amp, tmax_dur, tmin_dur;
    segment vals;
    angular_vals q0;
    RK4out overall_best;
    tpulseinfo *torque_array;
    RK4out *output_bests;
    
    // Provide initial values and values for running RK4 from command line input
    q0.q1 = atof(argv[1])*PI/180; // Initial position in Radians
    q0.q2 = atof(argv[2])*PI/180; // Initial velocity in Radians/s
    int t_n = atoi(argv[3]); // Number of values attempted for each torque parameter
    
    float h = 0.01; // Step size for RK4 solver method
    float sim_time = 0.5; //Ending time for RK4 solver in seconds
    
    // Allocate managed memory based on number of torque parameters being used
    cudaMallocManaged((void**)&torque_array, t_n*t_n*sizeof(tpulseinfo));
    cudaMallocManaged((void**)&output_bests, t_n*t_n*sizeof(RK4out));
  
    // Initialize Variables for timing
    std::chrono::duration<double, std::milli> ms1, ms2;
    std::chrono::high_resolution_clock::time_point start1, start2;
    std::chrono::high_resolution_clock::time_point end;

    // Populate physical system values
    vals.l = 0.867; // anthro table length of ankle to hip
    vals.lc = 0.589; // anthro table lenth of ankle to CM of legs
    vals.m = 26.30; // anthro table mass of lower leg segments
    vals.Icm = 1.4; // anthro table moment of intertia of leg segments around CM
    vals.I = vals.I+vals.m*vals.lc*vals.lc; // moment of intertia of leg segments around ankle

    overall_best.norm = 100;
    
    if (q0.q1 <= 0 )
    {
        tmin_amp = 0;
        tmax_amp = 50;
    }
    else
    {
        tmax_amp = 0;
        tmin_amp = -50;
    }

    tmax_dur = 0.3;
    tmin_dur = 0.05;
    float t_dur_step = (tmax_dur - tmin_dur)/t_n;
    float t_amp_step = (tmax_amp - tmin_amp)/t_n;

    torque_array[0].amp = tmin_amp;
    torque_array[0].duration = tmin_dur;

    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < t_n; i++){
        for (int j = 0; j<t_n; j++){
            torque_array[i*t_n + j].amp = tmin_amp + i*t_amp_step;
            torque_array[i*t_n + j].duration = tmin_dur + j*t_dur_step;
        }
    }
    int threads_per_block = 1024;
    int full_blocks_needed = t_n*t_n / threads_per_block;

    // Call to RK4 kernel and timing
    start2 = std::chrono::high_resolution_clock::now();
        single_RK4<<<full_blocks_needed+1,threads_per_block>>>(sim_time,h,torque_array, q0, vals,output_bests, t_n);
        cudaDeviceSynchronize();
        for (int i = 0; i < t_n*t_n; i++){
            if (overall_best.norm > output_bests[i].norm) overall_best = output_bests[i];
            }

    end = std::chrono::high_resolution_clock::now();
    ms1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start1);
    ms2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start2);

    std::cout << "Time for RK4 loop: "<< ms2.count() <<"\nTime for Torque Allocation + RK4 Loop: "<< ms1.count()<<"\nTorque Steps: "<<t_n<<"\nInitial Values - q1: "<<q0.q1*180/PI<<" Degrees q2: "<<q0.q2*180/PI<<" Degrees/s\n";
    std::cout << "Best Performance - Norm: " << overall_best.norm << " Torque Amp: " << overall_best.torque.amp << " Torque Duration: " << overall_best.torque.duration<<"\n";

    // Memory cleanup
    cudaFree(torque_array);
    cudaFree(output_bests);
}
