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
    // Get Command Line Input
    //q0[0]= atof(argv[1]);
    //q0[1] = atof(argv[2]);


    int t_n = 1000; // Number of values attempted for each torque parameter

    //Initialize Variables and values to pass to device
    float tmax_amp, tmin_amp, tmax_dur, tmin_dur;
    segment vals;
    angular_vals q0;
    RK4out overall_best;
    tpulseinfo *torque_array;
    RK4out *output_bests;
    
    cudaMallocManaged((void**)&torque_array, t_n*t_n*sizeof(tpulseinfo));
    cudaMallocManaged((void**)&output_bests, t_n*t_n*sizeof(RK4out));
  
    // Initialize Variables for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Provide initial values and values for running RK4
    q0.q1 = -5*PI/180; // Initial position in Radians
    q0.q2 = 0*PI/180; // Initial velocity in Radians/s

    float h = 0.01; // Step size for RK4 solver method
    float sim_time = 0.5; //Ending time for RK4 solver in seconds

    vals.l = 0.867; // anthro table length of ankle to hip
    vals.lc = 0.589; // anthro table lenth of ankle to CM of legs
    vals.m = 26.30; // anthro table mass of lower leg segments
    vals.I = 1.4; // anthro table moment of intertia of leg segments
    vals.Icm = vals.I+vals.m*vals.lc*vals.lc;

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

    for (int i = 0; i < t_n; i++){
        for (int j = 0; j<t_n; j++){
            torque_array[i*t_n + j].amp = tmin_amp + i*t_amp_step;
            torque_array[i*t_n + j].duration = tmin_dur + j*t_dur_step;
        }
    }
    int threads_per_block = 1024;
    int full_blocks_needed = t_n*t_n / threads_per_block;

    // Call to RK4 kernel and timing
    start = std::chrono::high_resolution_clock::now();
        single_RK4<<<full_blocks_needed+1,threads_per_block>>>(sim_time,h,torque_array, q0, vals,output_bests, t_n);
        cudaDeviceSynchronize();
        for (int i = 0; i < t_n*t_n; i++){
        //std::cout << output_bests[i].norm << "\n";
        if (overall_best.norm > output_bests[i].norm) overall_best = output_bests[i];
    }

    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);
    
    std::cout << "time for loop: "<< ms.count() <<"\n";
    std::cout << "best performance - norm: " << overall_best.norm << " torque amp: " << overall_best.torque.amp << " torque time: " << overall_best.torque.duration << "\n";

    // Memory cleanup
    cudaFree(torque_array);
    cudaFree(output_bests);
}
