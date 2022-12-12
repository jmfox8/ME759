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
    float q0[2], tmax_amp, tmin_amp, tmax_dur, tmin_dur;
    segment vals[2];
    // Get Command Line Input
    //q0[0]= atof(argv[1]);
    //q0[1] = atof(argv[2]);

    // Initialize Variables for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    q0[0] = -5*PI/180;
    q0[1] = 0*PI/180;

    float h = 0.01; // Step size for path solver [s]
    int t_n = 1000; // Number of values attempted for each torque parameter
    cudaMallocManaged((void**)&torque_array, t_n*t_n*sizeof(tpulseinfo));
    cudaMallocManaged((void**)&output_bests, t_n*t_n*sizeof(RK4out));
    RK4out overall_best;
    overall_best.norm = 100;
    
    vals[0].l = 0.867; // anthro table length of ankle to hip
    vals[0].lc = 0.589; // anthro table lenth of ankle to CM of legs
    vals[0].m = 26.30; // anthro table mass of lower leg segments
    vals[0].I = 1.4; // anthro table moment of intertia of leg segments
    vals[0].Icm = vals[0].I+vals[0].m*vals[0].lc*vals[0].lc;
    
    if (q0[0] <= 0 )
    {
        tmin_amp = 0;
        tmax_amp = 100;
    }
    else
    {
        tmax_amp = -50;
        tmin_amp = -100;
    }

    tmax_dur = 0.3;
    tmin_dur = 0.05;
    float t_dur_step = (tmax_dur - tmin_dur)/t_n;
    float t_amp_step = (tmax_amp - tmin_amp)/t_n;

    float sim_time = 0.5;
    torque_array[0].amp = tmin_amp;
    torque_array[0].duration = tmin_dur;

    for (int i = 0; i < t_n; i++){
        for (int j = 0; j<t_n; j++){
            torque_array[i*t_n + j].amp = tmin_amp + i*t_amp_step;
            torque_array[i*t_n + j].duration = tmin_dur + j*t_dur_step;
        }
    }
    start = std::chrono::high_resolution_clock::now();
    
        RK4<<<1,256>>(sim_time,h,torque_array[i], q0, vals,output_bests);
        for (int i = 0; i < t_n*t_n; i++){
        if (overall_best.norm > output_bests[i].norm) overall_best = output_bests[i];
    }

    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);
    std::cout << "time for loop: "<< ms.count() <<"\n";
    std::cout << "best performance - norm: " << overall_best.norm << "torque amp: " << overall_best.torque.amp << "torque time: " << overall_best.torque.duration << "\n";
}
