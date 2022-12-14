#include "single_definitions.h"
#include "RK4.h"
#include <cstddef>
#include <math.h>
#include <iostream>
#include <chrono>
//#include <omp.h>

#define PI 3.14159


// CONFIGURED FOR SINGLE INVERTED PENDULUM
int main(int argc, char *argv[]){
   
    // Initialize Variables
    float tmax_amp, tmin_amp, tmax_dur, tmin_dur;
    segment vals[2];
    angular_vals q0;
    RK4out overall_best;
    
    // Provide initial values and values for running RK4 from command line input
    q0.q1 = atof(argv[1])*PI/180; // Initial position in Radians
    q0.q2 = atof(argv[2])*PI/180; // Initial velocity in Radians/s
    int t_n = atoi(argv[3]); // Number of values attempted for each torque parameter

    float h = 0.01; // Step size for RK4 solver method
    float sim_time = 0.5; //Ending time for RK4 solver in seconds

    //Allocate memory based on number of torque parameters being used
    tpulseinfo* torque_array = new tpulseinfo[t_n*t_n];
    RK4out* output_bests = new RK4out[t_n*t_n];

    overall_best.norm = 100;

    // Initialize Variables for timing
    std::chrono::duration<double, std::milli> ms1, ms2;
    std::chrono::high_resolution_clock::time_point start1, start2;
    std::chrono::high_resolution_clock::time_point end;

    // Populate physical system values
    vals[0].l = 0.867; // anthro table length of ankle to hip
    vals[0].lc = 0.589; // anthro table lenth of ankle to CM of legs
    vals[0].m = 26.30; // anthro table mass of lower leg segments
    vals[0].Icm = 1.4; // anthro table moment of intertia of leg segment around CM
    vals[0].I = vals[0].I+vals[0].m*vals[0].lc*vals[0].lc; // moment of inertia of leg segment around ankle
    
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
    start2 = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel num_threads(20)
    {
        //#pragma omp  for
            for (int i = 0; i < t_n*t_n; i++){
                //std::cout<<torque_array[i].amp<<"\n";
                output_bests[i] = RK4(sim_time,h,torque_array[i], q0, vals);
                if (overall_best.norm > output_bests[i].norm) overall_best = output_bests[i];
            }
    }
    end = std::chrono::high_resolution_clock::now();
    ms1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start1);
    ms2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start2);
    std::cout << "Time for RK4 loop: "<< ms2.count() <<"\n Time for Torque Allocation + RK4 Loop: "<< ms1.count()<<"\nTorque Steps: "<<t_n<<"\nInitial Values - q1: "<<q0.q1*180/PI<<" degrees q2: "<<q0.q2*180/PI<<" degrees/s\n";
    std::cout << "Best Performance - Norm: " << overall_best.norm << " Torque Amp: " << overall_best.torque.amp << " Torque Duration: " << overall_best.torque.duration<<"\n";
}
