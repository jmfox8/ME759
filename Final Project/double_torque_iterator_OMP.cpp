#include "double_definitions.h"
#include "double_RK4.h"
#include <cstddef>
#include <math.h>
#include <iostream>
#include <chrono>
//#include <omp.h>

#define PI 3.14159

// CONFIGURED FOR SINGLE INVERTED PENDULUM
int main(int argc, char *argv[]){
    // Get Command Line Input
    //q0[0]= atof(argv[1]);
    //q0[1] = atof(argv[2]);
    
    // Initialize Variables
    float tmax_amp, tmin_amp, tmax_dur, tmin_dur, tmax_ratio, tmin_ratio;
    segment vals[2];
    angular_vals2 q0;
    float h = 0.01; // Step size for RK4 simulated motion
    int t_n = 100; // Number of values searched through for each torque parameter
    float sim_time = 0.3; //Length of motion simulated by RK4 method [s]

    tpulseinfo* torque_array = new tpulseinfo[t_n*t_n*t_n];
    RK4out* output_bests = new RK4out[t_n*t_n*t_n];
    RK4out overall_best;
    overall_best.norm = 100;

    // Initialize Variables for timing
    std::chrono::duration<double, std::milli> ms1, ms2;
    std::chrono::high_resolution_clock::time_point start1, start2;
    std::chrono::high_resolution_clock::time_point end;

    q0.q1 = -5*PI/180; // Set initial ankle angular position in radians
    q0.q2 = 0*PI/180; // Set initial ankle angular velocity in radians/s
    q0.q3 = -5*PI/180; // Set initial hip angular position in rad
    q0.q4 = 0*PI/180; // Set initialhip angular velocity in rad/s

    vals[0].l = 0.867; // anthro table length of ankle to hip
    vals[0].lc = 0.589; // anthro table lenth of ankle to CM of legs
    vals[0].m = 26.30; // anthro table mass of lower leg segments
    vals[0].Icm = 1.4; // anthro table moment of intertia of leg around CM
    vals[0].I = vals[0].I+vals[0].m*vals[0].lc*vals[0].lc;

    vals[1].l = 0.851; // anthro table length of hip to head [m]
    vals[1].lc = 0.332; // anthro table lenth of hip to CM of torso [m]
    vals[1].m = 42.88; // anthro table mass of torso [kg]
    vals[1].Icm = 2.227; // anthro table moment of intertia torso about hip [kg m^2]
    vals[1].I = vals[1].I+vals[1].m*vals[1].lc*vals[1].lc;
    
    if (q0.q1<= 0 )
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
    tmax_ratio = 1;
    tmin_ratio = 0.05;
    float t_dur_step = (tmax_dur - tmin_dur)/t_n;
    float t_amp_step = (tmax_amp - tmin_amp)/t_n;
    float t_rat_step = (tmax_ratio - tmin_ratio)/t_n;

    torque_array[0].amp = tmin_amp;
    torque_array[0].duration = tmin_dur;
    torque_array[0].ratio = tmin_ratio;

    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < t_n; i++){
        for (int j = 0; j<t_n; j++){
            for (int k = 0; k<t_n; k++){
            torque_array[t_n*t_n*i+t_n*j + k].amp = tmin_amp + i*t_amp_step;
            torque_array[t_n*t_n*i+t_n*j + k].duration = tmin_dur + j*t_dur_step;
            torque_array[t_n*t_n*i+t_n*j + k].ratio = tmin_ratio + k*t_rat_step;
            }
        }
    }
    start2 = std::chrono::high_resolution_clock::now();
//#pragma omp parallel num_threads(20)
    {
       //#pragma omp  for
            for (int i = 0; i < t_n*t_n*t_n; i++){
                //std::cout<<torque_array[i].amp<<"\n";
                output_bests[i] = double_RK4(sim_time, h, torque_array[i], q0, vals);
                if (overall_best.norm > output_bests[i].norm) overall_best = output_bests[i];
            }
    }
    end = std::chrono::high_resolution_clock::now();
    ms1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start1);
    ms2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start2);
    std::cout << "Time for RK4 loop: "<< ms2.count() <<"\n Time for Torque Allocation + RK4 Loop: "<< ms1.count()<<"\n Torque Steps: "<<t_n<<"\nInitial values - q1: "<<q0.q1*180/PI<<" degrees q2: "<<q0.q2*180/PI<<" degrees/s q3: "<<q0.q3*180/PI<<" degrees q4: "<<q0.q4*180/PI<<" degrees/s\n";
    std::cout << "Best Performance - Norm: " << overall_best.norm << " Torque Amp: " << overall_best.torque.amp << " Torque Duration: " << overall_best.torque.duration << "Torque Ratio: " << overall_best.torque.ratio<< "\n";
}
