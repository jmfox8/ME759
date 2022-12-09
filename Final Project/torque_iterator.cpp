#include "single_definitions.h"
#include "RK4.h"
#include <cstddef>
#include <math.h>
#include <iostream>

#define PI 3.14159
// CONFIGURED FOR SINGLE INVERTED PENDULUM
int main(int argc, char *argv[]){
    float q0[2], tmax_amp, tmin_amp, tmax_dur, tmin_dur;
    segment vals[2];
    // Get Command Line Input
    q0[0]= atoi(argv[1]);
    q0[1] = atoi(argv[2]);

    int h = 0.01; // Step size for path solver [s]
    int t_n = 10; // Number of values attempted for each torque parameter
    tpulseinfo* torque_array = new tpulseinfo[t_n*t_n];
    RK4out* output_bests = new RK4out[t_n*t_n];
    
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
        tmax_amp = 0;
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

    for (int i = 0; i < t_n*t_n; i++){

        output_bests[i] = RK4(sim_time,h,torque_array[i], q0, vals);
        std::cout<<output_bests[i].norm << "  " << output_bests->torque.amp << "\n";
    }


}
