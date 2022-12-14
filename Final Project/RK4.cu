#include "single_definitions.cuh"
#include "RK4.cuh"
#include <cstddef>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define PI 3.14159
#define g 9.81

__global__ void single_RK4(float tf, float h, tpulseinfo *tspecs, angular_vals q0, segment vals, RK4out *output, int t_n){
//CURRENT CONFIGURATION FOR SINGLE INVERTED PENDULUM
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    int thread_i = block*blocksize+thread;
    if (thread_i >= t_n*t_n);
    else{
        // initialize values needed for executing RK4 run based on inputs
        float q[2], qdot[2], min_norm, k[5][2], t0, torque_i, qn[2];
        // Set intial values from inputs
        q[0] = q0.q1; // Angular Position
        q[1] = q0.q2; // Angular Velocity
        t0 = 0; // Time
        tpulseinfo tspec_thread = tspecs[thread_i]; //Torque Pulse information
        min_norm = sqrt(q[0]*q[0] + q[1]*q[1]);
        // Define steps needed basd on inputs
        int n = tf/h;

        // Allocate memory based on steps needed
        float* norms = new float[n];

        for (int i = 0; i<n; i++){
            
            if(h*i >= tspec_thread.duration) torque_i = 0;
            else {
                torque_i = tspec_thread.amp*sin(PI*h*i/tspec_thread.duration);
            }

            //fsingle<<<1,1>>>(t0,q,qdot,torque_i[i], vals[0],0,0);
            qdot[0] = q[1] + 0;
            qdot[1] = torque_i/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+0);
            k[1][0] = h * qdot[0];
            k[1][1] = h * qdot[1];
            
            //fsingle<<<1,1>>>(t0+h/2,q,qdot,torque_i[i],vals[0],k[1][0]/2,k[1][1]/2);
            qdot[0] = q[1] + k[1][1]/2;
            qdot[1] = torque_i/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+k[1][0]/2);
            k[2][0] = h * qdot[0];
            k[2][1] = h * qdot[1];

            //fsingle<<<1,1>>>(t0+h/2,q,qdot,torque_i[i],vals[0],k[2][0]/2,k[2][1]/2);
            qdot[0] = q[1] + k[2][1]/2;
            qdot[1] = torque_i/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+k[2][0]/2);
            k[3][0] = h * qdot[0];
            k[3][1] = h * qdot[1];

            //fsingle<<<1,1>>>(t0+h,q,qdot,torque_i[i],vals[0],k[3][0],k[3][1]);
            qdot[0] = q[1] + k[3][1];
            qdot[1] = torque_i/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+k[3][0]);
            k[4][0] = h * qdot[0];
            k[4][1] = h * qdot[1];
            
            //std::cout<<k[4][0]<<" "<<k[4][1]<<"\n";
            k[0][0] = (k[1][0]+2*k[2][0]+2*k[3][0]+k[4][0])/6;
            k[0][1] = (k[1][1]+2*k[2][1]+2*k[3][1]+k[4][1])/6;

            qn[0] = q[0]+k[0][0];
            qn[1] = q[1]+k[0][1];
            norms[i] = sqrt(qn[0]*qn[0]+qn[1]*qn[1])*180/PI;
            
            t0 = t0+h;
            q[0] = qn[0];
            q[1] = qn[1];
        }

        for (int i = 0; i < n-1; i++){
            if (norms[i]<min_norm) min_norm = norms[i];
        }
        
        output[thread_i].norm = min_norm;
        output[thread_i].torque.amp = tspec_thread.amp;
        output[thread_i].torque.duration = tspec_thread.duration;
        output[thread_i].torque.ratio = 0;

        // Memory clean up
        delete[] norms;
    }
}
