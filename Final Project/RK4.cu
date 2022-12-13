#include "single_definitions.cuh"
#include "RK4.cuh"
#include <cstddef>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define PI 3.14159
#define g 9.81

__global__ void torquecalc(float *torque_i, float h, tpulseinfo tspec,  int n){
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    int i = block*blocksize+thread;
    if (i < n){
        if (h*i>=tspecs.duration) torque_i[i] = 0;
        else
        {
            torque_i[i] = tspec.amp * sin(PI*h*i/tspecs.duration);
        }   
    }
}

__global__ void RK4(float tf, float h, tpulseinfo *tspecs, float *q0, segment *vals, RK4out *output, int t_n){
//CURRENT CONFIGURATION FOR SINGLE INVERTED PENDULUM
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    int thread_i = block*blocksize+thread;
    if (thread_i >= t_n*t_n);
    else{
        // initialize values needed for executing RK4 run based on inputs
        float q[2], qdot[2], min_norm, k[5][2], t0;

        // Set intial values
        q[0] = q0[0]; // Angular Position
        q[1] = q0[1]; // Angular Velocity
        t0 = 0; // Time

        // Define steps needed basd on inputs
        int n = tf/h;

        // Allocate memory based on steps needed
        float* norms = new float[n];
        float* qn = new float[2*n]; 
        float *torque_i = new float[n];

        //tpulseinfo tspec_thread = tspecs[thread_i];
        //torquecalc<<<1,n>>>(torque_i, h, tspec_thread, n);

        for (int i = 0; i < n; i++){
            if (h*i >= tspecs[thread_i].duration) torque_i[i] = 0;
            else{
                torque_i[i] = tspecs[thread_i].amp*sin(PI*h*i/tspecs[thread_i].duration);
            }
        }

        //min_norm = sqrt(q[0]*q[0] + q[1]*q[1]);
        min_norm = 10;
        for (int i = 0; i<n; i++){

            //fsingle<<<1,1>>>(t0,q,qdot,torque_i[i], vals[0],0,0);
            qdot[0] = q[1] + 0;
            qdot[1] = torque_i[i]/vals[0].I+vals[0].m*g*vals[0].lc/vals[0].I*sin(q[0]+0);
            k[1][0] = h * qdot[0];
            k[1][1] = h * qdot[1];
            
            //fsingle<<<1,1>>>(t0+h/2,q,qdot,torque_i[i],vals[0],k[1][0]/2,k[1][1]/2);
            qdot[0] = q[1] + k[1][0]/2;
            qdot[1] = torque_i[i]/vals[0].I+vals[0].m*g*vals[0].lc/vals[0].I*sin(q[0]+k[1][1]/2);
            k[2][0] = h * qdot[0];
            k[2][1] = h * qdot[1];

            //fsingle<<<1,1>>>(t0+h/2,q,qdot,torque_i[i],vals[0],k[2][0]/2,k[2][1]/2);
            qdot[0] = q[1] + k[2][0]/2;
            qdot[1] = torque_i[i]/vals[0].I+vals[0].m*g*vals[0].lc/vals[0].I*sin(q[0]+k[2][1]/2);
            k[3][0] = h * qdot[0];
            k[3][1] = h * qdot[1];

            //fsingle<<<1,1>>>(t0+h,q,qdot,torque_i[i],vals[0],k[3][0],k[3][1]);
            qdot[0] = q[1] + k[3][0];
            qdot[1] = torque_i[i]/vals[0].I+vals[0].m*g*vals[0].lc/vals[0].I*sin(q[0]+k[3][1]);
            k[4][0] = h * qdot[0];
            k[4][1] = h * qdot[1];
            
            
            //std::cout<<k[4][0]<<" "<<k[4][1]<<"\n";
            k[0][0] = (k[1][0]+2*k[2][0]+2*k[3][0]+k[4][0])/6;
            k[0][1] = (k[1][1]+2*k[2][1]+2*k[3][1]+k[4][1])/6;

            qn[i*2] = q[0]+k[0][0];
            qn[i*2 + 1] = q[1]+k[0][1];
            norms[i] = sqrt(qn[i*2]*qn[i*2]+qn[i*2+1]*qn[i*2+1])*180/PI;
            
            //std::cout << qn[i*2]*180/PI <<"  "<< qn[i*2 + 1]*180/PI << "  " << torque_i[i] << "  " << t0 << "  " << norms[i] << "\n";
            
            t0 = t0+h;
            q[0] = qn[i*2];
            q[1] = qn[i*2+1];
        }

        for (int i = 0; i < n-1; i++){
            if (norms[i]<min_norm) min_norm = norms[i];
        }
        output[thread_i].norm = min_norm;
        output[thread_i].torque = tspecs[thread_i];

        // Memory clean up
        delete[] norms;
        delete[] qn;
        delete[] torque_i;
    }
}
