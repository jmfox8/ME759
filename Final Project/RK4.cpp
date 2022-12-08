#include "single_definitions.h"
#include "RK4.h"
#include <cstddef>
#include <math.h>
#include <iostream>

#define PI 3.14159

simresults RK4(float t0, float tf, int h, tpulseinfo tspecs, float *q0, segment *vals){
//CURRENT CONFIGURATION FOR SINGLE INVERTED PENDULUM

// initialize values needed for executing RK4 run based on inputs
float q[2], qdot[2], min_norm, k[5][2];
q[0] = q0[0];
q[1] = q[1];
// Define steps needed basd on inputs
int n = (tf - t0)/h;
float* norms = new float[n];
float* qn = new float[2*n]; 
simresults output;
// calculate the torques for each step based on inputs
float torque_i[n];

for (int i = 0; i<n; i++){
    torque_i[i] = tpulsecalc(tspecs,t0+(h*i));    
}

min_norm = sqrt(q[0]*q[0] + q[1]*q[1]);

for (int i = 0; i<n; i++){

    fsingle(t0,q,qdot,torque_i[i], vals[0],0,0);
    k[1][0] = h * qdot[0];
    k[1][1] = h * qdot[1];
    
    fsingle(t0+h/2,q,qdot,torque_i[i],vals[0],k[1][0]/2,k[1][1]/2);
    k[2][0] = h * qdot[0];
    k[2][1] = h * qdot[1];

    fsingle(t0+h/2,q,qdot,torque_i[i],vals[0],k[2][0]/2,k[2][1]/2);
    k[3][0] = h * qdot[0];
    k[3][1] = h * qdot[1];

    fsingle(t0+h,q,qdot,torque_i[i],vals[0],k[3][0],k[3][1]);
    k[4][0] = h * qdot[0];
    k[4][1] = h * qdot[1];

    k[0][0] = (k[1][0]+2*k[2][0]+2*k[3][0]+k[4][0])/6;
    k[0][1] = (k[1][1]+2*k[2][1]+2*k[3][1]+k[4][1])/6;

    qn[i*2] = q[0]+k[0][0];
    qn[i*2 + 1] = q[1]+k[1][1];
    norms[i] = sqrt(qn[i*2]*qn[i*2]+qn[i*2+1]*qn[i*2+1]);
    
    std::cout << qn[i*2]*180/PI <<"  "<< qn[i*2 + 1]*180/PI << "  " << torque_i[i] << "  " << t0 << "  " << norms[i] << "\n";
    
    t0 = t0+h;
    q[0] = qn[i*2];
    q[1] = qn[i*2+1];
}

for (int i = 0; i < n-1; i++){
    if (norms[i]<min_norm) min_norm = norms[i];
}
output.norm = min_norm;
output.torque = tspecs;
return(output);
}
