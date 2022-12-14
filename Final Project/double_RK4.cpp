// Author: Jackson Fox
#include "double_definitions.h"
#include "double_RK4.h"
#include <cstddef>
#include <math.h>
#include <iostream>

#define PI 3.14159
 
RK4out double_RK4(float tf, float h, tpulseinfo tspecs, angular_vals2 q0, segment *vals){

// initialize values needed for executing RK4 run based on inputs
float q[4], qdot[4], min_norm, k[5][4], t0, torque_i[2], qn[4], norm;
RK4out output;

// Set intial values
q[0] = q0.q1; // Angular Position lower Segment
q[1] = q0.q2; // Angular Velocity lower Segment
q[2] = q0.q3; // Angular Position Upper Segment
q[3] = q0.q4; // Angular Velocity Upper Segment
t0 = 0; // Time

// Define steps needed based on inputs
int n = tf/h;

min_norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])*180/PI;

for (int i = 0; i<n; i++){
    
    tpulsecalc(tspecs, h*i, torque_i);   

    f(t0,q,qdot,torque_i, vals,0,0,0,0);
    k[1][0] = h * qdot[0];
    k[1][1] = h * qdot[1];
    k[1][2] = h * qdot[2];
    k[1][3] = h * qdot[3];
    
    f(t0+h/2,q,qdot,torque_i,vals,k[1][0]/2,k[1][1]/2,k[1][2]/2,k[1][3]/2);
    k[2][0] = h * qdot[0];
    k[2][1] = h * qdot[1];
    k[2][2] = h * qdot[2];
    k[2][3] = h * qdot[3];

    f(t0+h/2,q,qdot,torque_i,vals,k[2][0]/2,k[2][1]/2,k[2][2]/2,k[2][3]/2);
    k[3][0] = h * qdot[0];
    k[3][1] = h * qdot[1];
    k[3][2] = h * qdot[2];
    k[3][3] = h * qdot[3];

    f(t0+h,q,qdot,torque_i,vals,k[3][0],k[3][1],k[3][2],k[3][3]);
    k[4][0] = h * qdot[0];
    k[4][1] = h * qdot[1];
    k[4][2] = h * qdot[2];
    k[4][3] = h * qdot[3];

    k[0][0] = (k[1][0]+2*k[2][0]+2*k[3][0]+k[4][0])/6;
    k[0][1] = (k[1][1]+2*k[2][1]+2*k[3][1]+k[4][1])/6;
    k[0][2] = (k[1][2]+2*k[2][2]+2*k[3][2]+k[4][2])/6;
    k[0][3] = (k[1][3]+2*k[2][3]+2*k[3][3]+k[4][3])/6;

    qn[0] = q[0]+k[0][0];
    qn[1] = q[1]+k[0][1];
    qn[2] = q[2]+k[0][2];
    qn[3] = q[3]+k[0][3];

    norm = sqrt(qn[0]*qn[0]+qn[1]*qn[1]+qn[2]*qn[2]+qn[3]*qn[3])*180/PI;
 
    if (norm<min_norm) min_norm = norm;

    t0 = t0+h;
    q[0] = qn[0];
    q[1] = qn[1];
    q[2] = qn[2];
    q[3] = qn[3];
}

output.norm = min_norm;
output.torque = tspecs;

// Return Output
return(output);
}
