#include "double_definitions.h"
#include<iostream>
#include<vector>
#include<math.h>

#define PI 3.14159

void f(float t, float *q, float *qdot, float *torques, segment vals[2], float *K){
    float G[4], C[4], M[4];
    Gmat(q,G,t,K);
    Cmat(q,C,t,K);
    Mmat(q,M,t,K);
    qdot[0] = q[1];
    qdot[1] = (M[3]*(torques[0]-C[0]*q[1] - C[1]*q[3] - G[0])+M[1]*(-torques[1]+C[2]*q[1] + C[3]*q[3] + G[2]))/(M[0]*M[3] - M[2]*M[1]);
    qdot[2] = q[3];
    qdot[3] = (M[0]*(torques[1]-C[2]*q[1] - C[3]*q[3] - G[2])+M[2]*(-torques[0]+C[0]*q[1] + C[1]*q[3] + G[0]))/(M[0]*M[3] - M[2]*M[1]);
}

void Gmat(float *q,float *G,float t, float *K, segment vals[2]){
G[0] = 9.81*(vals[0].m*vals[0].lc*sin(q[0]) + vals[1].m*(vals[0].l*sin(q[0])+vals[1].lc*sin(q[0]+q[2])));
G[1] = 9.81*(vals[1].m*vals[1].lc*sin(q[0]+q[2]));
}

void Cmat(float *q,float *C,float t, float *K,segment vals[2]){
C[0] = vals[1].m*vals[0].l*vals[0].lc*sin(q[2])*-2*q[3];
C[1] = vals[1].m*vals[0].l*vals[0].lc*sin(q[2])*-1*q[3];
C[3] = vals[1].m*vals[0].l*vals[0].lc*sin(q[2])*q[1];
C[4] = 0;
}

void Mmat(float *q, float *M, float t, float *K,segment vals[2]){
M[0] = vals[0].I+vals[1].I+vals[1].m*(vals[0].l*vals[0].l+vals[0].l*vals[1].lc*cos(q[2]));
M[1] = vals[1].I+vals[1].m*vals[0].l*vals[1].lc*cos(q[2]);
M[2] = vals[1].I+vals[1].m*vals[1].lc*vals[0].l*cos(q[2]);
M[3] = vals[1].I;
}

float tpulsecalc(tpulseinfo torque, float time, float *joint_t){
    float val;
    if (time>=torque.duration){
        joint_t[0] = 0;
        joint_t[1] = 0;
    }
    else
    joint_t[0] = torque.amp * sin(PI*time/torque.duration);
    joint_t[1] = joint_t[0] * torque.ratio;
}