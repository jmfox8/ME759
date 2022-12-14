// Author - Jackson Fox
#include "double_definitions.h"
#include<iostream>
#include<vector>
#include<math.h>

#define PI 3.14159

void f(float t, float *q, float *qdot, float *torques, segment vals[2], float k, float l, float m, float n){
    float G[2], C[4], M[4], K[4];
    K[0] = k;
    K[1] = l;
    K[2] = m;
    K[3] = n;
    Gmat(q,G,t,K,vals);
    Cmat(q,C,t,K,vals);
    Mmat(q,M,t,K,vals);
    qdot[0] = q[1] + K[1];
    qdot[1] = (M[3]*(torques[0]-C[0]*(q[1]+K[1]) - C[1]*(q[3]*K[3]) - G[0])+M[1]*(-torques[1]+C[2]*(q[1]+K[1]) + C[3]*(q[3]+K[3]) + G[1]))/(M[0]*M[3] - M[2]*M[1]);
    qdot[2] = q[3] + K[3];
    qdot[3] = (M[0]*(torques[1]-C[2]*(q[1]+K[1]) - C[3]*(q[3]+K[3]) - G[1])+M[2]*(-torques[0]+C[0]*(q[1]+K[1]) + C[1]*(q[3]+K[3]) + G[0]))/(M[0]*M[3] - M[2]*M[1]);
}

void Gmat(float *q,float *G,float t, float *K, segment vals[2]){
G[0] = 9.81*(vals[0].m*vals[0].lc*sin(q[0]+K[0]) + vals[1].m*(vals[0].l*sin((q[0]+K[0]))+vals[1].lc*sin((q[0]+K[0])+(q[2]+K[2]))));
G[1] = 9.81*(vals[1].m*vals[1].lc*sin((q[0]+K[0])+(q[2]+K[2])));
}

void Cmat(float *q,float *C,float t, float *K,segment vals[2]){
C[0] = vals[1].m*vals[0].l*vals[1].lc*sin(q[2]+K[2])*-2*(q[3]+K[3]);
C[1] = vals[1].m*vals[0].l*vals[1].lc*sin(q[2]+K[2])*-1*(q[3]+K[3]);
C[2] = vals[1].m*vals[0].l*vals[1].lc*sin(q[2]+K[2])*(q[1]+K[1]);
C[3] = 0;
}

void Mmat(float *q, float *M, float t, float *K,segment vals[2]){
M[0] = vals[0].I+vals[1].I+vals[1].m*(vals[0].l*vals[0].l+vals[0].l*vals[1].lc*cos(q[2]+K[2]));
M[1] = vals[1].I+vals[1].m*vals[0].l*vals[1].lc*cos(q[2]+K[2]);
M[2] = vals[1].I+vals[1].m*vals[1].lc*vals[0].l*cos(q[2]+K[2]);
M[3] = vals[1].I;
}

void tpulsecalc(tpulseinfo torque, float time, float *joint_t){
    if (time>=torque.duration){
        joint_t[0] = 0;
        joint_t[1] = 0;
    }
    else{
        joint_t[0] = torque.amp * sin(PI*time/torque.duration);
        joint_t[1] = joint_t[0] * torque.ratio;
    }
}