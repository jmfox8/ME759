#include "double_definitions.h"
#include<iostream>
#include<vector>

#define PI 3.14159

f(float t, float *q, float *qdot, float *torques, segment vals[2], float *K){
    float G[4], C[4], M[4];
    Gmat(q,G,t,K);
    Cmat(q,C,t,K);
    Mmat(q,M,t,K);
    qdot[0] = q[1];
    qdot[1] = (M[3]*(torques[0]-C[0]*q[1] - C[1]*q[3] - G[0])+M[1]*(-torques[1]+C[2]*q[1] + C[3]*q[3] + G[2]))/(M[0]*M[3] - M[2]*M[1]);
    qdot[2] = q[3];
    qdot[3] = (M[0]*(torques[1]-C[2]*q[1] - C[3]*q[3] - G[2])+M[2]*(-torques[0]+C[0]*q[1] + C[1]*q[3] + G[0]))/(M[0]*M[3] - M[2]*M[1]);
}

Gmat(float *q,float *G,float t, float *K, segment vals[2]){
G
}

Cmat(float *q,float *C,float t, float *K,segment vals[2]){

}

Mmat(float *q, float *M, float t, float *K,segment vals[2]){

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