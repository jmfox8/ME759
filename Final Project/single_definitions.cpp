#include "single_definitions.h"
#include<iostream>
#include<vector>

#define PI 3.14159
#define g 9.81

void f(float *qdot, float t, float *q, float torque, segment vals){
    qdot[0] = q[1] + k[1];
    qdot[1] = torque/vals.I+vals.m*9.81*vals.lc/vals.I*sin(q[0]+k[0]);
    return fvec;

}

float tcalc(tpulse torque, float t){
    float val;
    if (t>=torque.duration) return(0);
    else
    val = torque.amp * sin(PI*t/torque.duration);
    return val;
    }



    std::vector<float> f(float t, float *q,segment vals,float torque, float k, float l){
    std::vector<float> fvec(2,0);
    fvec[0] = q[1]+k;
    fvec[1] = torque/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+l);
    return fvec;
} 

float tcalc(tpulse torque, float t){
    float val;
    if (t>=torque.duration) return(0);
    else
    val = torque.amp * sin(PI*t/torque.duration);
    return val;
    }


void Gmat(float *q,float *G,float t, float *K);

void Cmat(float *q,float *C,float t, float *K);

void Mmat(float *q, float *M, float t, float *K);

float tpulsecalc(tpulseinfo torque, float time);