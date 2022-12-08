#include "single_definitions.h"
#include<iostream>
#include<vector>

#define PI 3.14159
#define g 9.81

void fsingle(float t, float *q, float *qdot, float torque, segment vals, float k, float l){
    qdot[0] = q[1] + k;
    qdot[1] = torque/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+l);
}

float tpulsecalc(tpulseinfo torque, float time){
    float val;
    if (time>=torque.duration) return(0);
    else
    val = torque.amp * sin(PI*time/torque.duration);
    return val;
    } 

