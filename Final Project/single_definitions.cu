#include "single_definitions.cuh"
#include<iostream>
#include<vector>

#define PI 3.14159
#define g 9.81

__device__ void fsingle(float t, float *q, float *qdot, float torque, segment vals, float k, float l){
    qdot[0] = q[1] + l;
    qdot[1] = torque/vals.I+vals.m*g*vals.lc/vals.I*sin(q[0]+k);
}
