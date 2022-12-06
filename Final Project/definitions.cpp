#include "definitions.h"
#include<iostream>
#include<vector>

#define PI 3.14159

std::vector<float> f(float t, float *q,segment vals,float torque, float k, float l){

}

float tcalc(tpulse torque, float t){
    float val;
    if (t>=torque.duration) return(0);
    else
    val = torque.amp * sin(PI*t/torque.duration);
    return val;
    }