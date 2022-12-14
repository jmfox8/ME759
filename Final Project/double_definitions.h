#ifndef DOUBLE_DEFINITIONS_H
#define DOUBLE_DEFINITIONS_H
#include <cstddef>

struct angular_vals2{
    float q1;
    float q2;
    float q3;
    float q4;
};

struct segment{
    float l; //Length of body segment [m]
    float lc; // Distance from distal joint to CoM of segment [m]
    float m; // mass of segment [kg]
    float I; //Moment of inertia of segment around distal joint [kgm^2]
    float Icm; //Moment of inertia of segment around CoM [kgm^2]
    float t; // Torque around distal joint
};

struct tpulseinfo{
    float amp; //Peak of torque waveform
    float duration; //Length of torque waveform
    float ratio; //ratio of hip to ankle torque, hip = ankle*ratio
};

struct RK4out{
    float norm;
    tpulseinfo torque;
};


// Function that calculates the q matrix values for Runge Kutta approach
void f(float t, float *q, float *qdot, float *torques, segment vals[2], float k, float l, float m, float n);

// Shell functions that calculate M, C and G matrix and torque values    
void Gmat(float *q,float *G,float t, float *K, segment vals[2]);

void Cmat(float *q,float *C,float t, float *K, segment vals[2]);

void Mmat(float *q, float *M, float t, float *K, segment vals[2]);

void tpulsecalc(tpulseinfo torque, float time, float *joint_t);

#endif