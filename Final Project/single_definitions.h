#ifndef SINGLE_DEFINITIONS_H
#define SINGLE_DEFINITIONS_H
#include <cstddef>
#define PI 3.14159
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
    float ratio; //ratio of hip to ankle torque
};

struct simresults{
    float norm;
    tpulseinfo torque;
};

// Function that calculates the q matrix values for Runge Kutta approach
// Shell function to call functions that calculate M, C and G matrix and torque values
void fsingle(float t, float *q, float *qdot, float torque, segment vals, float k, float l);

float tpulsecalc(tpulseinfo torque, float time);
#endif