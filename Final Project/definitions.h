#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cstddef>

typedef struct segment{
    float l; //Length of body segment [m]
    float lc; // Distance from distal joint to CoM of segment [m]
    float m; // mass of segment [kg]
    float I; //Moment of inertia of segment around distal joint [kgm^2]
    float Icm; //Moment of inertia of segment around CoM [kgm^2]
    float t; // Torque around distal joint
};

typedef struct tpulse{
    float amp; //Peak of torque waveform
    float duration; //Length of torque waveform
    float ratio; //ratio of hip to ankle torque
};

std::vector<float> f(float t, float *q,segment vals,float torque, float k, float l);
float tpulsecalc(tpulse torque, float time);
#endif