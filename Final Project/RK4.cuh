#ifndef RK4_CUH
#define RK4_CUH
#include "single_definitions.cuh"
#include <cstddef>

__global__ void RK4(float tf, float n, tpulseinfo *tspecs, float *q0, segment *vals, RK4out *output, int t_n);

__global__ void torquecalc(float *torque_i, float h, tpulseinfo tspecs, int n);


#endif