// Author: Jackson Fox
#ifndef RK4_CUH
#define RK4_CUH
#include "single_definitions.cuh"
#include <cstddef>

__global__ void single_RK4(float tf, float n, tpulseinfo *tspecs, angular_vals q0, segment vals, RK4out *output, int t_n);

#endif