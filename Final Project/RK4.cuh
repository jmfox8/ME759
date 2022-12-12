#ifndef RK4_CUH
#define RK4_CUH
#include "single_definitions.h"
#include <cstddef>

RK4out RK4(float tf, float n, tpulseinfo tspecs, float *q0, segment *vals, RK4out *output);

#endif