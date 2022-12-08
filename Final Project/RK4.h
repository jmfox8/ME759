#ifndef RK4_H
#define RK4_H
#include "single_definitions.h"
#include <cstddef>

simresults RK4(float t0, float tf, int n, tpulseinfo tspecs, float *q0, segment *vals);

#endif