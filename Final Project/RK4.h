#ifndef RK4_H
#define RK4_H
#include "single_definitions.h"
#include <cstddef>

RK4out RK4(float tf, float n, tpulseinfo tspecs, float *q0, segment *vals);

#endif