#ifndef double_RK4_H
#define double_RK4_H
#include "double_definitions.h"
#include <cstddef>

RK4out double_RK4(float tf, float n, tpulseinfo tspecs, angular_vals2 q0, segment *vals);

#endif