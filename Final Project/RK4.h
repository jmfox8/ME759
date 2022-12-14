// Author: Jackson Fox
#ifndef RK4_H
#define RK4_H
#include "single_definitions.h"
#include <cstddef>

 RK4out RK4(float tf, float n, tpulseinfo tspecs, angular_vals q0, segment *vals);

#endif