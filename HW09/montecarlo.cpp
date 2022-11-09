#include "montecarlo.h"
#include <cstdio>
#include <cstddef>
#include <cmath>
#include <omp.h>

// this function returns the number of points that lay inside
// a circle using OpenMP parallel for. 
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.

int montecarlo(const size_t n, const float *x, const float *y, const float radius){
    int incircle = 0;
    #pragma omp parallel for reduction(+:incircle) 
    for (int i = 0; i<n; i++){
            incircle += (x[i]*x[i] + y[j]*y[j]) <= 1 ? 1 : 0;
        }
return(incircle);
}
