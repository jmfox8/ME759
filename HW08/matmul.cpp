#include "matmul.h"
#include <cstddef>
#include <omp.h>

// This function produces a parallel version of matrix multiplication C = A B using OpenMP. 
// The resulting C matrix should be stored in row-major representation. 
// Use mmul2 from HW02 task3. You may recycle the code from HW02.

// The matrices A, B, and C have dimension n by n and are represented as 1D arrays.

void mmul(const float* A, const float* B, float* C, const std::size_t n){
    // HW2 Code
    int i;
    int j;
    int k;
    for (i = 0; i<n; i++){

    for (j = 0; j<n; j++){

        for (k = 0; k<n; k++){
            C[i*n+j] = C[i*n+j] + A[i*n+k]*B[k*n+j];
        }
    }
}
}
