#include "convolution.h"
#include <cstddef>
#include <omp.h>

// This function does a parallel version of the convolution process in HW02 task2
// using OpenMP. You may recycle your code from HW02.

// "image" is an n by n grid stored in row-major order.
// "mask" is an m by m grid stored in row-major order.
// "output" stores the result as an n by n grid in row-major order.

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    //HW2 Code
    float sum1 = 0;
    float sum2 = 0;
    int imagei;
    int imagej;
    int imageij;
    for (int x = 0; x<n; x++){
    for(int y = 0; y<n; y++){
        for (int i = 0;i<m;i++){
            for(int j=0;j<m;j++){
                imagei = x+i-((m-1)/2);
                imagej = y+j-((m-1)/2);
                if (n <= imagei || imagei < 0){
                    if (n <= imagej || imagej < 0){
                        imageij = 0;
                    }
                    else if (n > imagej && imagej >= 0){
                        imageij = 1;
                    }
                }
                else if (n <= imagej || imagej < 0){
                        imageij = 1;
                }
                else {
                imageij = image[n*imagei + imagej];
                }
                sum1 = sum1 + mask[m*i+j]*imageij;        
            }
            sum2 = sum2 + sum1;
            sum1 = 0;
        }
        output[n*x+y] = sum2;
        sum2 = 0;
    }
}
}
