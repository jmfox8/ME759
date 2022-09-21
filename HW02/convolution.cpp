// Author: Jackson Fox

#include "convolution.h"
#include<iostream>
#include <cstddef>

using namespace std;
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
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