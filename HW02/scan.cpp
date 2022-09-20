#include "scan.h"
#include<iostream>
#include <cstddef>

using namespace std;

void scan(const float *arr, float *output, std::size_t n){
    output[0] = arr[0];
    for (int i = 1; i<n+1; i++){

        output[i] = output[i-1]+ arr[i];
    }
}