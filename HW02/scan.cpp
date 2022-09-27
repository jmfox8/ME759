#include "scan.h"
#include<iostream>
#include <cstddef>

using namespace std;

void scan(const float *arr, float *output, std::size_t n){
    output[0] = arr[0];
    if (n == 0){
    
    }
    else {
        for (int i = 1; i<n; i++){
        output[i] = output[i-1]+ arr[i];
    }
    }
}