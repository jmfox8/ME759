#include "msort.h"
#include <cstddef>
#include <omp.h>


void copyarray(int* arr, int* copy, const std::size_t n);
void merge(int* arr, int ileft, int iright, int iend, int* copy);
void serialsort(int* arr, const std::size_t n);


// This function does a merge sort on the input array "arr" of length n. 
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would 
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling

void copyarray(int* arr, int* copy, const std::size_t n){
    for (int i = 0; i < n; i++){
        copy[i] = arr[i];
    }
}
void merge(int* arr, int ileft, int iright, int iend, int* copy){
    int i = ileft;
    int j = iright;
    for (int k = ileft; k< iend; k++){
        if (i < iright && (j >= iend || arr[i] <= arr[j])){
            copy[k] = arr[i];
            i += 1;
        }
        else{
            copy[k] = arr[j];
            j+= 1;
        }
    }
}

void serialsort(int* arr, const std::size_t n){
    int* arr2  = new int[n];
    for (int width = 1; width<n; width = 2*width){
        for (int i = 0; i < n; i = i + 2*width){
            if (i + width < n){
                if (i + 2 * width < n) merge(arr, i, i+width, i + 2*width,arr2);
                else merge(arr,i,i+width,n,arr2);
            }
            else merge(arr,i,n,n,arr2);
            //            merge(arr, i, std::min(i+width, n),min(i+2*width,n),arr2);
        }
        copyarray(arr2, arr, n);
    }
}

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    if (n<threshold) serialsort(arr,n);
}