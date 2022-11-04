#include "msort.h"
#include <cstddef>
#include <iostream>
#include <omp.h>
#include <algorithm>


void copyarray(int* arr, int* copy, const std::size_t n);
void merge(int* arr, int start, int mid, int end, int* arr2);
void splitmerge(int* arr2, int start, int end, int* arr, std::size_t threshold);
void parallelsort(int*arr, const std::size_t n, std::size_t threshold);


// This function does a merge sort on the input array "arr" of length n. 
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would 
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling

// Function to copy array into another
void copyarray(int* arr, int* copy, const std::size_t n){
    for (int i = 0; i < n; i++){
        copy[i] = arr[i];
    }
}

//Parallel merging function
void merge(int* arr, int start, int mid, int end, int* arr2){
    int i = start;
    int j = mid;
    #pragma omp task
    {
        for (int k = start; k< end; k++){
            if (i < mid && (j >= end || arr[i] <= arr[j])){
                arr2[k] = arr[i];
                i += 1;
            }
            else{
                arr2[k] = arr[j];
                j += 1;
            }
        }
    }
}



// Parallel Split function, recursive
void splitmerge(int* arr2, int start, int end, int* arr,std::size_t threshold){
    if (end - start < threshold) std::sort(arr+start,arr+end);
        else{
        int split = (start+end)/2;
        #pragma omp task
        {
            splitmerge(arr,start,split,arr2,threshold);
        }
        #pragma omp task
        {
            splitmerge(arr,split,end,arr2,threshold);
        }
        #pragma omp taskwait
        merge(arr2,start, split, end, arr);
    }
}

// Parallel Sorting Function - calls others
void parallelsort(int*arr, const std::size_t n, std::size_t threshold){
    int *arr2 = new int[n];
    copyarray(arr,arr2,n);
    #pragma omp parallel
    #pragma omp single
    {
    splitmerge(arr2,0,n,arr,threshold);
    }
}

// Main function
void msort(int* arr, const std::size_t n, const std::size_t threshold){
    if (n<threshold) std::sort(arr,arr+n);
    else parallelsort(arr,n,threshold);
}