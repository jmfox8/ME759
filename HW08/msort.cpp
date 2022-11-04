#include "msort.h"
#include <cstddef>
#include <iostream>
#include <omp.h>


void copyarray(int* arr, int* copy, const std::size_t n);
void serialmerge(int* arr, int ileft, int iright, int iend, int* copy);
void serialsort(int* arr, const std::size_t n);

void merge(int* arr, int start, int mid, int end, int* arr2);
void splitmerge(int* arr2, int start, int end, int* arr);
void parallelsort(int*arr, const std::size_t n);


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

// serial Function to re-merge two equally sized arrays in sorted order
void serialmerge(int* arr, int ileft, int iright, int iend, int* copy){
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

// Serial sorting function - calls copyarray and serialmerge
void serialsort(int* arr, const std::size_t n){
    int* arr2  = new int[n];
    for (int width = 1; width<n; width = 2*width){
        for (int i = 0; i < n; i = i + 2*width){
            if (i + width < n){
                if (i + 2 * width < n) serialmerge(arr, i, i+width, i + 2*width,arr2);
                else serialmerge(arr,i,i+width,n,arr2);
            }
            else serialmerge(arr,i,n,n,arr2);
        }
        copyarray(arr2, arr, n);
    }
}

//Parallel merging function
void merge(int* arr, int start, int mid, int end, int* arr2){
    int i = start;
    int j = mid;
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



// Parallel Split function, recursive
void splitmerge(int* arr2, int start, int end, int* arr){
    if (end > start+1){
        int split = (start+end)/2;
        #pragma omp task
        {
            splitmerge(arr,start,split,arr2);
        }
        #pragma omp task
        {
            splitmerge(arr,split,end,arr2);
        }
        #pragma omptaskwait
        #pragma omp critical(RES_lock)
        {
            merge(arr2,start, split, end, arr);
        }
    }
}

// Parallel Sorting Function - calls others
void parallelsort(int*arr, const std::size_t n){
    int *arr2 = new int[n];
    copyarray(arr,arr2,n);
    #pragma omp parallel
    #pragma omp single
    {
    splitmerge(arr2,0,n,arr);
    }
}

// Main function
void msort(int* arr, const std::size_t n, const std::size_t threshold){
    if (n<threshold) serialsort(arr,n);
    else parallelsort(arr,n);
}