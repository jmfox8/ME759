#include "count.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]
void count(const thrust::device_vector<int>& d_in, thrust::device_vector<int>& values, thrust::device_vector<int>& counts){

// create filler array of 1s as long as the input array
thrust::device_vector<int> fillers = d_in;
thrust::fill(fillers.begin(),fillers.end(),1);

// reduce by key to get out put of values and counts of the values
thrust::reduce_by_key(d_in.begin(),d_in.end(),fillers.begin(),values.begin(),counts.begin());
}