// Author: Jackson Fox

#include "matmul.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[]){
 
 //Define Matrices Dimensions
 int n = 2048;
 
 // Intialize and Allocate Memory for Arrays
 double* A_arr = new double[n*n];
 double* B_arr = new double[n*n];
 double* C_arr1 = new double[n*n];
 double* C_arr2 = new double[n*n];
 double* C_arr3 = new double[n*n];
 double* C_arr4 = new double[n*n];

// Initialize Vectors
vector<double>A_vec (n*n);
vector<double>B_vec (n*n);

// Initialization for timing
duration<double, std::milli> duration1_ms;
duration<double, std::milli> duration2_ms;
duration<double, std::milli> duration3_ms;
duration<double, std::milli> duration4_ms;
high_resolution_clock::time_point start;
high_resolution_clock::time_point end;

// Initialization for randomization
random_device entropy_source;
mt19937 generator(entropy_source());

// Generate and fill random values for Matrix/Vector A
uniform_real_distribution<double> distroA(-10.0,10.0);
for(auto& value_A : A_vec){
    value_A=distroA(generator);
}
for(int i = 0; i<n*n; i++){
    A_arr[i] = A_vec[i];
}

// Generate and fill random values for Matrix/Vector B
uniform_real_distribution<double> distroB(-10.0,10.0);
for(auto& value_B : B_vec){
    value_B=distroB(generator);
}
for(int i = 0; i<n*n; i++){
    B_arr[i] = B_vec[i];
}

// Execute Function Calls with timing
start = high_resolution_clock::now();
mmul1(A_arr, B_arr, C_arr1, n);
end = high_resolution_clock::now();
duration1_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

start = high_resolution_clock::now();
mmul2(A_arr, B_arr, C_arr2, n);
end = high_resolution_clock::now();
duration2_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

start = high_resolution_clock::now();
mmul3(A_arr, B_arr, C_arr3, n);
end = high_resolution_clock::now();
duration3_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

start = high_resolution_clock::now();
mmul4(A_vec, B_vec, C_arr4, n);
end = high_resolution_clock::now();
duration4_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

//Print Results
cout << n << "\n";
cout << duration1_ms.count() << "\n";
cout << C_arr1[n*n-1] << "\n";
cout << duration2_ms.count() << "\n";
cout << C_arr2[n*n-1] << "\n";
cout << duration3_ms.count() << "\n";
cout << C_arr3[n*n-1] << "\n";
cout << duration4_ms.count() << "\n";
cout << C_arr4[n*n-1] << "\n";

// //test print
// for (int i = 0; i<n*n; i++){
// cout<<C_arr1[i]<<" ";
// }
// cout << "\n";
// for (int i = 0; i<n*n; i++){
// cout<<C_arr2[i]<<" ";
// }
// cout << "\n";
// for (int i = 0; i<n*n; i++){
// cout<<C_arr3[i]<<" ";
// }
// cout << "\n";
// for (int i = 0; i<n*n; i++){
// cout<<C_arr4[i]<<" ";
// }
// cout << "\n";

// Deallocate Memory for Arrays
delete[] A_arr;
delete[] B_arr;
delete[] C_arr1;
delete[] C_arr2;
delete[] C_arr3;
delete[] C_arr4;
}