// Author: Jackson Fox

#include "convolution.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>


using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[]){


    // Define input variables from command line
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    // Allocate Memory for arrays
    float* image_arr = new float[n*n];
    float* mask_arr = new float[m*m];
    float* conv_arr = new float[n*n];

    // Create test arrays
    float test_image_arr[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
    float* test_conv_arr = new float[16];
    float test_mask_arr[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};

     // Create vectors for random number generation
    vector<float> random_val_n(n*n);
    vector<float> random_val_m(m*m);

    // Initialization for timing
    duration<double, std::milli> duration_ms;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;

    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());

    // // Generate and fill random values for the image matrix
    // uniform_real_distribution<float> distron(-10.0,10.0);
    // for(auto& value_n : random_val_n){
    //     value_n=distron(generator);
    // }
    // for(int i = 0; i<n*n; i++){
    //     image_arr[i] = random_val_n[i];
    // }

    // // Generate and fill random values for the mask matrix
    // uniform_real_distribution<float> distrom(-1.0,1.0);
    // for(auto& value_m : random_val_m){
    //     value_m=distrom(generator);
    // }
    // for(int i = 0; i<m*m; i++){
    //     mask_arr[i] = random_val_m[i];
    // }

    // Test Arrays
 


    //Execute and time Algorithm Execution
    start = high_resolution_clock::now();
    //convolve(image_arr, conv_arr, n, mask_arr, m); 
    convolve(test_image_arr, test_conv_arr, 4, test_mask_arr, 3);
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

    //Print results
    cout << duration_ms.count() <<"\n";
    cout << conv_arr[0] << "\n";
    cout << conv_arr[n*n-1] << "\n";

    //Deallocate Preallocated Arrays
    delete[] image_arr;
    delete[] conv_arr;
    delete[] mask_arr;
    delete[] test_conv_arr;
}