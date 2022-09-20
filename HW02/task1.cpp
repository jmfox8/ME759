#include "scan.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>


using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[]){

    // Define input variable from command line
    int n = atoi(argv[1]);
    
    // Create vector for random number generation
    vector<float> random_val(n);
    
    // Initialization for timing
    duration<double, std::milli> duration_ms;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    
    // Allocate Memory for arrays
    float* inputarray = new float[n];
    float* outputarray = new float[n];
    
    // Random Array generation
    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_real_distribution<float> distro(-1.0,1.0);
    for(auto& value : random_val){
        value = distro(generator);
    } 
    for(int i = 0; i<n; i++){
        inputarray[i] = random_val[i];
    }
    
    //Execute and time function
    start = high_resolution_clock::now();
    scan(inputarray, outputarray,n);
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

    //Print results
    std::cout << outputarray[0] << "\n";
    std::cout << outputarray[n-1] << "\n";
    std::cout << duration_ms.count() <<"\n";

    //Free Allocated Memory
    delete[] inputarray;
    delete[] outputarray;

    return 0;
}
