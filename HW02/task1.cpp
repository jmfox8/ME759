#include "scan.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>


using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[]){

    int n = atoi(argv[1]);
    float* inputarray = new float[n];
    float* outputarray = new float[n];
    vector<float> random_val(n);
    duration<double, std::milli> duration_ms;

    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_real_distribution<float> distro(-1.0,1.0);
    for(auto& value : random_val) {
        value = distro(generator);
    } 
    for(int i = 0; i<n; i++){
        inputarray[i] = random_val[i];
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    start = high_resolution_clock::now();
    scan(inputarray, outputarray,n);
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli> >(end-start);

    std::cout << outputarray[0] << "\n";
    std::cout << outputarray[n-1] << "\n";
    std::cout << duration_ms.count() <<"\n";
//will need to free memory malloced for the output array
}
