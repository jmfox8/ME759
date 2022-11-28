#include "optimize.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <algorithm> // std::sort
#include <chrono>


int main(int argc, char *argv[]){
    // Define input variable from command line
    int n = atoi(argv[1]);
    
    // Create vec
    vec v(n);

    //Create output variables
    data_t dest1, dest2, dest3, dest4, dest5;
    
    // Initialization for timing
    std::chrono::duration<double, std::milli> ms1, ms2, ms3, ms4, ms5;
    std::chrono::high_resolution_clock::time_point start1, start2, start3, start4, start5;
    std::chrono::high_resolution_clock::time_point end1, end2, end3, end4, end5;
    double ms1avg = 0.0, ms2avg = 0.0, ms3avg = 0.0, ms4avg = 0.0, ms5avg = 0.0;
    
    //int* arr = new int[n];
    data_t* arr = new data_t[n];

    // Populate Array arr
    for (int i = 0; i<n; i++){
            arr[i] = static_cast < data_t > (1);
    }
    //Assign values to the data pointer of the struct
    v.data = arr;

    // Run and time the optimizations
    for (int i = 0; i<10; i++){
        start1 = std::chrono::high_resolution_clock::now();
        optimize1(&v, &dest1);
        end1 = std::chrono::high_resolution_clock::now();
        ms1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end1 - start1);
        ms1avg += ms1.count();

        start2 = std::chrono::high_resolution_clock::now();
        optimize2(&v, &dest2);
        end2 = std::chrono::high_resolution_clock::now();
        ms2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end2 - start2);
        ms2avg += ms2.count();

        start3 = std::chrono::high_resolution_clock::now();
        optimize3(&v, &dest3);
        end3 = std::chrono::high_resolution_clock::now();
        ms3 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end3 - start3);
        ms3avg += ms3.count();

        start4 = std::chrono::high_resolution_clock::now();
        optimize4(&v, &dest4);
        end4 = std::chrono::high_resolution_clock::now();
        ms4 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end4 - start4);
        ms4avg += ms4.count();

        start5 = std::chrono::high_resolution_clock::now();
        optimize5(&v, &dest5);
        end5 = std::chrono::high_resolution_clock::now();
        ms5 = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end5 - start5);
        ms5avg += ms5.count();
    }
    //Print Results
    std::cout<<dest1<<"\n"<<ms1avg/10<<"\n";
    std::cout<<dest2<<"\n"<<ms2avg/10<<"\n";
    std::cout<<dest3<<"\n"<<ms3avg/10<<"\n";
    std::cout<<dest4<<"\n"<<ms4avg/10<<"\n";
    std::cout<<dest5<<"\n"<<ms5avg/10<<"\n";

    // Deallocate Memory
    delete[] arr;
    
    return(0);
}