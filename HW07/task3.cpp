#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>

int main(){
    int tnum = 4;
    omp_set_num_threads(tnum); 

    #pragma omp parallel
    #pragma omp master
        {
            std::cout<<"Number of Threads: "<<omp_get_num_threads()<<"\n";    
        }
    #pragma omp parallel
        {
            int thread = omp_get_thread_num();
            printf("I am thread No. %d\n",thread);
            int fact = 1;
            for(int i = thread; i < 8; i += tnum){
                for(int j = 1; j <= i+1; j++){
                    fact *= j; 
                }
                printf("%d!=%d\n",i+1,fact);
                fact = 1;
            }
        }
}