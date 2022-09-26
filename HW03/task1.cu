#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cuda.h>

__global__ void fac()
{
  int factorial = 1;
  int index = threadIdx.x;
  // Loop to have each thread execute and print the factorial for a different number
  for (int i = 1; i<= index+1; i++){
    factorial *= i;
  }
  std::printf("%d! = %d \n",index+1,factorial);
}

int main(int argc, char *argv[]){

  // Run kernel on the GPU
  fac <<<1, 8>>>();

  // Wait for GPU to finish
  cudaDeviceSynchronize();

return 0;
}
