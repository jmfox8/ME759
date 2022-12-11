#!/usr/bin/env zsh
#SBATCH --job-name=ptest
#SBATCH --partition=wacc
#SBATCH --nodes=1 --cpus-per-task=20
#SBATCH --ntasks=1  --cpus-per-task=20
#SBATCH --time=0-00:10:00
#SBATCH --output="ptest.out"
#SBATCH --error="ptest.err"

g++ torque_iterator_OMP.cpp single_definitions.cpp RK4.cpp -Wall -O3 -std=c++17 -o ptest -fopenmp

./ptest