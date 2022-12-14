#!/usr/bin/env zsh
#SBATCH --job-name=p2test
#SBATCH --partition=wacc
#SBATCH --nodes=1 --cpus-per-task=20
#SBATCH --ntasks=1  --cpus-per-task=20
#SBATCH --time=0-00:10:00
#SBATCH --output="p2test.out"
#SBATCH --error="p2test.err"

g++ double_torque_iterator_OMP.cpp double_definitions.cpp double_RK4.cpp -Wall -O3 -std=c++17 -o p2test -fopenmp

./p2test