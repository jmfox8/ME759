#!/usr/bin/env zsh
#SBATCH --job-name=test
#SBATCH --partition=wacc
#SBATCH --nodes=1 --cpus-per-task=20
#SBATCH --ntasks=1  --cpus-per-task=20
#SBATCH --time=0-00:10:00
#SBATCH --output="test.out"
#SBATCH --error="test.err"

g++ torque_iterator.cpp single_definitions.cpp RK4.cpp -Wall -O3 -std=c++17 -o test 

./test