#!/usr/bin/env zsh
#SBATCH --job-name=task8_1
#SBATCH --partition=wacc
#SBATCH --nodes=1 --cpus-per-task=20
#SBATCH --ntasks=1  --cpus-per-task=20
#SBATCH --time=0-00:10:00
#SBATCH --output="task8_1.out"
#SBATCH --error="task8_1.err"

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

./task1 4 5