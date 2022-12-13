#!/usr/bin/env zsh
#SBATCH --job-name=ctest
#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --cpus-per-task=1
#SBATCH --ntasks=1  --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="ctest.out"
#SBATCH --error="ctest.err"

module load nvidia/cuda/11.6.0

nvcc torque_iterator_gpu.cu single_definitions.cu RK4.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o ctest

./ctest