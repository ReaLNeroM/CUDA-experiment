# CUDA-experiment
Vector addition parallelized using CUDA.

# Documentation
- test.sh - Compiles and executes all the code files below.
- exec_log.sh - Execution log when I ran the shell script on gpu-node01.csug.rochester.edu.
- add.cpp - simple CPU implementation of vector addition. Prints the amount of error of vector addition.
- add.cu - GPU-parallelized implementation of vector addition, with 4 variants tested. Each vector is copied to the device and back to the host. For each variant, it prints out the amount of time taken, and then error amount (should be 0).
- addManaged.cu - GPU-parallelized implementation of vector addition, with the same 4 variants tested. Here, memcpy is avoided by allocating memory which can be accessed by the device and host at the same time.