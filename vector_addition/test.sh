#!/bin/zsh

echo Testing CPU file
g++ -std=c++11 add.cpp
./a.out > a
echo Finished CPU variant with stats: $(python3 reader.py a)

echo Testing GPU
nvcc -std=c++11 add.cu

echo Compiling done, trying each of 4 variants

nvprof ./a.out 1 > a 2>&1
echo Finished GPU variant 4 with stats: $(python3 reader.py a)

nvprof ./a.out 2 > a 2>&1
echo Finished GPU variant 2 with stats: $(python3 reader.py a)

nvprof ./a.out 3 > a 2>&1
echo Finished GPU variant 3 with stats: $(python3 reader.py a)

nvprof ./a.out 4 > a 2>&1
echo Finished GPU variant 4 with stats: $(python3 reader.py a)
