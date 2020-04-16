echo Testing CPU file
g++ -std=c++11 add.cpp && ./a.out

echo Testing GPU, unified memory version
nvcc -std=c++11 addManaged.cu

echo Compiling done, trying each of 4 variants
./a.out 1
./a.out 2
./a.out 3
./a.out 4

echo Testing GPU, separate memory version
nvcc -std=c++11 add.cu

echo Compiling done, trying each of 4 variants
./a.out 1
./a.out 2
./a.out 3
./a.out 4
