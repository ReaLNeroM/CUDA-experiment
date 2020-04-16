echo Testing CPU file
g++ add.cpp && ./a.out

echo Testing GPU, unified memory version
nvcc addManaged.cu

echo Compiling done, trying each of 4 variants
./a.out 1
./a.out 2
./a.out 3
./a.out 4

echo Testing GPU, separate memory version
nvcc add.cu

echo Compiling done, trying each of 4 variants
./a.out 1
./a.out 2
./a.out 3
./a.out 4
