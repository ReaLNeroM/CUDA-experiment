#!/bin/zsh

echo Testing CPU file
g++ -std=c++11 prefix_scan.cpp
rm -f a

./a.out >> a 2>&1

cat a
echo Finished prefix_scan.cpp with stats: $(python3 reader.py a)

foreach n ($(ls *.cu))
    nvcc -std=c++11 $n
    rm -f a

    echo Testing $n file
    nvprof ./a.out >> a 2>&1

    echo Finished $n with stats: $(python3 reader.py a)
end
