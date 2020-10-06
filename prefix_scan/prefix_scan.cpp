#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

// MUST BE ASSOCIATIVE
int f(int a, int b){
    return a + b;
}

int main(){
    int n = (1 << 28);

    int *x = (int *) malloc(n * sizeof(int));
    assert(x != NULL);

    for(int i = 0; i < n; i++){
        x[i] = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
    for(int i = 1; i < n; i++){
        x[i] += x[i - 1];
    }
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "prefix_scan: " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() / 1000.0 << "ms\n";
    std::cout << "memory usage: " << n * sizeof(int) << " bytes" << std::endl;
}
