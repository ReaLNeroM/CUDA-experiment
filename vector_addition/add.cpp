#include <iostream>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>

void add(int n, double *x, double *y, double *ans){
    for(int i = 0; i < n; i++){
        ans[i] = x[i] + y[i];
    }
}

int main(){
    int n = (1 << 25);

    double *x, *y, *ans;

    x = (double *) malloc(n * sizeof(double));
    y = (double *) malloc(n * sizeof(double));
    ans = (double *) malloc(n * sizeof(double));

    for(int i = 0; i < n; i++){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
    add(n, x, y, ans);
    finish = std::chrono::high_resolution_clock::now();

    std::cout << "vector_addition: " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() / 1000.0 << "ms\n";
    std::cout << "memory usage: " << n * sizeof(int) << " bytes" << '\n';

    double err = 0.0;
    for(int i = 0; i < n; i++){
        err += abs(ans[i] - 3.0);
    }

    std::cout << err << '\n';
}