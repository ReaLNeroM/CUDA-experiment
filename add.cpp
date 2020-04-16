#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>

void add(int n, float *x, float *y, float *ans){
    for(int i = 0; i < n; i++){
        ans[i] = x[i] + y[i];
    }
}

int main(){
    int n = (1 << 20);

    float *x, *y, *ans;

    x = (float *) malloc(n * sizeof(float));
    y = (float *) malloc(n * sizeof(float));
    ans = (float *) malloc(n * sizeof(float));

    for(int i = 0; i < n; i++){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    add(n, x, y, ans);
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "add CPU: " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << "us\n";

    float err = 0.0;
    for(int i = 0; i < n; i++){
        err += abs(ans[i] - 3.0);
    }

    std::cout << err << '\n';
}