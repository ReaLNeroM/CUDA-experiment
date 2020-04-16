#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>

void add(int n, double *x, double *y, double *ans){
    for(int i = 0; i < n; i++){
        ans[i] = x[i] + y[i];
    }
}

int main(){
    int n = (1 << 20);

    double *x, *y, *ans;

    x = (double *) malloc(n * sizeof(double));
    y = (double *) malloc(n * sizeof(double));
    ans = (double *) malloc(n * sizeof(double));

    for(int i = 0; i < n; i++){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    add(n, x, y, ans);

    double err = 0.0;
    for(int i = 0; i < n; i++){
        err += abs(ans[i] - 3.0);
    }

    std::cout << err << '\n';
}