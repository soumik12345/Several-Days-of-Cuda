#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <cmath>


__global__ void fibonacci_kernel(double* a, int n) {
    unsigned int index = threadIdx.x;
    if (index < n)
        a[index] = (pow((1 + sqrt(5.0)) / 2, index) - pow((1 - sqrt(5.0)) / 2, index)) / sqrt(5.0);
}


class Fibonacci {

public:

    int arrayLength;
    double phi;

    explicit Fibonacci(int arrayLength);

    void run(int numGrids, int numThreads) const;
    void displayResult(double* array, double* resultArray) const;
    void checkResult(const int* array, const int* resultArray) const;
};

Fibonacci::Fibonacci(int arrayLength) {
    this->arrayLength = arrayLength;
    this->phi = (1 - sqrt(5)) / 2.0;
}

void Fibonacci::displayResult(double *array, double* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        printf("Index %d: %f\n", i, resultArray[i]);
}

void Fibonacci::checkResult(const int *array, const int* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        assert(resultArray[i] == array[i] * 2);
    printf("Program Executed Successfully");
}

void Fibonacci::run(int numGrids, int numThreads) const {

    double * hostArray, * resultArray, * deviceArray;
    size_t arrayBytes = sizeof(int) * this->arrayLength;

    cudaMallocHost(&hostArray, arrayBytes);
    cudaMallocHost(&resultArray, arrayBytes);
    cudaMalloc(&deviceArray, arrayBytes);

    cudaMemcpy(deviceArray, hostArray, arrayBytes, cudaMemcpyHostToDevice);

    fibonacci_kernel<<<numGrids, numThreads>>>(deviceArray, arrayLength);
    cudaDeviceSynchronize();

    cudaMemcpy(resultArray, deviceArray, arrayBytes, cudaMemcpyDeviceToHost);

    displayResult(hostArray, resultArray);
//    checkResult(hostArray, resultArray);

    cudaFreeHost(hostArray);
    cudaFreeHost(resultArray);
    cudaFree(deviceArray);
}


void Demo() {
    Fibonacci program(16);
    program.run(1, 256);
}
