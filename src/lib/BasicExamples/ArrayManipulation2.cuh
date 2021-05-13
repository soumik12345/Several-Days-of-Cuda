#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>


__global__ void manipulate_array(int* a, int n) {
    unsigned int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        a[index] *= 2;
}


class ArrayManipulationProgram2 {

public:

    int arrayLength;

    explicit ArrayManipulationProgram2(int arrayLength);

    void initArray(int* array) const;
    void run(int numGrids, int numThreads) const;
    void displayResult(int* array, int* resultArray) const;
    void checkResult(const int* array, const int* resultArray) const;
};

ArrayManipulationProgram2::ArrayManipulationProgram2(int arrayLength) {
    this->arrayLength = arrayLength;
}

void ArrayManipulationProgram2::initArray(int *array) const {
    for(int i = 0; i < this->arrayLength; i++)
        array[i] = rand() % 100;
}

void ArrayManipulationProgram2::displayResult(int *array, int* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        printf("%d * 2 = %d\n", array[i], resultArray[i]);
}

void ArrayManipulationProgram2::checkResult(const int *array, const int* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        assert(resultArray[i] == array[i] * 2);
    printf("Program Executed Successfully");
}

void ArrayManipulationProgram2::run(int numGrids, int numThreads) const {

    int * hostArray, * resultArray, * deviceArray;
    size_t arrayBytes = sizeof(int) * this->arrayLength;

    cudaMallocHost(&hostArray, arrayBytes);
    cudaMallocHost(&resultArray, arrayBytes);
    cudaMalloc(&deviceArray, arrayBytes);

    initArray(hostArray);
    cudaMemcpy(deviceArray, hostArray, arrayBytes, cudaMemcpyHostToDevice);

    manipulate_array<<<numGrids, numThreads>>>(deviceArray, arrayLength);
    cudaDeviceSynchronize();

    cudaMemcpy(resultArray, deviceArray, arrayBytes, cudaMemcpyDeviceToHost);

    displayResult(hostArray, resultArray);
    checkResult(hostArray, resultArray);

    cudaFreeHost(hostArray);
    cudaFreeHost(resultArray);
    cudaFree(deviceArray);
}


void Demo() {
    ArrayManipulationProgram2 program1(16);
    program1.run(1, 16);
}
