#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>


__global__ void array_manipulation_kernel(int* a, int n) {
    unsigned int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        a[index] *= 2;
}


class ArrayManipulation {

public:

    int arrayLength;

    explicit ArrayManipulation(int arrayLength);

    void initArray(int* array) const;
    void run(int numGrids, int numThreads) const;
    void displayResult(int* array, int* resultArray) const;
    void checkResult(const int* array, const int* resultArray) const;
};

ArrayManipulation::ArrayManipulation(int arrayLength) {
    this->arrayLength = arrayLength;
}

void ArrayManipulation::initArray(int *array) const {
    for(int i = 0; i < this->arrayLength; i++)
        array[i] = rand() % 100;
}

void ArrayManipulation::displayResult(int *array, int* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        printf("%d * 2 = %d\n", array[i], resultArray[i]);
}

void ArrayManipulation::checkResult(const int *array, const int* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        assert(resultArray[i] == array[i] * 2);
    printf("Program Executed Successfully");
}

void ArrayManipulation::run(int numGrids, int numThreads) const {

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    int * hostArray, * resultArray, * deviceArray;
    size_t arrayBytes = sizeof(int) * this->arrayLength;

    cudaMallocHost(&hostArray, arrayBytes);
    cudaMallocHost(&resultArray, arrayBytes);
    cudaMalloc(&deviceArray, arrayBytes);

    initArray(hostArray);
    cudaMemcpy(deviceArray, hostArray, arrayBytes, cudaMemcpyHostToDevice);

    array_manipulation_kernel<<<numGrids, numThreads>>>(deviceArray, arrayLength);
    cudaDeviceSynchronize();

    cudaMemcpy(resultArray, deviceArray, arrayBytes, cudaMemcpyDeviceToHost);

    displayResult(hostArray, resultArray);
    checkResult(hostArray, resultArray);

    cudaFreeHost(hostArray);
    cudaFreeHost(resultArray);
    cudaFree(deviceArray);
}


int main() {
    ArrayManipulation program(16);
    program.run(1, 16);
}
