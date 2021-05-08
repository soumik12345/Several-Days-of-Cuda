#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cassert>


__global__ void manipulate_array(int* a, int n) {
    unsigned int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        a[index] *= 2;
}


class ArrayManipulationProgram {

public:

    int *array;
    int arrayLength;    // Number of elements in the array
    size_t arraySize;   // Size of the array in bytes

    explicit ArrayManipulationProgram(int length);

    void run(size_t numThreads);
    void DisplayArray() const;

private:
    void assertResult() const;
};

ArrayManipulationProgram::ArrayManipulationProgram(int length) {
    this->arrayLength = length;
    this->arraySize = length * sizeof(int);
    cudaMallocManaged(&array, arraySize);
    for(int i = 0; i < length; i++)
        this->array[i] = i;
}

void ArrayManipulationProgram::DisplayArray() const {
    for(int i = 0; i < this->arrayLength; i++)
        printf("Index %d: %d\n", i, this->array[i]);
}

void ArrayManipulationProgram::run(size_t numThreads) {
    size_t numBlocks = (arrayLength + numThreads - 1) / numThreads;
    manipulate_array<<<numBlocks, numThreads>>>(array, arraySize);
    cudaDeviceSynchronize();
    this->assertResult();
}

void ArrayManipulationProgram::assertResult() const {
    for(int i = 0; i < arrayLength; i++)
        assert (array[i] == i * 2);
}

void Demo() {
    ArrayManipulationProgram program(10);
    program.run(256);
    program.DisplayArray();
}
