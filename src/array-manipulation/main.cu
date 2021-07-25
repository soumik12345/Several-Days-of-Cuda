#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cassert>


__global__ void array_manipulation_kernel(int* a, int n) {
    unsigned int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        a[index] *= 2;
}


class ArrayManipulation {

public:

    int *array;
    int arrayLength;    // Number of elements in the array
    size_t arraySize;   // Size of the array in bytes

    explicit ArrayManipulation(int length);

    void run(size_t numGrids, size_t numThreads);
    void DisplayArray() const;

private:
    void assertResult() const;
};

ArrayManipulation::ArrayManipulation(int length) {
    this->arrayLength = length;
    this->arraySize = length * sizeof(int);
    cudaMallocManaged(&array, arraySize);
    for(int i = 0; i < length; i++)
        this->array[i] = i;
}

void ArrayManipulation::DisplayArray() const {
    for(int i = 0; i < this->arrayLength; i++)
        printf("Index %d: %d\n", i, this->array[i]);
}

void ArrayManipulation::run(size_t numGrids, size_t numThreads) {
    array_manipulation_kernel<<<numGrids, numThreads>>>(array, arraySize);
    cudaDeviceSynchronize();
    this->assertResult();
}

void ArrayManipulation::assertResult() const {
    for(int i = 0; i < arrayLength; i++)
        assert (array[i] == i * 2);
}

int main() {
    int arrayLength = 10;
    ArrayManipulation program(arrayLength);
    size_t numThreads = 256;
    size_t numGrids = (arrayLength + numThreads - 1) / numThreads;
    program.run(numGrids, numThreads);
    program.DisplayArray();
}
