#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#define ULI unsigned long int


__global__ void fibonacci_kernel(ULI* a, int start) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = i + start;
    if (i < 2 * start - 1)
        a[index] = (a[start - 2] * a[i]) + (a[start - 1] * a[i + 1]);
}


class FibonacciDynamicProgramming {

public:

    int numElements, sizeInBytes;

    explicit FibonacciDynamicProgramming(int numElements);

    void run(int numThreads) const;
};

FibonacciDynamicProgramming::FibonacciDynamicProgramming(int numElements) {
    this->numElements = numElements;
    this->sizeInBytes = numElements * sizeof(ULI);
}

void FibonacciDynamicProgramming::run(int numThreads) const {

    ULI startingElements[3] = {1, 1, 2};
    ULI* deviceArray;
    ULI resultArray[numThreads];

    cudaMalloc(&deviceArray, sizeInBytes);
    cudaMemcpy(deviceArray, startingElements, sizeof(startingElements), cudaMemcpyHostToDevice);

    unsigned int start = 3;
    while (start <= numElements / 2 ) {
        unsigned int numBlocks = (start - 1) / numThreads;
        if ((start - 1) % numThreads != 0)
            numBlocks++;
        fibonacci_kernel<<<numBlocks, numThreads>>>(deviceArray, start);
        start = 2 * start - 1;
    }

    cudaMemcpy(resultArray, deviceArray, sizeInBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements; i++) {
        printf("%d:\t%lu \n", i + 1, resultArray[i]);
    }

    cudaFree(deviceArray);

}


int main() {
    FibonacciDynamicProgramming program(16);
    program.run(16);
}
