#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>


__global__ void vector_addition_kernel(int* A, int* B, int* result, int vLen) {

    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadId < vLen)
        result[threadId] = A[threadId] + B[threadId];
}


class VectorAdditionProgram {

public:

    VectorAdditionProgram(int, int, int);

    void run();

private:

    int vectorLength, threadBlockSize, gridSize;

    void initVector(int*, int);

    void displayResult(int*, int*, int*, int);

    void checkResult(int*, int*, int*, int);
};


VectorAdditionProgram::VectorAdditionProgram(int vectorLength, int threadBlockSize, int gridSize) {

    this->vectorLength = vectorLength;
    this->threadBlockSize = threadBlockSize;
    this->gridSize = gridSize;
}

void VectorAdditionProgram::initVector(int* vector, int vLen) {

    for (int i = 0; i < vLen; i++)
        vector[i] = rand() % 100;
}

void VectorAdditionProgram::displayResult(int* A, int* B, int* result, int vLen) {

    for (int i = 0; i < vLen; i++)
        printf("%d + %d = %d\n", A[i], B[i], result[i]);

    printf("\n");
}

void VectorAdditionProgram::checkResult(int* A, int* B, int* result, int vLen) {

    for (int i = 0; i < vLen; i++)
        assert(result[i] == A[i] + B[i]);
}

void VectorAdditionProgram::run() {

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    int * hostA, * hostB, * hostResult;
    int * deviceA, * deviceB, * deviceResult;
    size_t vectorBytes = sizeof(int) * vectorLength;

    cudaMallocHost(&hostA, vectorBytes);
    cudaMallocHost(&hostB, vectorBytes);
    cudaMallocHost(&hostResult, vectorBytes);

    cudaMalloc(&deviceA, vectorBytes);
    cudaMalloc(&deviceB, vectorBytes);
    cudaMalloc(&deviceResult , vectorBytes);

    initVector(hostA, vectorLength);
    initVector(hostB, vectorLength);

    cudaMemcpy(deviceA, hostA, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, vectorBytes, cudaMemcpyHostToDevice);

    vector_addition_kernel << <threadBlockSize, gridSize >> > (deviceA, deviceB, deviceResult, vectorLength);
    cudaDeviceSynchronize();

    cudaMemcpy(hostResult, deviceResult, vectorBytes, cudaMemcpyDeviceToHost);

    if (vectorLength <= 1 << 4)
        displayResult(hostA, hostB, hostResult, vectorLength);
    else {

        checkResult(hostA, hostB, hostResult, vectorLength);
        printf("Program Successfully Executed");
    }

    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostResult);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);
}


int main() {

    int vectorLength = 1 << 16;
    int threadBlockSize = 1 << 10;
    int gridSize = (vectorLength + threadBlockSize - 1) / threadBlockSize;

    VectorAdditionProgram program = VectorAdditionProgram(vectorLength, threadBlockSize, gridSize);
    program.run();
}
