#pragma once


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


class VectorAdditionUnifiedMemory {

public:

	VectorAdditionUnifiedMemory(int, int, int);

	void run(bool);

private:

	int vectorLength, threadBlockSize, gridSize;

	void initVector(int*, int);

	void displayResult(int*, int*, int*, int);

	void checkResult(int*, int*, int*, int);

};


VectorAdditionUnifiedMemory::VectorAdditionUnifiedMemory(int vectorLength, int threadBlockSize, int gridSize) {

	this->vectorLength = vectorLength;
	this->threadBlockSize = threadBlockSize;
	this->gridSize = gridSize;
}

void VectorAdditionUnifiedMemory::initVector(int* vector, int vLen) {

	for (int i = 0; i < vLen; i++)
		vector[i] = rand() % 100;
}

void VectorAdditionUnifiedMemory::displayResult(int* A, int* B, int* result, int vLen) {
	
	for (int i = 0; i < vLen; i++)
		printf("%d + %d = %d\n", A[i], B[i], result[i]);
	
	printf("\n");
}

void VectorAdditionUnifiedMemory::checkResult(int* A, int* B, int* result, int vLen) {
	
	for (int i = 0; i < vLen; i++)
		assert(result[i] == A[i] + B[i]);
}

void VectorAdditionUnifiedMemory::run(bool prefetchMemory) {

	int deviceId = cudaGetDevice(&deviceId);

	printf("GPU Device ID: %d\n", deviceId);
	printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

	int * vectorA, * vectorB, * vectorResult;
	size_t vectorBytes = sizeof(int) * vectorLength;

	cudaMallocManaged(&vectorA, vectorBytes);
	cudaMallocManaged(&vectorB, vectorBytes);
	cudaMallocManaged(&vectorResult, vectorBytes);

	initVector(vectorA, vectorLength);
	initVector(vectorB, vectorLength);

	if (prefetchMemory) {

		cudaMemPrefetchAsync(vectorA, vectorBytes, deviceId);
		cudaMemPrefetchAsync(vectorB, vectorBytes, deviceId);
	}

	vector_addition_kernel<<<threadBlockSize, gridSize>>>(vectorA, vectorB, vectorResult, vectorLength);
	cudaDeviceSynchronize();

	if (prefetchMemory)
		cudaMemPrefetchAsync(vectorResult, vectorBytes, cudaCpuDeviceId);

	if (vectorLength <= 1 << 4)
		displayResult(vectorA, vectorB, vectorResult, vectorLength);
	else {
		checkResult(vectorA, vectorB, vectorResult, vectorLength);
		printf("Program Successfully Executed");
	}

	cudaFree(vectorA);
	cudaFree(vectorB);
	cudaFree(vectorResult);
}

inline void Demo() {

	int vectorLength = 1 << 16;
	int threadBlockSize = 1 << 10;
	int gridSize = (vectorLength + threadBlockSize - 1) / threadBlockSize;

	VectorAdditionUnifiedMemory program = VectorAdditionUnifiedMemory(vectorLength, threadBlockSize, gridSize);
	program.run(true);
}
