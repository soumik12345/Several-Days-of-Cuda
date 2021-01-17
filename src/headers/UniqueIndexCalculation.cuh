#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Parameters.cuh"


__global__ void unique_index_calculation_kernel(int* input) {

	int threadId = threadIdx.x;
	printf("threadId.x = %d, Value = %d\n", threadId, input[threadId]);
}


class UniqueIndexCalculation {

public:
	
	UniqueIndexCalculation(BlockParams, GridParams);
	UniqueIndexCalculation(BlockParams, ThreadParams);

	~UniqueIndexCalculation();

	void run(int*, int, int);

	BlockParams BlockParameters;
	GridParams GridParameters;
};


UniqueIndexCalculation::UniqueIndexCalculation(BlockParams bParams, GridParams gParams) {
	
	BlockParameters = bParams;
	GridParameters = gParams;
}


UniqueIndexCalculation::UniqueIndexCalculation(BlockParams bParams, ThreadParams tParams) {

	BlockParameters = bParams;
	GridParameters = {
		tParams.x / bParams.x,
		tParams.y / bParams.y,
		tParams.z / bParams.z,
	};
}


UniqueIndexCalculation::~UniqueIndexCalculation() {

	cudaDeviceReset();
}


void UniqueIndexCalculation::run(int* inputArray, int arraySize, int arraySizeBytes) {

	printf("Array Data: ");
	for(int i = 0; i < arraySize; i++)
		printf("%d ", inputArray[i]);
	printf("\n\n");

	int* gpuData;
	cudaMalloc((void**)&gpuData, arraySizeBytes);
	cudaMemcpy(gpuData, inputArray, arraySizeBytes, cudaMemcpyHostToDevice);

	dim3 block(arraySize, 1, 1);
	dim3 grid(1, 1, 1);

	unique_index_calculation_kernel << <grid, block >> > (gpuData);
	cudaDeviceSynchronize();
}


inline void Demo() {

	int arraySize = 8;
	int arraySizeBytes = sizeof(int) * arraySize;
	int inputArray[] = {0, 1, 1, 2, 3, 5, 8, 13};

	BlockParams blockParams = {
		arraySize, 1, 1
	};

	GridParams gridParams = {
		1, 1, 1
	};
	
	UniqueIndexCalculation program = UniqueIndexCalculation(blockParams, gridParams);
	program.run(inputArray, arraySize, arraySizeBytes);
}
