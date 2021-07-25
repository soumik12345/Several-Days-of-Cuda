#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct BlockParams {
    int x, y, z;
};


struct GridParams {
    int x, y, z;
};

struct ThreadParams {
    int x, y, z;
};


__global__ void unique_index_calculation_kernel(int* input) {

    int threadId = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int globalIndex = threadId + offset;
    printf(
            "threadIdx.x = %d, offset = %d, globalIndex = %d, Value = %d\n",
            threadId, offset, globalIndex, input[globalIndex]
    );
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

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    printf("Array Data: ");
    for(int i = 0; i < arraySize; i++)
        printf("%d ", inputArray[i]);
    printf("\n\n");

    int* gpuData;
    cudaMalloc((void**)&gpuData, arraySizeBytes);
    cudaMemcpy(gpuData, inputArray, arraySizeBytes, cudaMemcpyHostToDevice);

    dim3 block(BlockParameters.x, BlockParameters.y, BlockParameters.z);
    dim3 grid(GridParameters.x, GridParameters.y, GridParameters.z);

    unique_index_calculation_kernel << <grid, block >> > (gpuData);
    cudaDeviceSynchronize();
}


inline void Demo() {

    int arraySize = 16;
    int arraySizeBytes = sizeof(int) * arraySize;
    int inputArray[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 65, 99, 164, 263, 427, 690};

    int n_grids = 4;

    BlockParams blockParams = {
            arraySize / n_grids, 1, 1
    };

    GridParams gridParams = {
            n_grids, 1, 1
    };

    UniqueIndexCalculation program = UniqueIndexCalculation(blockParams, gridParams);
    program.run(inputArray, arraySize, arraySizeBytes);
}

int main() {
    Demo();
}
