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


__global__ void unique_index_calculation_2d_kernel(int* input) {

    int threadId = threadIdx.x;
    int offsetPerRow = gridDim.x * blockDim.x * blockIdx.y;
    int offsetPerBlock = blockIdx.x * blockDim.x;
    int globalIndex = threadId + offsetPerBlock + offsetPerRow;
    printf(
            "blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, offset_row = %d, offset_block = %d, globalIndex = %d, Value = %d\n",
            blockIdx.x, blockIdx.y, threadId, offsetPerRow, offsetPerBlock, globalIndex, input[globalIndex]
    );
}


class UniqueIndexCalculation2D {

public:

    UniqueIndexCalculation2D(BlockParams, GridParams);
    UniqueIndexCalculation2D(BlockParams, ThreadParams);

    ~UniqueIndexCalculation2D();

    void run(int*, int, int);

    BlockParams BlockParameters;
    GridParams GridParameters;
};


UniqueIndexCalculation2D::UniqueIndexCalculation2D(BlockParams bParams, GridParams gParams) {

    BlockParameters = bParams;
    GridParameters = gParams;
}


UniqueIndexCalculation2D::UniqueIndexCalculation2D(BlockParams bParams, ThreadParams tParams) {

    BlockParameters = bParams;
    GridParameters = {
            tParams.x / bParams.x,
            tParams.y / bParams.y,
            tParams.z / bParams.z,
    };
}


UniqueIndexCalculation2D::~UniqueIndexCalculation2D() {

    cudaDeviceReset();
}


void UniqueIndexCalculation2D::run(int* inputArray, int arraySize, int arraySizeBytes) {

    printf("Array Data: ");
    for(int i = 0; i < arraySize; i++)
        printf("%d ", inputArray[i]);
    printf("\n\n");

    int* gpuData;
    cudaMalloc((void**)&gpuData, arraySizeBytes);
    cudaMemcpy(gpuData, inputArray, arraySizeBytes, cudaMemcpyHostToDevice);

    dim3 block(BlockParameters.x, BlockParameters.y, BlockParameters.z);
    dim3 grid(GridParameters.x, GridParameters.y, GridParameters.z);

    unique_index_calculation_2d_kernel << <grid, block >> > (gpuData);
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
            n_grids / 2, n_grids / 2, 1
    };

    UniqueIndexCalculation2D program = UniqueIndexCalculation2D(blockParams, gridParams);
    program.run(inputArray, arraySize, arraySizeBytes);
}

int main() {
    Demo();
}
