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


__global__ void block_dim_demo_kernel() {

    printf(
            "threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d\n",
            threadIdx.x, threadIdx.y, threadIdx.z
    );

    printf(
            "blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n",
            blockIdx.x, blockIdx.y, blockIdx.z
    );

    printf(
            "blockDim.x = %d, blockDim.y = %d, blockDim.z = %d\n",
            blockDim.x, blockDim.y, blockDim.z
    );
}


class BlockDimDemo {

public:

    BlockDimDemo(BlockParams, GridParams);
    BlockDimDemo(BlockParams, ThreadParams);

    ~BlockDimDemo();

    void run();

    BlockParams BlockParameters;
    GridParams GridParameters;
};


BlockDimDemo::BlockDimDemo(BlockParams bParams, GridParams gParams) {

    BlockParameters = bParams;
    GridParameters = gParams;
}


BlockDimDemo::BlockDimDemo(BlockParams bParams, ThreadParams tParams) {

    BlockParameters = bParams;
    GridParameters = {
            tParams.x / bParams.x,
            tParams.y / bParams.y,
            tParams.z / bParams.z,
    };
}


BlockDimDemo::~BlockDimDemo() {

    cudaDeviceReset();
}


void BlockDimDemo::run() {

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    dim3 block(BlockParameters.x, BlockParameters.y, BlockParameters.z);
    dim3 grid(GridParameters.x, GridParameters.y, GridParameters.z);

    block_dim_demo_kernel << <grid, block >> > ();
    cudaDeviceSynchronize();
}


inline void Demo() {

    BlockParams blockParams = {
            8, 8, 1
    };

    ThreadParams threadParams = {
            16, 16, 1
    };

    BlockDimDemo program = BlockDimDemo(blockParams, threadParams);
    program.run();
}

int main() {
    Demo();
}
