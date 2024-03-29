#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void parallel_for_loop() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Current Iteration Number: %d\n", index);
}


class ParallelizedForLoopProgramMultipleBlocks {

public:
    int nBlocks, nThreads;

    ParallelizedForLoopProgramMultipleBlocks(int nBlocks, int nThreads);
    void run();
};

ParallelizedForLoopProgramMultipleBlocks::ParallelizedForLoopProgramMultipleBlocks(int nBlocks, int nThreads) {
    this->nBlocks = nBlocks;
    this->nThreads = nThreads;
}

void ParallelizedForLoopProgramMultipleBlocks::run() {

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    parallel_for_loop<<<this->nBlocks, this->nThreads>>>();
    cudaDeviceSynchronize();
}

int main() {

    ParallelizedForLoopProgramMultipleBlocks program(10, 1);
    program.run();
}
