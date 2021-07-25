#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void parallel_for_loop() {
    printf("Current Iteration Number: %d\n", threadIdx.x);
}


class ParallelizedForLoopProgram {

public:
    int n;

    ParallelizedForLoopProgram(int n);
    void run();
};

ParallelizedForLoopProgram::ParallelizedForLoopProgram(int n) {
    this->n = n;
}

void ParallelizedForLoopProgram::run() {

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    parallel_for_loop<<<1, this->n>>>();
    cudaDeviceSynchronize();
}

int main() {

    ParallelizedForLoopProgram program(10);
    program.run();
}
