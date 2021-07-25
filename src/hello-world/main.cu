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


__global__ void hello_world_kernel() {

    printf("Hello World!!!\n");
}


class HelloWorld {

public:

    HelloWorld(BlockParams, GridParams);
    HelloWorld(BlockParams, ThreadParams);

    ~HelloWorld();

    void run();

    BlockParams BlockParameters;
    GridParams GridParameters;
};


HelloWorld::HelloWorld(BlockParams bParams, GridParams gParams) {

    BlockParameters = bParams;
    GridParameters = gParams;
}


HelloWorld::HelloWorld(BlockParams bParams, ThreadParams tParams) {

    BlockParameters = bParams;
    GridParameters = {
            tParams.x / bParams.x,
            tParams.y / bParams.y,
            tParams.z / bParams.z,
    };
}


HelloWorld::~HelloWorld() {

    cudaDeviceReset();
}


void HelloWorld::run() {

    dim3 block(BlockParameters.x, BlockParameters.y, BlockParameters.z);
    dim3 grid(GridParameters.x, GridParameters.y, GridParameters.z);

    hello_world_kernel << <grid, block >> > ();
    cudaDeviceSynchronize();
}


inline void DemoHelloWorld1() {

    BlockParams blockParams = {
            8, 2, 1
    };

    GridParams gridParams = {
            2, 2, 1
    };

    HelloWorld helloWorldProgram = HelloWorld(blockParams, gridParams);
    helloWorldProgram.run();
}


inline void DemoHelloWorld2() {

    BlockParams blockParams = {
            8, 2, 1
    };

    ThreadParams threadParams = {
            16, 4, 1
    };

    HelloWorld helloWorldProgram = HelloWorld(blockParams, threadParams);
    helloWorldProgram.run();
}


int main() {
    printf("Demo 1:\n");
    DemoHelloWorld1();
    printf("Demo 2:\n");
    DemoHelloWorld2();
}
