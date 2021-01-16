#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Parameters.h"


__global__ void hello_world_kernel() {

	printf("Hello World!!!\n");
}


class HelloWorld {

public:
	
	HelloWorld(BlockParams, GridParams);
	HelloWorld(BlockParams, ThreadParams);

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


void HelloWorld::run() {

	dim3 block(BlockParameters.x, BlockParameters.y, BlockParameters.z);
	dim3 grid(BlockParameters.x, BlockParameters.y, BlockParameters.z);

	hello_world_kernel << <grid, block >> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
}
