#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void hello_world_kernel() {

	printf("Hello World!!!\n");
}


class HelloWorld {

public:

	static void run();
};


void HelloWorld::run() {

	hello_world_kernel << <1, 1 >> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
}
