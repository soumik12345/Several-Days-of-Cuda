
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Parameters.h"


__global__ void thread_id_demo_kernel() {

	printf("Hello World!!!");
}


class ThreadIdDemo {

public:
	
	ThreadIdDemo(BlockParams, GridParams);
	ThreadIdDemo(BlockParams, ThreadParams);

	~ThreadIdDemo();

	void run();

	BlockParams BlockParameters;
	GridParams GridParameters;
};


ThreadIdDemo::ThreadIdDemo(BlockParams bParams, GridParams gParams) {
	
	BlockParameters = bParams;
	GridParameters = gParams;
}


ThreadIdDemo::ThreadIdDemo(BlockParams bParams, ThreadParams tParams) {

	BlockParameters = bParams;
	GridParameters = {
		tParams.x / bParams.x,
		tParams.y / bParams.y,
		tParams.z / bParams.z,
	};
}


ThreadIdDemo::~ThreadIdDemo() {

	cudaDeviceReset();
}


void ThreadIdDemo::run() {

	dim3 block(BlockParameters.x, BlockParameters.y, BlockParameters.z);
	dim3 grid(GridParameters.x, GridParameters.y, GridParameters.z);

	thread_id_demo_kernel << <grid, block >> > ();
	cudaDeviceSynchronize();
}


inline void DemoThreadIdDemo1() {
	
	BlockParams blockParams = {
		8, 2, 1
	};

	GridParams gridParams = {
		2, 2, 1
	};

	ThreadIdDemo helloWorldProgram = ThreadIdDemo(blockParams, gridParams);
	helloWorldProgram.run();
}


inline void DemoThreadIdDemo2() {

	BlockParams blockParams = {
		8, 2, 1
	};

	ThreadParams threadParams = {
		16, 4, 1
	};
	
	ThreadIdDemo helloWorldProgram = ThreadIdDemo(blockParams, threadParams);
	helloWorldProgram.run();
}

