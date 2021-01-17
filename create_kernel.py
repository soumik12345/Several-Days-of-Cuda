import click


HEADER_FILE_SOURCE = '''
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Parameters.h"


__global__ void hello_world_kernel() {

	printf("Hello World!!!");
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

'''

def camel_case(string: str):
    temp = string.split('_')
    return ''.join([t.capitalize() for t in temp])


@click.command()
@click.option('--kernel_name', '-k')
def main(kernel_name):
    class_name = camel_case(kernel_name)
    source_code = HEADER_FILE_SOURCE.replace('hello_world_kernel', kernel_name + '_kernel')
    source_code = source_code.replace('HelloWorld', class_name)
    with open('./src/headers/{}.h'.format(class_name), 'w') as out_file:
        out_file.write(source_code)

if __name__ == "__main__":
    main()
