#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "headers/HelloWorld.h"
#include "headers/Parameters.h"


int main() {

	BlockParams blockParams = {
		8, 2, 1
	};

	GridParams gridParams = {
		2, 2, 1
	};

	ThreadParams threadParams = {
		16, 4, 1
	};

	HelloWorld helloWorldProgram = HelloWorld(blockParams, gridParams);
	helloWorldProgram.run();

	// HelloWorld helloWorldProgram = HelloWorld(blockParams, threadParams);
	// helloWorldProgram.run();

	return 0;
}
