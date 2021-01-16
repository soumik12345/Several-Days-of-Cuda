#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "headers/HelloWorld.h"


int main() {

	HelloWorld::run();
	return 0;
}
