#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "headers/ThreadIdDemo.cuh"


int main() {

	DemoThreadIdDemo2();

	return 0;
}
