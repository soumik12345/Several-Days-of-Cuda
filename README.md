# Several Days of Cuda

## Instructions

### Instructions for running locally

1. Download and install cuda toolkit for your platform.

2. Check the GPU you're using by `nvidia-smi -L`.

3. Use `run.sh <CUDA_VERSION>` to build and run if you're on Linux. If you are on Windows, clone and open us the directory using Visual Studio 2019 and build using CMake.

### Instructions for running on google colab

1. Run the `Cuda_Workspace.ipynb` Notebook on Google Colab [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/Several-Days-of-Cuda/blob/master/notebooks/Cuda_Workspace.ipynb)

2. Check the GPU you're using by `nvidia-smi -L`.

3. If you wish to ssh into the colab instance from your local VSCode, follow the instructions for `VSCode Remote SSH`. For setting up ssh connection between local VSCode to Google Colab, please follow the instructions this article: [Connect Local VSCode to Google Colab’s GPU Runtime](https://medium.com/swlh/connecting-local-vscode-to-google-colabs-gpu-runtime-bceda3d6cf64).

4. `cd /content/Several-Days-of-Cuda` and use `run.sh <CUDA_VERSION>` to build and run either on the Google Colab Terminal (if you're using Colab Pro) or on the VSCode in-built terminal.

5. In order to create starter code for a kernel use `python create_kernel.py --kernel_name <kernel_name_in_camel_case>`. Then import the respective kernel header and call the demo function in `main.cu`.

![](./assets/sample_execution_example.gif)

## Programs

1. [Hello World](./src/lib/HelloWorld.cuh): Print Hello World in a cuda kernel. 

2. [Accelerated For Loop](./src/lib/BasicExamples/ParallelizedLoop.cuh)

3. [Accelerated For Loop with Multiple ThreadBlocks](./src/lib/BasicExamples/ParallelizedLoopMultipleBlocks.cuh)

4. [Manipulate Array](./src/lib/BasicExamples/ArrayManipulation.cuh): Simple Array Manipulation on GPU, doubling 
   all elements in the array.
