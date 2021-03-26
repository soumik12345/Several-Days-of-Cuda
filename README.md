# Several Days of Cuda

## Instructions

### Instructions for running locally

1. Download and install cuda toolkit for your platform

2. Check the GPU you're using by `nvidia-smi -L`.

3. Use `run.sh` to build and run.

### Instructions for running on google colab

1. Run the `Cuda_Workspace.ipynb` Notebook on Google Colab [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/Several-Days-of-Cuda/blob/master/notebooks/Cuda_Workspace.ipynb)

2. Go to the Ngrok link where vscode is hosted. Go easy on vscode for the first few minutes if you're not using Ngrok paid version, otherwise you might end up with an error saying too many API calls in the last minute.

3. Check the GPU you're using by `nvidia-smi -L`.

4. Bring up the in-built terminal in vscode and use `run.sh` to build and run.

5. In order to create starter code for a kernel use `python create_kernel.py --kernel_name <kernel_name_in_camel_case>`. Then import the respective kernel header and call the demo function in `main.cu`.

![](./assets/sample_execution_example.gif)

## Programs

1. [Hello World](./src/headers/HelloWorld.cuh): Print Hello World in a cuda kernel. 

2. [Thread ID Demo](./src/headers/ThreadIdDemo.cuh): Display Thread IDs.

3. [Block ID Demo](./src/headers/BlockIdDemo.cuh): Display Thread IDs, Block IDs and Block Dimensions.

4. [Unique Indices Calculation](./src/headers/UniqueIndexCalculation.cuh): Calculate Thread ID for each element in an array.

5. [Unique Indices Calculation using 2D Grid](./src/headers/UniqueIndexCalculation2D.cuh): Calculate Thread ID for each element in an array using a 2D Grid.

5. [Vector Addition](./src/headers/VectorAddition.cuh): Calculate the sum of 2 vectors.
