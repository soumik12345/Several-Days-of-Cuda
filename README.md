# Several Days of Cuda

## Instructions

### Instructions for building locally

1. Download and install cuda toolkit for your platform.

2. Check the GPU you're using by `nvidia-smi -L`.

3. For building on Linux, use `mkdir build && cd build && cmake ../ && make`.

### Instructions for building and running on Google Colab

- In order to build on Google Colab, run the `colab_execution.ipynb` Notebook [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/Several-Days-of-Cuda/blob/refactor/notebooks/colab_execution.ipynb)

- In order to develop on Google Colab:
   
   - Run the `Cuda_Workspace.ipynb` Notebook on Google Colab [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/Several-Days-of-Cuda/blob/master/notebooks/Cuda_Workspace.ipynb)
   
   - Check the GPU you're using by `nvidia-smi -L`.
   
   - If you wish to ssh into the colab instance from your local VSCode, follow the instructions for `VSCode Remote SSH`. For setting up ssh connection between local VSCode to Google Colab, please follow the instructions this article: [Connect Local VSCode to Google Colab’s GPU Runtime](https://medium.com/swlh/connecting-local-vscode-to-google-colabs-gpu-runtime-bceda3d6cf64)

![](./assets/sample_execution_example.gif)

### Instructions for Development

In order to add a new program to the codebase, you can use a simple Python CLI. Simply install the required 
dependencies for the CLI using `python3 -m pip install -r requirements.txt`. Now you can use the CLI to add a new 
example to the codebase. Executing the `create_kernel.py` using the necessary parameters would add a cuda header file 
with some simple starter code. The usage of the CLI is described below:

```
Usage: create_example.py [OPTIONS]

Options:
  -k, --example_name TEXT  Example Name in Camel Case
  --help                   Show this message and exit.
```

## Examples

<table>
   <th>
      <td>Example</td>
      <td>Run Instructions (inside <code>build</code> directory)</td>
   </th>
   <tr>
      <td>1</td>
      <td><a href="src/hello-world">Hello World</a></td>
      <td><code>./src/hello-world/hello_world</code></td>
   </tr>
   <tr>
      <td>2</td>
      <td><a href="src/thread-id-demo">Thread-ID Demo</a></td>
      <td><code>./src/thread-id-demo/thread_id_demo</code></td>
   </tr>
   <tr>
      <td>3</td>
      <td><a href="src/block-dim-demo">Block-Dim Demo</a></td>
      <td><code>./src/block-dim-demo/block_dim_demo</code></td>
   </tr>
   <tr>
      <td>4</td>
      <td><a href="src/unique-index">Unique Index Calculation</a></td>
      <td><code>./src/unique-index/unique_index</code></td>
   </tr>
   <tr>
      <td>5</td>
      <td><a href="src/unique-index-2d">Unique Index Calculation in 2D</a></td>
      <td><code>./src/unique-index-2d/unique_index_2d</code></td>
   </tr>
   <tr>
      <td>6</td>
      <td><a href="src/parallel-loop">Parallelized Loops</a></td>
      <td><code>./src/parallel-loop/parallel_loop</code></td>
   </tr>
   <tr>
      <td>7</td>
      <td><a href="src/parallel-loop-multi-blocks">Parallelized Loops in Multiple Blocks</a></td>
      <td><code>./src/parallel-loop-multi-blocks/parallel_loop_multi_blocks</code></td>
   </tr>
   <tr>
      <td>8</td>
      <td><a href="src/array-manipulation">Array Manipulation</a></td>
      <td><code>./src/array-manipulation/array_manipulation</code></td>
   </tr>
   <tr>
      <td>9</td>
      <td><a href="src/array-manipulation-manual-memory">Array Manipulation with Manual Memory Allocation</a></td>
      <td><code>./src/array-manipulation-manual-memory/array_manipulation_manual_memory</code></td>
   </tr>
   <tr>
      <td>10</td>
      <td><a href="src/vector-addition">Vector Addition</a></td>
      <td><code>./src/vector-addition/vector_addition</code></td>
   </tr>
   <tr>
      <td>11</td>
      <td><a href="src/vector-addition-unified-memory">Vector Addition with Unified Memory</a></td>
      <td><code>./src/vector-addition-unified-memory/vector_addition_unified_memory</code></td>
   </tr>
</table>
