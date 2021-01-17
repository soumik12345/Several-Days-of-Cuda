# Several Days of Cuda

## Instructions for running locally

1. Download and install cuda toolkit for your platform
2. Use `run.sh` to build and run.

## Instructions for running on google colab

1. Run the `Cuda_Workspace.ipynb` Notebook on Google Colab [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/Several-Days-of-Cuda/blob/master/notebooks/Cuda_Workspace.ipynb)

2. Go to the Ngrok link where vscode is hosted. Go easy on vscode for the first few minutes if you're not using Ngrok paid version, otherwise you might end up with an error saying too many API calls in the last minute.

3. Bring up the in-built terminal in vscode and use `run.sh` to build and run.

4. In order to create starter code for a kernel use `python create_component.py --kernel_name <kernel_name_in_camel_case>`.
