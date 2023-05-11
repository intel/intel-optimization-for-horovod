Intel® Optimization for Horovod* is the distributed training framework for TensorFlow* and PyTorch*. The goal is to make distributed Deep Learning workload run faster and easier to use on Intel GPU devices. It's developed based on latest release version v0.26.1 of public [Horovod](https://github.com/horovod/horovod).

## Install

### Hardware Requirements
 - Intel® Data Center GPU Max Series, Driver Version: [602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html)


### Software Requirement
 - Note: The patched PyTorch 1.13.0a0+git6c9b55e is required to work with Intel® Extension for PyTorch* on Intel® graphics card for now.

|Software|Installation requirement|
|-|-|
|Intel® oneAPI Base Toolkit|[Install Intel® oneAPI Base Toolkit](https://github.com/intel/intel-extension-for-tensorflow/blob/r1.2/docs/install/install_for_gpu.md#install-oneapi-base-toolkit-packages)|
|TensorFlow|[Install tensorflow 2.12.0](https://www.tensorflow.org/install)|
|Intel® Extension for TensorFlow*|[Install Intel® Extension for TensorFlow*](https://github.com/intel/intel-extension-for-tensorflow/tree/r1.2#install) |
|Pytorch|[Install Pytorch 1.13.0a0+git6c9b55e](https://developer.intel.com/ipex-whl-stable-xpu)|
|Intel® Extension for Pytorch*|[Install Intel® Extension for Pytorch*](https://github.com/intel/intel-extension-for-pytorch#installation)|
|System|Ubuntu 22.04, RedHat 8.6 (64-bit), SUSE Linux Enterprise Server(SLES) 15 SP3/SP4|
|Python|3.8-3.10|
|Pip|19.0 or later (requires manylinux2014 support)|

### Install GPU Drivers

|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|
|Ubuntu 22.04, RedHat 8.6, SLES 15 SP3/SP4|Intel® Data Center GPU Max Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html), please append the specific version after components.|


### Installation Channel:
Intel® Optimization for Horovod* can be installed through the following channels:

|PyPI|Source|
|-|-|
|[Install from pip](https://pypi.org/project/intel-optimization-for-horovod) | [Build from source](xpu_docs/how_to_build.md)|



### Install for GPU
Installing Intel® Optimization for Horovod* with different frameworks is feasible. You could choose either Intel® Extension for TensorFlow* or Intel® Extension for Pytorch* as dependency.
 1. Installing Intel® Extension for TensorFlow* and Intel® Optimization for Horovod* with command:
    ```bash
    pip install tensorflow==2.12.0
    pip install --upgrade intel-extension-for-tensorflow[gpu]
    pip install intel-optimization-for-horovod
    ```

 2. Installing Intel® Extension for Pytorch* and Intel® Optimization for Horovod* with command:
    ```bash
    python -m pip install torch==1.13.0a0+git6c9b55e -f https://developer.intel.com/ipex-whl-stable-xpu
    python -m pip install intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
    pip install intel-optimization-for-horovod
    ```


## Running Intel® Optimization for Horovod*

The example commands below show how to run distributed training.
1. To run on a machine with 2 Intel GPUs, which have 4 titles totally.
    ```bash
    horovodrun -np 4 python train.py
    ```

2. To run on 4 machines with 2 GPUs(4 tiles) each:
    ```bash
    horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
    ```

### Running Intel® Optimization for Horovod* with tensorflow on Intel GPU
It is easy to train models with Intel® Extension for TensorFlow. You can refer to [tensorflow examples](xpu_docs/tensorflow_example.md) for more details.
