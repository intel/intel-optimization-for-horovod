Intel® Optimization for Horovod* is the distributed training framework for TensorFlow*. The goal is to make distributed Deep Learning workload run faster and easier to use on Intel GPU devices. It's developed based on latest release version v0.28.1 of public [Horovod](https://github.com/horovod/horovod).

## Install

### Hardware Requirements
 - Intel® Data Center GPU Max Series, Driver Version: [803](https://dgpu-docs.intel.com/releases/LTS_803.63_20240617.html)



|Software|Installation requirement|
|-|-|
|Intel® oneAPI Base Toolkit|[Install Intel® oneAPI Base Toolkit](https://github.com/intel/intel-extension-for-tensorflow/tree/r2.15/docs/install/install_for_xpu.md#install-oneapi-base-toolkit-packages)|
|TensorFlow|[Install tensorflow 2.15.1](https://www.tensorflow.org/install)|
|Intel® Extension for TensorFlow*|[Install Intel® Extension for TensorFlow*](https://github.com/intel/intel-extension-for-tensorflow/tree/r2.15#install) |
|System|Ubuntu 22.04, SUSE Linux Enterprise Server(SLES) 15 SP3/SP4|
|Python|3.9-3.11|
|Pip|19.0 or later (requires manylinux2014 support)|

### Install GPU Drivers

|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|
|Ubuntu 22.04, RedHat 8.6, SLES 15 SP3/SP4|Intel® Data Center GPU Max Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [803](https://dgpu-docs.intel.com/releases/LTS_803.63_20240617.html), please append the specific version after components.|


### Installation Channel:
Intel® Optimization for Horovod* can be installed through the following channels:

|PyPI|Source|
|-|-|
|[Install from pip](https://pypi.org/project/intel-optimization-for-horovod) | [Build from source](xpu_docs/how_to_build.md)|



### Install for GPU
Installing Intel® Optimization for Horovod* with different frameworks is feasible. You could choose Intel® Extension for TensorFlow* as dependency.
 1. Installing Intel® Extension for TensorFlow* and Intel® Optimization for Horovod* with command: <br/>
    ```bash
    pip install tensorflow==2.15.1
    pip install --upgrade intel-extension-for-tensorflow[xpu]
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
It is easy to train models with Intel® Extension for TensorFlow. You can refer to [tensorflow examples](xpu_docs/tensorflow_example.md) for more details
