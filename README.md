Intel® Optimization for Horovod* is the distributed training framework for TensorFlow* and PyTorch*. The goal is to make distributed Deep Learning workload run faster and easier to use on Intel GPU devices. It's developed based on latest release version v0.26.1 of public [Horovod](https://github.com/horovod/horovod).

## Install


### Software Requirement

|Package|GPU|Installation|
|-|-|-|
|Intel® oneAPI Base Toolkit|Y|[Install Intel® oneAPI Base Toolkit](#install-oneapi-base-toolkit-packages)|
|TensorFlow|Y|[Install tensorflow 2.11.0](https://www.tensorflow.org/install)|
|Intel® Extension for TensorFlow*|Y|[Install Intel® Extension for TensorFlow*](https://github.com/intel/intel-extension-for-tensorflow#install) |
|Pytorch|Y|[Install Pytorch 1.13.0](https://pytorch.org/get-started/locally/#linux-installation)|
|Intel® Extension for Pytorch*|Y|[Install Intel® Extension for Pytorch*](https://github.com/intel/intel-extension-for-pytorch#installation)|

### Installation Channel:
Intel® Optimization for Horovod* can be installed through the following channels:

|PyPI|Source|
|-|-|
|[Install from pip](https://test.pypi.org/project/intel-optimization-for-horovod/#description) | [Build from source](xpu_docs/how_to_build.md)|


### Install oneAPI Base Toolkit Packages

Need to install components of Intel® oneAPI Base Toolkit:
- Intel® oneAPI DPC++ Compiler
- Intel® oneAPI Threading Building Blocks (oneTBB)
- Intel® oneAPI Math Kernel Library (oneMKL)
- Intel® oneAPI Collective Communications Library (oneCCL)
- Intel® oneAPI MPI Library

Download and install the verified DPC++ compiler and oneMKL in Ubuntu 22.04.

```bash
# (todo: replace latest oneapi basekit link and package name)
$ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2023.0.0.25537_offline.sh
# 6 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, Threading Building Blocks, oneMKL, oneCCL and Intel MPI
$ sudo sh ./l_BaseKit_p_2023.0.0.25537_offline.sh
```

For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.


### Setup environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

A user may install more components than Intel® Optimization for Horovod* needs, and if required, `setvars.sh` can be customized to point to a specific directory by using a [configuration file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html):

```bash
source /opt/intel/oneapi/setvars.sh --config="full/path/to/your/config.txt"
```

### Install for GPU
Installing Intel® Optimization for Horovod* with different frameworks is feasiable. You could choose either Intel® Extension for TensorFlow* or Intel® Extension for Pytorch* as dependency.
 1. Installing Intel® Extension for TensorFlow* and Intel® Optimization for Horovod* with command:
    ```bash
    pip install tensorflow==2.11.0
    pip install --upgrade intel-extension-for-tensorflow[gpu]
    pip install -i https://test.pypi.org/simple/ intel-optimization-for-horovod
    ```

 2. Installing Intel® Extension for Pytorch* and Intel® Optimization for Horovod* with command:
    ```bash
    #(todo: replace new ipex version)
    python -m pip install torch==1.10.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
    python -m pip install intel_extension_for_pytorch==1.10.200+gpu -f https://developer.intel.com/ipex-whl-stable-xpu
    pip install -i https://test.pypi.org/simple/ intel-optimization-for-horovod
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
