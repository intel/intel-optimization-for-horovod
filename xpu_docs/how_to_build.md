# Build from Source Code

This guide shows how to build an Intel® Optimization for Horovod* PyPI package from source and install it.



## Prepare

### Install GPU Driver
An Intel GPU driver is needed to build with GPU support. Refer to [Install Intel GPU Driver](../README.md#install-gpu-drivers).



### Install oneAPI Base Toolkit

Refer to [Install oneAPI Base Toolkit Packages](https://github.com/intel/intel-extension-for-tensorflow/blob/r1.1/docs/install/install_for_gpu.md#install-oneapi-base-toolkit-packages).



## Build

Intel® Optimization for Horovod* depends on TensorFlow* or/and Pytorch* to build from source.

```bash
$ pip install tensorflow==2.11.0

$ pip install torch==1.13.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
```



### Download source code

```bash
$ git clone https://github.com/intel/intel-optimization-for-horovod
$ cd intel-optimization-for-horovod
# The repo defaults to the `master` branch. You can also check out a release branch:
$ git checkout <branch_name>

$ git submodule init && git submodule update
```



### Build wheel and install

```bash
$ rm -rf build/

# You could build with single framework support by changing `HOROVOD_WITHOUT_PYTORCH=1` or `HOROVOD_WITHOUT_TENSORFLOW=1`
$ CC=icx CXX=icpx HOROVOD_GPU=DPCPP \
HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITHOUT_GLOO=1 HOROVOD_GPU_OPERATIONS=CCL HOROVOD_WITH_MPI=1 \
python setup.py bdist_wheel

$ pip install dist/*.whl
```



## Runtime

Intel® Optimization for Horovod* depends on Intel® Extension for Tensorflow* or Intel® Extension for Pytorch* at runtime to run on intel GPU.



- To run TensorFlow* applications

```bash
$ pip install --upgrade intel-extension-for-tensorflow[gpu]
```



- To run Pytorch* applications.
To avoid incompatibility issue, please try to build latest Intel® Extension for PyTorch* from source code. Check [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#install-via-compiling-from-source) for detailed information. Source code is available at the [xpu-master branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-master).



You can then follow the steps in [Contributing guide](../CONTRIBUTING.md) to run some tests to confirm it works as expected.

