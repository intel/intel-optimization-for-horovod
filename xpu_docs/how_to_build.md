# Build from Source Code

This guide shows how to build an Intel® Optimization for Horovod* PyPI package from source and install it.

## Prepare

### Install GPU Driver
An Intel GPU driver is needed to build with GPU support. Refer to [Install Intel GPU Driver](../README.md#install-gpu-drivers)


### Install Intel® Extension for Tensorflow*

```bash
$ pip install tensorflow==2.11.0
$ pip install --upgrade intel-extension-for-tensorflow[gpu]
```

### Intel® Extension for PyTorch*
```bash
$ python -m pip install torch==1.13.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
$ python -m pip install intel_extension_for_pytorch==1.13.10+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

### Install oneAPI Base Toolkit

Refer to [Install oneAPI Base Toolkit Packages](https://github.com/intel/intel-extension-for-tensorflow/blob/r1.1/docs/install/install_for_gpu.md#install-oneapi-base-toolkit-packages)

### Install CMake and pybind11

To build Intel® Optimization for Horovod*, install CMake and pybind11(Only for Pytorch dependency).
```
apt install cmake
pip install pybind11 # Only for Pytorch
```

### Download the Intel® Optimization for Horovod* source code
```bash
$ git clone https://github.com/intel/intel-optimization-for-horovod
$ cd intel-optimization-for-horovod
```

Change to release branch (Optional):

The repo defaults to the `master` development branch. You can also check out a release branch to build:

```bash
$ git checkout branch_name
```

Pull submodule
```bash
$ git submodule init && git submodule update
```

Build wheel and install
```bash
$ rm -rf build/

# You could build with it with single framework support by changing `HOROVOD_WITHOUT_PYTORCH=1` or `HOROVOD_WITHOUT_TENSORFLOW=1`
$ CC=icx CXX=icpx HOROVOD_GPU=DPCPP \
HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITHOUT_GLOO=1 HOROVOD_GPU_OPERATIONS=CCL HOROVOD_WITH_MPI=1 \
python setup.py bdist_wheel

$ pip install dist/*.whl
```

## Uninstall
```
$ pip uninstall intel_optimization_for_horovod
```
