Contributing guidelines
============

## Contributing to Intel® Optimization for Horovod\*

We welcome community contributions to Intel® Optimization for Horovod*. Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to submit a bug or propose a new feature.
    - Log a bug or feedback with an [issue](https://github.com/intel/intel-optimization-for-horovod/issues).
    - Post about your intended feature in an [issue](https://github.com/intel/intel-optimization-for-horovod/issues) for design and implementation approval.
2. You want to implement a bug-fix or feature for an issue.
    - Search for your issue in the [issue list](https://github.com/intel/intel-optimization-for-horovod/issues).
    - Pick an issue and comment that you'd like to work on the bug-fix or feature.

* For bug-fix, please submit a Pull Request to project [github](https://github.com/intel/intel-optimization-for-horovod/pulls).

  Ensure that you can build the product and run all the examples with your patch.
  Submit a [pull request](https://github.com/intel/intel-optimization-for-horovod/pulls).

**See also:** [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## Developing Intel® Optimization for Horovod\*

Please refer to a full set of [instructions](xpu_docs/how_to_build.md) on installing Intel® Optimization for Horovod\* from source.

## Unit testing

Intel® Optimization for Horovod\* provides python unit tests. 

### Python Unit Testing
* Python unit tests are located at `intel-optimization-for-horovod/xpu_test`.

```
xpu_test/
├── parallel     # Parallel unit tests 
├── utils        # Some utils scripts required by unit test
```

* You need to install following packages to run these tests:

```bash
pip install pytest mock parameterized
```

* Running single unit test:

```
mpirun -np 2 pytest <path_to_python_unit_test>
```

* Running all the unit tests:

```
cd intel-optimization-for-horovod/xpu_test/parallel
for ut in $(find test -name "*.py"); do
    mpirun -np 2 pytest $ut
done
```

## Code style guide

Intel® Optimization for Horovod\* follows the same code style guide as public horovod. Refer to [Contributing to horovod](https://github.com/horovod/horovod/blob/master/CONTRIBUTING.md) for more details. 

1. Use [autopep8](https://github.com/hhatto/autopep8) to format the Python code.
2. Use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format C++ code.

