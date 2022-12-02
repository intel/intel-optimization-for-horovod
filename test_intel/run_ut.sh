#!/bin/bash

# Pass the absulute path of test folder
if [ $# -ne 1 ]; then
  echo "./run.pt [test_absolute_path]"
  exit 1
fi

# Save currnet dir 
TEST_DIR=$1
CUR_DIR=`pwd`

# Install test suit package
pip install pytest mock parameterized

# Test tensorflow example: 'mpirun -np 2 python xxx.py'
cd ${TEST_DIR}/examples/tensorflow2 && (ls -1 *.py | xargs -n 1 mpirun -np 2 python)
rm checkpoint*

# Test integration test: 'pytest xxx.py'
cd ${TEST_DIR}/integration && (ls -1 test_*.py | xargs -n 1 pytest)

# Test parallel test: 'mpirun -np 2 pytest xxx.py'
cd ${TEST_DIR}/test_parallel && (ls -1 test_*.py | xargs -n 1 mpirun -np 2 pytest)
mpirun -np 2 python Test_AllReduce_Precision.py

# Return to current directory
cd $TEST_DIR
