# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
# Modifications copyright (C) 2019-2022 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Tests for horovod.tensorflow.init._allreduce"""
import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow import test

class AllReducePrecisionTests(test.TestCase):
    def random_uniform(self, *args, **kwargs):
        if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(1234)
            return tf.random.uniform(*args, **kwargs)
        else:
            tf.set_random_seed(1234)
            return tf.random_uniform(*args, **kwargs)

    # verify fp32 by comparing with fp64
    def test_allreduce_gpu_fp32_precision(self):
        np.random.seed(1234)
        prescale_factor = np.random.uniform()
        postscale_factor = np.random.uniform()
        fp64_tensor = self.random_uniform(
            [1000], -1, 1, dtype=tf.dtypes.float64)
        fp64_reduced = hvd.mpi_ops._allreduce(fp64_tensor, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        
        fp32_tensor = tf.cast(fp64_tensor, tf.dtypes.float32)
        fp32_reduced = hvd.mpi_ops._allreduce(fp32_tensor, prescale_factor=prescale_factor, postscale_factor=postscale_factor)

        self.assertAllClose(fp64_reduced, fp32_reduced)

    # verify fp32 by comparing with fp64
    def test_allreduce_gpu_fp32_BatchD2DMemoryCopy_precision(self):
        np.random.seed(1234)
        prescale_factor = np.random.uniform()
        postscale_factor = np.random.uniform()
        fp64_tensors = []
        fp32_tensors = []
        for i in range(10):
            fp64_tensor = self.random_uniform(
            [1000], -1, 1, dtype=tf.dtypes.float64)
            fp64_tensors.append(fp64_tensor)
            fp32_tensor = tf.cast(fp64_tensor, tf.dtypes.float32)
            fp32_tensors.append(fp32_tensor)   

        fp64_reduceds = hvd.mpi_ops._grouped_allreduce(fp64_tensors, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        fp32_reduceds = hvd.mpi_ops._grouped_allreduce(fp32_tensors, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        
        for i in range(10):
            self.assertAllClose(fp64_reduceds[i], fp32_reduceds[i])

    # verify fp16 by comparing with fp32
    def test_allreduce_gpu_fp16_precision(self):
        np.random.seed(1234)
        prescale_factor = np.random.uniform()
        postscale_factor = np.random.uniform()
        fp32_tensor = self.random_uniform(
            [1000], -1, 1, dtype=tf.dtypes.float32)
        fp32_reduced = hvd.mpi_ops._allreduce(fp32_tensor, prescale_factor=prescale_factor, postscale_factor=postscale_factor)

        fp16_tensor = tf.cast(fp32_tensor, tf.dtypes.float16)
        fp16_reduced = hvd.mpi_ops._allreduce(fp16_tensor, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        fp16_reduced = tf.cast(fp16_reduced, tf.dtypes.float32)
        if hvd.size() <= 2:
            self.assertAllClose(fp32_reduced, fp16_reduced, rtol=1e-3, atol=1e-3)

    # verify fp16 by comparing with fp32
    def test_allreduce_gpu_fp16_BatchD2DMemoryCopy_precision(self):
        np.random.seed(1234)
        prescale_factor = np.random.uniform()
        postscale_factor = np.random.uniform()
        fp32_tensors = []
        fp16_tensors = []
        for i in range(10):
            fp32_tensor = self.random_uniform(
            [1000], -1, 1, dtype=tf.dtypes.float32)
            fp32_tensors.append(fp32_tensor)
            fp16_tensor = tf.cast(fp32_tensor, tf.dtypes.float16)
            fp16_tensors.append(fp16_tensor)   

        fp32_reduceds = hvd.mpi_ops._grouped_allreduce(fp32_tensors, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        fp16_reduceds = hvd.mpi_ops._grouped_allreduce(fp16_tensors, prescale_factor=prescale_factor, postscale_factor=postscale_factor)

        if hvd.size() <= 2:
            for i in range(10):
                self.assertAllClose(fp32_reduceds[i], fp16_reduceds[i], rtol=1e-3, atol=1e-3)

    # verify bf16 by comparing with fp32
    def test_allreduce_gpu_bf16_precision(self):
        np.random.seed(1234)
        prescale_factor = np.random.uniform()
        postscale_factor = np.random.uniform()
        fp32_tensor = self.random_uniform(
            [1000], -1, 1, dtype=tf.dtypes.float32)
        fp32_reduced = hvd.mpi_ops._allreduce(fp32_tensor, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        
        bf16_tensor = tf.cast(fp32_tensor, tf.dtypes.bfloat16)
        bf16_reduced = hvd.mpi_ops._allreduce(bf16_tensor, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        bf16_reduced = tf.cast(bf16_reduced, tf.dtypes.float32)

        if hvd.size() <= 2:
            self.assertAllClose(fp32_reduced, bf16_reduced, rtol=1e-2, atol=1e-2)

    # verify bf16 by comparing with fp32
    def test_allreduce_gpu_bf16_BatchD2DMemoryCopy_precision(self):
        np.random.seed(1234)
        prescale_factor = np.random.uniform()
        postscale_factor = np.random.uniform()
        fp32_tensors = []
        bf16_tensors = []
        for i in range(10):
            fp32_tensor = self.random_uniform(
            [1000], -1, 1, dtype=tf.dtypes.float32)
            fp32_tensors.append(fp32_tensor)
            bf16_tensor = tf.cast(fp32_tensor, tf.dtypes.bfloat16)
            bf16_tensors.append(bf16_tensor)   

        fp32_reduceds = hvd.mpi_ops._grouped_allreduce(fp32_tensors, prescale_factor=prescale_factor, postscale_factor=postscale_factor)
        bf16_reduceds = hvd.mpi_ops._grouped_allreduce(bf16_tensors, prescale_factor=prescale_factor, postscale_factor=postscale_factor)

        if hvd.size() <= 2:
            for i in range(10):
                self.assertAllClose(fp32_reduceds[i], bf16_reduceds[i], rtol=1e-2, atol=1e-2)

from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(AllReducePrecisionTests)

if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('XPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    tf.test.main()
