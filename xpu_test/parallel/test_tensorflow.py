"""Tests for horovod.tensorflow.mpi_ops."""

from packaging import version

import itertools
import numpy as np
import os
import platform
import math
import pytest
import sys
import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly
from tensorflow.python.ops import resource_variable_ops
try:
    from tensorflow.python.ops.variables import RefVariable
except ImportError:
    # TF 2.13+
    from tensorflow.python.ops.ref_variable import RefVariable

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import mpi_env_rank_and_size

import horovod.tensorflow as hvd

from base_test_tensorflow import *

_IS_TF2 = version.parse(tf.__version__) >= version.parse('2.0.0')
_is_mac = platform.system() == 'Darwin'


class TensorFlowTests(BaseTensorFlowTests):
    """
    Tests for ops in horovod.tensorflow.
    """
    def __init__(self, *args, **kwargs):
        super(TensorFlowTests, self).__init__(*args, **kwargs)
        hvd.init()
        if hvd.sycl_built():
            tf.config.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    def test_horovod_rank(self):
        """Test that the rank returned by hvd.rank() is correct."""
        mpi_rank, _ = mpi_env_rank_and_size()
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))

        # The mpi rank does not match gloo rank, we need to figure which one
        # we are using to run the test.
        is_mpi = gloo_rank == -1
        rank = hvd.rank()

        if is_mpi:
            assert mpi_rank == rank
        else:
            assert gloo_rank == rank

    def test_horovod_size(self):
        """Test that the size returned by hvd.size() is correct."""
        _, mpi_size = mpi_env_rank_and_size()
        gloo_size = int(os.getenv('HOROVOD_SIZE', -1))

        # The mpi size does not match gloo size, we need to figure which one
        # we are using to run the test.
        is_mpi = gloo_size == -1
        size = hvd.size()
        if is_mpi:
            assert mpi_size == size
        else:
            assert gloo_size == size

    def test_horovod_rank_op(self):
        """Test that the rank returned by hvd.rank_op() is correct."""
        rank = self.evaluate(hvd.rank_op())
        self.assertTrue(rank == hvd.rank(),
                        "hvd.rank_op produces incorrect results")

    def test_horovod_local_rank_op(self):
        """Test that the local rank returned by hvd.local_rank_op() is correct."""
        local_rank = self.evaluate(hvd.local_rank_op())
        self.assertTrue(local_rank == hvd.local_rank(),
                        "hvd.local_rank_op produces incorrect results")

    def test_horovod_size_op(self):
        """Test that the size returned by hvd.size_op() is correct."""
        size = self.evaluate(hvd.size_op())
        self.assertTrue(size == hvd.size(),
                        "hvd.size_op produces incorrect results")

    def test_horovod_local_size_op(self):
        """Test that the local size returned by hvd.local_size_op() is correct."""
        local_size = self.evaluate(hvd.local_size_op())
        self.assertTrue(local_size == hvd.local_size(),
                        "hvd.local_size_op produces incorrect results")

    def test_horovod_allreduce_gpu(self):
        """Test that the allreduce works on GPUs."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_gpu_fused(self):
        """Test that the allreduce works on GPUs with Tensor Fusion.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            test = max_difference <= threshold
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                tests.append(test)
        
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_multi_gpu(self):
        """Test that the allreduce works on multiple GPUs.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        local_rank = hvd.local_rank()
        size = hvd.size()
        local_size = hvd.local_size()

        # Only do this test if there are enough GPUs available.
        if len(tf.config.experimental.list_physical_devices('XPU')) < 2 * local_size:
            self.skipTest("Too few GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        
        iter = 0
        gpu_ids = [local_rank * 2, local_rank * 2 + 1]
        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            iter += 1
            with tf.device("/xpu:%d" % gpu_ids[(iter + local_rank) % 2]):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_gpu_prescale(self):
        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with prescaling"""

        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            return

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            return

        size = hvd.size()
        local_rank = hvd.local_rank()
        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32]
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%s" % local_rank):
                np.random.seed(1234)
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.uint8, tf.int8, tf.float16, tf.bfloat16] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum,
                                       prescale_factor=factor)

                # Scaling done in FP64 math for integer types.
                tensor = tf.cast(tensor, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * tensor, dtype) * size
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))
            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = 1e-3
            elif dtype == tf.bfloat16:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            
            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_gpu_postscale(self):
        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with postscaling"""

        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            return

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            return

        size = hvd.size()
        local_rank = hvd.local_rank()
        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32]
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%s" % local_rank):
                np.random.seed(1234)
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.uint8, tf.int8, tf.float16, tf.bfloat16] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum,
                                       postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types.
                multiplied = tf.cast(multiplied, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * multiplied, dtype)
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = 1e-3
            elif dtype == tf.bfloat16:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_grad_gpu(self):
        """Test the correctness of the allreduce gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.allreduce(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, op=hvd.Sum)

                grad_ys = tf.ones([5] * dim)
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce_gpu(self):
        """Test on GPU that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [tf.cast(self.random_uniform(
                    [17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)
            multiplied = [tensor * size for tensor in tensors]
            differences = [t1 - t2 for t1, t2 in zip(summed, multiplied)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold, "hvd.grouped_allreduce on GPU produces incorrect results")

    def test_horovod_grouped_allreduce_grad_gpu(self):
        """Test the correctness of the grouped allreduce gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                if _executing_eagerly():
                    tensors = [self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)) for _ in range(5)]
                    with tf.GradientTape(persistent=True) as tape:
                        summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)
                else:
                    tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)

                grads_ys = [tf.ones([5] * dim, dtype=dtype) for _ in range(5)]
                if _executing_eagerly():
                    grads_out = [tape.gradient(s, t, g) for s, t, g in zip(summed, tensors, grads_ys)]
                else:
                    grads = [tf.gradients(s, t, g)[0] for s, t, g in zip(summed, tensors, grads_ys)]
                    grads_out = [self.evaluate(grad) for grad in grads]

            expected = np.ones([5] * dim) * size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_average_gpu(self):
        """Test that the allreduce with average works on GPUs."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.bfloat16, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                averaged = hvd.allreduce(tensor, op=hvd.Average)
            # handle int8, uint8 overflows when allreduce sums up and averages the values
            tensor = tf.cast((tensor*size)/size, dtype=dtype)
            difference = tf.cast(averaged, dtype=dtype) - tensor

            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            if size <= 2 and dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")
    
    def test_horovod_allreduce_min_gpu(self):
        """Test on GPU that the allreduce correctly minimizes 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        
        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        rank = hvd.rank()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Min)
            reference = tf.math.reduce_min(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for min")

    def test_horovod_allreduce_max_gpu(self):
        """Test on GPU that the allreduce correctly maximizes 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        rank = hvd.rank()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Max)
            reference = tf.math.reduce_max(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for max")

    def test_horovod_allreduce_product_gpu(self):
        """Test on GPU that the allreduce correctly multiplies 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        rank = hvd.rank()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.bfloat16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Product)
            reference = tf.math.reduce_prod(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16, tf.float32, tf.float64]:
                self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for product")

    def test_horovod_allreduce_average_grad_gpu(self):
        """Test the correctness of the allreduce with average gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        averaged = hvd.allreduce(tensor, op=hvd.Average)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    averaged = hvd.allreduce(tensor, op=hvd.Average)

                grad_ys = tf.ones([5] * dim, dtype=dtype)
                if _executing_eagerly():
                    grad_out = tape.gradient(averaged, tensor, grad_ys)
                else:
                    grad = tf.gradients(averaged, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))


    def test_horovod_allgather_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/xpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)

            gathered_tensor = self.evaluate(gathered)
            self.assertEqual(list(gathered_tensor.shape),
                             [17 * size] + [17] * (dim - 1))

            for i in range(size):
                rank_tensor = tf.slice(gathered_tensor,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                self.assertEqual(list(rank_tensor.shape), [17] * dim)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_fused_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors
        with Tensor Fusion."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        tests = []
        shape_tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/xpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)

            shape_tests.append(
                tf.reduce_all(tf.equal(tf.shape(gathered),
                                       [17 * size] + [17] * (dim - 1))))

            for i in range(size):
                rank_tensor = tf.slice(gathered,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2

                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                tests.append(
                    tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value)))

            shape_tests_passed, value_tests_passed = \
                self.evaluate([tf.reduce_all(shape_tests), tf.reduce_all(tests)])

            self.assertTrue(shape_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")
            
            if size <= 2:
                self.assertTrue(value_tests_passed,
                                "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size_fused_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors with
        Tensor Fusion, even if those tensors have different sizes along the
        first dim."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        tests = []
        shape_tests = []

        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/xpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)
            shape_tests.append(
                tf.reduce_all(tf.equal(tf.shape(gathered),
                             [sum(tensor_sizes)] + [17] * (dim - 1))))

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = tf.slice(
                    gathered, [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                    rank_size)
                self.assertEqual(list(rank_tensor.shape), rank_size)
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2

                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                tests.append(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value)))

            shape_tests_passed, value_tests_passed = \
                self.evaluate([tf.reduce_all(shape_tests), tf.reduce_all(tests)])

            self.assertTrue(shape_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

            if size <= 2:
                self.assertTrue(value_tests_passed,
                                "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/xpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)

            gathered_tensor = self.evaluate(gathered)
            expected_size = sum(tensor_sizes)
            self.assertEqual(list(gathered_tensor.shape),
                             [expected_size] + [17] * (dim - 1))

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = tf.slice(
                    gathered, [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                    rank_size)
                self.assertEqual(list(rank_tensor.shape), rank_size)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                if size <= 2:
                    self.assertTrue(
                        self.evaluate(tf.reduce_all(
                            tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                        "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_grad_gpu(self):
        """Test the correctness of the allgather gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]

            with tf.device("/xpu:%d" % local_rank):
                if _executing_eagerly():
                    with tf.GradientTape() as tape:
                        tensor = self.tfe.Variable(
                            tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank)
                        if dtype == tf.bool:
                            tensor = tensor % 2
                        tensor = tf.cast(tensor, dtype=dtype)
                        gathered = hvd.allgather(tensor)
                        grad_list = []
                        for r, tensor_size in enumerate(tensor_sizes):
                            g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                            grad_list.append(g)
                        grad_ys = tf.concat(grad_list, axis=0)
                    grad_out = tape.gradient(gathered, tensor, grad_ys)
                else:
                    tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
                    if dtype == tf.bool:
                        tensor = tensor % 2
                    tensor = tf.cast(tensor, dtype=dtype)
                    gathered = hvd.allgather(tensor)

                    grad_list = []
                    for r, tensor_size in enumerate(tensor_sizes):
                        g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                        grad_list.append(g)
                    grad_ys = tf.concat(grad_list, axis=0)

                    grad = tf.gradients(gathered, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(
                [tensor_sizes[rank]] + [17] * (dim - 1)
            ) * rank
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" %
                            (grad_out, expected, str(err)))

    def test_horovod_grouped_allgather_gpu(self):
        """Test that the grouped allgather correctly gathers 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensors = [tf.ones([17] * dim) * rank for _ in range(5)]
            if dtype == tf.bool:
                tensors = [tensor % 2 for tensor in tensors]
            tensors = [tf.cast(tensor, dtype=dtype) for tensor in tensors]
            with tf.device("/xpu:%d" % local_rank):
                gathered = hvd.grouped_allgather(tensors)

            gathered_tensors = self.evaluate(gathered)
            for gathered_tensor in gathered_tensors:
                self.assertEqual(list(gathered_tensor.shape),
                                 [17 * size] + [17] * (dim - 1))

            for i in range(size):
                rank_tensors = [tf.slice(gathered_tensor,
                                         [i * 17] + [0] * (dim - 1),
                                         [17] + [-1] * (dim - 1))
                                for gathered_tensor in gathered_tensors]
                self.assertEqual([rank_tensor.shape for rank_tensor in rank_tensors], len(tensors) * [[17] * dim])
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(all(self.evaluate(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value))) for rank_tensor in rank_tensors),
                    "hvd.grouped_allgather produces incorrect gathered tensor")


    def test_horovod_broadcast_gpu(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = tf.ones([17] * dim) * rank
            root_tensor = tf.ones([17] * dim) * root_rank
            if dtype == tf.bool:
                tensor = tensor % 2
                root_tensor = root_tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            root_tensor = tf.cast(root_tensor, dtype=dtype)
            with tf.device("/xpu:%d" % local_rank):
                broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            self.assertTrue(
                self.evaluate(tf.reduce_all(tf.equal(
                    tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                "hvd.broadcast produces incorrect broadcasted tensor")
    
    @pytest.mark.skip(reason="ccl does not support empty input yet.")
    def test_horovod_alltoall_empty_gpu(self):
        """Test that the alltoall correctly deals with an empty input tensor."""
        # ncclGroupEnd failed: invalid usage

        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        for dtype in dtypes:
            with tf.device("/xpu:%s" % local_rank):
                vals = [[] for i in range(size)]
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                collected = hvd.alltoall(tensor)

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), 0)),
                    "hvd.alltoall collected wrong number of values")
    
    
    def test_horovod_broadcast_error(self):
        """Test that the broadcast returns an error if any dimension besides
        the first is different among the tensors being broadcasted."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [17] * 3
        tensor_size[1] = 10 * (rank + 1)
        tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.broadcast(tensor, 0))

    def test_horovod_broadcast_type_error(self):
        """Test that the broadcast returns an error if the types being broadcasted
        differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [17] * 3
        dtype = tf.int32 if rank % 2 == 0 else tf.float32
        tensor = tf.ones(tensor_size, dtype=dtype) * rank
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.broadcast(tensor, 0))

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
        specify different root rank."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor = tf.ones([17] * 3, dtype=tf.float32)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.broadcast(tensor, rank))

    @pytest.mark.skip(reason="stock tensorflow has bug on alltoall int32 Op.")
    def test_horovod_alltoall_gpu(self):
        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                splits = tf.convert_to_tensor([rank+1] * size, dtype=tf.int32)
                collected, received_splits = hvd.alltoall(tensor, splits)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), [rk + 1 for rk in range(size)],
                                         "hvd.alltoall returned incorrect received_splits")

    @pytest.mark.skip(reason="stock tensorflow has bug on alltoall int32 Op.")
    def test_horovod_alltoall_equal_split_gpu(self):
        """Test that the alltoall correctly distributes 1D tensors with default splitting on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                collected = hvd.alltoall(tensor)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

    @pytest.mark.skip(reason="stock tensorflow has bug on alltoall int32 Op.")
    def test_horovod_alltoall_grad_gpu(self):
        """Test the correctness of the alltoall gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    with tf.GradientTape() as tape:
                        collected, received_splits = hvd.alltoall(tensor, splits)
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    collected, received_splits = hvd.alltoall(tensor, splits)

                grad_ys = tf.ones(tf.shape(collected))
                if _executing_eagerly():
                    grad_out = tape.gradient(collected, tensor, grad_ys)
                else:
                    grad = tf.gradients(collected, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(tensor.get_shape().as_list())
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    @pytest.mark.skip(reason="stock tensorflow has bug on alltoall int32 Op.")
    def test_horovod_alltoall_equal_split_grad_gpu(self):
        """Test the correctness of the alltoall gradient with default splitting on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    with tf.GradientTape() as tape:
                        collected = hvd.alltoall(tensor)
                else:
                    collected = hvd.alltoall(tensor)

                grad_ys = tf.ones(tf.shape(collected))
                if _executing_eagerly():
                    grad_out = tape.gradient(collected, tensor, grad_ys)
                else:
                    grad = tf.gradients(collected, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(tensor.get_shape().as_list())
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    
    def test_horovod_reducescatter_gpu(self):
        """Test that the reducescatter works on GPUs."""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=red_op)
            if red_op == hvd.Sum:
                expected = tf.cast(tensor[rank * 4:(rank + 1) * 4] * size, reduced.dtype)
            elif red_op == hvd.Average:
                expected = tf.cast(tensor[rank * 4:(rank + 1) * 4], reduced.dtype)
            max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            diff = self.evaluate(max_difference)
            if size <= 2:
                self.assertTrue(diff <= threshold,
                                "hvd.reducescatter on GPU produces incorrect results")

    def test_horovod_reducescatter_gpu_prescale(self):
        """Test that the reducescatter works on GPUs with prescaling."""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32]
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(123456)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.float16, tf.bfloat16] else 1
                minval = -maxval
                tensor = self.random_uniform([size * 4] * dim, minval, maxval, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=hvd.Sum, prescale_factor=factor)

                # Scaling done in FP64 math for integer types
                tensor = tf.cast(tensor, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                expected = tf.cast(factor * tensor[rank * 4:(rank + 1) * 4], reduced.dtype) * size
                max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = 1e-3
            elif dtype == tf.bfloat16:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            
            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.reducescatter produces incorrect results")        

    def test_horovod_reducescatter_gpu_postscale(self):
        """Test on GPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling."""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32]
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(123456)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.float16, tf.bfloat16] else 1
                minval = -maxval
                tensor = self.random_uniform([size * 4] * dim, minval, maxval, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=hvd.Sum, postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types.
                multiplied = tf.cast(multiplied, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)

                expected = tf.cast(factor * multiplied[rank * 4:(rank + 1) * 4], reduced.dtype)
                max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = 1e-3
            elif dtype == tf.bfloat16:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            
            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu_fused(self):
        """Test that the reducescatter works on GPUs with Tensor Fusion.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            test = max_difference <= threshold
            tests.append(test)

        if size <= 2:
            self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu_uneven(self):
        """Test on GPU that the reducescatter correctly sums and scatters tensors that cannot
           be distributed evenly over the Horovod processes"""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        for dtype in dtypes:
            with tf.device("/xpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4 + size // 2], -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)

            if rank < size // 2:
                low = rank * (4 + 1)
                high = low + (4 + 1)
            else:
                low = (size // 2) * (4 + 1) + (rank - size // 2) * 4
                high = low + 4
            expected = tensor[low:high] * size

            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff, expected_shape, summed_shape = self.evaluate([max_difference, tf.shape(expected),
                                                                tf.shape(summed)])
            self.assertSequenceEqual(expected_shape, summed_shape,
                                     "hvd.reducescatter produces incorrect shapes")
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu_uneven_fused(self):
        """Test on GPU that the reducescatter correctly sums and scatters tensors that cannot
           be distributed evenly over the Horovod processes, with Tensor Fusion"""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        indices = [0, 1, 2, 3]
        tests = []
        infos = []
        for dtype, index in itertools.product(dtypes, indices):
            with tf.device("/xpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4 + size // 2], -100, 100,
                    seed=1234 + index,
                    dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)

            if rank < size // 2:
                low = rank * (4 + 1)
                high = low + (4 + 1)
            else:
                low = (size // 2) * (4 + 1) + (rank - size // 2) * 4
                high = low + 4
            expected = tensor[low:high] * size

            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            test = max_difference <= threshold
            tests.append(test)
            # infos.append({"0_t": tensor, "1_e": expected, "2_s": summed, "3_ok": tf.reduce_all(test)})
        i = self.evaluate([tf.reduce_all(tests)] + infos)
        succesful = i.pop(0)
        # pprint(i)
        if size <= 2:
            self.assertTrue(succesful,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_grad_gpu(self):
        """Test the correctness of the reducescatter gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.reducescatter(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                    summed = hvd.reducescatter(tensor, op=hvd.Sum)

                grad_ys = tf.ones([4] + [size * 4] * (dim - 1))
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([size * 4] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))
    
    def test_horovod_grouped_reducescatter_gpu(self):
        """Test that the grouped reducescatter works on GPUs."""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                           for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=red_op)
            if red_op == hvd.Sum:
                expected = [tf.cast(tensor[rank * 4:(rank + 1) * 4] * size, reduced[0].dtype)
                            for tensor in tensors]
            elif red_op == hvd.Average:
                expected = [tf.cast(tensor[rank * 4:(rank + 1) * 4], reduced[0].dtype)
                            for tensor in tensors]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            diff = self.evaluate(max_difference)
            if size <= 2:
                self.assertTrue(diff <= threshold,
                                "hvd.grouped_reducescatter on GPU produces incorrect results")

    def test_horovod_grouped_reducescatter_gpu_prescale(self):
        """Test on GPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with prescaling."""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32]
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.float16, tf.bfloat16] else 1
                minval = -maxval
                tensors = [self.random_uniform([size * 4] * dim, minval, maxval, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=hvd.Sum, prescale_factor=factor)

                # Scaling done in FP64 math for integer types
                tensors = [tf.cast(t, tf.float64 if dtype in int_types else dtype) for t in tensors]
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)

                expected = [tf.cast(factor * t[rank * 4:(rank + 1) * 4], reduced[0].dtype) * size
                            for t in tensors]
                max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = 1e-3
            elif dtype == tf.bfloat16:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            
            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.grouped_reducescatter produces incorrect results")

    def test_horovod_grouped_reducescatter_gpu_postscale(self):
        """Test on GPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling"""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32]
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.float16, tf.bfloat16] else 1
                minval = -maxval
                tensors = [self.random_uniform([size * 4] * dim, minval, maxval, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=hvd.Sum, postscale_factor=factor)

                multiplied = [t * size for t in tensors]
                # Scaling done in FP64 math for integer types
                multiplied = [tf.cast(t, tf.float64 if dtype in int_types else dtype) for t in multiplied]
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)

                expected = [tf.cast(factor * m[rank * 4:(rank + 1) * 4], reduced[0].dtype)
                            for m in multiplied]
                max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = 1e-3
            elif dtype == tf.bfloat16:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            
            diff = self.evaluate(max_difference)
            if size <= 2 or dtype not in [tf.float16, tf.bfloat16]:
                self.assertTrue(diff <= threshold,
                                "hvd.grouped_reducescatter produces incorrect results")

    def test_compression_fp16(self):
        valid_dtypes = [tf.float16, tf.float32, tf.float64]
        invalid_dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                          tf.int32, tf.int64, tf.bool]

        tensor_size = [17] * 3
        compression = hvd.Compression.fp16

        for dtype in valid_dtypes:
            tensor = tf.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, tf.float16)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            actual = self.evaluate(tensor_decompressed)
            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - actual)
            self.assertLess(err, 0.00000001)

        for dtype in invalid_dtypes:
            tensor = tf.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, dtype)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            actual = self.evaluate(tensor_decompressed)
            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - actual)
            self.assertLess(err, 0.00000001)
    
    def test_horovod_broadcast_eager_mode_error(self):
        """Test that tries to broadcast tensorflow global variables
        in eager execution mode. This call should raise a RuntimeError."""

        if not hvd.util._executing_eagerly():
            self.skipTest("Only in eager execution mode")

        with self.assertRaises(RuntimeError):
            hvd.broadcast_global_variables(root_rank=0)

    def test_horovod_broadcast_graph_mode(self):
        """Test that tries to broadcast tensorflow global variables
        in graph execution mode. This call should not raise any exception."""

        if hvd.util._executing_eagerly():
            self.skipTest("Not in eager execution mode")

        hvd.broadcast_global_variables(root_rank=0)

    def test_horovod_broadcast_inplace_gpu(self):
        """Test that the inplace broadcast correctly broadcasts 1D, 2D, 3D variables on GPU."""
        if version.parse(tf.__version__) < version.parse('2.6.0'):
            self.skipTest("Custom Ops using resource variables only work with TF 2.6+")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # dtypes that are supported both for variable assignments and by Horovod
        dtypes = [tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for use_resource in [False, True]:
            if not use_resource and _executing_eagerly():
                continue
            for counter, (dtype, dim, root_rank) in enumerate(itertools.product(dtypes, dims, root_ranks)):
                with tf.device("/xpu:0"):
                    if dtype == tf.bool:
                        initial_value = tf.cast((tf.ones([17] * dim) * rank) % 2, dtype)
                    else:
                        initial_value = tf.cast(tf.ones([17] * dim) * rank, dtype)
                    root_tensor = tf.ones([17] * dim) * root_rank
                    if dtype == tf.bool:
                        root_tensor = root_tensor % 2
                    if not hvd._executing_eagerly():
                        if use_resource:
                            var = resource_variable_ops.ResourceVariable(initial_value)
                        else:
                            var = RefVariable(initial_value)
                        init = tf.compat.v1.global_variables_initializer()
                        self.evaluate(init)
                    else:
                        assert use_resource
                        var = self.tfe.Variable(initial_value)
                    broadcasted_tensor, = hvd.broadcast_([var], root_rank)
                    self.assertEqual(var.dtype.base_dtype, dtype)
                    self.assertEqual(broadcasted_tensor.dtype.base_dtype, dtype)
                    np.testing.assert_array_equal(self.evaluate(broadcasted_tensor), self.evaluate(var),
                                                  err_msg="broadcasted_var and var may not differ, actually they should have the same underlying buffer")
                    self.assertTrue(
                        self.evaluate(tf.reduce_all(tf.equal(
                            tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                        "Inplace hvd.broadcast_ produces incorrect broadcasted variable value")


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowTests)

if __name__ == '__main__':
    tf.test.main()
