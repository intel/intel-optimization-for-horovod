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
from tensorflow.python.ops import variables as tf_ops_variables

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
                summed = hvd.allreduce(tensor, average=False)
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
                summed = hvd.allreduce(tensor, average=False)
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
                summed = hvd.allreduce(tensor, average=False)
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
                summed = hvd.allreduce(tensor, average=False,
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
                summed = hvd.allreduce(tensor, average=False,
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
                        summed = hvd.allreduce(tensor, average=False)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)

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
        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [tf.cast(self.random_uniform(
                    [17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                summed = hvd.grouped_allreduce(tensors, average=False)
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
                        summed = hvd.grouped_allreduce(tensors, average=False)
                else:
                    tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    summed = hvd.grouped_allreduce(tensors, average=False)

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
                averaged = hvd.allreduce(tensor, average=True)
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
            self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")
    

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
                        averaged = hvd.allreduce(tensor, average=True)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    averaged = hvd.allreduce(tensor, average=True)

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


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowTests)

if __name__ == '__main__':
    tf.test.main()
