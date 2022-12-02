"""Tests for horovod.tensorflow.mpi_ops using multiple process sets.

With TensorFlow 2.9 and MPI the option HOROVOD_DYNAMIC_PROCESS_SETS has been observed to cause significant
slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
tests for multiple process sets in this script that initializes Horovod with static process sets.
"""

from packaging import version

import itertools
import numpy as np
import platform
import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly

import horovod.tensorflow as hvd

from base_test_tensorflow import *

from horovod.runner.common.util.env import get_env_rank_and_size

_IS_TF2 = version.parse(tf.__version__) >= version.parse('2.0.0')
_is_mac = platform.system() == 'Darwin'


class TensorFlowProcessSetsTests(BaseTensorFlowTests):
    """
    Tests for ops in horovod.tensorflow using multiple process sets.
    """
    def __init__(self, *args, **kwargs):
        super(TensorFlowProcessSetsTests, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        """Initializes Horovod with two process sets"""
        _, size = get_env_rank_and_size()

        cls.even_ranks = [rk for rk in range(0, size) if rk % 2 == 0]
        cls.odd_ranks = [rk for rk in range(0, size) if rk % 2 == 1]
        cls.even_set = hvd.ProcessSet(cls.even_ranks)
        cls.odd_set = hvd.ProcessSet(cls.odd_ranks)

        hvd.init(process_sets=[cls.even_set, cls.odd_set])
        if hvd.sycl_built():
            tf.config.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    def tearDown(self):
        """Prevent that one process shuts down Horovod too early"""
        with tf.device("/cpu:0"):
            b = hvd.allreduce(tf.constant([0.]), name="global_barrier_after_test")
            _ = self.evaluate(b)

    def test_horovod_size_op_process_set(self):
        """Test that the size returned by hvd.size_op(process_set_id) is correct."""
        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        size = self.evaluate(hvd.size_op(process_set_id=self.even_set.process_set_id))
        self.assertEqual(size, self.even_set.size(),
                        "hvd.size_op produces incorrect results for a process set")

    def test_horovod_process_set_included_op(self):
        """Test that the result of hvd.process_set_included_op(process_set_id) is correct."""
        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        included = self.evaluate(hvd.process_set_included_op(process_set_id=self.even_set.process_set_id))

        if hvd.rank() in self.even_ranks:
            self.assertEqual(included, 1)
        else:
            self.assertEqual(included, 0)

    def test_horovod_allreduce_gpu_process_sets(self):
        """ Test on GPU that allreduce correctly sums if restricted to non-global process sets"""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        rank = hvd.rank()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                even_rank_tensor = self.random_uniform([17] * dim, -100, 100)
                even_rank_tensor = tf.cast(even_rank_tensor, dtype=dtype)
                odd_rank_tensor = self.random_uniform([17] * dim, -100, 100)
                odd_rank_tensor = tf.cast(odd_rank_tensor, dtype=dtype)
                if rank in self.even_ranks:
                    summed = hvd.allreduce(even_rank_tensor, average=False, process_set=self.even_set)
                    multiplied = even_rank_tensor * len(self.even_ranks)
                if rank in self.odd_ranks:
                    summed = hvd.allreduce(odd_rank_tensor, average=False, process_set=self.odd_set)
                    multiplied = odd_rank_tensor * len(self.odd_ranks)
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            max_process_set_size = max(len(self.even_ranks), len(self.odd_ranks))
            if max_process_set_size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif max_process_set_size < 10:
                threshold = 1e-4
            elif max_process_set_size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_gpu_process_sets(self):
        """Test on GPU that the grouped allreduce correctly sums if restricted to non-global process sets"""
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")
        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        rank = hvd.rank()
        local_rank = hvd.local_rank()

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/xpu:%d" % local_rank):
                even_rank_tensors = [tf.cast(self.random_uniform(
                    [17] * dim, -100, 100), dtype=dtype) for _ in range(5)]
                odd_rank_tensors = [tf.cast(self.random_uniform(
                    [17] * dim, -100, 100), dtype=dtype) for _ in range(5)]
                if rank in self.even_ranks:
                    summed = hvd.grouped_allreduce(even_rank_tensors, average=False, process_set=self.even_set)
                    multiplied = [tensor * len(self.even_ranks) for tensor in even_rank_tensors]
                elif rank in self.odd_ranks:
                    summed = hvd.grouped_allreduce(odd_rank_tensors, average=False, process_set=self.odd_set)
                    multiplied = [tensor * len(self.odd_ranks) for tensor in odd_rank_tensors]
            differences = [t1 - t2 for t1, t2 in zip(summed, multiplied)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            max_process_set_size = max(len(self.even_ranks), len(self.odd_ranks))
            if max_process_set_size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif max_process_set_size < 10:
                threshold = 1e-4
            elif max_process_set_size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")
    
    def test_horovod_broadcast_gpu_process_sets(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on GPU
         if restricted to non-global process sets"""
        # Only do this test if there are GPUs available.
        if not hvd.sycl_built():
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(set_ranks)
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = tf.ones([17] * dim) * rank
            root_tensor = tf.ones([17] * dim) * root_rank
            if dtype == tf.bool:
                tensor = tensor % 2
                root_tensor = root_tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            root_tensor = tf.cast(root_tensor, dtype=dtype)
            with tf.device("/xpu:%d" % local_rank):
                broadcasted_tensor = hvd.broadcast(tensor, root_rank, process_set=this_set)
            self.assertTrue(
                self.evaluate(tf.reduce_all(tf.equal(
                    tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                "hvd.broadcast produces incorrect broadcasted tensor")


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowProcessSetsTests)

if __name__ == '__main__':
    tf.test.main()
