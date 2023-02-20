# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2019 Intel Corporation
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
# ==============================================================================

from packaging import version

import inspect
import itertools
import os
import platform
import sys
import unittest
import warnings
import time
import json

from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch
import horovod.torch as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import mpi_env_rank_and_size, skip_or_fail_gpu_test, temppath

_1_12_api = version.parse(torch.__version__) >= version.parse('1.12.0')
_1_5_api = version.parse(torch.__version__) >= version.parse('1.5.0')
_is_mac = platform.system() == 'Darwin'

# Set environment variable for dynamic timeline API test
os.environ["HOROVOD_TIMELINE"] = "DYNAMIC"

# Set environment variable to enable adding/removing process sets after initializing Horovod.
os.environ["HOROVOD_DYNAMIC_PROCESS_SETS"] = "1"

class TorchTests(unittest.TestCase):
    """
    Tests for ops in horovod.torch.
    """

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def setup(self):
        hvd.init()

    def tearDown(self):
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))
        if hvd.is_initialized() and not _is_mac and gloo_rank != -1:
            hvd.barrier()
            hvd.shutdown()

    def convert_cpu_fp16_to_fp32(self, *values):
        # PyTorch doesn't support any CPU ops on FP16 tensors.
        # In case we need to do ops, we will convert tensor to FP32 here.
        result = []
        for value in values:
            if value.dtype in [torch.float16, torch.HalfTensor] and not value.is_xpu:
                result.append(value.float())
            else:
                result.append(value)
        return result

    # TODO: use wa, correct it after PYTORCHDGQ-2379
    def cast_and_place(self, tensor, dtype):
        if torch.xpu.is_available():
            local_rank = "xpu:{}".format(hvd.local_rank())
            # return tensor.xpu(local_rank).type(dtype)
            return tensor.type(dtype).xpu(local_rank)
        return tensor.type(dtype)

    def test_gpu_required(self):
        if not torch.xpu.is_available():
            skip_or_fail_gpu_test(self, "No GPUs available")

    def test_horovod_rank(self):
        """Test that the rank returned by hvd.rank() is correct."""
        mpi_rank, _ = mpi_env_rank_and_size()
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))

        # The mpi rank does not match gloo rank, we need to figure which one
        # we are using to run the test.
        is_mpi = gloo_rank == -1
        hvd.init()
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
        hvd.init()
        size = hvd.size()
        if is_mpi:
            assert mpi_size == size
        else:
            assert gloo_size == size

    def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False)
            tensor, summed = self.convert_cpu_fp16_to_fp32(tensor, summed)
            multiplied = tensor * size

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_average(self):
        """Test that the allreduce correctly averages 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            tensor = self.cast_and_place(tensor, dtype)
            averaged = hvd.allreduce(tensor, average=True)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                averaged, tensor, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_min(self):
        """Test that the allreduce correctly minimizes 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = torch.FloatTensor(size, *([17] * dim)).random_(-100, 100)
            tensors = self.cast_and_place(tensors, dtype)
            tensor = tensors[rank, ...]
            result = hvd.allreduce(tensor, op=hvd.Min)

            reference = tensors.min(0).values

            assert torch.equal(
                result, reference), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_max(self):
        """Test that the allreduce correctly maximizes 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = torch.FloatTensor(size, *([17] * dim)).random_(-100, 100)
            tensors = self.cast_and_place(tensors, dtype)
            tensor = tensors[rank, ...]
            result = hvd.allreduce(tensor, op=hvd.Max)

            reference = tensors.max(0).values
            assert torch.equal(
                result, reference), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_product(self):
        """Test that the allreduce correctly multiplies 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        local_rank = "xpu:{}".format(hvd.local_rank())
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensors = torch.FloatTensor(
                size, *([17] * dim)).random_(minval, maxval)
            tensors = self.cast_and_place(tensors, dtype)
            tensor = tensors[rank, ...]
            result = hvd.allreduce(tensor, op=hvd.Product)

            reference = tensors.prod(0).type(dtype)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                result.to('cpu'), reference, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_inplace(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            multiplied = self.cast_and_place(tensor * size, dtype)
            tensor = self.cast_and_place(tensor, dtype)
            hvd.allreduce_(tensor, average=False)
            tensor, multiplied = self.convert_cpu_fp16_to_fp32(
                tensor, multiplied)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                tensor, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_async_fused(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        tests = []
        is_hvd_poll_false_once = False
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            tensor = self.cast_and_place(tensor, dtype)
            handle = hvd.allreduce_async(tensor, average=False)
            if not hvd.poll(handle):
                is_hvd_poll_false_once = True
            tensor, = self.convert_cpu_fp16_to_fp32(tensor)
            multiplied = tensor * size
            tests.append((dtype, multiplied, handle))

        # Make sure it's an asynchronous operation.
        assert is_hvd_poll_false_once, 'hvd.poll() always returns True, not an async op?'

        for dtype, multiplied, handle in tests:
            summed = hvd.synchronize(handle)
            summed, = self.convert_cpu_fp16_to_fp32(summed)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_multi_gpu(self):
        """Test that the allreduce works on multiple GPUs."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        # Skip the test if there are not enough GPUs.
        if torch.xpu.device_count() < size:
            self.skipTest("Not enough GPUs available")

        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            device = local_rank
            device = "xpu:{}".format(device)
            # tensor = tensor.xpu(device).type(dtype)
            tensor = tensor.type(dtype).xpu(device)
            multiplied = tensor * size
            hvd.allreduce_(tensor, average=False)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            assert torch.allclose(
                tensor, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_prescale(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors with prescaling."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]
        int_types = [torch.IntTensor, torch.LongTensor]
        half_types = [torch.HalfTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            np.random.seed(1234)
            factor = np.random.uniform()
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False,
                                   prescale_factor=factor)

            factor = torch.tensor(factor, dtype=torch.float64)
            local_rank = "xpu:{}".format(hvd.local_rank())
            factor = factor.xpu(local_rank)
            # For integer types, scaling done in FP64
            factor = factor.type(
                torch.float64 if dtype in int_types else dtype)
            tensor = tensor.type(
                torch.float64 if dtype in int_types else dtype)
            multiplied = factor * tensor
            multiplied = multiplied.type(dtype).to(local_rank)
            summed, multiplied = self.convert_cpu_fp16_to_fp32(
                summed, multiplied)
            multiplied *= size

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == torch.HalfTensor:
                threshold = 1e-3
            elif dtype == torch.BFloat16Tensor:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break
            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_postscale(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors with postscaling."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        int_types = [torch.IntTensor, torch.LongTensor]
        half_types = [torch.HalfTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            np.random.seed(1234)
            factor = np.random.uniform()
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensor = torch.FloatTensor(*([17] * dim)).random_(minval, maxval)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False,
                                   postscale_factor=factor)

            factor = torch.tensor(factor, dtype=torch.float64)
            local_rank = "xpu:{}".format(hvd.local_rank())
            factor = factor.xpu(local_rank)
            # For integer types, scaling done in FP64
            factor = factor.type(
                torch.float64 if dtype in int_types else dtype)
            tensor = tensor.type(
                torch.float64 if dtype in int_types else dtype)
            multiplied = size * tensor
            multiplied = multiplied * factor
            multiplied = multiplied.type(dtype).xpu(local_rank)
            summed, multiplied = self.convert_cpu_fp16_to_fp32(
                summed, multiplied)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == torch.HalfTensor:
                threshold = 1e-3
            elif dtype == torch.BFloat16Tensor:
                threshold = 1e-2
            elif size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
        perform reduction on CPU and GPU."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        dims = [17] * 3
        if rank % 2 == 0:
            tensor = torch.xpu.FloatTensor(*dims)
        else:
            tensor = torch.FloatTensor(*dims)

        try:
            hvd.allreduce(tensor)
            assert False, 'hvd.allreduce did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_allreduce_grad(self):
        """Test the correctness of the allreduce gradient."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            summed = hvd.allreduce(tensor, average=False)

            summed.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones([17] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_grad_average(self):
        """Test the correctness of the allreduce averaged gradient."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            summed = hvd.allreduce(tensor, average=True)

            summed.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones([17] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce(self):
        """Test that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(minval, maxval) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            summed = hvd.grouped_allreduce(tensors, average=False)
            tensors, summed = zip(
                *[self.convert_cpu_fp16_to_fp32(t, s) for t, s in zip(tensors, summed)])
            multiplied = [tensor * size for tensor in tensors]

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([torch.allclose(t1, t2, threshold) for t1, t2 in zip(summed, multiplied)]), \
                'hvd.grouped_allreduce produces incorrect results'

    def test_horovod_grouped_allreduce_average(self):
        """Test that the grouped allreduce correctly averages 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(minval, maxval) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            averaged = hvd.grouped_allreduce(tensors, average=True)
            tensors, averaged = zip(
                *[self.convert_cpu_fp16_to_fp32(t, m) for t, m in zip(tensors, averaged)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([torch.allclose(t1, t2, threshold) for t1, t2 in zip(averaged, tensors)]), \
                'hvd.grouped_allreduce produces incorrect results for average'

    def test_horovod_grouped_allreduce_inplace(self):
        """Test that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            multiplied = [self.cast_and_place(
                tensor * size, dtype) for tensor in tensors]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            hvd.grouped_allreduce_(tensors, average=False)
            tensors, multiplied = zip(
                *[self.convert_cpu_fp16_to_fp32(t, m) for t, m in zip(tensors, multiplied)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
                if dtype == torch.HalfTensor:
                    threshold = 1e-3
                elif dtype == torch.BFloat16Tensor:
                    threshold = 1e-2
            else:
                break

            assert all([torch.allclose(t1, t2, threshold) for t1, t2 in zip(tensors, multiplied)]), \
                'hvd.grouped_allreduce_ produces incorrect results'

    def test_horovod_grouped_allreduce_cpu_gpu_error(self):
        """Test that the grouped allreduce raises an error if the input tensor
        list contains a mix of tensors on CPU and GPU."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        tensors = [torch.FloatTensor(
            10) if i % 2 else torch.xpu.FloatTensor(10) for i in range(5)]
        try:
            hvd.grouped_allreduce(tensors, average=False)
            assert False, 'hvd.allreduce did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_grouped_allreduce_grad(self):
        """Test the correctness of the grouped allreduce gradient."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            for tensor in tensors:
                tensor.requires_grad_()
            summed = hvd.grouped_allreduce(tensors, average=False)

            for s in summed:
                s.backward(self.cast_and_place(torch.ones([17] * dim), dtype))

            grads_out = [tensor.grad.data.cpu().numpy() for tensor in tensors]

            expected = np.ones([17] * dim) * size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce_grad_average(self):
        """Test the correctness of the grouped allreduce averaged gradient."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        size = hvd.size()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            maxval = 100
            minval = -100
            if dtype in [torch.HalfTensor, torch.BFloat16Tensor] and size > 2:
                maxval = 1
                minval = 0

            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(minval, maxval) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            for tensor in tensors:
                tensor.requires_grad_()
            summed = hvd.grouped_allreduce(tensors, average=True)

            for s in summed:
                s.backward(self.cast_and_place(torch.ones([17] * dim), dtype))

            grads_out = [tensor.grad.data.cpu().numpy() for tensor in tensors]

            expected = np.ones([17] * dim)
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor]:
                threshold = 1e-8
            elif size < 10:
                threshold = 1e-7
            elif size < 15:
                threshold = 5e-6
            else:
                break

            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, threshold,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(
                *([17] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)
            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            tensor, root_tensor, broadcasted_tensor = \
                self.convert_cpu_fp16_to_fp32(
                    tensor, root_tensor, broadcasted_tensor)
            if rank != root_rank:
                assert (tensor == root_tensor).max() == 0, \
                    'hvd.broadcast modifies source tensor'
            assert (broadcasted_tensor.data == root_tensor).min() == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(
                *([17] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)
            broadcasted_tensor = hvd.broadcast_(tensor, root_rank)
            tensor, root_tensor, broadcasted_tensor = \
                self.convert_cpu_fp16_to_fp32(
                    tensor, root_tensor, broadcasted_tensor)
            assert (tensor == broadcasted_tensor).min() == 1, \
                'hvd.broadcast does not modify source tensor'
            assert (broadcasted_tensor == root_tensor).min() == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_grad(self):
        """Test the correctness of the broadcast gradient."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()

            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            broadcasted_tensor.backward(
                self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            c = 1 if rank == root_rank else 0
            expected = np.ones([17] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall(self):
        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest(
                "NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor,
                  torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
                vals += [i] * (rank + 1)

            tensor = torch.Tensor(vals)
            for _ in range(dim - 1):
                tensor = tensor.unsqueeze(1)
                tensor = torch.cat((tensor, tensor), dim=1)

            splits = torch.tensor([rank + 1] * size, dtype=torch.int32)
            tensor = self.cast_and_place(tensor, dtype)
            collected, received_splits = hvd.alltoall(tensor, splits)
            tensor, collected = self.convert_cpu_fp16_to_fp32(
                tensor, collected)

            assert collected.data.min() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.data.max() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.numel() == size * (size + 1) // 2 * 2**(dim -
                                                                     1), 'hvd.alltoall collected wrong number of values'
            self.assertSequenceEqual(received_splits.tolist(), [rk + 1 for rk in range(size)],
                                     "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_equal_split(self):
        """Test that the alltoall correctly distributes 1D tensors with default splitting."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest(
                "NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor,
                  torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
                vals += [i] * (rank + 1)

            tensor = torch.Tensor(vals)
            for _ in range(dim - 1):
                tensor = tensor.unsqueeze(1)
                tensor = torch.cat((tensor, tensor), dim=1)

            tensor = self.cast_and_place(tensor, dtype)
            collected = hvd.alltoall(tensor)
            tensor, collected = self.convert_cpu_fp16_to_fp32(
                tensor, collected)

            assert collected.data.min() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.data.max() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.numel() == size * (size + 1) // 2 * 2**(dim -
                                                                     1), 'hvd.alltoall collected wrong number of values'

    def test_horovod_alltoall_splits_on_gpu(self):
        """Test that the alltoall works correctly when the splits argument is a tensor on GPU."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest(
                "NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor,
                  torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
                vals += [i] * (rank + 1)

            tensor = torch.Tensor(vals)
            for _ in range(dim - 1):
                tensor = tensor.unsqueeze(1)
                tensor = torch.cat((tensor, tensor), dim=1)

            splits = torch.tensor(
                [rank + 1] * size, dtype=torch.int32, device="cuda")
            tensor = self.cast_and_place(tensor, dtype)
            collected, received_splits = hvd.alltoall(tensor, splits)
            tensor, collected = self.convert_cpu_fp16_to_fp32(
                tensor, collected)

            assert collected.data.min() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.data.max() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.numel() == size * (size + 1) // 2 * 2**(dim -
                                                                     1), 'hvd.alltoall collected wrong number of values'
            self.assertEqual(received_splits.device.type, "cuda",
                             "received_splits should be on GPU here")
            self.assertSequenceEqual(received_splits.tolist(), [rk + 1 for rk in range(size)],
                                     "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_grad(self):
        """Test the correctness of the alltoall gradient."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest(
                "NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
                vals += [i] * (rank + 1)

            tensor = torch.Tensor(vals)
            for _ in range(dim - 1):
                tensor = tensor.unsqueeze(1)
                tensor = torch.cat((tensor, tensor), dim=1)

            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            splits = torch.tensor([rank + 1] * size, dtype=torch.int32)
            collected, received_splits = hvd.alltoall(tensor, splits)

            collected.backward(self.cast_and_place(
                torch.ones(collected.shape), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones(tensor.shape)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall_equal_split_grad(self):
        """Test the correctness of the alltoall gradient with default splitting."""
        # Only do this test if there are GPUs available.
        if not torch.xpu.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest(
                "NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor, torch.BFloat16Tensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
                vals += [i] * (rank + 1)

            tensor = torch.Tensor(vals)
            for _ in range(dim - 1):
                tensor = tensor.unsqueeze(1)
                tensor = torch.cat((tensor, tensor), dim=1)

            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            collected = hvd.alltoall(tensor)

            collected.backward(self.cast_and_place(
                torch.ones(collected.shape), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones(tensor.shape)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))


if __name__ == "__main__":
    unittest.main()

