import argparse
import time
import os

parser = argparse.ArgumentParser(description='TensorFlow Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--compression', type=str, default='all', choices=('none','fp16','bf16','all'),
                    help='use none/fp16/bf16/all compression during allreduce')
parser.add_argument('--dtype', type=str, default='float32',
                    help='tensor type')
parser.add_argument('--xpu', type=str, default='gpu',
                    help='use cpu or gpu for running collective operations')
parser.add_argument('--framework', type=str, default='tf',
                    help='use tf or pt for running with tensorflow or pytorch')
args = parser.parse_args()

# common methods
def check_shape(data, summed):
  if (summed.shape[0] != data.shape[0]):
    raise Exception('unexpected summed tensor shape ', summed.shape[0], ", expected ", data.shape[0])

def gather_time(hvd, N, elem_cnt, ops_sec, start, end, count_):
  if hvd.local_rank() == 0:
    elem_cnt.append(N)
    ops_sec.append(((end - start)/1000)/count_)

def print_time(args, hvd, elem_cnt, ops_sec):
  symb = "." if hvd.local_rank() == 0 else "-"
  print(symb, end=" ", flush=True)
  if hvd.local_rank() == 0:
    print("Device=%s" % args.xpu)
    print("============================")
    for i in range(len(elem_cnt)):
      print("Elements_count=%s         allreduce_elapsed_time=%.2f usec" % (elem_cnt[i], ops_sec[i]))
    print("============DONE============")

# common variables
N = 1
NMAX = 100000
VAL = 2
elem_cnt = []
ops_sec = []
count_ = 1

class tf_runner:
  device = ""
  def __init__(self, args, hvd, tf_obj):
    self.args = args
    self.hvd = hvd
    self.tf_obj = tf_obj

  def get_config(self):
    args = self.args
    tf = self.tf_obj
    hvd = self.hvd

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
    config.graph_options.rewrite_options.disable_meta_optimizer = 1 # disable meta optimizer to avoid const folding.
    if args.xpu == "gpu":
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      tf_runner.device = "/xpu:" + str(hvd.local_rank())
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
      config.gpu_options.allow_growth = False
      config.gpu_options.visible_device_list = ""
      tf_runner.device = "/cpu:0"

  def run(self, N, NMAX, VAL, dtype, elem_cnt, ops_sec, count_, compression):
    tf = self.tf_obj
    hvd = self.hvd

    with tf.device(tf_runner.device):
      while N < NMAX:
        data = []
        for dt in [dtype]:
          for i in range(count_):
            data.append(tf.cast(tf.constant([VAL]*N), dt)) # count_ tensors of size N in data list
          summed = []

          start = time.perf_counter_ns() # counts in ns
          for tensor in data:
            summed.append(hvd.allreduce(tensor, average=False, name="allreduce", compression=compression))
          end = time.perf_counter_ns()

          print("data: ", data[0], ", summed: ", summed[0])
          # checking
          expected = tf.constant([hvd.size()*VAL]*N)
          for i in range(len(data)):
            if (not all(tf.equal(tf.cast(summed[i], tf.int32), expected))):
              raise Exception('unexpected summed tensor ', summed[i], ", expected ", expected)
            check_shape(data[i], summed[i])
          gather_time(hvd, N, elem_cnt, ops_sec, start, end, count_)
        N *= 2
        print_time(args, hvd, elem_cnt, ops_sec)

class pt_runner:
  def __init__(self, hvd, torch):
    self.hvd = hvd
    self.torch = torch

  def run(self, N, NMAX, dtype, elem_cnt, ops_sec, count_):
    hvd = self.hvd
    torch = self.torch

    while N <= NMAX:
      data = []
      for dt in [dtype]:
        for i in range(count_):
          data.append(torch.FloatTensor([VAL] * N).type(dt))
        handles = []
        summed = []
        real_idx = 0

        start = time.perf_counter_ns() # counts in ns
        for idx in range(len(data)):
          real_idx = idx
          if hvd.local_rank() == 1:
            real_idx = len(data) - real_idx - 1
          name = "allreduce_" + str(real_idx)
          handles.append(hvd.allreduce_async(data[idx].to("xpu"), op=hvd.Sum, name=name))
        for h in handles:
          summed.append(hvd.synchronize(h))
        end = time.perf_counter_ns()

        print("data: ", data[0], ", summed: ", summed[0])
        # checking
        expected = torch.FloatTensor([hvd.size()*VAL]*N).type(dtype).to("xpu")
        for i in range(len(data)):
          ret = torch.equal(summed[i], expected)
          if(ret == False):
            raise Exception('unexpected summed tensor ', summed[i], ", expected ", expected)
          check_shape(data[i], summed[i])
        gather_time(hvd, N, elem_cnt, ops_sec, start, end, count_)
      N *= 2
      print_time(args, hvd, elem_cnt, ops_sec)

if args.framework == "tf":
  import tensorflow as tf
  import horovod.tensorflow as hvd
  from tensorflow.python.framework import ops

  hvd.init()
  dtype = tf.float16 if args.dtype == 'float16' else tf.float32

  compression_map = {
    'none':[hvd.Compression.none],
    'fp16':[hvd.Compression.fp16],
    'bf16':[hvd.Compression.bf16],
    'all': [hvd.Compression.none, hvd.Compression.fp16, hvd.Compression.bf16]
  }

  compression_list = compression_map[args.compression]

  tf_obj = tf_runner(args, hvd, tf)
  tf_obj.get_config()
  for compression in compression_list:
    N = 1
    elem_cnt = []
    ops_sec = []
    tf_obj.run(N, NMAX, VAL, dtype, elem_cnt, ops_sec, count_, compression)

if args.framework == "pt":
  import numpy as np
  import timeit

  import torch
  import intel_extension_for_pytorch
  import horovod.torch as hvd

  hvd.init()
  torch.xpu.set_device(hvd.local_rank())
  dtype = torch.FloatTensor

  pt = pt_runner(hvd, torch)
  pt.run(N, NMAX, dtype, elem_cnt, ops_sec, count_)
