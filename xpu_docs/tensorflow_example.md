# Tensorflow examples with Intel® Optimization for Horovod

Running Intel® Optimization for Horovod with Intel® Extension for Tensorflow* is similar to normal routine: [official guide](https://github.com/intel-innersource/frameworks.ai.horovod/blob/master/docs/tensorflow.rst). Just need to replace device name from **GPU** to **XPU** while pinning each XPU to a single process. Here is the simple version of training process:

To use Intel® Optimization for Horovod with Intel extension for TensorFlow*, you need to make the following modifications to your training script:

1. Run hvd.init()

2. Pin each XPU to a single process.
   
    With the typical setup of one XPU per process, set this to local rank. The first process on the server will be allocated the first XPU, the second process will be allocated the second XPU, and so forth(Public horovod use **GPU** instead, you need to use **XPU** to run examples on Intel device).
   
    For Tensorflow v2:
   
   ```python3
   gpus = tf.config.experimental.list_physical_devices('XPU')
   for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
   if gpus:
     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'XPU')
   ```

3. Wrap the optimizer in hvd.DistributedOptimizer
   
    For TensorFlow v2, when using a tf.GradientTape, wrap the tape in hvd.DistributedGradientTape instead of wrapping the optimizer.Just like the following example:
   
   ```python3
   tape = hvd.DistributedGradientTape(tape, compression=compression)
   
   gradients = tape.gradient(loss, model.trainable_variables)
   opt.apply_gradients(zip(gradients, model.trainable_variables))
   ```

4. Start training with mpirun.
    Use mpirun command to start training. Just like:
   
   ```
   mpirun -np 2 python tensorflow2_keras_mnist.py
   ```

The following examples are borrowed from public horovod example. We have made necessary changes to pin Intel GPU at the beginning, just as said above. In **xpu_examples/tensorflow2/** folder:
  1. tensorflow2_keras_mnist.py
  2. tensorflow2_keras_synthetic_benchmark.py
  3. tensorflow2_mnist.py
  4. tensorflow2_synthetic_benchmark.py

You could run examples directly with the following command:

```bash
mpirun -np 2 -l python tensorflow2_keras_mnist.py
```
