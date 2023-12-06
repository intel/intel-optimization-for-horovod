import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import os
import math
import time
from tqdm import tqdm
try:
    import intel_extension_for_pytorch as ipex
except:
    ipex = None
import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=16,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu for training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--synthetic', action='store_true', default=False,
                    help='Use synthetic data')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--groups', type=int, default=1,
                    help='horovod num_groups for group fusion(default: 1)')
parser.add_argument('--metrics', type=int, default=0,
                    help='display additional metrics in training loop every n iterations')
parser.add_argument('--maxiter', type=int, default=0,
                    help='exit after n training iterations')
parser.add_argument('--baseline-test', action='store_true', default=False,
                    help='Run without Horovod optimizer for baselining test')
parser.add_argument('--no-checkpoint', action='store_true', default=False,
                    help='run without checkpointing')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='write to logger for tensorboard')
parser.add_argument('--cuda-tf32', action='store_true', default=False,
                    help='Use cuda tf32')

def _time(sync=False):
    if sync:
        if args.cuda:
            torch.cuda.synchronize()
        if args.xpu:
            torch.xpu.synchronize()
    return time.time()

# track per iteration throughput
score = 0
def train(epoch):
    global score
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    train_throughput, train_dataloader = [], []

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        end = _time()
        for batch_idx, (data, target) in train_loaders[epoch]:
            adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if args.xpu:
                data, target = data.xpu(), target.xpu()
            if args.metrics:
                # track dataloader + H2D overhead
                train_dataloader.append(_time(True) - end)
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                loss = F.cross_entropy(output, target_batch)
                train_accuracy.update(accuracy(output, target_batch))
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            if args.metrics:
                # track throughput for full iteration including dataloader
                train_throughput.append(len(data)/(_time(True) - end))
                if batch_idx and (batch_idx+1) % args.metrics == 0:
                    avg = lambda x: sum(x) / len(x)
                    score = avg(train_throughput)
                    print(f'rank{hvd.local_rank()} i{batch_idx+1}: loss {train_loss.avg.item():.4f}, accuracy {100. * train_accuracy.avg.item():.4f}, dataloader {avg(train_dataloader):.4f} ({min(train_dataloader):.4f}-{max(train_dataloader):.4f}), throughput {score:.1f} ({min(train_throughput):.1f}-{max(train_throughput):.1f})')
                    train_throughput, train_dataloader = [], []
                end = _time()
            else:
                t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
                t.update(1)
            if args.maxiter and (batch_idx+1) >= args.maxiter: break
    if args.maxiter: train_loaders[epoch] = None

    if args.tensorboard:
        reduce_loss = train_loss.reduce()
        reduce_accuracy = train_accuracy.reduce()
        if log_writer:
            log_writer.add_scalar('train/loss', reduce_loss, epoch)
            log_writer.add_scalar('train/accuracy', reduce_accuracy, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for batch_idx, (data, target) in val_loaders[epoch]:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                if args.xpu:
                    data, target = data.xpu(), target.xpu()
                output = model(data)

                # update metrics with allreduce
                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)
                if args.maxiter and (batch_idx+1) >= args.maxiter: break
    if args.maxiter: val_loaders[epoch] = None

    if args.tensorboard:
        reduce_loss = val_loss.reduce()
        reduce_accuracy = val_accuracy.reduce()
        if log_writer:
            log_writer.add_scalar('val/loss', reduce_loss, epoch)
            log_writer.add_scalar('val/accuracy', reduce_accuracy, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    def reduce(self):
        return hvd.allreduce(self.avg, op=hvd.Average, name=self.name)

    @property
    def avg(self):
        return self.sum / self.n

def model_warmup(model, bs):
    data = torch.randn(bs, 3, 224, 224)
    target = torch.tensor([1] * bs).long()
    if args.xpu:
        data, target = data.xpu(), target.xpu()
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    model.train()
    out = model(data)
    loss = F.cross_entropy(out, target)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(data.detach())

if __name__ == '__main__':
    args = parser.parse_args()
    args.xpu = not args.cpu and ipex and torch.xpu.is_available()
    args.cuda = not args.cpu and torch.cuda.is_available() and not args.xpu

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0
    if verbose:
        print(f'Using {"XPU" if args.xpu else "CUDA" if args.cuda else "CPU"} device.')

    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank() if torch.cuda.device_count() > 1 else 0)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        if not args.cuda_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    if args.xpu:
        torch.xpu.set_device(hvd.local_rank() if torch.xpu.device_count() > 1 else 0)
        torch.xpu.manual_seed(args.seed)

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda or args.xpu else {}
    if 'pin_memory' in kwargs and args.xpu:
        kwargs['pin_memory_device'] = 'xpu'
    if "PYTORCH_WORKER_AFFINITY" in os.environ:
        offsets = os.environ["PYTORCH_WORKER_AFFINITY"].split(",")
        if len(offsets) < hvd.local_size():
            if verbose: print(f"PYTORCH_WORKER_AFFINITY({len(offsets)}) < local ranks({hvd.local_size()})")
        else:
            o_ = int(offsets[hvd.local_rank()])
            kwargs['worker_init_fn'] = lambda id: os.sched_setaffinity(0, range(id+o_,id+o_+1))
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    if args.synthetic:
        if verbose: print("Synthetic data is used!")
        # limit dataset to maxiter
        _size = 1281167 if not args.maxiter else allreduce_batch_size*args.maxiter*hvd.local_size()
        train_dataset = datasets.FakeData(_size, (3, 224, 224), 1000,
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000,
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
    else:
        train_dataset = \
            datasets.ImageFolder(args.train_dir,
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
        val_dataset = \
            datasets.ImageFolder(args.val_dir,
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    train_loaders = [enumerate(train_loader) for _ in range(args.epochs)]

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)
    val_loaders = [enumerate(val_loader)]


    # Set up standard ResNet-50 model.
    model = models.resnet50()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()
    if args.xpu:
        # Move model to XPU.
        model.xpu()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    if args.xpu:
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer)

    # Horovod: wrap optimizer with DistributedOptimizer.
    if not args.baseline_test:
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=args.batches_per_allreduce,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor, groups=args.groups)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model_warmup(model, allreduce_batch_size)
    # Horovod: broadcast parameters & optimizer state.
    if not args.baseline_test:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    t0 = t0_ = time.time()
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        if verbose:
            et = time.time()-t0_
            eti = args.maxiter if args.maxiter else len(train_loader)
            ett = allreduce_batch_size * eti / et
            print(f'epoch {epoch+1}, train {et:.3f} secs, throughput {ett:.1f} imgs/sec/rank, nodes/ranks {int(hvd.size()/hvd.local_size())}x{hvd.local_size()}')
        t0_ = time.time()
    t1 = time.time()
    validate(0)
    t2 = time.time()
    if not args.no_checkpoint:
        save_checkpoint(0)
    t3 = time.time()
    if verbose:
        print(f'TotalTime: {t3-t0:.3f}, train {t1-t0:.3f}, validate {t2-t1:.3f}, checkpoint {t3-t2:.3f}')

    # print training throughput
    if score and args.baseline_test:
        min_score = hvd.allreduce(torch.tensor([score]), op=hvd.Min)[0]
        max_score = hvd.allreduce(torch.tensor([score]), op=hvd.Max)[0]
    if score and verbose:
        if args.baseline_test:
            print(f'Baseline Benchmark Min Score: {min_score:.1f} imgs/sec/rank, nodes/ranks {int(hvd.size()/hvd.local_size())}x{hvd.local_size()}')
            print(f'Baseline Benchmark Max Score: {max_score:.1f} imgs/sec/rank, nodes/ranks {int(hvd.size()/hvd.local_size())}x{hvd.local_size()}')
        else:
            print(f'Benchmark Score: {score:.1f} imgs/sec/rank, nodes/ranks {int(hvd.size()/hvd.local_size())}x{hvd.local_size()}')
