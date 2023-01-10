import torch
import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import os
import math
from tqdm import tqdm
import time

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# Training settings
parser = argparse.ArgumentParser(
    description='PyTorch ImageNet Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir',
                    default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir',
                    default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir',
                    default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format',
                    default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce',
                    action='store_true',
                    default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce',
                    type=int,
                    default=1,
                    help='number of batches processed locally before '
                    'executing allreduce across workers; it multiplies '
                    'total batch size.')
parser.add_argument('--use-adasum',
                    action='store_true',
                    default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument(
    '--gradient-predivide-factor',
    type=float,
    default=1.0,
    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--hvdgroups', type=int, default=0,
                    help='horovod num_groups for fusion')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size',
                    type=int,
                    default=16,
                    help='input batch size for training')
parser.add_argument('--val-batch-size',
                    type=int,
                    default=16,
                    help='input batch size for validation')
parser.add_argument('--epochs',
                    type=int,
                    default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr',
                    type=float,
                    default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs',
                    type=float,
                    default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005, help='weight decay')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--xpu',
                    action='store_true',
                    default=False,
                    help='use xpu')
parser.add_argument('--nohorovod',
                    action='store_true',
                    default=False,
                    help='no horovod')
parser.add_argument('--nhwc',
                    action='store_true',
                    default=False,
                    help='using channel last tensor layout')
parser.add_argument('--bf16',
                    action='store_true',
                    default=False,
                    help='using bfp16')
parser.add_argument('--warm', type=int, default=10, help='warmup for fps')
parser.add_argument('--iter', type=int, default=None, help='limit iterations')


def _time(args):
    if args.xpu:
        torch.xpu.synchronize(device)
    if args.cuda:
        torch.cuda.synchronize()
    return time.time()


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose, ascii=True) as t:
        t1 = _time(args)
        imgs = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == args.warm:
                t1 = _time(args)
            if batch_idx >= args.warm:
                imgs += args.batch_size
            adjust_learning_rate(epoch, batch_idx)

            if args.nhwc:
                data = data.to(memory_format=torch.channels_last)
            if args.bf16:
                data = data.bfloat16()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if args.xpu:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)  # zero the gradient buffers
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({
                'loss': train_loss.avg.item(),
                'accuracy': 100. * train_accuracy.avg.item(),
                'fps': imgs / (_time(args) - t1)
            })
            t.update(1)
            if args.iter and batch_idx == args.warm + args.iter:
                break

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose, ascii=True) as t:
        t1 = _time(args)
        imgs = 0
        with torch.no_grad():
            for data, target in val_loader:
                imgs += args.val_batch_size
                if args.nhwc:
                    data = data.to(memory_format=torch.channels_last)
                if args.bf16:
                    data = data.bfloat16()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                if args.xpu:
                    data, target = data.to(device), target.to(device)
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({
                    'loss': val_loss.avg.item(),
                    'accuracy': 100. * val_accuracy.avg.item(),
                    'fps': imgs / (_time(args) - t1)
                })
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch *
                                    (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size(
        ) * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    if args.xpu:
        return pred.eq(target.view_as(pred)).to(device).float().mean()
    else:
        return pred.eq(target.view_as(pred)).float().mean()


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
        if args.xpu:
            self.sum = torch.tensor(0.).to(device)
            self.n = torch.tensor(0.).to(device)
        else:
            self.sum = torch.tensor(0.)
            self.n = torch.tensor(0.)

    def update(self, val):
        if args.xpu:
            self.sum += hvd.allreduce(val.detach().to(device), name=self.name)
        else:
            self.sum += hvd.allreduce(val.detach(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and not args.xpu and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    torch.manual_seed(args.seed)

    device = 'cpu'
    if args.xpu:
        import intel_extension_for_pytorch
        torch.xpu.memory_stats()

    if not args.nohorovod:
        import horovod.torch as hvd
        hvd.init()

    if args.cuda:
        import torch.backends.cudnn as cudnn
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    if args.xpu:
        device = "xpu:{}".format(hvd.local_rank())
        torch.xpu.set_device(device)
        torch.xpu.manual_seed(args.seed)

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch),
                                      root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4,
              'pin_memory': True} if args.xpu or args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context')
            and mp._supports_context
            and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=allreduce_batch_size,
                                               sampler=train_sampler,
                                               **kwargs)

    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             sampler=val_sampler,
                                             **kwargs)

    # Set up standard ResNet-50 model.
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size(
    ) if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()
    if args.xpu:
        model.to(device)

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr * lr_scaler),
                          momentum=args.momentum,
                          weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor, num_groups=args.hvdgroups)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    if args.bf16:
        model = model.bfloat16()
    if args.nhwc:
        model = model.to(memory_format=torch.channels_last)
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        if args.iter is not None:
            break
        validate(epoch)
        save_checkpoint(epoch)
    hvd.shutdown()
