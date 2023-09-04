import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
#import horovod.torch as hvd
import os
import math
from tqdm import tqdm
import time

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
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--groups', type=int, default=0,
                    help='horovod num_groups for group fusion(default: 0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
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

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--xpu', action='store_true', default=False,
                    help='use Intel GPU for training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
#parser.add_argument('--synthetic', action='store_true', default=False,
parser.add_argument('--synthetic', type=int , default=0,
                    help='Use synthetic data in the amount, 0 is disabled, +1000 uses transforms , -1000 does not.')

parser.add_argument('--checkpoint-interval', type=int , default=1,
                    help=' checkpoint interval in epochs , 0 disables, ')

parser.add_argument('--iter-metric-logging', action='store_true', default=False,
                    help='enables per iteration loss and accuracy metric logging')

parser.add_argument('--metric-allreduce', action='store_true', default=False,
                    help='enable metric allreduce across ranks for logging')

parser.add_argument('--independent', action='store_true', default=False,
                    help="use PyTorch optimzier instead of Horovod's")

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss', metric_allreduce=args.metric_allreduce)
    train_accuracy = Metric('train_accuracy', metric_allreduce=args.metric_allreduce)
    TQDM_UPDATE_INTERVAL = len(train_loader)//5
    with tqdm(total=len(train_loader)*allreduce_batch_size,
              desc='Train Epoch     #{}'.format(epoch + 1),
              unit = " i/r",
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if args.xpu:
                data, target = data.xpu(), target.xpu()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                # this check is there to prevent unnecessary reformating
                # of tensor when batch_size is same as data size - prevents
                # performance  and layout related issues.
                if len(data) == args.batch_size:
                    data_batch = data
                    target_batch = target
                else:
                    data_batch = data[i:i + args.batch_size]
                    target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                #loss = F.cross_entropy(output, target_batch)
                loss = criterion(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                # this check prevents unecessary div.
                if len(data) == args.batch_size:
                    pass
                else:
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            # the following logging adds overhead due to d2h, hence disabled by default
            if args.iter_metric_logging:
                t.set_postfix,({'loss': train_loss.avg.item(),
                               'accuracy': 100. * train_accuracy.avg.item()})
            if batch_idx > 0 and batch_idx%TQDM_UPDATE_INTERVAL == 0:
                t.update(TQDM_UPDATE_INTERVAL*allreduce_batch_size)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

    return batch_idx


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader)*args.val_batch_size,
              desc='Validate Epoch  #{}'.format(epoch + 1),
              unit = "i",
              disable=not verbose) as t:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate (val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                if args.xpu:
                    data, target = data.xpu(), target.xpu()
                output = model(data)

                #val_loss.update(F.cross_entropy(output, target))
                loss = criterion(output, target)
                val_loss.update(loss)
                #val_accuracy.update(accuracy(output, target))
                acc = accuracy(output, target)
                val_accuracy.update(acc)

                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1*args.val_batch_size)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

    return batch_idx


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
    #return pred.eq(target.view_as(pred)).cpu().float().mean()
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
    def __init__(self, name, metric_allreduce=False):
        self.name = name
        self. metric_allreduce= metric_allreduce
        if args.xpu:
            device = 'xpu'
        elif args.cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        self.sum = torch.tensor(0.).to(device)
        self.n = torch.tensor(0.).to(device)

    def update(self, val):
        # disabled by default as this allreduce is only used for logging
        if self.metric_allreduce:
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

import numpy as np
from PIL import Image
class SyntheticDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, transform=None, num_of_batches=100, imgsize=224, batch_size=16, device="xpu"):
        'Initialization'
        self.data =   np.random.randint(0,255, (imgsize, imgsize, 3), 'uint8')
        self.target = 1
        self.num_of_batches = num_of_batches
        self.batch_size = batch_size
        self.transform=transform

    def __len__(self):
        'Denotes the total number of samples'
        return self.batch_size*self.num_of_batches

    def __getitem__(self, index:int):
        'Generates one sample of data'
        time0 = time.time()
        self.img = Image.fromarray(self.data)
        #self.img = torch.from_numpy(self.data)
        if self.transform is not None:
            self.img = self.transform(self.img)
        data_ = self.img
        y = self.target
        X = data_
        #print ( "index : ", index)
        return X, y

class StubHorovodTorch():
    class Compression():
        fp16 = None
        none = None

    def __init__(self, ):
        print ( "Using StubHorovodTorch")

    def init(self,):
        return

    def rank(self,):
        rid = os.getenv("PALS_RANKID", 0)
        return int(rid)

    def local_rank(self,):
        lrid = os.getenv("PALS_LOCAL_RANKID", 0)
        return int(lrid)

    def local_size(self,):
        lsz = os.getenv("PALS_LOCAL_SIZE", 1)
        return int(lsz)

    def size(self,):
        msz = os.getenv("MPI_SIZE", 1)
        return int(msz)

    def broadcast(self, tensor, root_rank, name=None, ignore_name_scope=False, process_set=None):
        return tensor

    def broadcast_parameters(self, params, root_rank=0):
        return

    def broadcast_optimizer_state(self, optimizer, root_rank, model=None):
        return


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available() and not args.xpu

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    if args.xpu:
        # intel extension for pytorch is needed for XPU
        import intel_extension_for_pytorch as ipex

    if args.independent:
        hvd = StubHorovodTorch()
    else:
        import horovod.torch as hvd
        hvd.init()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        print ("using cuda")
    elif args.xpu:
        try:
            torch.xpu.set_device(hvd.local_rank())
            torch.xpu.manual_seed(args.seed)
            print ("using xpu")
        except:
            if verbose: print("unable to load intel_extension_for_pytorch package")
            args.xpu = False
    else:
        print ("using cpu")

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
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'


    if args.synthetic != 0:
        if args.synthetic < 0 :
            if verbose: print("Synthetic data is used without transforms!")
            #train_dataset = datasets.FakeData(-args.synthetic, (3, 224, 224), 1000,
            #                                  transforms.ToTensor())
            train_dataset = SyntheticDataset(transform=transforms.ToTensor(),
                                            num_of_batches=-args.synthetic//args.batch_size)
            #val_dataset = datasets.FakeData(-args.synthetic//20, (3, 224, 224), 1000,
            #                                transforms.ToTensor())
            val_dataset = SyntheticDataset(transform=transforms.ToTensor(),
                                            num_of_batches=-args.synthetic//args.batch_size//20)
        elif  args.synthetic > 0 :
            if verbose: print("Synthetic data is used with transforms!")
            #train_dataset = datasets.FakeData(args.synthetic, (3, 224, 224), 1000,
            train_dataset = SyntheticDataset(
                                 num_of_batches=args.synthetic//args.batch_size,
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
            #val_dataset = datasets.FakeData(args.synthetic//20, (3, 224, 224), 1000,
            val_dataset = SyntheticDataset(
                                 num_of_batches=args.synthetic//args.batch_size//25,
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

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


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
    elif args.xpu:
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
        model, optimizer = ipex.optimize(
            model=model,
            optimizer=optimizer,
            level="O1")
    # Horovod: wrap optimizer with DistributedOptimizer.
    if args.independent:
        pass
    else:
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=args.batches_per_allreduce,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor,
            groups = args.groups)

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

    criterion = torch.nn.CrossEntropyLoss()
    if args.cuda:
        # Move model to GPU.
        criterion.cuda()
    elif args.xpu:
        # Move model to XPU.
        criterion.xpu()

    for epoch in range(resume_from_epoch, args.epochs):
        time0 = time.time()
        train(epoch)
        timet =  time.time()
        validate(epoch)
        timev = time.time()
        if args.checkpoint_interval > 0 and (epoch+1)%args.checkpoint_interval == 0 :
            save_checkpoint(epoch)
        timec = time.time()
        if log_writer:
            timages = len(train_dataset)
            vimages = len(val_dataset)
            train_time = timet - time0 + 0.000001
            val_time = timev - timet + 0.000001
            ckpt_time = timec - timev
            train_ipspr = (timages/train_time)/hvd.size()
            val_ipspr = vimages/val_time/hvd.size()
            print ( f" epoch {epoch} times(s): train {train_time} val {val_time} checkpoint {ckpt_time}" )
            print ( f" epoch {epoch} throughput(i/s/r): train {train_ipspr} val {val_ipspr}" )
            log_writer.add_scalar('time/train(s)', train_time, epoch)
            log_writer.add_scalar('time/val(s)', val_time, epoch)
            log_writer.add_scalar('time/checkpoint(s)', ckpt_time, epoch)
            log_writer.add_scalar('perf/train(ipspr)', train_ipspr, epoch)
            log_writer.add_scalar('perf/val(ipspr)', val_ipspr, epoch)
    if log_writer:
        print ( "completed")
