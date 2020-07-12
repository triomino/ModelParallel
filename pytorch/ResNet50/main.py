import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

from models import resnet50, ModelParallelResNet50, PipelineParallelResNet50
from data.loader import get_dataloader
from utils import AverageMeter, accuracy, adjust_learning_rate

def parse_args():

    parser = argparse.ArgumentParser('A Parser')
    parser.add_argument('--method', '-m', type=str, default='single_gpu', help='Method to use',
     choices=['single_gpu', 'split', 'pipeline_in_batch', 'pipeline_batch', 'multi_stream', 'multiprocess', 'multiprocess_parallel'])
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch_size')
    parser.add_argument('--no-validate', action='store_true', help='Skip Validation')
    parser.add_argument('--dataset', '-d', type=str, default='imagenette', help='Dataset', choices=['imagenette', 'imagenet', 'random'])
    parser.add_argument('--split-size', '-s', type=int, default=16, help='Split a batch')
    parser.add_argument('--sync', action='store_true', help='Do cuda synchronization to meature time.')

    return parser.parse_args()

model_dict = {
    'split': ModelParallelResNet50, 
    'single_gpu': resnet50,
    'pipeline_in_batch': PipelineParallelResNet50,
    'pipeline_batch': ModelParallelResNet50,
    'multiprocess': resnet50,
}

num_classes = 10

def run(args):
    print('Loading...')
    model_args = {'num_classes': num_classes}
    if args.method == 'pipeline_in_batch':
        model_args['split_size'] = args.split_size
    model = model_dict[args.method](**model_args)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    if args.method == 'single_gpu':
        model.cuda()

    train_loader, val_loader = get_dataloader(batch_size=args.batch_size, 
        no_validate=args.no_validate, dataset=args.dataset)

    time_epoch = AverageMeter()
    forward_epoch = AverageMeter()
    for i in range(args.epoch):
        train_acc = AverageMeter()
        val_acc = AverageMeter()
        train_loss = AverageMeter()
        val_loss = AverageMeter()
        data_time = AverageMeter()
        forward_time = AverageMeter()

        model.train(True)

        if args.method in ['split', 'single_gpu', 'pipeline_in_batch', 'multi_stream']:
            start = time.time()
            batch_start = start
            for index, (images, targets) in enumerate(train_loader):
                forward_start = time.time()
                data_time.update(forward_start - batch_start)

                images = images.to('cuda:0')
                outputs = model(images)
                targets = targets.to(outputs.device)

                loss = loss_fn(outputs, targets)

                if args.sync:
                    torch.cuda.synchronize()
                forward_time.update(time.time() - forward_start)

                train_acc.update(accuracy(outputs, targets)[0], images.size(0))
                train_loss.update(loss.item(), images.size(0))

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if args.sync:
                    torch.cuda.synchronize()
                batch_start = time.time()
        elif args.method == 'pipeline_batch':
            start = time.time()
            batch_start = start
            for index, (images, targets) in enumerate(train_loader):
                data_time.update(time.time() - batch_start)

                images = images.to('cuda:0')
                outputs = model(images)
                targets = targets.to(outputs.device)

                loss = loss_fn(outputs, targets)

                train_acc.update(accuracy(outputs, targets)[0], images.size(0))
                train_loss.update(loss.item(), images.size(0))

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_start = time.time()

        if args.no_validate:
            print("Skip Validation.")
        else:
            model.eval()
            with torch.no_grad():
                for index, (images, targets) in enumerate(val_loader):
                    images = images.cuda()
                    
                    outputs = model(images)
                    targets = targets.to(outputs.device)
                    loss = loss_fn(outputs, targets)

                    val_acc.update(accuracy(outputs, targets)[0], images.size(0))
                    val_loss.update(loss.item(), images.size(0))

        time_epoch.update(time.time() - start)
        forward_epoch.update(forward_time.sum)
        print('Epoch %2d Total Time %0.3fs Forward %0.3f Data %0.3fs' % (i,
             time_epoch.val, forward_time.sum, data_time.sum))
        if not args.no_validate:
            print('Epoch %2d TrainAcc&Loss %0.3f, %0.3f TestAcc&Loss %0.3f, %0.3f' % (i,
             train_acc.avg, train_loss.avg, val_acc.avg, val_loss.avg))
    
    print('Epoch Avg %0.3fs Forward Avg %0.3fs' % (time_epoch.avg, forward_epoch.avg))

def worker(gpu, args):
    args.gpu = gpu

    args.dist_url = 'tcp://127.0.0.1:23451'
    args.dist_backend = 'nccl'
    args.rank = args.gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    print('GPU %d Loading...' % args.gpu)
    model_args = {'num_classes': num_classes}
    if args.method == 'pipeline_in_batch':
        model_args['split_size'] = args.split_size
    model = model_dict[args.method](**model_args)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = args.batch_size // args.ngpu

    loss_fn = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    train_loader, val_loader, train_sampler = get_dataloader(batch_size=args.batch_size, 
        no_validate=args.no_validate, dataset=args.dataset,
        distributed=(args.method =='multiprocess'))

    time_epoch = AverageMeter()
    forward_epoch = AverageMeter()
    for i in range(args.epoch):
        train_acc = AverageMeter()
        val_acc = AverageMeter()
        train_loss = AverageMeter()
        val_loss = AverageMeter()
        data_time = AverageMeter()
        forward_time = AverageMeter()

        model.train()
        # adjust_learning_rate(optimizer, i+1, args)

        train_sampler.set_epoch(i+1)
        start = time.time()
        batch_start = start
        for index, (images, targets) in enumerate(train_loader):
            forward_start = time.time()
            data_time.update(forward_start - batch_start)

            images = images.cuda(args.gpu, non_blocking=True)
            outputs = model(images)
            targets = targets.cuda(args.gpu, non_blocking=True)

            loss = loss_fn(outputs, targets)

            if args.sync:
                torch.cuda.synchronize()
            forward_time.update(time.time() - forward_start)

            train_acc.update(accuracy(outputs, targets)[0], images.size(0))
            train_loss.update(loss.item(), images.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if args.sync:
                torch.cuda.synchronize()
            batch_start = time.time()

        if args.no_validate:
            print("Skip Validation.")
        else:
            model.eval()
            with torch.no_grad():
                for index, (images, targets) in enumerate(val_loader):
                    images = images.cuda()
                    targets = targets.cuda()
                    
                    outputs = model(images)
                    loss = loss_fn(outputs, targets)

                    val_acc.update(accuracy(outputs, targets)[0], images.size(0))
                    val_loss.update(loss.item(), images.size(0))

        time_epoch.update(time.time() - start)
        forward_epoch.update(forward_time.sum)
        print('GPU %d Epoch %2d Total Time %0.3fs Forward %0.3f Data %0.3fs' % (args.gpu, i,
             time_epoch.val, forward_time.sum, data_time.sum))
        if not args.no_validate:
            print('GPU %d Epoch %2d TrainAcc&Loss %0.3f, %0.3f TestAcc&Loss %0.3f,%0.3f' % (args.gpu, i,
             train_acc.avg, train_loss.avg, val_acc.avg, val_loss.avg))
    
    print('GPU %d Epoch Avg %0.3fs Forward Avg %0.3fs' % (args.gpu, time_epoch.avg, forward_epoch.avg))    

def main():
    if not torch.cuda.is_available():
        print('GPU unavailable')
        return
    
    args = parse_args()

    start = time.time()

    args.lr = 0.01
    if args.method == 'multiprocess':
        args.ngpu = 2
        args.world_size = args.ngpu
        mp.spawn(worker, args=(args,), nprocs=args.ngpu)
    else:
        run(args)

    print('Total Time Used: %0.4f' % (time.time() - start))
        
if __name__ == '__main__':
    main()