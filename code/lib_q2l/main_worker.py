
import math
import os, sys
import random
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


# from engine import train, validate
from engine import train, validate
from dataset.get_dataset import get_datasets, distributedsampler

import models
import models.loss.aslloss
from models.query2label import build_q2l
from utils.misc import clean_state_dict, init_distributed_and_seed
from utils.util import add_weight_decay, ModelEma, save_checkpoint, kill_process, load_model, WarmUpLR
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter

best_mAP = 0
def main_worker(args, logger):
    global best_mAP

    # build model
    model = build_q2l(args)

    model = model.cuda()


    ema_m = ModelEma(model, args.ema_decay) # 0.9997^641 = 0.82503551031 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # criterion
    criterion = {}
    criterion['aslloss'] = models.loss.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    # args.lr_mult = args.batch_size / 112
    args.lr_mult = 1.0
    logger.info("lr: {}".format(args.lr_mult * args.lr))

    if args.pattern_parameters == 'mutil_lr':
        backbone, others = [], []
        for n, p in model.module.named_parameters():
            if 'backbone' in n and p.requires_grad:
                backbone.append(p)
            elif 'backbone' not in n and p.requires_grad:
                others.append(p)
            else:
                continue
        parameters = [
            {'params': backbone, 'lr': args.lr * args.lrp},
            {'params': others, 'lr': args.lr}
        ]
    elif args.pattern_parameters == 'single_lr':
        parameters = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
    elif args.pattern_parameters == 'add_weight':
        parameters = add_weight_decay(model, args.weight_decay)
    else:
        raise NotImplementedError



    if args.optim == 'AdamW':
        optimizer = getattr(torch.optim, args.optim)(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    # optionally resume from a checkpoint
    if args.resume:
        load_model(args, logger, model)

    # Data loading code
    train_dataset, val_dataset = get_datasets(args)
    # Distributed Sampler
    train_loader, val_loader, train_sampler = distributedsampler(args, train_dataset, val_dataset)

    if args.evaluate:
        _, mAP = validate(val_loader, model, criterion, args, logger)

        logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAPs, losses_ema, mAPs_ema],
        prefix='=> Test Epoch: ')

    if args.lr_scheduler == 'OneCycleLR':
        # one cycle learning rate
        # args.lr_mult = args.batch_size / 112
        # args.lr_mult = args.batch_size / 128
        args.lr_mult = 1.0
        warmup_scheduler = None
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * args.lr_mult, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.2)
    
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.epoch_step, gamma=0.1, last_epoch=-1)
        if args.warmup_scheduler == True:
            warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warmup_epoch)
        else:
            warmup_scheduler = None

    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, verbose=True)
        if args.warmup_scheduler == True:
            warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warmup_epoch)
        else:
            warmup_scheduler = None
    
    else:
        raise Exception('lr scheduler {} not found!'.format(args.lr_scheduler))  


    # global value
    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger, warmup_scheduler)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            # evaluate on validation set
            loss, mAP = validate(val_loader, model, criterion, args, logger)
            loss_ema, mAP_ema = validate(val_loader, ema_m.module, criterion, args, logger)
            losses.update(loss)
            mAPs.update(mAP)
            losses_ema.update(loss_ema)
            mAPs_ema.update(mAP_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            progress.display(epoch, logger)


            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)
                summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)
            
            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)

            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))
            # filename=os.path.join(args.output, 'checkpoint_{:04d}.pth.tar'.format(epoch))

            if math.isnan(loss) or math.isnan(loss_ema):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)


            # early stop
            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if dist.get_rank() == 0 and args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 
                        break

    logger.info("Best mAP {}:".format(best_mAP))

    if summary_writer:
        summary_writer.close()
    
    return 0