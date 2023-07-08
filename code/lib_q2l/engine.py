import torch
import time
import torch
import time
import os
import math
import numpy as np
import _init_paths
import torch.distributed as dist
import torch.nn as nn
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import voc_mAP

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger, warmup_scheduler=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return [param_group][-1]['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    # CMix
    # cmix = CMix(mode='cmix', grids=(4,), n_grids=(0,), mix_prob=1., is_pair=False)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # print(target.shape)
        # images, target, _ = cmix.cmix(images, target)
        # print(target.shape)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            # output, features, prototype = model(images)
            output = model(images)

            loss = criterion['aslloss'](output, target)

            if args.loss_dev > 0:
                loss *= args.loss_dev

        # record loss
        losses.update(loss.item(), images.size(0))

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update learning rate
        if args.lr_scheduler == 'OneCycleLR':
            # print('OneCycleLR')
            scheduler.step()

        lr.update(get_learning_rate(optimizer))
        if epoch >= args.ema_epoch:
            ema_m.update(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

        if i % args.print_freq == 0:
            progress.display(i, logger)

        if epoch < args.warmup_epoch and args.lr_scheduler != 'OneCycleLR':
            # print('warmup_epoch')
            warmup_scheduler.step()

    # adjust learning rate
    if args.lr_scheduler != 'OneCycleLR':
        scheduler.step()

    return losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # to Device
    # prototype_list = []
    # label_list = []
    # num_prototype = 2000
    # to_device = True
    # num_count = 0

    # truelabes = [1,
    #             2, 2, 2, 2, 2, 2, 2, 2,       # Transportation
    #             3, 3, 3, 3, 3,                # outdoor
    #             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, # animal
    #             5, 5, 5, 5, 5,                # accessory
    #             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, # sports
    #             7, 7, 7, 7, 7, 7, 7,          # kitchen
    #             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, # food
    #             9, 9, 9, 9, 9, 9,             # furniture
    #             10, 10, 10, 10, 10, 10,       # electronic
    #             11, 11, 11, 11, 11,           # appliance
    #             12, 12, 12, 12, 12, 12, 12    # indoor
    #             ]


    # switch to evaluate mode
    saveflag = False
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):

                output, vis_prototype = model(images)
                loss = criterion['aslloss'](output, target)
                if args.loss_dev > 0:
                    loss *= args.loss_dev
                output_sm = torch.sigmoid(output)
                if torch.isnan(loss):
                    saveflag = True

            
            # target_nz = target.nonzero()
            # for label_info in target_nz:
            #     batch_id, label = label_info
            #     batch_id, label = int(batch_id), int(label)
            #     if num_count < num_prototype:
            #         prototype_list.append(vis_prototype.data[batch_id][label].cpu().numpy())
            #         # label_list.append(target[batch_id][label].cpu().numpy())
            #         label_list.append(truelabes[label])
            #         num_count += 1
            
            # if num_count == 2000:
            #     prototype_list = np.array(prototype_list)
            #     print(prototype_list.shape)
            #     np.save('/media/data2/maleilei/MLIC/Query2Labels_base/data/TSNE/vis_embed_list_sadcl.npy', prototype_list)
            #     label_list = np.array(label_list)
            #     print(label_list.shape)

            #     np.save('/media/data2/maleilei/MLIC/Query2Labels_base/data/TSNE/label_list_sadcl.npy',label_list)
            #     num_count += 1



            # record loss
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data
            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            # del output_sm
            # del target
            saved_data.append(_item)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)






        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )

        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating mAP:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP                
            mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
            
            logger.info("  mAP: {}".format(mAP))
            if args.out_aps:
                logger.info("  aps: {}".format(np.array2string(aps, precision=5)))
        else:
            mAP = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, mAP

def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()