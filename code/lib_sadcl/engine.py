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
from utils.misc import concat_all_gather

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger, warmup_scheduler=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    losses_bce = AverageMeter('Loss_bce', ':5.3f')
    losses_contras = AverageMeter('loss_contras', ':5.3f')
    losses_contras_s2s = AverageMeter('losses_contras_s2s', ':5.3f')
    losses_contras_p2s = AverageMeter('losses_contras_p2s', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, losses_bce, losses_contras_s2s, losses_contras_p2s ,mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return [param_group][-1]['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    # # for resnet101
    # if hasattr(model, 'module'):
    #     model.module.backbone[0].body.conv1.eval()
    #     model.module.backbone[0].body.bn1.eval()
    #     model.module.backbone[0].body.relu.eval()
    #     model.module.backbone[0].body.maxpool.eval()

    # else:
    #     model.backbone[0].body.conv1.eval()
    #     model.backbone[0].body.bn1.eval()
    #     model.backbone[0].body.relu.eval()
    #     model.backbone[0].body.maxpool.eval()

    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            output, embedding, vis_embed, memorybank, prototype = model(images)

            features_vis_memorybank = torch.cat([vis_embed.unsqueeze(1), memorybank.unsqueeze(1)], dim=1)
            
            bce_loss = criterion['aslloss'](output, target)
            contras_loss1 = 100*criterion['contrasloss'](features_vis_memorybank, target)
                
            if math.isnan(contras_loss1):
                import sys
                logger.info('contras_loss1 is NaN, break')

            contras_loss2 =  50*criterion['sem2pro'](vis_embed, prototype, target)
                     
            if math.isnan(contras_loss2):
                import sys
                logger.info('contras_loss2 is NaN, break')
            
            contras_loss = contras_loss1 + contras_loss2
            loss = bce_loss + contras_loss

            if contras_loss2 < 0:
                # output_gather = concat_all_gather(output)
                # target_gather    = concat_all_gather(target)
                # vis_embed_gather = concat_all_gather(vis_embed)
                # prototype_gather = concat_all_gather(prototype)
                debug = '/media/data2/maleilei/MLIC/Query2Labels_base/debug2/'
                np.save(debug+'output.{}.npy'.format(dist.get_rank()), output.cpu().detach().numpy())
                np.save(debug+'target.{}.npy'.format(dist.get_rank()), target.cpu().detach().numpy())
                np.save(debug+'vis_embed.{}.npy'.format(dist.get_rank()), vis_embed.cpu().detach().numpy())
                np.save(debug+'prototype.{}.npy'.format(dist.get_rank()), prototype.cpu().detach().numpy())


            if math.isnan(bce_loss):
                import sys
                # output, embedding, vis_embed, memorybank
                debug = '/media/data2/maleilei/MLIC/Query2Labels_base/debug/'
                np.save(debug+'output.{}.npy'.format(dist.get_rank()), output.cpu().detach().numpy())
                np.save(debug+'target.{}.npy'.format(dist.get_rank()), target.cpu().detach().numpy())
                np.save(debug+'embedding.{}.npy'.format(dist.get_rank()), embedding.cpu().detach().numpy())
                np.save(debug+'vis_embed.{}.npy'.format(dist.get_rank()), vis_embed.cpu().detach().numpy())
                np.save(debug+'memorybank.{}.npy'.format(dist.get_rank()), memorybank.cpu().detach().numpy())
                np.save(debug+'prototype.{}.npy'.format(dist.get_rank()), prototype.cpu().detach().numpy())

                logger.info('bce_loss is NaN, break')
                sys.exit(1)
            
            if math.isnan(contras_loss):
                import sys
                if math.isnan(contras_loss1):
                    logger.info('contras_loss1 is NaN, break')

                if math.isnan(contras_loss2):
                    logger.info('contras_loss2 is NaN, break')
                # output, embedding, vis_embed, prototype
                debug = '/media/data2/maleilei/MLIC/Query2Labels_base/debug/'
                np.save(debug+'output.{}.npy'.format(dist.get_rank()), output.cpu().detach().numpy())
                np.save(debug+'target.{}.npy'.format(dist.get_rank()), target.cpu().detach().numpy())
                np.save(debug+'embedding.{}.npy'.format(dist.get_rank()), embedding.cpu().detach().numpy())
                np.save(debug+'vis_embed.{}.npy'.format(dist.get_rank()), vis_embed.cpu().detach().numpy())
                np.save(debug+'memorybank.{}.npy'.format(dist.get_rank()), memorybank.cpu().detach().numpy())
                np.save(debug+'prototype.{}.npy'.format(dist.get_rank()), prototype.cpu().detach().numpy())
                logger.info('contras_loss is NaN, break')
                sys.exit(1)


            if args.loss_dev > 0:
                loss *= args.loss_dev

            # update prototype
            if hasattr(model, 'module'):
                model.module.memorybank_update(embedding, target, output)
            else:
                model.memorybank_update(embedding, target, output)

        # record loss
        losses.update(loss.item(), images.size(0))
        losses_bce.update(bce_loss.item(), images.size(0))
        losses_contras_s2s.update(contras_loss1.item(), images.size(0))
        losses_contras_p2s.update(contras_loss2.item(), images.size(0))
        losses_contras.update(contras_loss.item(), images.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        # if hasattr(model, 'module'):
        #     torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=args.max_clip_grad_norm)
        # else:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_clip_grad_norm)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update prototype
        if args.lr_scheduler == 'OneCycleLR':
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
            warmup_scheduler.step()

    # adjust learning rate
    if args.lr_scheduler != 'OneCycleLR':
        scheduler.step()

    return losses.avg, losses_bce.avg, losses_contras.avg

@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    losses_bce = AverageMeter('Loss_bce', ':5.3f')
    losses_contras = AverageMeter('loss_contras', ':5.3f')
    losses_contras_s2s = AverageMeter('losses_contras_s2s', ':5.3f')
    losses_contras_p2s = AverageMeter('losses_contras_p2s', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, losses_bce, losses_contras, losses_contras_s2s,losses_contras_p2s, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
        # for i, (images, target, iamge_name) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                # output, embedding, vis_embed, memorybank, prototype = model(images) 
                output, _, vis_embed, memorybank, prototype = model(images)

                if vis_embed.size() == memorybank.size():
                    # output_gather    = concat_all_gather(output)
                    # target_gather    = concat_all_gather(target)
                    # vis_embed_gather = concat_all_gather(vis_embed)
                    # prototype_gather = concat_all_gather(prototype)
                    
                    features_vis_memorybank = torch.cat([vis_embed.unsqueeze(1), memorybank.unsqueeze(1)], dim=1)
                    
                    bce_loss = criterion['aslloss'](output, target)
                    contras_loss1 = 100*criterion['contrasloss'](features_vis_memorybank, target)
                    contras_loss2 = 50*criterion['sem2pro'](vis_embed, prototype, target)
                    contras_loss = contras_loss1 + contras_loss2
                    loss = bce_loss + contras_loss
                else:
                    bce_loss = criterion['aslloss'](output, target)
                    loss = bce_loss

                if args.loss_dev > 0:
                    loss *= args.loss_dev
                
                output_sm = torch.sigmoid(output)
                if torch.isnan(loss):
                    saveflag = True
            

            # test_for_voc2012(output, iamge_name, args)

            # record loss
            losses.update(loss.item(), images.size(0))
            losses_bce.update(bce_loss.item(), images.size(0))
            if vis_embed.size() == memorybank.size():
                losses_contras.update(contras_loss.item(), images.size(0))
                losses_contras_s2s.update(contras_loss1.item(), images.size(0))
                losses_contras_p2s.update(contras_loss2.item(), images.size(0))
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
        loss_bce_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses_bce]
        )
        loss_contras_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses_contras]
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

    return loss_avg, loss_bce_avg, loss_contras_avg, mAP


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()



def test_for_voc2012(output, iamge_name, args):
    if not os.path.exists(os.path.join(args.output, 'results', 'VOC2012', 'Main')):
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Main'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Layout'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Action'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Segmentation'), exist_ok=True)

    temp = output.cpu().data.numpy()
    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

    # temp = output_sm.cpu().data.numpy()
    temp = output.cpu().data.numpy()
    for row in range(temp.shape[0]):
        for col in range(temp.shape[1]):
            name = iamge_name[row]
            val = temp[row, col]
            temp_categories_file = os.path.join(args.output, 'results', 'VOC2012', 'Main', 'comp1_cls_test_' + object_categories[col] + '.txt')
            with open(temp_categories_file, 'a') as file:
                file.write(name + ' ' + str(val) + '\n')



            # output=>predict, embedding=>update memorybank, 
            # vis_embed and memorybank=>semple ontrastive learning, 
            # prototype=> prototype contrastive learning 
            # 
            # return out, hs[-1].clone().detach(), vis_embed, memorybank, prototype

            # output_gather = concat_all_gather(output)
            # target_gather    = concat_all_gather(target)
            # vis_embed_gather = concat_all_gather(vis_embed)
            # prototype_gather = concat_all_gather(prototype)
            # debug = '/media/data2/maleilei/MLIC/SADCL/debug2/'
            # np.save(debug+'output.{}.npy'.format(dist.get_rank()), output.cpu().detach().numpy())
            # np.save(debug+'target.{}.npy'.format(dist.get_rank()), target.cpu().detach().numpy())
            # np.save(debug+'vis_embed.{}.npy'.format(dist.get_rank()), vis_embed.cpu().detach().numpy())
            # np.save(debug+'prototype.{}.npy'.format(dist.get_rank()), prototype.cpu().detach().numpy())