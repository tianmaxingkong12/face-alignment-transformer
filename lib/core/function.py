# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
from torch.utils import data

from lib.core.evaluate import get_transformer_coords, compute_nme
from lib.core.inference import get_final_preds_match
from torchvision import transforms
from PIL import Image


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    criterion.train()

    image_size = 256
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time()-end)
        
        # compute output
        outputs = model(input) 
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
            output = outputs[-1]
        elif isinstance(outputs, dict):
            output = outputs
            loss_dict, pred = criterion(outputs, target, target_weight)
            pred *= image_size
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)

            bs = input.size(0)
            # for k, v in loss_dict.items():
            #     add_to_metrics(f'{k}_unscaled', v.item(), bs)
            #     if k in weight_dict:
            #         add_to_metrics(k, (v * weight_dict[k]).item(), bs)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        preds = get_transformer_coords(pred, meta, [256,256])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('heatmap_loss', heatmap_losses.val, global_steps)
                # writer.add_scalar('coordinate_loss', coordinate_losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} nme_count:{}'\
        .format(epoch, batch_time.avg, losses.avg, nme, nme_count)
    logger.info(msg)

def validate(config, val_loader, model, criterion, epoch, writer_dict):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()
    criterion.eval()

    nme_count = 0
    nme_batch_sum = 0
    nme_batch_loss0 = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    image_size = 256

    with torch.no_grad():
        for i, (input, target, target_weight,meta) in enumerate(val_loader):
            # measure data time
            data_time.update(time.time()-end)
            num_images = input.size(0)
            outputs = model(input)
            target = target.cuda(non_blocking=True) 
            target_weight = target_weight.cuda(non_blocking=True)

            output = outputs

            loss_dict, _  = criterion(outputs, target, target_weight)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys() if k in weight_dict)
            
            # preds = get_transformer_coords(pred, meta, [256,256])
            # preds_loss0 = get_transformer_coords(meta['tpts'],meta,[256,256])
            preds, _, _ = get_final_preds_match(config, outputs, meta['center'], meta['scale'], meta['rotate'])
            # NME
            nme_temp = compute_nme(preds, meta)
            # nme_temp_loss0 = compute_nme(preds_loss0, meta)

            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010
            

            nme_batch_sum += np.sum(nme_temp)
            # nme_batch_loss0 += np.sum(nme_temp_loss0)
            nme_count = nme_count + preds.shape[0]

            # for n in range(output.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    nme_loss0 = nme_batch_loss0 / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f} nme_loss0:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate, nme_loss0)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions



# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 20:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
