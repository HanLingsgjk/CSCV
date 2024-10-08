# coding=utf-8
from __future__ import print_function, division
import sys

sys.path.append('core')
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from core.scale_flowpp import ResScale_AB
import dc_flow_eval as evaluate
import core.dataset_exp_orin as datasets
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 50
VAL_FREQ = 2000

def sequence_loss(flow_preds, dc_preds, flow_gt, dc_gt, valid, gamma=0.8, use_conf=0, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    dc_loss = 0.0

    #
    gt_dc = dc_gt[:, 0:1, :, :]
    mask_dc = (dc_gt[:, 1:, :, :].type(torch.bool))
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    validu = torch.cat([valid[:, None], valid[:, None]], dim=1)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        d_loss = (dc_preds[i] - gt_dc).abs()
        if use_conf:
            epes = torch.exp(-0.05 * torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1))[:, None, :, :]
            d_loss = d_loss * (epes.detach())

        flow_loss += i_weight * (i_loss[validu]).mean()
        dc_loss += i_weight * (d_loss[mask_dc]).mean()
    if len(flow_preds)<len(dc_preds):
        d_loss = (dc_preds[-1] - gt_dc).abs()
        dc_loss += 1 * (d_loss[mask_dc]).mean()


    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    epe1 = torch.sum((flow_preds[-2] - flow_gt) ** 2, dim=1).sqrt()
    epe1 = epe1.view(-1)[valid.view(-1)]

    epe2 = torch.sum((flow_preds[-3] - flow_gt) ** 2, dim=1).sqrt()
    epe2 = epe2.view(-1)[valid.view(-1)]


    ede = ((dc_preds[-1] - gt_dc) ** 2).sqrt()
    ede = ede[mask_dc]

    ede2 = ((dc_preds[-2] - gt_dc) ** 2).sqrt()
    ede2 = ede2[mask_dc]

    ede3 = ((dc_preds[-3] - gt_dc) ** 2).sqrt()
    ede3 = ede3[mask_dc]

    metrics = {
        '6epe': ede.mean().item()* 10,
        '7ede': ede2.mean().item() * 10,
        '8ede2': ede3.mean().item() * 10,

        '4epe': epe.mean().item(),
        '5ede': epe1.mean().item(),
        '6ede2': epe2.mean().item(),

        '1px': (epe < 1).float().mean().item(),
        '2px': (epe < 3).float().mean().item(),
        '3px': (epe < 5).float().mean().item(),
    }

    return flow_loss, dc_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.last_time = time.time()
        self.name = name

    def _print_training_status(self):
        now_time = time.time()
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        time_str = ("time = %.2f, " % (now_time - self.last_time))
        nameall = "/home/lh/CSCV_occ/ans/" + self.name + ".txt"
        data = open(nameall, 'a')
        # print the training status
        print(training_str + metrics_str + time_str)
        data.write(training_str + metrics_str + time_str + "\n")
        data.close()
        self.last_time = now_time
        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(ResScale_AB(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    '''
    pretrained_dict = torch.load(args.restore_ckpt)
    old_list = {}
    
    for k, v in pretrained_dict.items():
        # if k.find('encoder.convc1')<0 :
        old_list.update({k: v})
    model.load_state_dict(old_list, strict=False)
    '''
    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    name = args.name
    logger = Logger(model, scheduler, name)

    VAL_FREQ = 2000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, dc_change, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions, dc_predictions = model(image1, image2, iters=6)

            loss1, loss2, metrics = sequence_loss(flow_predictions, dc_predictions, flow,
                                                  dc_change, valid, args.gamma, args.useconf)

            loss = loss1 + loss2
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:

                    if val_dataset == 'kitti':
                        evaluate.validate_Sintel_train(model.module,iftest=True)
                        outerr = evaluate.validate_kitti_test(model.module, ci=5, name=name)
                        results.update(outerr)
                logger.write_dict(results)

                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--traneval', default=0, type=int)
    parser.add_argument('--useconf', default=0, type=int)
    parser.add_argument('--usechonggou', default=0, type=int)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)