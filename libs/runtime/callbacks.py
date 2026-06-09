# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from runtime.loggers import Logger
from runtime.metrics import MeanAbsoluteError


class BaseCallback(ABC):
    def on_fit_start(self, optimizer, args):
        pass

    def on_fit_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_start(self):
        pass

    def on_validation_step(self, input, target, pred):
        pass

    def on_validation_end(self, epoch=None):
        pass

    def on_checkpoint_load(self, checkpoint):
        pass

    def on_checkpoint_save(self, checkpoint):
        pass


class LRSchedulerCallback(BaseCallback):
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.scheduler = None

    @abstractmethod
    def get_scheduler(self, optimizer, args):
        pass

    def on_fit_start(self, optimizer, args):
        self.scheduler = self.get_scheduler(optimizer, args)

    def on_checkpoint_load(self, checkpoint):
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def on_checkpoint_save(self, checkpoint):
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

    def on_epoch_end(self,loss=None):
        if self.logger is not None:
            self.logger.log_metrics({'learning rate': self.scheduler.get_last_lr()[0]}, step=self.scheduler.last_epoch)
        if not loss==None:
            self.scheduler.step(metrics=loss)
        else:
            self.scheduler.step()


class QM9MetricCallback(BaseCallback):
    """ Logs the rescaled mean absolute error for QM9 regression tasks """

    def __init__(self, logger, targets_std, prefix=''):
        self.mae = MeanAbsoluteError()
        self.logger = logger
        self.targets_std = targets_std
        self.prefix = prefix
        self.best_mae = float('inf')

    def on_validation_step(self, input, target, pred):
        self.mae(pred.detach(), target.detach())

    def on_validation_end(self, epoch=None):
        mae = self.mae.compute() * self.targets_std
        logging.info(f'{self.prefix} MAE: {mae}')
        self.logger.log_metrics({f'{self.prefix} MAE': mae}, epoch)
        self.best_mae = min(self.best_mae, mae)

    def on_fit_end(self):
        if self.best_mae != float('inf'):
            self.logger.log_metrics({f'{self.prefix} best MAE': self.best_mae})
import math
from torch.optim.lr_scheduler import _LRScheduler,ReduceLROnPlateau

class tmp_add(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]
class tmp_rlp(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier=100, total_epoch=10):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler =tmp_add(optimizer,factor=0.75,patience=4,threshold=0.001,verbose=True) 
        self.finished = False
        super(tmp_rlp, self).__init__(optimizer)

    def get_last_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != tmp_add:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_last_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None,loss=None,metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr


class QM9LRSchedulerCallback(LRSchedulerCallback):
    def __init__(self, logger, epochs):
        super().__init__(logger)
        self.epochs = epochs

    def get_scheduler(self, optimizer, args):
        min_lr = args.min_learning_rate if args.min_learning_rate else 1e-7
        print(args.sch_type)
        if args.sch_type=='rlp':
            return tmp_rlp(optimizer)
        if args.sch_type=='cosine':
            for pg in optimizer.param_groups:
                pg['lr'] = min_lr
            return CosineAnnealingWarmUpRestarts(
                optimizer, T_0=200, eta_max=args.learning_rate, T_up=10, gamma=0.75
            )
        #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.epochs, eta_min=min_lr)


class PerformanceCallback(BaseCallback):
    def __init__(self, logger, batch_size: int, warmup_epochs: int = 1, mode: str = 'train'):
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.timestamps = []
        self.mode = mode
        self.logger = logger

    def on_batch_start(self):
        if self.epoch >= self.warmup_epochs:
            self.timestamps.append(time.time() * 1000.0)

    def _log_perf(self):
        stats = self.process_performance_stats()
        for k, v in stats.items():
            logging.info(f'performance {k}: {v}')

        self.logger.log_metrics(stats)

    def on_epoch_end(self):
        self.epoch += 1

    def on_fit_end(self):
        if self.epoch > self.warmup_epochs:
            self._log_perf()
            self.timestamps = []

    def process_performance_stats(self):
        timestamps = np.asarray(self.timestamps)
        deltas = np.diff(timestamps)
        throughput = (self.batch_size / deltas).mean()
        stats = {
            f"throughput_{self.mode}": throughput,
            f"latency_{self.mode}_mean": deltas.mean(),
            f"total_time_{self.mode}": timestamps[-1] - timestamps[0],
        }
        for level in [90, 95, 99]:
            stats.update({f"latency_{self.mode}_{level}": np.percentile(deltas, level)})

        return stats
