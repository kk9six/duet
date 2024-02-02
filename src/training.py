import os,sys,time
from copy import deepcopy
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

import torch
from copy import deepcopy

class TrainStepRunner:
    def __init__(self, net, loss_fn, optimizer):
        self.net, self.loss_fn = net, loss_fn
        self.optimizer = optimizer

    def __call__(self, X, y):
        self.net.train()
        preds = self.net(X)
        loss = self.loss_fn(preds, (X, y))

        #backward()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

class TrainEpochRunner:
    def __init__(self,steprunner):
        self.steprunner = steprunner

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        for X, y in dataloader:
            loss = self.steprunner(X, y)
            total_loss += loss
            step += 1
        epoch_loss = total_loss / step
        return epoch_loss

class ValStepRunner:
    def __init__(self, net, metrics_dict):
        self.net = net
        self.metrics_dict = metrics_dict

    @torch.no_grad()
    def __call__(self, X, y):
        self.net.eval()
        preds = self.net(X.unsqueeze(0))
        metrics = {name: metric_fn(preds, y).item() for name, metric_fn in self.metrics_dict.items()}
        return metrics


class ValEpochRunner:
    def __init__(self,steprunner):
        self.steprunner = steprunner

    def __call__(self, data):
        metrics = [self.steprunner(X, y) for X, y in data]
        names = metrics[0].keys()
        return {name: np.mean([metric[name] for metric in metrics]) for name in names}

def train_model(net, optimizer, loss_fn, metrics_dict, metrics_for_early_stopping,
                train_data, val_data=None,
                epochs=1000, patience=10, ckpt_path='models/checkpoint.pt'):

    metrics = {}
    best_metrics = {}
    counter = 0
    times = []
    with tqdm(total = epochs) as pbar:
        for epoch in range(1, epochs+1):
            # 1，train -------------------------------------------------
            train_step_runner = TrainStepRunner(net = net, loss_fn = loss_fn, optimizer = optimizer)
            train_epoch_runner = TrainEpochRunner(train_step_runner)
            torch.cuda.synchronize()
            start_epoch = time.time()
            epoch_loss = train_epoch_runner(train_data)
            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            times.append(elapsed)

            pbar.update(1)
            pbar.set_description("Epoch {0} / {1}".format(epoch, epochs))
            postfix = {'Loss': epoch_loss}

            # 2，validate -------------------------------------------------
            if val_data and epoch > 1200:
                val_step_runner = ValStepRunner(net = net, metrics_dict=deepcopy(metrics_dict))
                val_epoch_runner = ValEpochRunner(val_step_runner)
                with torch.no_grad():
                    metrics = val_epoch_runner(val_data)
                postfix.update(metrics)

                if best_metrics == {}:
                    best_metrics = {name: metrics[name] for name in metrics_for_early_stopping}

                if sum([best_metrics[name] >= metrics[name] for name in metrics_for_early_stopping]) == len(metrics_for_early_stopping):
                    best_metrics = {name: metrics[name] for name in metrics_for_early_stopping}
                    torch.save(deepcopy(net.state_dict()), ckpt_path)
                    counter = 0
                else:
                    counter += 1

                if counter == patience:
                    postfix.update({f'Best {name}': value for name, value in best_metrics.items()})
                    pbar.set_postfix(postfix)
                    break

                else:
                    postfix.update({'Counter': counter})
                    pbar.set_postfix(postfix)

            pbar.set_postfix(postfix)

        if patience == -1:
            torch.save(deepcopy(net.state_dict()), ckpt_path)

    return np.mean(times)

