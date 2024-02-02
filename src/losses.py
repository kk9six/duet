from torch import nn
import time
import torch
from .lie_algebra import SO3
from .utils import bbmv
import numpy as np

class GyroLoss(nn.Module):
    def __init__(self, delta=0.005, dt=1/200, T=64):
        super().__init__()
        self.g = torch.Tensor([0, 0, 9.80804]).cuda()
        self.delta = delta
        self.dt = dt
        self.T = T
        self.huber = torch.nn.HuberLoss(reduction='mean', delta=delta)

    def f_loss(self, x):
        return self.huber(x, torch.zeros_like(x)) * 1e5

    def forward(self, pred, targ):
        '''
        x: rotsT, (num_batchs, n, 3, 3)
        hat_x: angular rate, (num_batchs, n, 3)
        '''
        _, _, hat_gyro, _ = pred
        _, (_, _, rots, _, _) = targ
        T = int(self.T / 2)

        n_batches, n_samples, _ = hat_gyro.shape
        hat_rots = SO3.exp(hat_gyro.reshape(-1, 3) * self.dt).reshape(n_batches, n_samples, 3, 3)
        rots = rots[:, ::T]
        if int(np.log2(T)) == np.log2(T):
            for _ in range(int(np.log2(T))):
                hat_rots = torch.einsum('bnij, bnjk -> bnik', hat_rots[:, ::2], hat_rots[:, 1::2])
        else:
            _hat_rots = hat_rots[:, ::T]
            for i in range(T-1):
                _hat_rots = torch.einsum('bnij, bnjk -> bnik', _hat_rots, hat_rots[:, 1+i::T])
            hat_rots = _hat_rots
        diff_rot = SO3.log(torch.einsum('bnij, bnik -> bnjk', hat_rots, rots).reshape(-1, 3, 3))

        l1 = self.f_loss(diff_rot)
        hat_rots = torch.einsum('bnij, bnjk -> bnik', hat_rots[:, ::2], hat_rots[:, 1::2])
        rots = torch.einsum('bnij, bnjk -> bnik', rots[:, ::2], rots[:, 1::2])
        diff_rot = SO3.log(torch.einsum('bnij, bnik -> bnjk', hat_rots, rots).reshape(-1, 3, 3))
        l2 = self.f_loss(diff_rot)

        return l1 + l2 / 2

class AccLoss(nn.Module):
    def __init__(self, delta=0.005, dt=1/200, T=64, target='pos'):
        super().__init__()
        self.g = torch.Tensor([0, 0, 9.80804]).cuda()
        self.delta = delta
        self.dt = dt

        self.T = T
        self.x1 = torch.arange(1, self.T+1).cuda()
        self.x2 = torch.arange(1, self.T).cuda()
        self.x3 = torch.arange(0, self.T).cuda()

        self.huber = torch.nn.HuberLoss(reduction='mean', delta=delta)
        self.mse = torch.nn.MSELoss(reduction='mean', )

        if target == 'acc':
            self.forward = self.forward_with_acceleration

    def f_loss(self, x):
        return self.huber(x, torch.zeros_like(x)) * 1e4

    def forward(self, pred, targ):
        T = self.T

        _, _, _, hat_acc = pred
        _, (_, rots, _, _, ps) = targ
        n_batches, n_samples, n_dims = hat_acc.shape

        linear_acc = bbmv(rots, hat_acc) - self.g
        linear_acc = linear_acc.reshape(n_batches, int(n_samples / 2 / T), 2 * T, n_dims)

        diff_hat = (self.x1.unsqueeze(1) * linear_acc[:, :, self.x1 - 1, ] * (self.dt ** 2)).sum(dim=2) + \
                ((T - self.x2).unsqueeze(1) * linear_acc[:, :, T + self.x2 - 1] * (self.dt ** 2)).sum(dim=2) + \
                ((linear_acc[:, :, T + self.x3] - linear_acc[:, :, self.x3]) * (self.dt ** 2)).sum(dim=2) / 2.

        ps = ps.reshape(n_batches, int(n_samples / 2 / T), 2 * T, 3)
        diff_targ = (ps[:, :, -1] - ps[:, :, T]) - (ps[:, :, T] - ps[:, :, 0])
        # ps_2T = torch.cat((ps[:, 1:, 0], ps[:, -1, -1].unsqueeze(1)), dim=1)
        # ps_T = ps[:, :, T]
        # ps_0 = ps[:, :, 0]
        # diff_targ = (ps_2T - ps_T) - (ps_T - ps_0)

        l1 = self.f_loss(diff_hat - diff_targ)
        return l1

    def forward_with_acceleration(self, pred, targ):
        _, _, _, hat_acc = pred
        _, (accels, _, _, _) = targ
        return self.mse(hat_acc, accels)

class IMULoss(nn.Module):
    def __init__(self, target='pos', dt=1/200, T=64):
        super(IMULoss, self).__init__()
        self.params = torch.nn.Parameter(torch.ones(2, requires_grad=True))
        self.acc_loss = AccLoss(target=target, dt=dt, T=T)
        self.gyro_loss = GyroLoss(dt=dt, T=T)

        self.times_gyro = []
        self.times_accel = []

    def forward(self, pred, targ):
        torch.cuda.synchronize()
        start_epoch = time.time()
        loss_sum = 0.5 / (self.params[0] ** 2) * self.acc_loss(pred, targ) + torch.log(1 + self.params[0] ** 2)
        torch.cuda.synchronize()
        end_epoch = time.time()
        self.times_accel.append(end_epoch - start_epoch)

        torch.cuda.synchronize()
        start_epoch = time.time()
        loss_sum += 0.5 / (self.params[1] ** 2) * self.gyro_loss(pred, targ) + torch.log(1 + self.params[1] ** 2)
        torch.cuda.synchronize()
        end_epoch = time.time()
        self.times_gyro.append(end_epoch - start_epoch)

        return loss_sum

    def init_timer(self):
        self.times_accel = []
        self.times_gyro = []
