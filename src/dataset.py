from .utils import bmtm, btmv
from .lie_algebra import SO3
from torch.utils.data.dataset import Dataset
import numpy as np
import torch

class EuRoC(Dataset):
    """
        Dataloader for the EUROC Data Set.
    """
    def __init__(self, original_data_dir, processed_data_dir, train_seqs, test_seqs, train_length=16000, T=64):
        super().__init__()
        self.dt = 1/200
        self.g = torch.Tensor([0, 0, 9.80804]).cuda() # local gravity
        self.T = T # time interval, default = 64
        self.train_seq_names = train_seqs
        self.val_seq_names = self.train_seq_names.copy()
        self.test_seq_names = test_seqs

        self.seq_names = self.train_seq_names + self.test_seq_names # all sequence names

        self.n_train = len(self.train_seq_names)
        self.n_val = len(self.val_seq_names)
        self.n_test = len(self.test_seq_names)
        self.train_length = train_length

        self.process_data(original_data_dir, processed_data_dir)
        self.data = self.read_data(processed_data_dir)
        self.init_normalize_factor()

        self.is_train = True

    def interpolate(self, x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        """
        # vector interpolation
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            if i in [4, 5, 6, 7]:
                continue
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int

    def process_data(self, original_data_dir, processed_data_dir):
        try:
            name = self.seq_names[0]
            torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_imu.csv', delimiter=',')).cuda()[:-1]
        except:
            for name in self.seq_names:
                imu = np.genfromtxt(f'{original_data_dir}/{name}/mav0/imu0/data.csv', delimiter=',', skip_header=1)
                gt = np.genfromtxt(f'{original_data_dir}/{name}/mav0/state_groundtruth_estimate0/data.csv', delimiter=',', skip_header=1)

                t0 = np.max([gt[0, 0], imu[0, 0]])
                t_end = np.min([gt[-1, 0], imu[-1, 0]])
                idx0_imu = np.searchsorted(imu[:, 0], t0)
                idx0_gt = np.searchsorted(gt[:, 0], t0)
                idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
                idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

                imu = imu[idx0_imu: idx_end_imu]
                gt = gt[idx0_gt: idx_end_gt]
                ts = imu[:, 0]/1e9

                gt = self.interpolate(gt, gt[:, 0]/1e9, ts)
                gt[:, 1:4] = gt[:, 1:4] - gt[0, 1:4]
                q_gt = torch.Tensor(gt[:, 4:8]).double()
                q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
                gt[:, 4:8] = q_gt

                gt = torch.Tensor(gt).double()
                imu = torch.Tensor(imu[:, 1:]).double()

                np.savetxt(f'{processed_data_dir}/{name}_imu.csv', imu, delimiter=',')
                np.savetxt(f'{processed_data_dir}/{name}_gt.csv', gt, delimiter=',')

    def read_data(self, processed_data_dir):
        data = dict()
        T = self.T
        for name in self.seq_names:
            seq = dict()
            seq['name'] = name
            seq['imu'] = torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_imu.csv', delimiter=',')).cuda()[:-1]
            gt = torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_gt.csv', delimiter=',')).cuda()
            seq['ps'], seq['qs'], seq['vs'], seq['bgs'], seq['bas'] = gt[:, 1:4], gt[:, 4:8], gt[:, 8:11], gt[:, 11:14], gt[:, 14:]
            seq['rots'] = SO3.from_quaternion(seq['qs'], ordering='wxyz')
            seq['rotsT'] = bmtm(seq['rots'][:-int(T/2)], seq['rots'][int(T/2):])
            seq['angular_velocities'] = SO3.log(bmtm(seq['rots'][:-1], seq['rots'][1:])) / self.dt
            seq['accelerations'] = btmv(seq['rots'][:-1], (seq['vs'][1:] - seq['vs'][:-1]) / self.dt + self.g)

            data[name] = seq

        return data

    def init_normalize_factor(self):
        self.mean = torch.zeros((6, )).cuda()
        self.std = torch.zeros((6, )).cuda()
        n_samples = 0
        for name in self.train_seq_names:
            imu = self.data[name]['imu']
            self.mean += imu.sum(dim=0)
            n_samples += imu.shape[0]
        self.mean = self.mean / n_samples

        for name in self.train_seq_names:
            imu = self.data[name]['imu']
            self.std += ((imu - self.mean) ** 2).sum(dim=0)
        self.std = (self.std / n_samples).sqrt()

    @property
    def val(self):
        return [(
            self.data[name]['imu'].cuda(),
            (self.data[name]['accelerations'].cuda(),
             self.data[name]['rots'].cuda(),
             self.data[name]['rotsT'].cuda(),
             self.data[name]['vs'].cuda(),
             self.data[name]['ps'].cuda())
        ) for name in self.val_seq_names]

    @property
    def test_seqs(self):
        return dict({name: self.data[name] for name in self.test_seq_names})

    def __len__(self):
        return self.n_train

    def __getitem__(self, idx):
        name = self.train_seq_names[idx % self.n_train]

        start = torch.randint(0, self.T * 2, (1, ))
        X = self.data[name]['imu'][start:self.train_length + start].cuda()

        return X, \
            (self.data[name]['accelerations'][start:self.train_length+start].cuda(), \
             self.data[name]['rots'][start:self.train_length+start].cuda(),  \
             self.data[name]['rotsT'][start:self.train_length+start].cuda(), \
             self.data[name]['vs'][start:self.train_length+start].cuda(), \
             self.data[name]['ps'][start+1:self.train_length+start+1].cuda())


class TUMVICali(Dataset):
    """
        Dataloader for the EUROC Data Set.
    """
    def __init__(self, original_data_dir, processed_data_dir, train_seqs, test_seqs, train_length=16000, T=64):
        super().__init__()
        self.dt = 1/200
        self.g = torch.Tensor([0, 0, 9.80804]).cuda() # local gravity
        self.T = T # time interval, default = 64
        self.train_seq_names = train_seqs
        self.val_seq_names = self.train_seq_names.copy()
        self.test_seq_names = test_seqs

        self.seq_names = self.train_seq_names + self.test_seq_names # all sequence names

        self.n_train = len(self.train_seq_names)
        self.n_val = len(self.val_seq_names)
        self.n_test = len(self.test_seq_names)
        self.train_length = train_length

        self.process_data(original_data_dir, processed_data_dir)
        self.data = self.read_data(processed_data_dir)
        self.init_normalize_factor()

        self.is_train = True

    def interpolate(self, x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        """
        # vector interpolation
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            if i in [4, 5, 6, 7]:
                continue
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int

    def process_data(self, original_data_dir, processed_data_dir):
        try:
            name = self.seq_names[0]
            torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_imu.csv', delimiter=',')).cuda()[:-1]
        except:
            for name in self.seq_names:
                imu = np.genfromtxt(f'{original_data_dir}/{name}/mav0/imu0/data.csv', delimiter=',', skip_header=1)
                gt = np.genfromtxt(f'{original_data_dir}/{name}/mav0/mocap0/data.csv', delimiter=',', skip_header=1)

                t0 = np.max([gt[0, 0], imu[0, 0]])
                t_end = np.min([gt[-1, 0], imu[-1, 0]])
                idx0_imu = np.searchsorted(imu[:, 0], t0)
                idx0_gt = np.searchsorted(gt[:, 0], t0)
                idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
                idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

                imu = imu[idx0_imu: idx_end_imu]
                gt = gt[idx0_gt: idx_end_gt]
                ts = imu[:, 0]/1e9

                gt = self.interpolate(gt, gt[:, 0]/1e9, ts)
                gt[:, 1:4] = gt[:, 1:4] - gt[0, 1:4]
                q_gt = torch.Tensor(gt[:, 4:8]).double()
                q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
                gt[:, 4:8] = q_gt

                gt = torch.Tensor(gt).double()
                imu = torch.Tensor(imu[:, 1:]).double()

                np.savetxt(f'{processed_data_dir}/{name}_imu.csv', imu, delimiter=',')
                np.savetxt(f'{processed_data_dir}/{name}_gt.csv', gt, delimiter=',')

    def read_data(self, processed_data_dir):
        data = dict()
        T = self.T
        for name in self.seq_names:
            seq = dict()
            seq['name'] = name
            seq['imu'] = torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_imu.csv', delimiter=',')).cuda()[:-1]
            gt = torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_gt.csv', delimiter=',')).cuda()
            seq['ps'], seq['qs'] = gt[:, 1:4], gt[:, 4:8]
            seq["vs"] = (seq["ps"][1:] - seq["ps"][:-1]) / self.dt
            seq['rots'] = SO3.from_quaternion(seq['qs'], ordering='wxyz')
            seq['rotsT'] = bmtm(seq['rots'][:-int(T/2)], seq['rots'][int(T/2):])

            data[name] = seq

        return data

    def init_normalize_factor(self):
        self.mean = torch.zeros((6, )).cuda()
        self.std = torch.zeros((6, )).cuda()
        n_samples = 0
        for name in self.train_seq_names:
            imu = self.data[name]['imu']
            self.mean += imu.sum(dim=0)
            n_samples += imu.shape[0]
        self.mean = self.mean / n_samples

        for name in self.train_seq_names:
            imu = self.data[name]['imu']
            self.std += ((imu - self.mean) ** 2).sum(dim=0)
        self.std = (self.std / n_samples).sqrt()

    @property
    def val(self):
        return [(
            self.data[name]['imu'].cuda(),
            (None,
             self.data[name]['rots'].cuda(),
             self.data[name]['rotsT'].cuda(),
             None,
             self.data[name]['ps'].cuda())
        ) for name in self.val_seq_names]

    @property
    def test_seqs(self):
        return dict({name: self.data[name] for name in self.test_seq_names})

    def __len__(self):
        return self.n_train

    def __getitem__(self, idx):
        name = self.train_seq_names[idx % self.n_train]

        start = torch.randint(0, self.T * 2, (1, ))
        X = self.data[name]['imu'][start:self.train_length + start].cuda()

        return X, \
            (torch.tensor(0).cuda(), \
             self.data[name]['rots'][start:self.train_length+start].cuda(),  \
             self.data[name]['rotsT'][start:self.train_length+start].cuda(), \
             torch.tensor(0).cuda(), \
             self.data[name]['ps'][start+1:self.train_length+start+1].cuda())

class TUMVIUncali(Dataset):
    """
        Dataloader for the EUROC Data Set.
    """
    def __init__(self, original_data_dir, processed_data_dir, train_seqs, test_seqs, train_length=16000, T=64):
        super().__init__()
        self.dt = 1/200
        self.g = torch.Tensor([0, 0, 9.80804]).cuda() # local gravity
        self.T = T # time interval, default = 64
        self.train_seq_names = train_seqs
        self.val_seq_names = self.train_seq_names.copy()
        self.test_seq_names = test_seqs

        self.seq_names = self.train_seq_names + self.test_seq_names # all sequence names

        self.n_train = len(self.train_seq_names)
        self.n_val = len(self.val_seq_names)
        self.n_test = len(self.test_seq_names)
        self.train_length = train_length

        self.accel_bias = torch.Tensor([-1.30318, -0.391441, 0.380509]).cuda()
        self.accel_scale = torch.Tensor([
            1.00422, 0, 0,
            -7.82123e-05, 1.00136, 0,
            -0.0097745, -0.000976476, 0.970467
        ]).cuda().reshape(3, 3)
        self.gyro_bias = torch.Tensor([0.0283122, 0.00723077,0.0165292]).cuda()
        self.gyro_scale = torch.Tensor([
            0.943611,  0.00148681, 0.000824366,
            0.000369694,1.09413,-0.00273521,
            -0.00175252,0.00834754,1.01588
        ]).cuda().reshape(3, 3)
        self.multiplier = torch.zeros((6, 6)).cuda()
        self.multiplier[:3, :3] = self.gyro_scale
        self.multiplier[3:, 3:] = self.accel_scale
        self.accel_noise_density = 0.0014
        self.gyro_noise_density = 0.00008
        self.accel_bias_instability = 1e-4
        self.gyro_bias_instability = 1e-4

        self.process_data(original_data_dir, processed_data_dir)
        self.data = self.read_data(processed_data_dir)
        self.init_normalize_factor()

        self.is_train = True

    def add_error(self, x):
        n_samples = x.shape[0]
        noise = torch.zeros_like(x).cuda()
        noise[:, :3] = torch.normal(0, self.gyro_noise_density, (n_samples, 3)).cuda()
        noise[:, 3:] = torch.normal(0, self.accel_noise_density, (n_samples, 3)).cuda()

        bias = torch.zeros_like(x).cuda()
        bias[:, :3] += torch.normal(0, self.gyro_bias_instability, (n_samples, 3)).cuda()
        bias[:, 3:] += torch.normal(0, self.accel_bias_instability, (n_samples, 3)).cuda()
        bias += torch.cat((self.gyro_bias, self.accel_bias)).expand(x.shape).cuda()

        return torch.matmul(torch.inverse(self.multiplier), (x - (bias + noise)).transpose(0, 1)).transpose(0, 1)

    def interpolate(self, x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        """
        # vector interpolation
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            if i in [4, 5, 6, 7]:
                continue
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int

    def process_data(self, original_data_dir, processed_data_dir):
        try:
            name = self.seq_names[0]
            torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_imu.csv', delimiter=',')).cuda()[:-1]
        except:
            for name in self.seq_names:
                imu = np.genfromtxt(f'{original_data_dir}/{name}/mav0/imu0/data.csv', delimiter=',', skip_header=1)
                gt = np.genfromtxt(f'{original_data_dir}/{name}/mav0/mocap0/data.csv', delimiter=',', skip_header=1)

                t0 = np.max([gt[0, 0], imu[0, 0]])
                t_end = np.min([gt[-1, 0], imu[-1, 0]])
                idx0_imu = np.searchsorted(imu[:, 0], t0)
                idx0_gt = np.searchsorted(gt[:, 0], t0)
                idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
                idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

                imu = imu[idx0_imu: idx_end_imu]
                gt = gt[idx0_gt: idx_end_gt]
                ts = imu[:, 0]/1e9

                gt = self.interpolate(gt, gt[:, 0]/1e9, ts)
                gt[:, 1:4] = gt[:, 1:4] - gt[0, 1:4]
                q_gt = torch.Tensor(gt[:, 4:8]).double()
                q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
                gt[:, 4:8] = q_gt

                gt = torch.Tensor(gt).double()
                imu = torch.Tensor(imu[:, 1:]).double()

                np.savetxt(f'{processed_data_dir}/{name}_imu.csv', imu, delimiter=',')
                np.savetxt(f'{processed_data_dir}/{name}_gt.csv', gt, delimiter=',')

    def read_data(self, processed_data_dir):
        data = dict()
        T = self.T
        for name in self.seq_names:
            seq = dict()
            seq['name'] = name
            seq['imu'] = self.add_error(torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_imu.csv', delimiter=',')).cuda()[:-1])
            gt = torch.Tensor(np.loadtxt(f'{processed_data_dir}/{name}_gt.csv', delimiter=',')).cuda()
            seq['ps'], seq['qs'] = gt[:, 1:4], gt[:, 4:8]
            seq["vs"] = (seq["ps"][1:] - seq["ps"][:-1]) / self.dt
            seq['rots'] = SO3.from_quaternion(seq['qs'], ordering='wxyz')
            seq['rotsT'] = bmtm(seq['rots'][:-int(T/2)], seq['rots'][int(T/2):])

            data[name] = seq

        return data

    def init_normalize_factor(self):
        self.mean = torch.zeros((6, )).cuda()
        self.std = torch.zeros((6, )).cuda()
        n_samples = 0
        for name in self.train_seq_names:
            imu = self.data[name]['imu']
            self.mean += imu.sum(dim=0)
            n_samples += imu.shape[0]
        self.mean = self.mean / n_samples

        for name in self.train_seq_names:
            imu = self.data[name]['imu']
            self.std += ((imu - self.mean) ** 2).sum(dim=0)
        self.std = (self.std / n_samples).sqrt()

    @property
    def val(self):
        return [(
            self.data[name]['imu'].cuda(),
            (None,
             self.data[name]['rots'].cuda(),
             self.data[name]['rotsT'].cuda(),
             None,
             self.data[name]['ps'].cuda())
        ) for name in self.val_seq_names]

    @property
    def test_seqs(self):
        return dict({name: self.data[name] for name in self.test_seq_names})

    def __len__(self):
        return self.n_train

    def __getitem__(self, idx):
        name = self.train_seq_names[idx % self.n_train]

        start = torch.randint(0, self.T * 2, (1, ))
        X = self.data[name]['imu'][start:self.train_length + start].cuda()

        return X, \
            (torch.tensor(0).cuda(), \
             self.data[name]['rots'][start:self.train_length+start].cuda(),  \
             self.data[name]['rotsT'][start:self.train_length+start].cuda(), \
             torch.tensor(0).cuda(), \
             self.data[name]['ps'][start+1:self.train_length+start+1].cuda())
