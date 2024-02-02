import torch
from torch import nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, dilation, dropout=0.2):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, dilation=dilation, padding=padding)
        self.net = nn.Sequential(
            self.conv1,
            nn.BatchNorm1d(out_channel,),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.gelu = nn.GELU()
        self.init_weights(self.conv1)
        self.init_weights(self.conv2)

    def init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
        
    def forward(self, x):
        return self.gelu(self.net(x) + self.conv2(x))

class BaseNet(nn.Module):
    def __init__(self, in_channel, layer_channels, out_channel, kernel_size, dropout=0.2):
        super().__init__()
        layers = []
#         padding = 0
        for i, channel in enumerate(layer_channels):
            dilation = 4 ** i
            padding = (kernel_size - 1) * dilation
            layers += [
                Block(in_channel if i == 0 else layer_channels[i-1], 
                      layer_channels[i], kernel_size, padding, dilation, dropout)
            ]
        self.net = nn.Sequential(
            # nn.ReplicationPad1d((padding, 0)), 
            *layers,
            # nn.ReplicationPad1d((0, 0))
        )
        
    def forward(self, x):
        return self.net(x)
    
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        
    def forward(self, x, y):
        return torch.cat((x, y), dim=2)

class IMUNet(nn.Module):
    def __init__(self, in_channel, layer_channels, out_channel, kernel_size, dropout=0.2, mean=None, std=None):
        super().__init__()
        self.error_param_acc = torch.nn.Parameter((0.01 * torch.randn(3, 3)).cuda())
        self.error_param_gyro = torch.nn.Parameter((0.01 * torch.randn(3, 3)).cuda())
        self.g_sen = torch.nn.Parameter((0.0001 * torch.randn(3, 3)).cuda())
        self.eye = torch.eye(3).cuda()
        
        self.acc_layer = BaseNet(in_channel, layer_channels, out_channel, kernel_size, dropout=dropout).cuda()
        self.gyro_layer = BaseNet(in_channel, layer_channels, out_channel, kernel_size, dropout=dropout).cuda()
        
        self.acc_output_layer = nn.Conv1d(layer_channels[-1], out_channel, 1)
        self.gyro_output_layer = nn.Conv1d(layer_channels[-1], out_channel, 1)
        self.cat = Concat()
        self.mean1 = torch.nn.Parameter(mean.cuda(), requires_grad=False)
        self.std1 = torch.nn.Parameter(std.cuda(), requires_grad=False)
        
        self.mean2 = torch.nn.Parameter(torch.zeros((6, )).cuda(), requires_grad=False)
        self.std2 = torch.nn.Parameter(torch.ones((6, )).cuda(), requires_grad=False)
        self._n_samples = 0
        
        self.init_weights(self.acc_output_layer)
        self.init_weights(self.gyro_output_layer)
        
    def init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)

    def norm1(self, x):
        return (x - self.mean1) / self.std1
    def norm2(self, x):
        return (x - self.mean2) / self.std2
    
    def update_normalize_factor(self, x):
        self.mean2.data *= self._n_samples
        self.mean2.data += x.view(-1, 6).sum(dim=0)
        self._n_samples += x.view(-1, 6).shape[0]
        self.mean2.data /= self._n_samples
        self.std2.data = self.std2.pow(2) * self._n_samples
        self.std2.data += ((x.view(-1, 6) - self.mean2) ** 2).sum(dim=0)
        self.std2.data = (self.std2 / self._n_samples).sqrt()
        
    def forward(self, x):
        C_gyro_transpose = torch.inverse(self.eye + self.error_param_gyro)
        C_acc_transpose = torch.inverse(self.eye + self.error_param_acc)
        
        corr_gyro = self.gyro_output_layer(self.gyro_layer(self.norm1(x).permute(0, 2, 1))).permute(0, 2, 1)
        x_gyro = torch.matmul(C_gyro_transpose, (x[:, :, :3] - corr_gyro).transpose(1, 2)).transpose(1, 2)
        
        cat_x = self.cat(x_gyro, x[:, :, 3:])
        if self.training:
            self.update_normalize_factor(cat_x)

        corr_acc = self.acc_output_layer(self.acc_layer(self.norm2(cat_x).permute(0, 2, 1))).permute(0, 2, 1)
        x_acc = torch.matmul(C_acc_transpose, (x[:, :, 3:] - corr_acc).transpose(1, 2)).transpose(1, 2)
        
        return corr_gyro, corr_acc, x_gyro, x_acc
