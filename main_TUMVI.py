import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

print("cuda: %s" % torch.cuda.is_available())

from src.dataset import TUMVICali, TUMVIUncali
from src.networks import IMUNet
from src.losses import IMULoss
from src.metrics import (
    metric_aoe_test,
    metric_aoe_training,
    metric_ave_test,
    metric_ave_training,
    metric_aye_test,
    metric_rte_improvement_test,
    metric_rte_test,
)
from src.training import train_model
from src.test import test_model
import random

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

T = 84
training_samples = 16000 // (T * 2) * (T * 2)
train_seqs = [
    "dataset-room1_512_16",
    "dataset-room3_512_16",
    "dataset-room5_512_16",
]
test_seqs = ["dataset-room2_512_16", "dataset-room4_512_16", "dataset-room6_512_16"]

print("Loading data...")
data = TUMVIUncali(
    "original_datasets/tumviuncali",
    "data/tumviuncali",
    train_seqs,
    test_seqs,
    training_samples,
    T,
)

in_channel = 6
out_channel = 3
layer_channels = [16, 32, 64, 128]
batch_size = 128
kernel_size = 5
dropout = 0.1
lr = 0.01
num_epochs = 1800
ckpt_path = "models/tumviuncali.pt"

# train
train_iter = DataLoader(data, batch_size=batch_size, shuffle=False)
val_data = data.val
net = IMUNet(
    in_channel, layer_channels, out_channel, kernel_size, dropout, data.mean, data.std
).cuda()
loss_func = IMULoss(T=T).cuda()
optimiser = torch.optim.Adam(
    [
        {
            "params": net.parameters(),
            "lr": lr,
            "weight_decay": 0.1,
        },
        {"params": loss_func.parameters(), "weight_decay": 0},
    ]
)

metrics_dict_training = {"AOE": lambda x, y: metric_aoe_training(x, y)}
metrics_for_early_stopping = ["AOE"]
running_time = train_model(
    net,
    optimiser,
    loss_func,
    metrics_dict_training,
    metrics_for_early_stopping,
    train_iter,
    val_data,
    epochs=num_epochs,
    patience=-1,
    ckpt_path=ckpt_path,
)
print(
    f"training time: {running_time} s/epoch (time of gyro loss: {np.mean(loss_func.times_gyro)}, time of acc loss: {np.mean(loss_func.times_accel)})"
)
print("=".ljust(20, "="))

# test
metric_dict_test = {
    "Absolute Orientation Error (AOE)": lambda x, y: metric_aoe_test(x, y),
    "Absolute Yaw Error (AYE)": lambda x, y: metric_aye_test(x, y),
    "Relative Translation Error (RTE)": lambda x, y: metric_rte_test(
        x, y, 5, 50
    ),  # rte: 5s 50 reps
    "Improvement of Relative Translation Error (RAW, DUET, Improvement)": lambda x, y: metric_rte_improvement_test(
        x, y, 5, 50
    ),
}
net = IMUNet(
    in_channel, layer_channels, out_channel, kernel_size, dropout, data.mean, data.std
).cuda()
net.load_state_dict(torch.load(ckpt_path))
net.eval()
running_time = test_model(net, metric_dict_test, data.test_seqs)
print(f"running time: {running_time}")
