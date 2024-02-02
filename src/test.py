import torch
import numpy as np
import time

def est_from_net(net, X):
    net.eval()
    with torch.no_grad():
        corr_gyro, corr_acc, hat_gyro, hat_acc = net(X.view(1, -1, 6))
        return corr_gyro[0], corr_acc[0], hat_gyro[0], hat_acc[0]


def test_model(net, metrics_dict, test_data, saved_path=None, ckpt_path=None):
    # data: imu, ps, qs, vs, bgs, bas, rots, angular_velocities, accelerations 
    times = []
    for name, data in test_data.items():
        print(f'test {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 

        torch.cuda.synchronize()
        start_epoch = time.time()
        pred = est_from_net(net, imu)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed / imu.shape[0])

        if saved_path is not None and ckpt_path is not None:
            with open(f"{saved_path}_{name}.csv", "a") as f:
                f.write(f'{ckpt_path},')
                for metric_name, metric_fn in metrics_dict.items():
                    metric = metric_fn(pred, data)
                    try:
                        len(metric)
                        for m in metric:
                            f.write(f'{m},')
                    except:
                        try:
                            f.write(f'{metric.cpu().item()},')
                        except:
                            f.write(f'{metric},')
                f.write('\n')
                f.close()
        else:
            for metric_name, metric_fn in metrics_dict.items():
                print(f'{metric_name}: {metric_fn(pred, data)}')

    with open("test_time.csv", "a") as f:
        f.write(f"{ckpt_path},{np.mean(times)}\n")
        f.close()

    return np.mean(times)

def test_model_raw(metrics_dict, test_data):
    # data: imu, ps, qs, vs, bgs, bas, rots, angular_velocities, accelerations 
    for name, data in test_data.items():
        print(f'test {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 
        for metric_name, metric_fn in metrics_dict.items():
            print(f'{metric_name}: {metric_fn([None, None, imu[:, :3], imu[:, 3:]], data)}')
