from .lie_algebra import SO3, rad2deg, rot2euler
import numpy as np
from .kinematic import inf_orientation, inf_trajectory_from_signals, inf_velocity
from .utils import btmm

def RMSE(diff, report_individual_axis=False):
    if report_individual_axis:
        return diff.pow(2).mean(dim=0).sqrt()
    else:
        return diff.pow(2).mean(dim=0).sqrt().norm()

def metric_ave_training(pred, targ):
    _, _, _, hat_acc = pred
    _, rots, _, vs, _ = targ
    hat_vs = inf_velocity(hat_acc[0], vs[0], rots)
    n = min(hat_vs.shape[0], vs.shape[0])
    return RMSE(hat_vs[:n] - vs[:n])

def metric_aoe_training(pred, targ):
    _, _, hat_gyro, _ = pred
    _, rots, _, _, _ = targ
    dt = 1./200
    diff = rad2deg((SO3.log(btmm(SO3.exp(hat_gyro[0] * dt), btmm(rots[:-1], rots[1:])))))
    return RMSE(diff)

def metric_ave_test(pred, targ):
    _, _, _, hat_acc = pred
    rots, vs = targ['rots'], targ['vs']
    hat_vs = inf_velocity(hat_acc, vs[0], rots)
    n = hat_vs.shape[0] - 5 * 200 # truncation to align the length
    return RMSE(hat_vs[:n] - vs[:n])


def metric_aoe_test(pred, targ):
    _, _, hat_gyro, _ = pred
    rots = targ['rots']
    n = hat_gyro.shape[0] - 5 * 200 # truncation to align the length
    hat_rots = inf_orientation(hat_gyro, rots[0])
    diff = rad2deg((SO3.log(btmm(hat_rots[:n], rots[:n]))))
    # diff = rad2deg((rot2euler(btmm(hat_rots[:n], rots[:n]))))
    return RMSE(diff)

def metric_aye_test(pred, targ):
    _, _, hat_gyro, _ = pred
    rots = targ['rots']
    n = hat_gyro.shape[0] - 5 * 200 # truncation to align the length
    hat_rots = inf_orientation(hat_gyro, rots[0])
    diff = rad2deg((SO3.log(btmm(hat_rots[:n], rots[:n]))))
    errors_rpy = RMSE(diff, report_individual_axis=True)
    return errors_rpy[2]

def metric_rte_test(pred, targ, duration, repetitions):
    _, _, hat_gyro, hat_acc = pred
    acc = targ['imu'][:, 3:]
    rots = targ['rots']
    ps = targ['ps']
    vs = targ['vs']
    
    duration = duration * 200
    rtes = []
    n = hat_gyro.shape[0] - 5 * 200 # truncation to align the length

    for _ in range(repetitions): #repetitions
        start = np.random.randint(0, n - duration - 1)
        p0 = ps[start]
        ps_gt = ps[start:start+duration+1]
        v0 = vs[start]
        r0 = rots[start]
        hat_ps = inf_trajectory_from_signals(
                acc[start:start+duration],
                hat_gyro[start:start+duration],
                r0, v0, p0)
        rtes.append(RMSE(hat_ps - ps_gt).cpu().numpy())

    return np.mean(rtes)

def metric_rte_wo_acc_test(pred, targ, duration, repetitions):
    _, _, hat_gyro, hat_acc = pred
    rots = targ['rots']
    ps = targ['ps']
    vs = targ['vs']
    
    duration = duration * 200
    rtes = []
    n = hat_gyro.shape[0] - 5 * 200 # truncation to align the length

    for _ in range(repetitions): #repetitions
        start = np.random.randint(0, n - duration - 1)
        p0 = ps[start]
        ps_gt = ps[start:start+duration+1]
        v0 = vs[start]
        r0 = rots[start]
        hat_ps = inf_trajectory_from_signals(
                hat_acc[start:start+duration],
                hat_gyro[start:start+duration],
                r0, v0, p0)
        rtes.append(RMSE(hat_ps - ps_gt).cpu().numpy())

    return np.mean(rtes)


def metric_rte_improvement_test(pred, targ, duration, repetitions):
    _, _, hat_gyro, hat_acc = pred
    rots = targ['rots']
    raw_gyro, raw_acc = targ['imu'][:, :3], targ['imu'][:, 3:]
    ps = targ['ps']
    vs = targ['vs']
    
    duration = duration * 200
    hat_rtes = []
    raw_rtes = []
    rte_improvement = []
    n = hat_gyro.shape[0] - 5 * 200 # truncation to align the length


    for _ in range(repetitions): #repetitions
        start = np.random.randint(0, n - duration - 1)
        p0 = ps[start]
        ps_gt = ps[start:start+duration+1]
        v0 = vs[start]
        r0 = rots[start]
        hat_ps = inf_trajectory_from_signals(
                hat_acc[start:start+duration],
                hat_gyro[start:start+duration],
                r0, v0, p0)
        raw_ps = inf_trajectory_from_signals(
                raw_acc[start:start+duration],
                raw_gyro[start:start+duration],
                r0, v0, p0)
        hat_rmse = RMSE(hat_ps - ps_gt)
        raw_rmse = RMSE(raw_ps - ps_gt)
        hat_rtes.append(hat_rmse.cpu().numpy())
        raw_rtes.append(raw_rmse.cpu().numpy())

        rte_improvement.append((1 - hat_rmse/raw_rmse).cpu().numpy())
    return np.mean(raw_rtes), np.mean(hat_rtes), np.mean(rte_improvement)
