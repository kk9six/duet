import torch
from .utils import bmv
from .lie_algebra import SO3, euler2rot


def inf_orientation(angular_velocity, r0, dt=1./200):
    """ compute orientation from angular velocity.
    :param angular_velocity: (n, 3) tensor angular velocity
    :param r0: (3, 3) tensor initial rotation matrix
    :param dt: float time step
    :return: (n+1, 3, 3) tensor rotation matrix
    """
    delta_rots = SO3.exp(angular_velocity * dt)
    # delta_rots = euler2rot(angular_velocity * dt)
    rots = torch.zeros((angular_velocity.shape[0] + 1, 3, 3)).cuda()
    rots[0] = r0
    for i in range(len(delta_rots)):
        rots[i+1] = rots[i].matmul(delta_rots[i])
    return rots

def inf_velocity(accel, v0, rots, dt=1./200):
    """ compute velocity from acceleration and orientation.
    :param accel: (n, 3) tensor acceleration
    :param v0: (3, ) tensor initial velocity
    :param rots: (n+1, 3, 3) tensor rotation matrix
    :param dt: float time step
    :return: (n, 3) tensor velocity
    """
    n = accel.shape[0]
    g = torch.Tensor([0, 0, 9.80804]).cuda()
    accel_global = bmv(rots[:n], accel) - g
    vels = torch.concat((v0.unsqueeze(0), accel_global * dt)).cumsum(dim=0)
    return vels

def inf_trajectory_from_signals(accel, angular_velocity, r0, v0, p0, dt=1./200):
    """ compute trajectory from acceleration and angular velocity.
    :param accel: (n, 3) tensor acceleration
    :param angular_velocity: (n, 3) tensor angular velocity
    :param r0: (3, 3) tensor initial rotation matrix
    :param v0: (3, ) tensor initial velocity
    :param p0: (3, ) tensor initial position
    :param dt: float time step
    :return positions: (n, 3) tensor position
    """
    rots = inf_orientation(angular_velocity, r0, dt)
    n = accel.shape[0]
    g = torch.Tensor([0, 0, 9.80804]).cuda()
    accel_global = bmv(rots[:n], accel) - g
    vels = torch.concat((v0.unsqueeze(0), accel_global * dt)).cumsum(dim=0)
    positions = torch.concat((p0.unsqueeze(0),
                              vels[:-1] * dt + 1 / 2 * accel_global * (dt ** 2))).cumsum(dim=0)

    return positions

def inf_trajectory_from_accel(accel, rots, v0, p0, dt=1./200):
    """ compute trajectory from acceleration and given orientation.
    :param accel: (n, 3) tensor acceleration
    :param rots: (n+1, 3, 3) tensor rotation matrix
    :param v0: (3, ) tensor initial velocity
    :param p0: (3, ) tensor initial position
    :param dt: float time step
    :return positions: (n, 3) tensor position
    """
    g = torch.Tensor([0, 0, 9.80804]).cuda()
    accel_global = bmv(rots[:-1], accel) - g
    vels = torch.concat((v0.unsqueeze(0), accel_global * dt)).cumsum(dim=0)
    positions = torch.concat((p0.unsqueeze(0),
                    vels[:-1] * dt + 1 / 2 * accel_global * (dt ** 2))).cumsum(dim=0)

    return positions
