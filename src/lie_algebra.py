from src.utils import *
import numpy as np


class SO3:
    #  tolerance criterion
    TOL = 1e-8
    Id = torch.eye(3).cuda().float()
    dId = torch.eye(3).cuda().double()

    @classmethod
    def exp(cls, phi):
        angle = phi.norm(dim=1, keepdim=True)
        mask = angle[:, 0] < cls.TOL
        dim_batch = phi.shape[0]
        Id = cls.Id.expand(dim_batch, 3, 3)

        axis = phi[~mask] / angle[~mask]
        c = angle[~mask].cos().unsqueeze(2)
        s = angle[~mask].sin().unsqueeze(2)

        Rot = phi.new_empty(dim_batch, 3, 3)
        Rot[mask] = Id[mask] + SO3.wedge(phi[mask])
        Rot[~mask] = c*Id[~mask] + \
            (1-c)*cls.bouter(axis, axis) + s*cls.wedge(axis)
        return Rot

    @classmethod
    def log(cls, Rot):
        dim_batch = Rot.shape[0]
        Id = cls.Id.expand(dim_batch, 3, 3)

        cos_angle = (0.5 * cls.btrace(Rot) - 0.5).clamp(-0.999999, 0.999999)
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        angle = cos_angle.acos()
        mask = angle < cls.TOL
        if mask.sum() == 0:
            angle = angle.unsqueeze(1).unsqueeze(1)
            return cls.vee((0.5 * angle/angle.sin())*(Rot-Rot.transpose(1, 2)))
        elif mask.sum() == dim_batch:
            # If angle is close to zero, use first-order Taylor expansion
            return cls.vee(Rot - Id)
        phi = cls.vee(Rot - Id)
        angle = angle
        phi[~mask] = cls.vee((0.5 * angle[~mask]/angle[~mask].sin()).unsqueeze(
            1).unsqueeze(2)*(Rot[~mask] - Rot[~mask].transpose(1, 2)))
        return phi

    @staticmethod
    def vee(Phi):
        return torch.stack((Phi[:, 2, 1],
                            Phi[:, 0, 2],
                            Phi[:, 1, 0]), dim=1)

    @staticmethod
    def wedge(phi):
        dim_batch = phi.shape[0]
        zero = phi.new_zeros(dim_batch)
        return torch.stack((zero, -phi[:, 2], phi[:, 1],
                            phi[:, 2], zero, -phi[:, 0],
                            -phi[:, 1], phi[:, 0], zero), 1).view(dim_batch,
                            3, 3)

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        return cls.rotz(yaw).bmm(cls.roty(pitch).bmm(cls.rotx(roll)))

    @classmethod
    def rotx(cls, angle_in_radians):
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 0, 0] = 1
        mat[:, 1, 1] = c
        mat[:, 2, 2] = c
        mat[:, 1, 2] = -s
        mat[:, 2, 1] = s
        return mat

    @classmethod
    def roty(cls, angle_in_radians):
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 1, 1] = 1
        mat[:, 0, 0] = c
        mat[:, 2, 2] = c
        mat[:, 0, 2] = s
        mat[:, 2, 0] = -s
        return mat

    @classmethod
    def rotz(cls, angle_in_radians):
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 2, 2] = 1
        mat[:, 0, 0] = c
        mat[:, 1, 1] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        return mat

    @classmethod
    def isclose(cls, x, y):
        return (x-y).abs() < cls.TOL

    @classmethod
    def to_rpy(cls, Rots):
        """Convert a rotation matrix to RPY Euler angles."""

        pitch = torch.atan2(-Rots[:, 2, 0],
            torch.sqrt(Rots[:, 0, 0]**2 + Rots[:, 1, 0]**2))
        yaw = pitch.new_empty(pitch.shape)
        roll = pitch.new_empty(pitch.shape)

        near_pi_over_two_mask = cls.isclose(pitch, np.pi / 2.)
        near_neg_pi_over_two_mask = cls.isclose(pitch, -np.pi / 2.)

        remainder_inds = ~(near_pi_over_two_mask | near_neg_pi_over_two_mask)

        yaw[near_pi_over_two_mask] = 0
        roll[near_pi_over_two_mask] = torch.atan2(
            Rots[near_pi_over_two_mask, 0, 1],
            Rots[near_pi_over_two_mask, 1, 1])

        yaw[near_neg_pi_over_two_mask] = 0.
        roll[near_neg_pi_over_two_mask] = -torch.atan2(
            Rots[near_neg_pi_over_two_mask, 0, 1],
            Rots[near_neg_pi_over_two_mask, 1, 1])

        sec_pitch = 1/pitch[remainder_inds].cos()
        remainder_mats = Rots[remainder_inds]
        yaw = torch.atan2(remainder_mats[:, 1, 0] * sec_pitch,
                          remainder_mats[:, 0, 0] * sec_pitch)
        roll = torch.atan2(remainder_mats[:, 2, 1] * sec_pitch,
                           remainder_mats[:, 2, 2] * sec_pitch)
        rpys = torch.cat([roll.unsqueeze(dim=1),
                        pitch.unsqueeze(dim=1),
                        yaw.unsqueeze(dim=1)], dim=1)
        return rpys

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        """
        if ordering == 'xyzw':
            qx = quat[:, 0]
            qy = quat[:, 1]
            qz = quat[:, 2]
            qw = quat[:, 3]
        elif ordering == 'wxyz':
            qw = quat[:, 0]
            qx = quat[:, 1]
            qy = quat[:, 2]
            qz = quat[:, 3]

        # Form the matrix
        mat = quat.new_empty(quat.shape[0], 3, 3)

        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
        mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
        mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

        mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
        mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
        mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

        mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
        mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
        mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)
        return mat

    @classmethod
    def to_quaternion(cls, Rots, ordering='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        """
        tmp = 1 + Rots[:, 0, 0] + Rots[:, 1, 1] + Rots[:, 2, 2]
        tmp[tmp < 0] = 0
        qw = 0.5 * torch.sqrt(tmp)
        qx = qw.new_empty(qw.shape[0])
        qy = qw.new_empty(qw.shape[0])
        qz = qw.new_empty(qw.shape[0])

        near_zero_mask = qw.abs() < cls.TOL

        if near_zero_mask.sum() > 0:
            cond1_mask = near_zero_mask * \
                (Rots[:, 0, 0] > Rots[:, 1, 1])*(Rots[:, 0, 0] > Rots[:, 2, 2])
            cond1_inds = cond1_mask.nonzero()

            if len(cond1_inds) > 0:
                cond1_inds = cond1_inds.squeeze()
                R_cond1 = Rots[cond1_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond1[:, 0, 0] -
                    R_cond1[:, 1, 1] - R_cond1[:, 2, 2]).view(-1)
                qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
                qx[cond1_inds] = 0.25 * d
                qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
                qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

            cond2_mask = near_zero_mask * (Rots[:, 1, 1] > Rots[:, 2, 2])
            cond2_inds = cond2_mask.nonzero()

            if len(cond2_inds) > 0:
                cond2_inds = cond2_inds.squeeze()
                R_cond2 = Rots[cond2_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond2[:, 1, 1] -
                                R_cond2[:, 0, 0] - R_cond2[:, 2, 2]).squeeze()
                tmp = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
                qw[cond2_inds] = tmp
                qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
                qy[cond2_inds] = 0.25 * d
                qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

            cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
            cond3_inds = cond3_mask

            if len(cond3_inds) > 0:
                R_cond3 = Rots[cond3_inds].view(-1, 3, 3)
                d = 2. * \
                    torch.sqrt(1. + R_cond3[:, 2, 2] -
                    R_cond3[:, 0, 0] - R_cond3[:, 1, 1]).squeeze()
                qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
                qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
                qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
                qz[cond3_inds] = 0.25 * d

        far_zero_mask = near_zero_mask.logical_not()
        far_zero_inds = far_zero_mask
        if len(far_zero_inds) > 0:
            R_fz = Rots[far_zero_inds]
            d = 4. * qw[far_zero_inds]
            qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
            qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
            qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

        # Check ordering last
        if ordering == 'xyzw':
            quat = torch.stack([qx, qy, qz, qw], dim=1)
        elif ordering == 'wxyz':
            quat = torch.stack([qw, qx, qy, qz], dim=1)
        return quat

    @classmethod
    def normalize(cls, Rots):
        U, _, V = torch.svd(Rots)
        S = cls.Id.clone().repeat(Rots.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(U) * torch.det(V)
        return U.bmm(S).bmm(V.transpose(1, 2))

    @classmethod
    def dnormalize(cls, Rots):
        U, _, V = torch.svd(Rots)
        S = cls.dId.clone().repeat(Rots.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(U) * torch.det(V)
        return U.bmm(S).bmm(V.transpose(1, 2))

    @classmethod
    def qmul(cls, q, r, ordering='wxyz'):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        """
        terms = cls.bouter(r, q)
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        xyz = torch.stack((x, y, z), dim=1)
        xyz[w < 0] *= -1
        w[w < 0] *= -1
        if ordering == 'wxyz':
            q = torch.cat((w.unsqueeze(1), xyz), dim=1)
        else:
            q = torch.cat((xyz, w.unsqueeze(1)), dim=1)
        return q / q.norm(dim=1, keepdim=True)

    @staticmethod
    def sinc(x):
        return x.sin() / x

    @classmethod
    def qexp(cls, xi, ordering='wxyz'):
        """
        Convert exponential maps to quaternions.
        """
        theta = xi.norm(dim=1, keepdim=True)
        w = (0.5*theta).cos()
        xyz = 0.5*cls.sinc(0.5*theta/np.pi)*xi
        return torch.cat((w, xyz), 1)

    @classmethod
    def qlog(cls, q, ordering='wxyz'):
        """
        Applies the log map to quaternions.
        """
        n = 0.5*torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
        n = torch.clamp(n, min=1e-8)
        q = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))
        r = q / n
        return r

    @classmethod
    def qinv(cls, q, ordering='wxyz'):
        "Quaternion inverse"
        r = torch.empty_like(q)
        if ordering == 'wxyz':
            r[:, 1:4] = -q[:, 1:4]
            r[:, 0] = q[:, 0]
        else:
            r[:, :3] = -q[:, :3]
            r[:, 3] = q[:, 3]
        return r

    @classmethod
    def qnorm(cls, q):
        "Quaternion normalization"
        return q / q.norm(dim=1, keepdim=True)

    @classmethod
    def qinterp(cls, qs, t, t_int):
        idxs = np.searchsorted(t, t_int)
        idxs0 = idxs-1
        idxs0[idxs0 < 0] = 0
        idxs1 = idxs
        idxs1[idxs1 == t.shape[0]] = t.shape[0] - 1
        q0 = qs[idxs0]
        q1 = qs[idxs1]
        tau = torch.zeros_like(t_int)
        dt = (t[idxs1]-t[idxs0])[idxs0 != idxs1]
        tau[idxs0 != idxs1] = (t_int-t[idxs0])[idxs0 != idxs1]/dt
        return cls.slerp(q0, q1, tau)

    @classmethod
    def slerp(cls, q0, q1, tau, DOT_THRESHOLD = 0.9995):
        """Spherical linear interpolation."""

        dot = (q0*q1).sum(dim=1)
        q1[dot < 0] = -q1[dot < 0]
        dot[dot < 0] = -dot[dot < 0]

        q = torch.zeros_like(q0)
        tmp = q0 + tau.unsqueeze(1) * (q1 - q0)
        tmp = tmp[dot > DOT_THRESHOLD]
        q[dot > DOT_THRESHOLD] = tmp / tmp.norm(dim=1, keepdim=True)

        theta_0 = dot.acos()
        sin_theta_0 = theta_0.sin()
        theta = theta_0 * tau
        sin_theta = theta.sin()
        s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
        s1 = (sin_theta / sin_theta_0).unsqueeze(1)
        q[dot < DOT_THRESHOLD] = ((s0 * q0) + (s1 * q1))[dot < DOT_THRESHOLD]
        return q / q.norm(dim=1, keepdim=True)

    @staticmethod
    def bouter(vec1, vec2):
        """batch outer product"""
        return torch.einsum('bi, bj -> bij', vec1, vec2)

    @staticmethod
    def btrace(mat):
        """batch matrix trace"""
        return torch.einsum('bii -> b', mat)

# radian to degree
def rad2deg(rad):
    return rad / torch.pi * 180
# degree to radian
def deg2rad(angle):
    return angle / 180 * torch.pi

def euler2rot(eulers):
    '''
    input: 
        [roll, pitch, yaw]
        [[roll, pitch, yaw], [...]]
        [[[roll, pitch, yaw], [...]], [[...]]]
    '''
    n_batchs, n_samples = count(eulers, 1)
    if n_batchs != 0:
        eulers = eulers.reshape(-1, 3)
    elif n_samples == 0:
        eulers = eulers.view(1, 3)

    n = eulers.shape[0]

    ones = torch.ones(n).cuda()
    zeros = torch.zeros(n).cuda()

    roll = eulers[:, 0]
    pitch = eulers[:, 1]
    yaw = eulers[:, 2]

    s_roll = roll.sin()
    c_roll = roll.cos()
    rot_roll = torch.stack((ones, zeros, zeros,
                            zeros, c_roll, -s_roll,
                            zeros, s_roll, c_roll), dim=1).view(n, 3, 3)

    s_pitch = pitch.sin()
    c_pitch = pitch.cos()
    rot_pitch = torch.stack((c_pitch, zeros, s_pitch,
                             zeros, ones, zeros,
                             -s_pitch, zeros, c_pitch), dim=1).view(n, 3, 3)

    s_yaw = yaw.sin()
    c_yaw = yaw.cos()
    rot_yaw = torch.stack((c_yaw, -s_yaw, zeros,
                           s_yaw, c_yaw, zeros,
                           zeros, zeros, ones), dim=1).view(n, 3, 3)
    
    if n_batchs != 0:
        return rot_yaw.bmm(rot_pitch.bmm(rot_roll)).reshape(n_batchs, -1, 3, 3).cuda()
    elif n_samples == 0:
        return rot_yaw.bmm(rot_pitch.bmm(rot_roll))[0].cuda()
    else:
        return rot_yaw.bmm(rot_pitch.bmm(rot_roll)).cuda()
    
def quat2rot(qs, order='wxyz'):
    '''
    dim: 
        2 (n, 4)
    '''
    n_batchs, n_samples = count(qs, 1)
    if n_batchs != 0:
        qs = qs.reshape(-1, 4)
    
    if order == 'wxyz':
        qw = qs[:, 0]
        qx = qs[:, 1]
        qy = qs[:, 2]
        qz = qs[:, 3]
    # if order == 'xyzw':
    else:
        qw = qs[:, 3]
        qx = qs[:, 0]
        qy = qs[:, 1]
        qz = qs[:, 2]

    rots = torch.empty(qs.shape[0], 3, 3)
    rots[:, 0, 0] = 1. - 2. * (qy * qy + qz * qz)
    rots[:, 0, 1] = 2. * (qx * qy - qw * qz)
    rots[:, 0, 2] = 2. * (qx * qz + qw * qy)
    rots[:, 1, 0] = 2. * (qx * qy + qw * qz)
    rots[:, 1, 1] = 1. - 2. * (qx * qx + qz * qz)
    rots[:, 1, 2] = 2. * (qy * qz - qw * qx)
    rots[:, 2, 0] = 2. * (qx * qz - qw * qy)
    rots[:, 2, 1] = 2. * (qy * qz + qw * qx)
    rots[:, 2, 2] = 1. - 2. * (qx * qx + qy * qy)
    
    if n_batchs != 0:
        return rots.reshape(n_batchs, -1, 3, 3).cuda()
    return rots.cuda()

def quat2euler(qs):
    return rot2euler(quat2rot(qs)).cuda()
    

def rot2euler(rots):
    '''
    dim: 
        2: (3, 3)
        3: (n, 3, 3)
        4: (b, n, 3, 3)
    '''
    # reshape input as (n_samples, features)
    n_batchs, n_samples = count(rots, 2)
    if n_batchs != 0:
        rots = rots.reshape(-1, 3, 3)
    elif n_samples == 0:
        rots = rots.view(1, 3, 3)

    pitch = torch.atan2(-rots[:, 2, 0], torch.sqrt(rots[:, 0, 0]**2 +rots[:, 1, 0]**2))
    yaw = pitch.new_empty(pitch.shape)
    roll = pitch.new_empty(pitch.shape)
    
    near_pos_pi_mask = isclose(pitch, np.pi/2.)
    near_neg_pi_mask = isclose(pitch, -np.pi/2.)
    mask = ~(near_pos_pi_mask | near_neg_pi_mask)

    yaw[near_pos_pi_mask] = 0
    roll[near_pos_pi_mask] = torch.atan2(rots[near_pos_pi_mask, 0, 1], rots[near_pos_pi_mask, 1, 1])
    yaw[near_neg_pi_mask] = 0
    roll[near_neg_pi_mask] = -torch.atan2(rots[near_neg_pi_mask, 0, 1], rots[near_neg_pi_mask, 1, 1])
    
    c_pitch = 1 / pitch[mask].cos()
    yaw[mask] = torch.atan2(rots[mask][:, 1, 0] * c_pitch, rots[mask][:, 0, 0] * c_pitch)
    roll[mask] = torch.atan2(rots[mask][:, 2, 1] * c_pitch, rots[mask][:, 2, 2] * c_pitch)
    
    if n_batchs != 0:
        return torch.stack((roll, pitch, yaw), dim=1).reshape(n_batchs, -1, 3)
    elif n_samples == 0:
        return torch.stack((roll, pitch, yaw), dim=1)[0]
    else:
        return torch.stack((roll, pitch, yaw), dim=1).cuda()
