# copied from https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/803291bd0ee91c7c13fb5cc42195383c5ade7d15/camera.py#L44

import torch


def se3_to_SE3(wu):  # [...,3]
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
    return Rt


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                      torch.stack([w2, O, -w0], dim=-1),
                      torch.stack([-w1, w0, O], dim=-1)], dim=-2)
    return wx


def taylor_A(x, nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        if i > 0: denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_B(x, nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_C(x, nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def compose(pose_list):
    # compose a sequence of poses together
    # pose_new(x) = poseN o ... o pose2 o pose1(x)
    pose_new = pose_list[0]
    for pose in pose_list[1:]:
        pose_new = compose_pair(pose_new, pose)
    return pose_new


def compose_pair(pose_a, pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
    R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
    R_new = R_b @ R_a
    t_new = (R_b @ t_a + t_b)[..., 0]
    pose_new = torch.zeros(*pose_a.shape[:-2], 4, 4, device=R_new.device, dtype=R_new.dtype)
    pose_new[..., :3, :3] = R_new
    pose_new[..., :3, 3] = t_new
    return pose_new
