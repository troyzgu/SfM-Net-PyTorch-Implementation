import torch
import numpy as np

def obj_transform(pc, obj_mask, obj_t, obj_p, obj_r, num_masks=3):
    """
    input:
        pc Input points cloud, shape of (batch_size, channel = 3, height, width)
        obj_mask The mask of the object, shape of (batch_size, K = number of the objects, height, width)
        obj_t The translation of the object, shape of (batch_size, 3*K)
        obj_p 
        obj_r The rotation of the object, shape of (batch_size, 3*K)
    """
    pc = torch.movedim(pc, 1, 3)
    print(pc.shape)
    b, h, w, c = pc.shape

    p = _pivot_point(obj_p)
    R = _r_mat(torch.reshape(obj_r, [-1, 3]))
    print(p.shape)
    p = torch.reshape(p, [b, 1, 1, num_masks, 3])
    t = torch.reshape(obj_t, [b, 1, 1, num_masks, 3])
    R = torch.reshape(R, [b, 1, 1, num_masks, 3, 3])
    R = torch.tile(R, [1, h, w, 1, 1, 1])

    pc = torch.reshape(pc, [b, h, w, 1, 3])
    mask = torch.reshape(obj_mask, [b, h, w, num_masks, 1])

    pc_t = pc - p
    pc_t = _apply_r(pc_t, R)
    pc_t = pc_t + t - pc
    motion_maps = mask * pc_t

    pc = torch.reshape(pc, [b, h, w, 3])
    # pc_t = pc + tf.reduce_sum(motion_maps, -2)
    pc_t = pc + torch.sum(motion_maps, -2)
    return motion_maps, pc_t

def cam_transform(pc, cam_t, cam_p, cam_r):
    b, h, w, c = pc.shape

    p = _pivot_point(cam_p)
    R = _r_mat(cam_r)

    p = torch.reshape(p, [b, 1, 1, 3])
    t = torch.reshape(cam_t, [b, 1, 1, 3])
    R = torch.reshape(R, [b, 1, 1, 3, 3])
    R = torch.tile(R, [1, h, w, 1, 1])

    pc_t = pc - p
    pc_t = _apply_r(pc_t, R)
    pc_t = pc_t + t
    return pc_t

def _pivot_point(p):
    """
    This fucntion is used to calculate the pivot point of each object
    """
    p = torch.reshape(p, [-1, 20, 30])
    p_x = torch.sum(p, 1)
    p_y = torch.sum(p, 2)

    x_l = torch.from_numpy(np.linspace(-30.0, 30.0, 30))
    y_l = torch.from_numpy(np.linspace(-20.0, 20.0, 20))

    P_x = torch.sum(p_x * x_l, -1)
    P_y = torch.sum(p_y * y_l, -1)
    ground = torch.ones_like(P_x)

    P = torch.stack([P_x, P_y, ground], 1)
    return P

def _apply_r(pc, R):
    # for some reason matmul stopped working in tf 1.13
    # pc = tf.expand_dims(pc, -2)
    pc = torch.unsqueeze(pc, -2)
    return torch.sum(R * pc, -1)

def _r_mat(r):
    alpha = r[:, 0] * torch.pi
    beta = r[:, 1] * torch.pi
    gamma = r[:, 2] * torch.pi

    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)

    R_x = torch.stack([
        torch.stack([torch.cos(alpha), -torch.sin(alpha), zero], -1),
        torch.stack([torch.sin(alpha), torch.cos(alpha), zero], -1),
        torch.stack([zero, zero, one], -1),
    ], -2)

    R_y = torch.stack([
        torch.stack([torch.cos(beta), zero, torch.sin(beta)], -1),
        torch.stack([zero, one, zero], -1),
        torch.stack([-torch.sin(beta), zero, torch.cos(beta)], -1),
    ], -2)

    R_z = torch.stack([
        torch.stack([one, zero, zero], -1),
        torch.stack([zero, torch.cos(gamma), -torch.sin(gamma)], -1),
        torch.stack([zero, torch.sin(gamma), torch.cos(gamma)], -1),
    ], -2)

    return R_x @ R_y @ R_z

if __name__ == "__main__":
    pc = torch.rand(8, 3, 128, 384)
    obj_mask = torch.rand(8, 128, 384)
    obj_t = torch.rand(8, 9)
    obj_p = torch.rand(8, 3, 600)
    obj_r = torch.rand(8, 9)
    obj_transform(pc, obj_mask, obj_t, obj_p, obj_r)