import torch

def _2d_project(pc, camera_intrinsics):
    cx, cy, cf = camera_intrinsics

    X = pc[:, :, :, 0]
    Y = pc[:, :, :, 1]
    Z = pc[:, :, :, 2]

    x = cf * X / Z + cx
    y = cf * Y / Z + cy
    return torch.stack([x, y], dim = -1)

def optical_flow(pc, camera_intrinsics=(0.5, 0.5, 1.0)):
    points = _2d_project(pc, camera_intrinsics)
    b, h, w, c = points.shape

    x_l = torch.linspace(0.0, 1.0, steps = w)
    y_l = torch.linspace(0.0, 1.0, steps = h)
    x, y = torch.meshgrid(x_l, y_l, indexing ='xy')
    pos = torch.stack([x, y], dim = -1)
    flow = points - pos
    return points, flow
