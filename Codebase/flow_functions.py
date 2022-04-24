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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = _2d_project(pc, camera_intrinsics)
    b, h, w, c = points.shape

    x_l = torch.linspace(0.0, 1.0, steps = w)
    y_l = torch.linspace(0.0, 1.0, steps = h)
    x, y = torch.meshgrid(x_l, y_l, indexing ='xy')
    pos = torch.stack([x, y], dim = -1).to(device)
    flow = points - pos
    flow = torch.movedim(flow, -1, 1)

    return points, flow

if __name__ == "__main__":
    pc = torch.rand(8, 128, 384, 3)
    optical_flow(pc)
