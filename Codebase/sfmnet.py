import torch
import torch.nn as nn
import torch.nn.functional as F
from MotionNet import MotionNet
from StructureNet import StructureNet
from transform import obj_transform, cam_transform
from flow_functions import optical_flow

class sfmnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure = StructureNet()
        self.motion = MotionNet(6)

    
    def forward(self, frame0, frame1, sharpness_multiplier):
        depth, points = self.structure(frame0)
        objs, cams = self.motion(frame0, frame1, sharpness_multiplier)
        obj_mask, obj_t, obj_p, obj_r = objs
        cam_t, cam_p, cam_r = cams
        motion_map, points = obj_transform(points, obj_mask, obj_t, obj_p, obj_r)
        
        points = cam_transform(points, cam_t, cam_p, cam_r)
        # print("type after transform:", points.dtype)
        points_2d, flow = optical_flow(points)

        return depth, points, objs, cams, motion_map, points_2d, flow
