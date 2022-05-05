import csv
import os
import torch
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from torchvision.datasets.vision import VisionDataset
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from transform import r_to_homo

device = "cuda" if torch.cuda.is_available() else "cpu"

class kitti_depth(VisionDataset):
    """
    Structure of the store format
    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

        <root>
            └── kitti_depth
                └─ raw
                    ├── training
                    |   ├── img
                    |   |     ├── 2011_09_26_drive_0001_sync/image_02/data
                    |   |     ├── ...
                    |   |     └── 2011_10_03_drive_0058_sync/image_02/data
                    |   └── labels
                    |         ├── 2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/data
                    |         ├── ...
                    |         └── 2011_10_03_drive_0058_sync/proj_depth/groundtruth/image_02/data
                    |
                    └── validate
                        ├── img
                        |     ├── 2011_09_26_drive_0001_sync/image_02/data
                        |     ├── ...
                        |     └── 2011_10_03_drive_0058_sync/image_02/data
                        └── labels
                                ├── 2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/data
                                ├── ...
                                └── 2011_10_03_drive_0058_sync/proj_depth/groundtruth/image_02/data


    """

    image_dir_name = "img"
    labels_dir_name = "labels"

    image_local = "image_02/data"
    label_local = "proj_depth/groundtruth/image_02"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "validate"

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
            for img_direc in os.listdir(labels_dir):

                for img_file in os.listdir(os.path.join(labels_dir,img_direc, self.label_local)):
                    label_file = os.path.join(labels_dir, img_direc, self.label_local, img_file)
                    input_file = os.path.join(image_dir, img_direc, self.image_local, img_file)
                    if (os.path.exists(label_file) and os.path.exists(input_file)):
                        self.targets.append(label_file)
                        self.images.append(input_file)        
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        image = plt.imread(self.images[index])
        if self.train:          
            target = plt.imread(self.targets[index])

        transform_method = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 384)),
            ]
        )

        image = transform_method(image)
        target = transform_method(target)

        return image.float().to(device), target.float().to(device)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)


class kitti_odom(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

        <root>
            └── odometry
                ├── image_02
                └── 00.txt


    """

    image_dir_name = "image_2"
    labels_dir_name = "labels"

    def __init__(
        self,
        root: str = "/mnt/back_data/Kitti/odometry",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root = "/mnt/back_data/Kitti/odometry",
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._frame_dis = 1

        image_dir = os.path.join(self.root, self.image_dir_name)
        labels_dir = os.path.join(self.root, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            input_file = os.path.join(image_dir, img_file)
            self.images.append(input_file)
        
        label_file = os.path.join(labels_dir, "04.txt")
        with open(label_file, "r") as f:
            self.targets = f.readlines()
        
        
    def _parse_target(self, index: int) -> List:
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "type": line[0],
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return target 
    
    def get_initial_pose(self):
        str = self.targets[0].replace('\n', '').split(' ')
        matrix = torch.tensor([float(i) for i in str]).reshape((3, 4)).float().to(device)
        homo = torch.vstack((matrix, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(device)))
        return homo

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        
        image_00 = plt.imread(self.images[index])
        image_01 = plt.imread(self.images[index+self._frame_dis])   
        transform_method = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 384)),
            ]
        )
        image_00 = transform_method(image_00)
        image_01 = transform_method(image_01)
        str_00 = self.targets[index].replace('\n', '').split(' ')
        matrix_00 = torch.tensor([float(i) for i in str_00]).reshape((3, 4)).float().to(device)
        homo_00 = torch.vstack((matrix_00, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(device)))
        str_01 = self.targets[index+self._frame_dis].replace('\n', '').split(' ')
        matrix_01 = torch.tensor([float(i) for i in str_01]).reshape((3, 4)).float().to(device)
        homo_01 = torch.vstack((matrix_01, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(device)))
        trans_homo = homo_01@torch.inverse(homo_00)
        R = trans_homo[:3, :3]
        t = trans_homo[:3, 3]

        return (image_00.float().to(device), image_01.float().to(device)), (R, t)


    def __len__(self) -> int:
        return (len(self.images) - self._frame_dis)

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)



if __name__ == "__main__":
    dataset = kitti_odom()
    dataset.__getitem__(0)