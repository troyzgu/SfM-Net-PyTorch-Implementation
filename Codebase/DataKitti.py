import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

# from .utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torchvision

import numpy as np
import matplotlib.pyplot as plt


class kitti_depth(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── kitti_depth
                        └─ raw
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    # data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    # resources = [
    #     "data_object_image_2.zip",
    #     "data_object_label_2.zip",
    # ]
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

        # if download:
        #     self.download()
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
                
                break
        
        # for img_file in os.listdir(image_dir):
        #     self.images.append(os.path.join(image_dir, img_file))
            # if self.train:
            #     self.targets.append(os.path.join(labels_dir, img_file))
        
        

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
            # print("plt:", np.max(tmp_target))
            # target = Image.open(self.targets[index])

        # target = self._parse_target(index) if self.train else None
        # if self.transforms:
        #     image, target = self.transforms(image, target)
        
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(target)
        # plt.show()
        # print(np.max(target))
        transform_method = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 384)),
            ]
        )
        # image = torchvision.transforms.ToTensor(image)
        # target = torchvision.transforms.ToTensor()(target)
        image = transform_method(image)
        target = transform_method(target)

        return image.float().to('cuda'), target.float().to('cuda')


    # def _parse_target(self, index: int) -> List:
    #     target = []
    #     with open(self.targets[index]) as inp:
    #         content = csv.reader(inp, delimiter=" ")
    #         for line in content:
    #             target.append(
    #                 {
    #                     "type": line[0],
    #                     "truncated": float(line[1]),
    #                     "occluded": int(line[2]),
    #                     "alpha": float(line[3]),
    #                     "bbox": [float(x) for x in line[4:8]],
    #                     "dimensions": [float(x) for x in line[8:11]],
    #                     "location": [float(x) for x in line[11:14]],
    #                     "rotation_y": float(line[14]),
    #                 }
    #             )
    #     return target

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

