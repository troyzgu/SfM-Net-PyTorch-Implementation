# SfM-Net-PyTorch-Implementation

This is the github repo for the implementation of a modified version of the SfM-Net in PyTorch. 

We use Kitti Dataset for training and testing. You need to download the Kitti-Dataset and format the structure of the data as shown in the KittiDataset.py

## Dense Depth Map Generation

We use conv-deconv structure for depth map estimation. In our implementation, you can designate the learning rate and other hyper-parameters from the command line. Also, you have to the modify the root path your file. Then, you can start training by running:

python train_structure.py --root_path "YOUR ROOT PATH"

## KITTI Odometry Evaluation

We use a pre-trained VGG-Net for rotation and translation matrix estimation. For running our code, you should organize your data format as the KittiDataset.kittiodom() shows. Then, you could just run:

python train_motion.py --root_path "YOUR ROOT PATH"

## Training model using unsupervised learning

The structure of the data is the same as the kittiodom(). For training our model unsupervisedly, you can just run:

python train.py --root_path "YOUR ROOT PATH"
