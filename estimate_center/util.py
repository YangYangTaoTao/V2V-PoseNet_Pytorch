'''
    直接在深度图回归出
    input:
        points: world space points
        refpoints: reference points (it's the label)
'''

import os
import numpy as np
import sys
import struct
import h5py
import torch
from torch.utils.data import Dataset


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    return p_x, p_y


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels


# db: side or top
# db_type: train or test
# fileID: ID
def load_depthmap(root, fileID, db, db_type, img_width, img_height):
    mapPath = os.path.join(root, "ITOP_" + str(db) + "_" + str(db_type) + "_depth_map.h5")
    f = h5py.File(mapPath, 'r')
    depth_image = f['data'][fileID]
    # depth_image = torch.from_numpy(depth_image)
    # depth_image = depth_image.view(img_height, img_width)
    return depth_image


# root = './datasets/depthmap'
# db = 'side|top'
# center_dir = './datasets/center/ITOP_center'
# mode = 'train|test'
class ITOPCenterDataset(Dataset):
    def __init__(self, root, db, center_dir, mode, transform=None):
        self.img_width = 320
        self.img_height = 240
        self.fx = 285.71
        self.fy = 285.71
        self.joint_num = 15
        self.world_dim = 3

        self.root = root
        self.db = db
        self.center_dir = center_dir
        self.mode = mode
        self.transform = transform

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')

        if not self._check_exists(): raise RuntimeError('Invalid ITOP dataset')

        self._load()

    def __getitem__(self, index):
        depthmap = load_depthmap(self.root, self.names[index], self.db, self.mode, self.img_width, self.img_height)
        points = depthmap2points(depthmap, self.fx, self.fy)
        # 进行数据增强：引入平移，旋转，放缩等
        trans = np.random.rand(3) * 5
        angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi
        time_size = np.random.rand() * 2
        # resize
        points = points * time_size
        labels = self.ref_pts[index] * time_size
        # print(points.shape, labels.shape)
        # rotation, 只考虑xy方向
        points[:, :, 0] = points[:, :, 0]*np.cos(angle) - points[:, :, 1]*np.sin(angle)
        points[:, :, 1] = points[:, :, 0]*np.sin(angle) + points[:, :, 1]*np.cos(angle)
        labels[0] = labels[0]*np.cos(angle) - labels[1]*np.sin(angle)
        labels[1] = labels[0]*np.sin(angle) + labels[1]*np.cos(angle)
        # 平移
        points = points + trans
        labels = labels + trans

        points = torch.from_numpy(points)
        labels = torch.from_numpy(labels)
        points = points.permute(2, 0, 1)

        return points, labels

    def __len__(self):
        return self.num_samples
        
    # _load: Load all the data at once (This is not a good choice when the data set is large)
    # (The fid can be used as an index to load a small amount of data at a time)
    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        if self.mode == 'train':
            ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "center_train.txt")
        else:
            ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "center.txt")
            # ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "center_test.txt")

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
            ref_pt_str = [l.rstrip() for l in f]
        
        # for center and joints; select valid data
        frame_id = 0
        invalid_frameNum = 0
        for fid in range(self.num_samples):
            splitted = ref_pt_str[fid].split(" ")
            if splitted[0] == 'invalid':
                invalid_frameNum += 1
            else:
                self.ref_pts[frame_id, 0] = splitted[0]
                self.ref_pts[frame_id, 1] = splitted[1]
                self.ref_pts[frame_id, 2] = splitted[2]
                self.names.append(fid)
                frame_id += 1
        self.ref_pts = self.ref_pts[:-invalid_frameNum]
        self.num_samples = self.num_samples - invalid_frameNum

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 39795, 10501

    def _check_exists(self):
        # Check depth map
        depthmap_path = os.path.join(self.root, "ITOP_" + str(self.db) + "_" + str(self.mode) + "_depth_map.h5")
        if not os.path.exists(depthmap_path):
            print('Error: depth map file {} does not exist'.format(depthmap_path))
            return False

        # Check precomputed centers by v2v-hand model's author(referrence)
        ref_path = os.path.join(self.center_dir, "ITOP_" + str(self.db) + "_view", "center_" + str(self.mode) + ".txt")
        if not os.path.exists(ref_path):
            print('Error: precomputed center files do not exist')
            return False

        return True

# 产生的 center points 保存到 txt 文件中
def estimate_points(root, db, mode, model, device, center_dir, ref_pt_file, savepath):
    f = open(savepath, 'w')
    mapPath = os.path.join(root, "ITOP_" + str(db) + "_" + str(mode) + "_depth_map.h5")
    labelPath = os.path.join(root, "ITOP_" + str(db) + "_" + str(mode) + "_labels.h5")
    with open(os.path.join(center_dir, ref_pt_file)) as pf:
        ref_pt_str = [l.rstrip() for l in pf]
    
    depthmaps = (h5py.File(mapPath, "r"))['data']
    labels = (h5py.File(labelPath, 'r'))['real_world_coordinates']
    for fid in range(len(labels)):
        splitted = ref_pt_str[fid].split(" ")
        if(splitted[0] == "invalid"):
            print("invalid invalid invalid")
            f.write("invalid invalid invalid\n")
        else:
            img = depthmaps[fid]
            points = depthmap2points(img, 285.71, 285.71)
            # points[:,:,2] = points[:,:,2] + np.random.rand()*3
            points = (torch.from_numpy(points)).permute(2, 0, 1).unsqueeze(dim=0)
            points = points.to(device)
            output = model(points)
            print(output)
            f.write(str(output[0][0].item()) + ' ' + str(output[0][1].item()) + ' ' + str(output[0][2].item()) + '\n')
    f.close()


