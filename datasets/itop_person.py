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
class ITOPDataset(Dataset):
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

    def get_data(self):
        return self.names, self.joints_world, self.ref_pts

    def __getitem__(self, index):
        depthmap = load_depthmap(self.root, self.names[index], self.db, self.mode, self.img_width, self.img_height)
        points = depthmap2points(depthmap, self.fx, self.fy)
        points = points.reshape((-1, 3))

        sample = {
            'name': self.names[index],
            'points': points,
            'joints': self.joints_world[index],
            'refpoint': self.ref_pts[index]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples
        
    # _load: Load all the data at once (This is not a good choice when the data set is large)
    # (The fid can be used as an index to load a small amount of data at a time)
    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        if self.mode == 'train':
            ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "my_train_center.txt")
            # ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "center_train.txt")
        else:
            ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "my_test_center.txt")
            # ref_pt_file = os.path.join("ITOP_" + str(self.db) + "_view", "center_test.txt")

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
            ref_pt_str = [l.rstrip() for l in f]
        # read file
        labelfile = h5py.File(os.path.join(self.root, "ITOP_" + str(self.db) + "_" + str(self.mode) + "_labels.h5"), 'r')
        joints_world_original = labelfile['real_world_coordinates']
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
                self.joints_world[frame_id] = joints_world_original[fid]
                self.names.append(fid)
                frame_id += 1
        self.joints_world = self.joints_world[:-invalid_frameNum]
        self.ref_pts = self.ref_pts[:-invalid_frameNum]
        self.num_samples = self.num_samples - invalid_frameNum

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 39795, 10501

    def _check_exists(self):
        # Check depth map
        depthmap_path = os.path.join(self.root, "ITOP_" + str(self.db) + "_" + str(self.mode) + "_depth_map.h5")
        if not os.path.exists(depthmap_path):
            print('Error: depth map file {} does not exist'.format(annot_file))
            return False

        # Check basic data(label)
        label_path = os.path.join(self.root, "ITOP_" + str(self.db) + "_" + str(self.mode) + "_labels.h5")
        if not os.path.exists(label_path):
            print('Error: annotation file {} does not exist'.format(annot_file))
            return False

        # Check precomputed centers by v2v-hand model's author(referrence)
        ref_path = os.path.join(self.center_dir, "ITOP_" + str(self.db) + "_view", "center_" + str(self.mode) + ".txt")
        if not os.path.exists(ref_path):
            print('Error: precomputed center files do not exist')
            return False

        return True
