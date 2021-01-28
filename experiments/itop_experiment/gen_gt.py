##%%
import os
import h5py
import numpy as np
import sys
import struct
import torch
from datasets.itop_person import ITOPDataset
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

# ##%%   here
# # Generate train_subject3_gt.txt and test_subject3_gt.txt
# data_dir = './datasets'
# center_dir = './datasets/center/ITOP_center'
# db = 'side'


##%%
def save_keypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    print(keypoints.shape)
    np.savetxt(filename, keypoints, fmt='%0.4f')


# ##%% experiments\itop_experiment   here
# train_dataset = ITOPDataset(root=data_dir, center_dir=center_dir, db=db, mode='train')
# names, joints_world, ref_pts = train_dataset.get_data()
# print('save train reslt ..')
# save_keypoints('./experiments/itop_experiment/train_side_gt.txt', joints_world)
# # save_keypoints('./experiments/itop_experiment/train_side_res.txt', ref_pts)
# print('done ..')


# ##%%
# test_dataset = ITOPDataset(root=data_dir, center_dir=center_dir, db=db, mode='test')
# names, joints_world, ref_pts = test_dataset.get_data()
# print('save test reslt ..')
# save_keypoints('./experiments/itop_experiment/test_side_gt.txt', joints_world)
# # save_keypoints('./experiments/itop_experiment/test_side_res.txt', ref_pts)
# print('done ..')
