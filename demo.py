import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import h5py
import matplotlib.pyplot as plt
import cv2

from src.v2v_model import V2VModel
from src.v2v_util import V2VVoxelization
from estimate_center.model import resnet18
from mpl_toolkits import mplot3d

# load depthmap
# 这里以存储在h5文件内部的深度图为例
# 如果采用RealSense需要考虑分辨率的为题
def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z

def depth2point(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points

def points2pixels(h, w, keypoints, fx, fy):
    pixels = np.zeros((15, 2), dtype=np.float32)
    pixels[:, 0], pixels[:, 1] = keypoints[:, 0] * fx / keypoints[:, 2] + w / 2, h / 2 - keypoints[:, 1] * fy / keypoints[:, 2]
    return pixels

def get_realworld_points(depthmaps, fx, fy, is_h5, fileID):
    if is_h5:
        depthmap = depthmaps[fileID]
    else:
        depthmap = depthmaps
    realpoints = depth2point(depthmap, fx, fy)
    return realpoints

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", default=0, type=int, help="depth map file id choice")
    parser.add_argument("--is_h5", action='store_true', default=False, help="flag use")
    parser.add_argument("--fx", default=285.71, type=float, help="camera internal parameter fx")
    parser.add_argument("--fy", default=285.71, type=float, help="camera internal parameter fy")
    parser.add_argument("--custom_data", default="custom_image_data/mytest.npy", type=str, help="custom data select")
    args = parser.parse_args()
    fileID = args.fid
    is_h5 = args.is_h5
    fx = args.fx
    fy = args.fy
    cus_data = args.custom_data
    return fileID, is_h5, fx ,fy, cus_data

def config_cuda(device):
    if device == torch.device('cuda'):
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        print('cudnn.enabled: ', torch.backends.cudnn.enabled)

def plot_2Dresult(depthmap, keypoints, fx, fy):
    h, w = depthmap.shape
    pixels = points2pixels(h, w, keypoints, fx, fy)
    depthmap = ((depthmap - np.min(depthmap)) / (np.max(depthmap) - np.min(depthmap))).astype(np.float)
    fig = plt.figure(1)
    plt.imshow(depthmap)
    plt.plot(pixels[:, 0], pixels[:, 1], 'o')
    plt.show()

def plot_3Dresult(depthmap, keypoints, fx, fy):
    realpoints = depth2point(depthmap, fx, fy)
    keypoints = keypoints.transpose(1, 0)
    fig = plt.figure(2)
    ax = plt.axes(projection='3d')
    print(keypoints[2,:])
    # ax.scatter3D(realpoints[:,:,0].flatten(), realpoints[:,:,1].flatten(), realpoints[:,:,2].flatten(), c=realpoints[:,:,0].flatten(), cmap='Greens')
    ax.scatter3D(keypoints[0,:], keypoints[1,:], keypoints[2,:], c=keypoints[2,:], cmap='Greens')
    ax.plot3D([keypoints[0,0], keypoints[0,1]], [keypoints[1,0],keypoints[1,1]], [keypoints[2,0],keypoints[2,1]], 'b') # neck to head

    ax.plot3D([keypoints[0,1], keypoints[0,2]], [keypoints[1,1],keypoints[1,2]], [keypoints[2,1],keypoints[2,2]], 'c') # neck to r_shoulder
    ax.plot3D([keypoints[0,2], keypoints[0,4]], [keypoints[1,2],keypoints[1,4]], [keypoints[2,2],keypoints[2,4]], 'g') # r_shoulder to r_elbow
    ax.plot3D([keypoints[0,4], keypoints[0,6]], [keypoints[1,4],keypoints[1,6]], [keypoints[2,4],keypoints[2,6]], 'k') # r_elbow to r_hand
    
    ax.plot3D([keypoints[0,1], keypoints[0,3]], [keypoints[1,1],keypoints[1,3]], [keypoints[2,1],keypoints[2,3]], 'm') # neck to l_shoulder
    ax.plot3D([keypoints[0,3], keypoints[0,5]], [keypoints[1,3],keypoints[1,5]], [keypoints[2,3],keypoints[2,5]], 'r') # l_shoulder to l_elbow
    ax.plot3D([keypoints[0,5], keypoints[0,7]], [keypoints[1,5],keypoints[1,7]], [keypoints[2,5],keypoints[2,7]], 'y') # l_elbow to l_hand

    ax.plot3D([keypoints[0,1], keypoints[0,8]], [keypoints[1,1],keypoints[1,8]], [keypoints[2,1],keypoints[2,8]], 'c') # neck to torso

    ax.plot3D([keypoints[0,8], keypoints[0,9]], [keypoints[1,8],keypoints[1,9]], [keypoints[2,8],keypoints[2,9]], 'b') # torso to r_hip
    ax.plot3D([keypoints[0,9], keypoints[0,11]], [keypoints[1,9],keypoints[1,11]], [keypoints[2,9],keypoints[2,11]], 'k') # r_hip to r_knee
    ax.plot3D([keypoints[0,11], keypoints[0,13]], [keypoints[1,11],keypoints[1,13]], [keypoints[2,11],keypoints[2,13]], 'r') # r_knee to r_foot

    ax.plot3D([keypoints[0,8], keypoints[0,10]], [keypoints[1,8],keypoints[1,10]], [keypoints[2,8],keypoints[2,10]], 'b') # torso to r_hip
    ax.plot3D([keypoints[0,10], keypoints[0,12]], [keypoints[1,10],keypoints[1,12]], [keypoints[2,10],keypoints[2,12]], 'k') # r_hip to r_knee
    ax.plot3D([keypoints[0,12], keypoints[0,14]], [keypoints[1,12],keypoints[1,14]], [keypoints[2,12],keypoints[2,14]], 'r') # r_knee to r_foot
    plt.show()


if __name__ == "__main__":
    fileID, is_h5, fx, fy, cus_data = get_parse()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if is_h5:
        depthmap_path = "datasets/depthmap/ITOP_side_test_depth_map.h5"
        depthmaps = (h5py.File(depthmap_path))['data']
        realpoints = get_realworld_points(depthmaps, fx, fy, is_h5, fileID)
    else:
        depthmaps = np.load(cus_data) / 1000
        depthmaps = cv2.resize(depthmaps, (320, 240), interpolation=cv2.INTER_AREA)
        realpoints = get_realworld_points(depthmaps, fx, fy, is_h5, fileID)

    input2resnet = torch.from_numpy(realpoints)
    input2resnet = (input2resnet.permute(2, 0, 1).unsqueeze(dim=0)).to(device)
    resnet = (resnet18()).to(device)
    resnet.load_state_dict(torch.load("estimate_center/pretrain/epoch90.pth")["model_state_dict"])
    resnet.eval()
    refpoint = (resnet(input2resnet)).squeeze()
    # 现在将点转为体素
    print(refpoint.cpu().detach().numpy())
    voxelization_val = V2VVoxelization(cubic_size=2.0, augmentation=False)
    input2v2v = voxelization_val.voxelize(realpoints.reshape(-1, 3), refpoint.cpu().detach().numpy())
    # V2V 模型加载
    net = V2VModel(input_channels=1, output_channels=15)
    ## load weights
    state_dicts = torch.load("./experiments/itop_experiment/checkpoint/epoch14.pth")
    net.load_state_dict(state_dicts['model_state_dict'])
    net = net.to(device, torch.float)
    config_cuda(device)
    ## 数据输入
    outputs = net(torch.from_numpy(input2v2v).to(device, torch.float).unsqueeze(dim=0))
    keypoints = np.squeeze(voxelization_val.evaluate(outputs.cpu().detach().numpy(), refpoint.cpu().detach().numpy()), 0)
    if is_h5:
        # plot_2Dresult(depthmaps[fileID], keypoints, fx, fy)
        plot_3Dresult(depthmaps[fileID], keypoints, fx, fy)
    else:
        plot_2Dresult(depthmaps, keypoints, fx, fy)
        plot_3Dresult(depthmaps, keypoints, fx, fy)
    # print(keypoints)
    # label_path = "datasets/depthmap/ITOP_side_test_labels.h5"
    # label = (h5py.File(label_path))['real_world_coordinates']
    # print(label[fileID] - keypoints)

    



