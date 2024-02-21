#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/26/2020 4:35 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : modelnet.py
# @Software: PyCharm
import copy
import os
import random
import sys

import h5py
import torch.utils.data as data
import torch
from typing import Union
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from transforms import *

# modelnet10_label = np.array([2, 3, 9, 13, 15, 23, 24, 31, 34, 36]) - 1

def knn_trans(points, knn=16, metric="euclidean"):
    """
    Args:
      :param knn: default=16
      :param points: Nx3
      :param metric: distance type
    """
    assert (knn > 0)
    kdt = KDTree(points, leaf_size=30, metric=metric)
    # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
    dist, idx = kdt.query(points, k=knn + 1, return_distance=True)
    trans_pts = np.take(points, idx[:, 1: knn], axis=0).mean(1)
    return trans_pts


def _load_data_file(name:str)->np.ndarray:    
    """Loads an hdf5 file as numpy array

    Args:
        name (str): name of the hdf5 file to be loaded

    Returns:
        np.ndarray: A numpy array with all the instances in the hdf5 file
    """    
    with h5py.File(name, 'r') as f:
        data = f['data'][:]
    
    return data

class Abc(data.Dataset):
    def __init__(self, root:str, train:bool=True, unsup:bool=True, use_normal:bool=False, train_val_percent:float = 0.85, train_test_split:float = 0.95, val:bool = False,
                 is_crop:bool=True, n_point:int=2048, aug:str='jitter', angle:float=0.0, p_keep:bool=None):
        """Intitalises the class for the ABC dataset

        Args:
            root (str): path to the directory containing the ABC dataset as hdf5 files
            train (bool, optional): Whenever training is to be performed. Defaults to True.
            unsup (bool, optional): Whether the training in unsupervised. Defaults to True.
            use_normal (bool, optional): Whether the normal information is to be used. Defaults to False.
            train_val_percent (float, optional): The ratio of train-validation split for the training dataset. Defaults to 0.85.
            train_test_split (float, optional): The ratio of train-test split for the entire dataset. Defaults to 0.95.
            val (bool, optional): Whether validation dataset is to be generated. Defaults to False.
            is_crop (bool, optional): Whether random cropping is performed. Defaults to True.
            n_point (int, optional): The number of points inthe point cloud for each object. Defaults to 2048.
            aug (str, optional): The type of augmentation that is to be performed. can be 'jitter', 'jiknn' or 'rotation' Defaults to 'jitter'.
            angle (float, optional): The angle for random rotation along any axis. Defaults to 0.0.
            p_keep (bool, optional): The proportion of points to be kept after random cropping. Defaults to None.

        """        
        super(Abc, self).__init__()
        if p_keep is None:
            p_keep = [0.85, 0.85]
        self.crop = RandomCrop(p_keep=p_keep)
        self.is_crop = is_crop
        if aug == 'jitter':
            self.aug = Jitter(sigma=0.001, clip=0.0025)
        elif aug == 'jiknn':
            self.aug = KNNJitter(sigma=0.001, clip=0.0025, knn=4, metric="euclidean")
        else:
            self.aug = RandomRotation(angle)
        self.n_points = n_point
        self.unsup = unsup
        self.train = train
        self.train_val_percent = train_val_percent
        self.train_test_split = train_test_split
        self.files = os.listdir(root)
        self.all_files = []
        for file in self.files:
            file_path = os.path.join(root,file)
            data = _load_data_file(file_path)
            self.all_files.append(data)
        self.all_points = np.concatenate(self.all_files,axis=0)
        self.indices = np.arange(self.all_points.shape[0], dtype=int)
        random.seed(42)
        random.shuffle(self.indices)
        self.train_indices = np.sort(self.indices[:round(self.all_points.shape[0]*self.train_test_split)])
        self.test_indices = np.sort(self.indices[round(self.all_points.shape[0]*self.train_test_split):])

        random.seed(42)
        random.shuffle(self.train_indices)

        self.val_indices = np.sort(self.train_indices[round(len(self.train_indices)*self.train_val_percent):])
        self.train_indices = np.sort(self.train_indices[:round(len(self.train_indices)*self.train_val_percent)])

        if train and not val:
            self.points = self.all_points[self.train_indices]
            self.points = self.points if self.points.ndim > 2 else np.expand_dims(self.points, axis=0)
        elif train and val:
            self.points = self.points = self.all_points[self.val_indices]
            self.points = self.points if self.points.ndim > 2 else np.expand_dims(self.points, axis=0)
        elif not train:
            self.points = self.all_points[self.test_indices]
            self.points = self.points if self.points.ndim > 2 else np.expand_dims(self.points, axis=0)
        if not use_normal:
            self.points = self.points[:, :, :3]
        print('Successfully load Abc with', self.points.shape[0], 'instances')
            

        self.num = self.points.shape[0]




    def __getitem__(self, index:int)->Union[list,tuple]:
        """Get a single instance of the ABC dataset

        Args:
            index (int): The instance that is to be retrieved

        Returns:
            Union[list,tuple]: return the original and the augmented point cloud is unsup is True. Else returns only the original point cloud as a single element in a list. 
        """        
        raw_num = self.points.shape[1]
        
        pt_idxs = np.arange(0, raw_num)
        if self.train:
            np.random.shuffle(pt_idxs)

        point_set = self.points[index, pt_idxs].copy()        
        # point_set[:, 0:3] = pc_normalize_np(point_set[:, 0:3])
        aug_set = None

        if self.unsup:
            if self.is_crop:           
                sample = {'src': point_set, 'ref': point_set}
                pts_dict = self.crop(sample)
                point_set = pts_dict['src']
                aug_set = pts_dict['ref']
            else:
                aug_set = copy.deepcopy(point_set)
            if self.n_points < aug_set.shape[0]:
                aug_set = farthest_point_sample(aug_set, self.n_points)
            # np.random.shuffle(pt_idxs)
            aug_set = torch.from_numpy(aug_set)
            aug_set[:, 0:3] = self.aug(aug_set[:, 0:3])
            aug_set[:, 0:3] = pc_normalize(aug_set[:, 0:3])
        if self.n_points < raw_num:
            point_set = farthest_point_sample(point_set, self.n_points)
        point_set = torch.from_numpy(point_set)
        if self.train:
            point_set[:, 0:3] = self.aug(point_set[:, 0:3])        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.unsup:
            return point_set, aug_set
        else:
            return [point_set]

    def __len__(self)-> int:        
        """get the number of point clouds in an instance of the ABC dataset

        Returns:
            int: the number of point clouds in the instance of the ABC dataset
        """        
        return self.points.shape[0]


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    # src = np.random.random((100, 3))
    # s = knn_trans(src, p=6, metric="euclidean")
    # d_path = '/data/gmei/data/modelnet40_normal_resampled/dataset/'
    # data = ModelNet(d_path, angle=12)
    d_path = '/mnt/data/das-sb/dataset_hdf5'
    data = Abc(d_path, angle=12, aug='rotation')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=6, shuffle=True)
    count = 0
    for point, points2 in DataLoader:
        count += 1
        print(point.shape)
        print(points2.shape)
        break
        # if count > 10:
        #     break
