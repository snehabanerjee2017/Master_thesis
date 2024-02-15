import datetime
import logging
import sys
from pathlib import Path
import time

import numpy as np
import torch
torch.cuda.is_available()
import os
# print(torch.cuda.get_device_name(1))
from tensorboardX import SummaryWriter
from tqdm import tqdm
from lib.dbscan_utils import validate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.parse import parse_args, load_model
from dataset.dataloader import data_loader
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.spatial.distance import directed_hausdorff
from DBCV.DBCV_multiproc import DBCV
# from DBCV.DBCV import DBCV

config = parse_args()
test_data_loader = data_loader(config)
pca = PCA(n_components=config['pca']['n_components'])
tsne = TSNE(n_components=config['tsne']['n_components'],n_jobs=-1)
with torch.cuda.device(config['util']['gpu']):
    ema_net = load_model(config)
    all_points = []
    start = time.time()
    for batch_id, test_data in tqdm(enumerate(test_data_loader, 0), total=len(test_data_loader), smoothing=0.9,desc = 'test dataset'):
        x1_test = test_data[0]
        x1_test = x1_test.cuda()
        x1_test = x1_test.transpose(2, 1)
        emb =  ema_net(x1_test,return_embedding=config['training']['return_embedding'])
        all_points.append(emb[0].detach().cpu().numpy())
    del ema_net
    del test_data_loader
    all_points = np.concatenate(all_points,axis=0)
    np.random.seed(42)
    all_points = np.take(all_points, np.random.choice(np.array(list(range(0,all_points.shape[0]))), config['dbcv']['num_points'], replace=False), axis=0, out=None, mode='raise')
    end = time.time()
    print(f'Dimension of dataset {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
    start = time.time()
    all_points = pca.fit_transform(all_points)
    end = time.time()
    print(f'Dimesnion after PCA {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
    start = time.time()
    all_points = tsne.fit_transform(all_points)
    end = time.time()
    print(f'Dimesnion after TSNE {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
    clf = HDBSCAN(min_cluster_size=2,allow_single_cluster=True,n_jobs=-1,store_centers='medoid')
    start = time.time()
    pred_val = clf.fit_predict(all_points)
    num_clusters = len(set(pred_val)) - (1 if -1 in pred_val else 0)  # excluding outliers
    end = time.time()
    print(f'Clustering validation dataset took {end-start} seconds')
    start = time.time()
    dbcv_val = DBCV(all_points,pred_val)
    print(dbcv_val)
    end = time.time()
    print(f'DBCV took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours for test dataset')
        