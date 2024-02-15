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
from utils.parse import parse_args, load_model
from dataset.dataloader import data_loader
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
# from DBCV.DBCV_multiproc import DBCV
from DBCV.DBCV import DBCV
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
import gc   

config = parse_args()
config['training']['train'] = True
train_data_loader, val_data_loader = data_loader(config)
with torch.cuda.device(config['util']['gpu']):
    ema_net = load_model(config)
    all_points = []
    for batch_id, train_data in tqdm(enumerate(train_data_loader, 0), total=len(train_data_loader), smoothing=0.9,desc = 'train dataset'):
        x1, _ = train_data
        x1 = x1.cuda()
        x1 = x1.transpose(2, 1)
        emb =  ema_net(x1,return_embedding=config['training']['return_embedding'])
        all_points.append(emb[0].detach().cpu().numpy())
    del train_data_loader
    gc.collect()
    torch.cuda.empty_cache() 
    for batch_id, val_data in tqdm(enumerate(val_data_loader, 0), total=len(val_data_loader), smoothing=0.9,desc = 'val dataset'):
        x1, _ = val_data
        x1 = x1.cuda()
        x1 = x1.transpose(2, 1)
        emb =  ema_net(x1,return_embedding=config['training']['return_embedding'])
        all_points.append(emb[0].detach().cpu().numpy())
    del val_data_loader
    gc.collect()
    torch.cuda.empty_cache() 
del ema_net
gc.collect()
torch.cuda.empty_cache() 
all_points = np.concatenate(all_points,axis=0)
print(f'Dimension of dataset {all_points.shape}')

pca = PCA(n_components=config['pca']['n_components'])
start = time.time()
all_points = pca.fit_transform(all_points)
end = time.time()
print(f'Dimesnion after PCA {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')

tsne = TSNE(n_components=config['tsne']['n_components'],n_jobs=-1)
start = time.time()
all_points = tsne.fit_transform(all_points)
end = time.time()
print(f'Dimesnion after TSNE {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')

clf = HDBSCAN(min_cluster_size=2,allow_single_cluster=True,n_jobs=-1,store_centers='medoid')
start = time.time()
pred_val = clf.fit_predict(all_points)
num_clusters = num_clusters = len(set(pred_val)) - (1 if -1 in pred_val else 0)  # excluding outliers
end = time.time()
print(f'Clustering validation dataset took {end-start} seconds')

with open('/home/das-sb/GIT/source_library/medoids.npy', 'wb') as f:
    np.save(f, clf.medoids_)