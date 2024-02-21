import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.is_available()
import os
from tqdm import tqdm
from utils.parse import parse_args, load_model, get_pca, get_tsne, get_clusters
from dataset.dataloader import data_loader
import gc   

config = parse_args()
os.makedirs(config['results']['dir_path'],exist_ok=True)
train_data_loader, val_data_loader = data_loader(config)
with torch.cuda.device(config['util']['gpu']):
    ema_net = load_model(config)
    emb_all_points = []
    all_points = []
    for batch_id, train_data in tqdm(enumerate(train_data_loader, 0), total=len(train_data_loader), smoothing=0.9,desc = 'train dataset'):
        x1, _ = train_data
        x1 = x1.cuda()
        x1 = x1.transpose(2, 1)
        emb =  ema_net(x1,return_embedding=config['training']['return_embedding'])
        emb_all_points.append(emb[0].detach().cpu().numpy())
        all_points.append(train_data[0].detach().cpu().numpy())
    del train_data_loader
    gc.collect()
    torch.cuda.empty_cache() 
    for batch_id, val_data in tqdm(enumerate(val_data_loader, 0), total=len(val_data_loader), smoothing=0.9,desc = 'val dataset'):
        x1, _ = val_data
        x1 = x1.cuda()
        x1 = x1.transpose(2, 1)
        emb =  ema_net(x1,return_embedding=config['training']['return_embedding'])
        emb_all_points.append(emb[0].detach().cpu().numpy())
        all_points.append(val_data[0].detach().cpu().numpy())
    del val_data_loader
    gc.collect()
    torch.cuda.empty_cache() 
del ema_net
gc.collect()
torch.cuda.empty_cache() 
all_points = np.concatenate(all_points,axis=0)
emb_all_points = np.concatenate(emb_all_points,axis=0)
print(f'Dimension of dataset {all_points.shape}')
print(f'Dimension of dataset embedding {emb_all_points.shape}')

emb_all_points = get_pca(n_components=config['pca']['n_components'],data=emb_all_points)

emb_all_points = get_tsne(n_components=config['tsne']['n_components'], data=emb_all_points)

clf, pred = get_clusters(data = emb_all_points,store_centers = 'medoid', classifer='hdbscan')

with open(os.path.join(config['results']['dir_path'],'all_medoids.npy'), 'wb') as f:
    np.save(f, clf.medoids_)
all_medoids = np.load(os.path.join(config['results']['dir_path'],'all_medoids.npy'))

all_medoid_indices = []
for medoid in tqdm(all_medoids):
    if medoid in emb_all_points:
        all_medoid_indices.append(np.where(emb_all_points==medoid)[0][0])
    else:
        raise Exception(f'medoid {medoid} not found in embedding of all points')
    
all_rep_emb = np.take(emb_all_points,all_medoid_indices,axis=0)
print(f'Dimension of embedding of all representative  objects {all_rep_emb.shape}')

with open(os.path.join(config['results']['dir_path'],'all_rep_emb.npy'), 'wb') as f:
    np.save(f, all_rep_emb)

all_rep_objects = np.take(all_points,all_medoid_indices,axis=0)
print(f'Dimension of all representative  objects {all_rep_objects.shape}')

with open(os.path.join(config['results']['dir_path'],'all_rep_objects.npy'), 'wb') as f:
    np.save(f, all_rep_objects)