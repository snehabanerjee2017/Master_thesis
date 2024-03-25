import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.is_available()
import os
from tqdm import tqdm
from utils.parse import parse_args, load_model, get_pca, get_tsne, get_clusters, get_hier_clusters, get_medoid_indices
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

red_emb_all_points = get_pca(n_components=config['pca']['n_components'],data=emb_all_points)

# red_emb_all_points = get_pca(n_components=config['pca_2']['n_components'], data=red_emb_all_points)

red_emb_all_points = get_tsne(n_components=config['tsne']['n_components'], data=red_emb_all_points)

red_emb_rel_points = red_emb_all_points.copy()

clf, pred, num_clusters = get_clusters(data = red_emb_rel_points,store_centers = 'medoid', classifier=config['results']['classifier'],min_samples=config['results']['min_samples']) 
original_clusters = pred.copy()
cluster_members = []
clusters = np.unique(original_clusters).tolist()
for cluster in clusters:
    idx = (original_clusters == cluster).nonzero()[0]
    cluster_points = np.take(red_emb_all_points,idx,axis=0)
    cluster_members.append(cluster_points)

count=0

while True:
    red_emb_rel_points, original_clusters, num_clusters, clf = get_hier_clusters(clf = clf, all_points=red_emb_all_points,rel_points=red_emb_rel_points,  original_clusters=original_clusters, clusters = clusters, cluster_members=cluster_members, min_samples=config['results']['min_samples'], classifier=config['results']['classifier'],count=count)

    with open(os.path.join(config['results']['dir_path'],f"all_medoids_{config['results']['classifier']}_tsne.npy"), 'wb') as f:
        np.save(f, clf.medoids_)
    all_medoids = np.load(os.path.join(config['results']['dir_path'],f"all_medoids_{config['results']['classifier']}_tsne.npy"))

    all_medoid_indices = get_medoid_indices(medoids=all_medoids,data=red_emb_all_points)
    all_rep_emb = np.take(emb_all_points,all_medoid_indices,axis=0)
    print(f'Dimension of embedding of all representative  objects {all_rep_emb.shape}')

    all_red_rep_emb = np.take(red_emb_all_points,all_medoid_indices,axis=0)
    print(f'Dimension of reduced embedding embedding of all representative  objects {all_red_rep_emb.shape}')

    with open(os.path.join(config['results']['dir_path'],f"all_rep_emb_{config['results']['classifier']}_tsne.npy"), 'wb') as f:
        np.save(f, all_rep_emb)

    all_rep_objects = np.take(all_points,all_medoid_indices,axis=0)
    print(f'Dimension of all representative  objects {all_rep_objects.shape}')

    with open(os.path.join(config['results']['dir_path'],f"all_rep_objects_{config['results']['classifier']}_tsne.npy"), 'wb') as f:
        np.save(f, all_rep_objects)

    if num_clusters<=config['results']['n_clusters']:
        break
    else:
        emb_all_points = all_rep_emb
        red_emb_all_points = all_red_rep_emb
        all_points = all_rep_objects
        count+=1

        