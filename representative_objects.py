import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.is_available()
import os
from tqdm import tqdm
from utils.parse import parse_args, load_model, get_pca, get_tsne, get_clusters, get_medoids
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

red_emb_all_points = get_pca(n_components=config['pca']['n_components'],data=np.copy(emb_all_points))

red_emb_all_points = get_pca(n_components=config['pca_2']['n_components'], data=red_emb_all_points)

emb_chunks = [red_emb_all_points[i:i+config['results']['chunk_size']] for i in range(0,len(red_emb_all_points),config['results']['chunk_size'])]
obj_chunks = [all_points[i:i+config['results']['chunk_size']] for i in range(0,len(all_points),config['results']['chunk_size'])]

medoids = []
embs = []
objs = []
for emb_chunk, obj_chunk in tqdm(zip(emb_chunks,obj_chunks), total=len(emb_chunks), smoothing=0.9,desc = 'Chunk of entire dataset'):
    clf, pred, num_clusters = get_clusters(data = emb_chunk,store_centers = 'medoid', classifier=config["results"]["classifier"],n_clusters=config['results']['n_clusters'])

    with open(os.path.join(config['results']['dir_path'],f'all_medoids_{config["results"]["classifier"]}.npy'), 'wb') as f:
        if config["results"]["classifier"] in ['kmedoids', 'kmeans']:
            np.save(f, clf.cluster_centers_)
        elif config["results"]["classifier"] in ['hdbcsan', 'dbscan']:
            np.save(f, clf.medoids_)
        elif config["results"]["classifier"] in ['agglomerative', 'spectral']:
            np.save(f, get_medoids(data=obj_chunk,pred_labels=pred))
        else:
            raise NotImplementedError
    all_medoids = np.load(os.path.join(config['results']['dir_path'],f'all_medoids_{config["results"]["classifier"]}.npy'))
    medoids.extend(all_medoids)

    if config["results"]["classifier"] in ['agglomerative', 'spectral']:
        all_medoid_indices = []
        for medoid in tqdm(all_medoids):
            if medoid in obj_chunk:
                all_medoid_indices.append(np.where(obj_chunk==medoid)[0][0])
            else:
                raise Exception(f'medoid {medoid} not found in embedding of all points')
    else:
        all_medoid_indices = []
        for medoid in tqdm(all_medoids):
            if medoid in emb_chunk:
                all_medoid_indices.append(np.where(emb_chunk==medoid)[0][0])
            else:
                raise Exception(f'medoid {medoid} not found in embedding of all points')
        
    all_rep_emb = np.take(emb_chunk,all_medoid_indices,axis=0)
    print(f'Dimension of embedding of all representative  objects {all_rep_emb.shape}')
    embs.extend(all_rep_emb)

    with open(os.path.join(config['results']['dir_path'],f'all_rep_emb_{config["results"]["classifier"]}.npy'), 'wb') as f:
        np.save(f, all_rep_emb)

    all_rep_objects = np.take(obj_chunk,all_medoid_indices,axis=0)
    print(f'Dimension of all representative  objects {all_rep_objects.shape}')
    objs.extend(all_rep_objects)

    with open(os.path.join(config['results']['dir_path'],f'all_rep_objects_{config["results"]["classifier"]}.npy'), 'wb') as f:
        np.save(f, all_rep_objects)

medoids = np.array(medoids)
embs = np.array(embs)
objs = np.array(objs)

print(f'medoids shape is {medoids.shape}')
print(f'embeddings shape is {embs.shape}')
print(f'objects shape is {objs.shape}')

clf, pred, num_clusters = get_clusters(data = embs,store_centers = 'medoid', classifier=config["results"]["classifier"],n_clusters=config['results']['n_clusters'])

with open(os.path.join(config['results']['dir_path'],f'all_medoids_{config["results"]["classifier"]}.npy'), 'wb') as f:
    if config["results"]["classifier"] in ['kmedoids', 'kmeans']:
        np.save(f, clf.cluster_centers_)
    elif config["results"]["classifier"] in ['hdbcsan', 'dbscan']:
        np.save(f, clf.medoids_)
    elif config["results"]["classifier"] in ['agglomerative', 'spectral']:
        np.save(f, get_medoids(data=objs,pred_labels=pred))
all_medoids = np.load(os.path.join(config['results']['dir_path'],f'all_medoids_{config["results"]["classifier"]}.npy'))

if config["results"]["classifier"] in ['agglomerative', 'spectral']:
    all_medoid_indices = []
    for medoid in tqdm(all_medoids):
        if medoid in objs:
            all_medoid_indices.append(np.where(objs==medoid)[0][0])
        else:
            raise Exception(f'medoid {medoid} not found in embedding of all points')
else:
    all_medoid_indices = []
    for medoid in tqdm(all_medoids):
        if medoid in embs:
            all_medoid_indices.append(np.where(embs==medoid)[0][0])
        else:
            raise Exception(f'medoid {medoid} not found in embedding of all points')
    
all_rep_emb = np.take(embs,all_medoid_indices,axis=0)
print(f'Dimension of embedding of all representative  objects {all_rep_emb.shape}')

with open(os.path.join(config['results']['dir_path'],f'all_rep_emb_{config["results"]["classifier"]}.npy'), 'wb') as f:
        np.save(f, all_rep_emb)

all_rep_objects = np.take(objs,all_medoid_indices,axis=0)
print(f'Dimension of all representative  objects {all_rep_objects.shape}')

with open(os.path.join(config['results']['dir_path'],f'all_rep_objects_{config["results"]["classifier"]}.npy'), 'wb') as f:
    np.save(f, all_rep_objects)
        
