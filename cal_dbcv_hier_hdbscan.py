import time
import numpy as np
import torch
torch.cuda.is_available()
from tqdm import tqdm
from utils.parse import parse_args, load_model, get_clusters, get_pca, get_tsne
from dataset.dataloader import data_loader
from DBCV.DBCV_multiproc import DBCV
# from DBCV.DBCV_neighbor import DBCV
# from DBCV.DBCV import DBCV

config = parse_args()
test_data_loader = data_loader(config)
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
    end = time.time()
    np.random.seed(42)
    all_points = np.take(all_points, np.random.choice(np.array(list(range(0,all_points.shape[0]))), config['dbcv']['num_points'], replace=False), axis=0, out=None, mode='raise')
    
    print(f'Dimension of dataset {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
    
    all_points = get_pca(n_components=config['pca']['n_components'],data=all_points)
    
    # all_points = get_pca(n_components=config['pca_2']['n_components'],data=all_points)

    all_points = get_tsne(n_components=config['tsne']['n_components'], data=all_points)

    rel_points = all_points.copy()

    clf, pred, num_clusters = get_clusters(data = rel_points,store_centers = 'medoid', classifier=config['results']['classifier'],min_samples=config['results']['min_samples']) 

    original_clusters = pred.copy()
    original_cluster_members = []
    clusters = np.unique(original_clusters).tolist()
    for cluster in clusters:
        idx = (original_clusters == cluster).nonzero()[0]
        cluster_points = np.take(all_points,idx,axis=0)
        original_cluster_members.append(cluster_points)

    count=0

    while True:
        if count!=0:
            clf, pred, num_clusters = get_clusters(data = rel_points,store_centers = 'medoid', classifier=config['results']['classifier'],min_samples=config['results']['min_samples']) 
        all_medoids = clf.medoids_

        print(all_medoids.shape)
        cluster_pointer = num_clusters 
        if count!=0:
            for i, label in enumerate(pred):
                if label==-1:
                    for j, cluster in enumerate(clusters[1:]):
                        cluster_members = original_cluster_members[j]
                        if np.any(cluster_members==rel_points[i]):
                            for member in cluster_members:
                                idx =  np.where(all_points==member)[0][0] 
                                original_clusters[idx] = cluster_pointer
                            cluster_pointer+=1
                            break
        for j, point in enumerate(rel_points):
            if pred[j]!=-1:
                for i, cluster in enumerate(clusters[1:]):
                    cluster_members = original_cluster_members[i]
                    if np.any(cluster_members==point):
                        idx_point = np.where(rel_points==point)[0][0]
                        label_point = pred[idx_point]
                            
                        for member in cluster_members:
                            idx =  np.where(all_points==member)[0][0] 
                            original_clusters[idx] = label_point
                        break
        print(f'Now number of clusters are {len(np.unique(original_clusters).tolist())}') 
        print(original_clusters)

        start = time.time()
        dbcv_score = DBCV(all_points,original_clusters)
        print(f"DBCV score is {dbcv_score}")
        end = time.time()
        print(f'DBCV took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours for {all_points.shape[0]} datapoints')
        if num_clusters<=config['results']['n_clusters']:
            break
        else:
            all_medoid_indices = []
            for medoid in tqdm(all_medoids):
                if medoid in rel_points:
                    all_medoid_indices.append(np.where(rel_points==medoid)[0][0])
                else:
                    raise Exception(f'medoid {medoid} not found in embedding of all points')
                
            rel_points = np.take(rel_points,all_medoid_indices,axis=0)
            print(f'Dimension of embedding of all representative  objects {rel_points.shape}')
            count+=1
