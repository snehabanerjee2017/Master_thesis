import argparse
import yaml
from model.encoders import PointNet, DGCNN, SCGEncoder
from model.model import MLP, SiamCluster
from scipy.spatial.distance import directed_hausdorff
import torch
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, HDBSCAN, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
from multiprocessing import Pool,cpu_count
from functools import partial
from DBCV.DBCV_multiproc import DBCV
# from DBCV.DBCV_neighbor import DBCV
# from DBCV.DBCV import DBCV

def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument("-c", "--config", required=True,
                    help="JSON configuration string for this operation")
    
    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)
    return config

def load_model(config,logger=None):
    dims = config['model']['dims']
    l_dim = dims
    if config['model']['md'] == 'pn':
        net = PointNet(dims=config['model']['dims'], is_normal=False, feature_transform=True)
    elif config['model']['md'] == 'scg':
        dims = 512 * 3
        l_dim = 512 * 2
        net = SCGEncoder(last_dims=config['model']['dims'], is_normal=config['training']['normal'], n_rkhs=512)
    elif config['model']['md'] == 'dg':
        net = DGCNN(dims=dims, k=config['model']['neighs'])
    else:
        raise NotImplementedError
    net = net.cuda()
    projector = MLP(in_size=dims, out_size=config['model']['proj_dim']).cuda()
    predictor = MLP(in_size=config['model']['proj_dim'], out_size=config['model']['proj_dim'], hidden_size=512, used='pred').cuda()
    # decoder = FoldingNet(dims, k=32).cuda()
    # decoder = DecoderFC(latent_dim=dims, output_pts=config['dataset']['num_]).cuda()
    decoder = None
    ema_net = SiamCluster(net, projector, predictor, dim=l_dim, clusters=config['model']['K'],
                        tau=config['model']['tau'], l_type=config['training']['l_type'], decoder=decoder).cuda()
    try:
        checkpoint = torch.load(config['model']['path'])
        ema_net.load_state_dict(checkpoint['model_state_dict'])
        if logger is not None:
            logger.info('Use pretrain model')
    except Exception as e:
        if logger is not None:
            logger.info('No existing model, starting training from scratch {}'.format(e))
    if logger is not None:
        return ema_net, logger
    else:
        return ema_net
    
def calculate_similarity(points,rep_objs):
    mins = []
    for point in tqdm(points):
        hds = []
        for rep_obj in rep_objs:
            haus_dist = directed_hausdorff(point, rep_obj)
            hds.append(haus_dist)
        mins.append(min(hds))
    return mins
    
def train(config, logger, train_loader, val_loader, checkpoints_dir,exp='sinkhorn'):
    """MODEL LOADING"""    
    with torch.cuda.device(config['util']['gpu']):
        ema_net, logger = load_model(config,logger)
        start_epoch = 0
        global_epoch = 0
        global_step = 0
        best_loss = float('inf')
        if config['training']['early_stopping']:
            early_stopping_counter = 0
        best_val_loss = float('inf')
        if config['training']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(ema_net.parameters(), lr=config['training']['learning_rate'], betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=config['training']['decay_rate'])
        elif config['training']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(ema_net.parameters(), lr=config['training']['learning_rate'], betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=config['training']['decay_rate'], amsgrad=False)
        else:
            optimizer = torch.optim.SGD(ema_net.parameters(), lr=0.01, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['lr_decay'], patience=config['training']['lr_patience'])
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=config['training']['lr_decay'])
        is_con = False
        writer = SummaryWriter(str(checkpoints_dir) + '/log')
        if exp == 'contras':
            is_con = True
        start = time.time()
        for epoch in tqdm(range(start_epoch, config['training']['epoch']), total=config['training']['epoch']):
            logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, config['training']['epoch']))
            train_loss = []
            s_loss = list()
            for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9,desc = 'training dataset'):
                x1, x2 = data
                x1 = x1.cuda()
                x1 = x1.transpose(2, 1)
                x2 = x2.cuda()
                x2 = x2.transpose(2, 1)
                optimizer.zero_grad()
                ema_net = ema_net.train()
                g_loss, l_loss = ema_net(x1, x2, is_con=is_con)
                loss = g_loss + 0.5 * l_loss
                loss.backward()
                optimizer.step()
                global_step += 1
                s_loss.append(l_loss.item())
                train_loss.append(loss.item())
                niter = epoch * len(train_loader) + batch_id
                writer.add_scalars(f'{config["training"]["log_dir"]}_Loss', {'train_loss': loss.data.item()}, niter)
            mean_loss = np.mean(train_loss)
            # scheduler.step()
            logger.info('Train mean loss:{}, assistant loss: {} \n'.format(mean_loss, np.mean(s_loss)))
            print('Train mean loss:{}, assistant loss: {}'.format(mean_loss, np.mean(s_loss)))
            writer.add_scalars(f'{config["training"]["log_dir"]}_Loss', {'train_loss_mean': mean_loss}, epoch)
            val_loss = []  # Initialize the list to store val losses
            for batch_id, val_data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9,desc = 'validation dataset'):
                x1_val, x2_val = val_data
                x1_val = x1_val.cuda()
                x1_val = x1_val.transpose(2, 1)
                x2_val = x2_val.cuda()
                x2_val = x2_val.transpose(2, 1)
                ema_net.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    g_loss_val, l_loss_val = ema_net(x1_val, x2_val, is_con=is_con)
                    loss_val = g_loss_val + 0.5 * l_loss_val
                    val_loss.append(loss_val.item())
                    niter = epoch * len(train_loader) + batch_id
                    writer.add_scalars(f'{config["training"]["log_dir"]}_Loss', {'val_loss': loss_val}, niter)                    
            mean_val_loss = np.mean(val_loss)
            scheduler.step(mean_val_loss)
            logger.info('val mean loss: {} \n'.format(mean_val_loss))
            print('val mean loss: {}'.format(mean_val_loss))
            writer.add_scalars(f'{config["training"]["log_dir"]}_Loss', {'val_loss_mean': mean_val_loss}, epoch)
            ema_net.train()
            if config['training']['early_stopping']:
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    early_stopping_counter = 0
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/self_best_model.pth'
                    logger.info('Saving at %s' % savepath)
                    print('Saving at %s' % savepath)
                    state = {
                        'best_val_loss': best_val_loss,
                        'model_state_dict': ema_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= config['training']['early_stopping_patience']:
                    print("Early stopping triggered. Training stopped.")
                    logger.info("Early stopping triggered. Training stopped.")
                    end = time.time()
                    logger.info(f'Training {epoch} epochs took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
                    break
            else:
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/self_best_model.pth'
                    logger.info('Saving at %s' % savepath)
                    print('Saving at %s' % savepath)
                    state = {
                        'best_val_loss': best_val_loss,
                        'model_state_dict': ema_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)

            global_epoch += 1          

        end = time.time()
        print(f'Training {global_epoch} epochs took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')


def get_pca(n_components:int,data:np.ndarray):
    pca = PCA(n_components=n_components)
    start = time.time()
    data = pca.fit_transform(data)
    end = time.time()
    print(f'Dimesnion after PCA {data.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
    return data

def get_tsne(n_components:int,data:np.ndarray):
    tsne = TSNE(n_components=n_components,n_jobs=-1)
    start = time.time()
    data = tsne.fit_transform(data)
    end = time.time()
    print(f'Dimesnion after TSNE {data.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')
    return data

def get_clusters(data:np.ndarray,store_centers:str = 'medoid',classifier:str='hdbscan',eps:float=0.5,min_samples:int=5,n_clusters:int=175,batch_size:int=50000):
    print(f'{classifier} Clustering')
    if classifier == 'hdbscan':
        clf = HDBSCAN(min_cluster_size=min_samples,n_jobs=-1,store_centers=store_centers)
    # elif classifier == 'dbscan':
    #     clf = DBSCAN(eps=eps,min_samples=min_samples,n_jobs=-1)
    elif classifier == 'kmeans':
        clf = MiniBatchKMeans(n_clusters=n_clusters,random_state=42,init='k-means++',batch_size=batch_size,n_init=1)
    # elif classifier == 'kmedoids':
    #     clf = KMedoids(n_clusters=n_clusters,random_state=42,init='k-medoids++')
    elif classifier == 'agglomerative':
        clf = AgglomerativeClustering(n_clusters = n_clusters)
    elif classifier == 'spectral':
        clf = SpectralClustering(n_clusters=n_clusters,random_state=42,assign_labels='cluster_qr')
    else:
        raise Exception(f"{classifier} not implemented")
    start = time.time()
    pred_val = clf.fit_predict(data)
    if classifier in ['kmeans', 'kmedoids', 'agglomerative','spectral']:
        num_clusters=n_clusters
    else:
        num_clusters = len(set(pred_val)) - (1 if -1 in pred_val else 0)  # excluding outliers
        print(f'The value of min_cluster_size is {min_samples}')
        print(f"Number of outliers {pred_val.tolist().count(-1)}")    
    
    print(f"Number of clusters excluding outliers {num_clusters}")
    end = time.time()
    print(f'Clustering validation dataset took {end-start} seconds')

    return clf, pred_val, num_clusters

def calc_dbcv(data:np.ndarray, labels:np.ndarray):
    start = time.time()
    dbcv_score = DBCV(data,labels)
    print(f"DBCV score is {dbcv_score}")
    end = time.time()
    print(f'DBCV took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours for {data.shape[0]} datapoints')

    return dbcv_score


def get_hier_clusters(clf, all_points:np.ndarray, original_clusters:np.ndarray, clusters:list, cluster_members:list, rel_points: np.ndarray, min_samples:int = 2, classifier = 'hdbscan',count:int=0):

    if count!=0:
        clf, pred, num_clusters = get_clusters(data = rel_points,store_centers = 'medoid', classifier=classifier,min_samples=min_samples) 
    all_medoids = clf.medoids_

    print(all_medoids.shape)
     
    if count!=0:
        cluster_pointer = num_clusters
        # No new outliers from label 2
        for i, label in enumerate(pred):
            if label==-1:
                for j, cluster in enumerate(clusters[1:]):
                    members = cluster_members[j]
                    if np.any(members==rel_points[i]):
                        for member in members:
                            idx =  np.where(all_points==member)[0][0] 
                            original_clusters[idx] = cluster_pointer
                        cluster_pointer+=1
                        break

        # reassign all members of the original clusters
        for j, point in enumerate(rel_points):
            if pred[j]!=-1:
                for i, cluster in enumerate(clusters[1:]):
                    members = cluster_members[i]
                    if np.any(members==point):                            
                        for member in members:
                            idx =  np.where(all_points==member)[0][0] 
                            original_clusters[idx] = pred[j]
                        break
        clusters = np.unique(original_clusters).tolist()
        cluster_members = []
        for cluster in clusters:
            idx = (original_clusters == cluster).nonzero()[0]
            cluster_points = np.take(all_points,idx,axis=0)
            cluster_members.append(cluster_points)
        print(f'Now number of clusters are {len(np.unique(original_clusters).tolist())}') 
        print(f'The number are outliers in level {count} are {original_clusters.tolist().count(-1)}')

    all_medoid_indices = get_medoid_indices(medoids=all_medoids,data=rel_points)

    rel_points = np.take(rel_points,all_medoid_indices,axis=0)
    print(f'Dimension of embedding of all representative  objects {rel_points.shape}')

    if count==0:
        num_clusters = len(np.unique(original_clusters).tolist())
    return rel_points, original_clusters, clusters, cluster_members, num_clusters, clf
        
        

def calculate_medoid(cluster_points):
    """
    Function to calculate the medoid of a cluster.
    """
    num_points = len(cluster_points)
    distances = np.zeros((num_points, num_points))
    
    # Calculate pairwise distances between points
    for i in range(num_points):
        for j in range(i+1, num_points):
            distances[i][j] = distances[j][i] = directed_hausdorff(cluster_points[i], cluster_points[j])[0]
    
    # Calculate total distance for each point
    total_distances = np.sum(distances, axis=0)
    
    # Find index of point with minimum total distance
    medoid_index = np.argmin(total_distances)
    
    return cluster_points[medoid_index]

def calc_medoid(clusters,all_points,pred_val):

    medoid_dict = {cluster: None for cluster in clusters}    
    for cluster in clusters:
        idx = (pred_val == cluster).nonzero()[0]
        cluster_points = np.take(all_points,idx,axis=0)
        print(f'Number of points in cluster {cluster} is {len(cluster_points)}')
        medoid_dict[cluster] = calculate_medoid(cluster_points)

    return medoid_dict

def get_medoids(data:np.ndarray,pred_labels:np.ndarray):
    start = time.time()
    clusters = np.unique(pred_labels).tolist()
    processes = cpu_count()
    chunk_size = len(clusters)//processes + 1

    chunks = [clusters[i:i+chunk_size] for i in range(0,len(clusters),chunk_size)]
    g = partial(calc_medoid,all_points = data,pred_val=pred_labels)

    with Pool(processes) as p:
        res = p.map(g,chunks)

    result = {}
    for ele in res:
        result.update(ele)

    medoids = np.array(list(result.values()))
    end = time.time()

    print(f'medoid calculation for dataset dimension {data.shape} took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')

    return medoids

def get_medoid_indices(medoids:np.ndarray, data:np.ndarray)->list:
    all_medoid_indices = []
    for medoid in tqdm(medoids):
        if medoid in data:
            all_medoid_indices.append(np.where(data==medoid)[0][0])
        else:
            raise Exception(f'medoid {medoid} not found in embedding of all points')
        
    return all_medoid_indices


