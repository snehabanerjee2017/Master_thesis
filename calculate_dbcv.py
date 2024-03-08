import time
import numpy as np
import torch
torch.cuda.is_available()
from tqdm import tqdm
from utils.parse import parse_args, load_model, get_clusters, get_pca, get_tsne
from dataset.dataloader import data_loader
# from DBCV.DBCV_multiproc import DBCV
from DBCV.DBCV_neighbor import DBCV
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

    clf, pred, num_clusters = get_clusters(data=all_points,store_centers='medoid',classifier=config['results']['classifier'],min_samples=2)
    
    start = time.time()
    dbcv_score = DBCV(all_points,pred)
    print(f"DBCV score is {dbcv_score}")
    end = time.time()
    print(f'DBCV took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours for {all_points.shape[0]} datapoints')
        