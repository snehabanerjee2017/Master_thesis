import numpy as np
import time
import torch
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import DBCV
from cdbw import CDbw

def evaluate_dbscan(train_features, val_features, md = 'hdbscan', cb = False):
    if md == 'hdbscan':
        clf = HDBSCAN()
    elif md == 'dbscan':
        clf = DBSCAN()
    
    start = time.time()
    pred_train = clf.fit_predict(train_features)
    end = time.time()
    print(f'Clustering training dataset took {end-start} seconds')
    start = time.time()
    pred_val = clf.fit_predict(val_features)
    num_clusters = num_clusters = len(set(pred_val)) - (1 if -1 in pred_val else 0)  # excluding outliers
    end = time.time()
    print(f'Clustering valing dataset took {end-start} seconds')

    if cb:        
        cb_train = CDbw(X=train_features, labels=pred_train, metric="euclidean")
        cb_val = CDbw(X=val_features, labels=pred_val, metric="euclidean")

        return cb_train, cb_val, num_clusters

    # if dbcv:
    #     dbcv_train = DBCV.DBCV(train_features,pred_train)
    #     dbcv_val = DBCV.DBCV(train_features,pred_val)

    #     return dbcv_train, dbcv_val, num_clusters

    else:

        # Silhouette Score
        silhouette_avg_train = silhouette_score(X=train_features, labels=pred_train)
        silhouette_avg_val = silhouette_score(X=val_features, labels=pred_val)

        # Davies-Bouldin Index
        db_index_train = davies_bouldin_score(X=train_features, labels=pred_train)
        db_index_val = davies_bouldin_score(X=val_features, labels=pred_val)
        return silhouette_avg_train, db_index_train, silhouette_avg_val, db_index_val, num_clusters


def dbscan_data(loader, encoder):
    encoder.eval()
    features = list()
    # label = list()
    for _, data in enumerate(loader, 0):
        points= data[0]                             # the training dataset returns the original points and the augmented points because unsup is true
        points = points.cuda()
        feature = encoder.backbone(points.transpose(2, 1), True)
        features.append(feature[0])
    features = torch.cat(features, dim=0)
    # label = torch.cat(label, dim=0)

    return features


def validate(train_loader, val_loader, encoder, logger,epoch, best_silhouette_avg_train = None, best_db_index_train = None,best_silhouette_avg_val = None, best_db_index_val = None, best_cb_train = None, best_cb_val = None, md='hdbscan', cb = True):
    # feature extraction
    with torch.no_grad():
        train_features = dbscan_data(train_loader, encoder)
        val_features = dbscan_data(val_loader, encoder)

    if cb:
        cb_train, cb_val, num_clusters = evaluate_dbscan(train_features.data.cpu().numpy(), val_features.data.cpu().numpy(), md)
        if cb_val > best_cb_val :
            cb_val = cb_val

        if cb_train > best_cb_train:
            cb_train = cb_train

        encoder.train()
        logger.info(f'Epoch {epoch} Clustering results: cb train={cb_train},\t best cb train={best_cb_train}')
        logger.info(f'Epoch {epoch} Clustering results: cb val={cb_val},\t best cb val={best_cb_val}')
        print(f'Clustering results: cb train={cb_train},\t best cb train={best_cb_train}')
        print(f'Clustering results: cb val={cb_val},\t best cb val={best_cb_val}')
        return cb_train, cb_val, num_clusters

    else:
        # train svm
        silhouette_avg_train, db_index_train, silhouette_avg_val, db_index_val, num_clusters = evaluate_dbscan(train_features.data.cpu().numpy(), val_features.data.cpu().numpy(), md, cb = False)

        if silhouette_avg_val > best_silhouette_avg_val and db_index_val < best_db_index_val:
            best_silhouette_avg_val = silhouette_avg_val
            best_db_index_val = db_index_val

        if silhouette_avg_train > best_silhouette_avg_train and db_index_train < best_db_index_train:
            best_silhouette_avg_train = silhouette_avg_train
            best_db_index_train = db_index_train

        encoder.train()
        logger.info(f'Epoch {epoch} Clustering results: silhouette avg train={silhouette_avg_train},\t best silhouette avg train={best_silhouette_avg_train}')
        logger.info(f'Epoch {epoch} Clustering results: db index train={db_index_train},\t best db index train={best_db_index_train}')
        logger.info(f'Epoch {epoch} Clustering results: silhouette avg val={silhouette_avg_val},\t best silhouette avg val={best_silhouette_avg_val}')
        logger.info(f'Epoch {epoch} Clustering results: db index val={db_index_val},\t best db index val={best_db_index_val}')
        print(f'Clustering results: silhouette avg train={silhouette_avg_train},\t best silhouette avg train={best_silhouette_avg_train}')
        print(f'Clustering results: db index train={db_index_train},\t best db index train={best_db_index_train}')
        print(f'Clustering results: silhouette avg val={silhouette_avg_val},\t best silhouette avg val={best_silhouette_avg_val}')
        print(f'Clustering results: db index val={db_index_val},\t best db index val={best_db_index_val}')
        return silhouette_avg_train, db_index_train, silhouette_avg_val, db_index_val, num_clusters
