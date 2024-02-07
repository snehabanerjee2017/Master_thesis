#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:39 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : main.py
# @Software: PyCharm
import argparse
import datetime
import logging
import sys
from pathlib import Path
import time

import numpy as np
import torch
import yaml

torch.cuda.is_available()
import os
# print(torch.cuda.get_device_name(1))
from tensorboardX import SummaryWriter
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from lib.dbscan_utils import validate
from model.encoders import PointNet, DGCNN, SCGEncoder
from model.model import MLP, SiamCluster
from dataset.dataloader import data_loader


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument("-c", "--config", required=True,
                    help="JSON configuration string for this operation")
    
    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)
    return config


def train(config, logger, train_loader, val_loader, exp='sinkhorn'):
    """MODEL LOADING"""    
    with torch.cuda.device(config['util']['gpu']):
        dims = config['model']['dims']
        # print(torch.cuda.current_device())
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
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/svm_best_model.pth')
            ema_net.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Use pretrain model')
        except Exception as e:
            logger.info('No existing model, starting training from scratch {}'.format(e))
        start_epoch = 0
        global_epoch = 0
        global_step = 0
        best_loss = float('inf')
        if config['training']['early_stopping']:
            early_stopping_counter = 0
        best_val_loss = float('inf')
        # if config['training']['cb']:
        #     best_cb_train = 0.0
        #     best_cb_val = 0.0
        # else:
        #     best_silhouette_avg_train = -1.0
        #     best_db_index_train = float('inf')
        #     best_silhouette_avg_val = -1.0
        #     best_db_index_val = float('inf')
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
            # if epoch > 100:
            #     if config['training']['cb']:
            #         cb_train, cb_val, num_clusters = validate(train_loader, val_loader, ema_net, logger,epoch=epoch, best_cb_train=best_cb_train,  best_cb_val=best_cb_val, md = 'hdbscan', cb = config['training']['cb'])
            #         if cb_val > best_cb_val :
            #             best_cb_val = cb_val

            #         if cb_train > best_cb_train:
            #             best_cb_train = cb_train

            #             savepath = str(checkpoints_dir) + '/svm_best_model.pth'
            #             state = {
            #                 'best_loss': best_loss,
            #                 'model_state_dict': ema_net.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #             }
            #             torch.save(state, savepath)
            #         print(f'Epoch {epoch} predicts {num_clusters} clusters in the val set having {len(val_loader)} instances')
            #     else:
            #         silhouette_avg_train, db_index_train, silhouette_avg_val, db_index_val, num_clusters = validate(train_loader, val_loader, ema_net, logger, epoch=epoch, best_silhouette_avg_train=best_silhouette_avg_train, best_db_index_train=best_db_index_train, best_silhouette_avg_val=best_silhouette_avg_val, best_db_index_val=best_db_index_val, md = 'hdbscan', cb = config['training']['cb'])
            #         if silhouette_avg_val > best_silhouette_avg_val and db_index_val < best_db_index_val:
            #             best_silhouette_avg_val = silhouette_avg_val
            #             best_db_index_val = db_index_val
            #             savepath = str(checkpoints_dir) + '/svm_best_model.pth'
            #             state = {
            #                 'best_loss': best_loss,
            #                 'model_state_dict': ema_net.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #             }
            #             torch.save(state, savepath)
            #         if silhouette_avg_train > best_silhouette_avg_train and db_index_train < best_db_index_train:
            #             best_silhouette_avg_train = silhouette_avg_train
            #             best_db_index_train = db_index_train
            #         print(f'Epoch {epoch} predicts {num_clusters} clusters in the val set having {len(val_loader)} instances')
            # k = 10
            # if config['model']['md'] == 'pn' or config['training']['l_type'] == 'l':
            #     k = 5
            # if (epoch + 1) % k == 0 and epoch <= 100:
            #     if config['training']['cb']:
            #         cb_train, cb_val, num_clusters = validate(train_loader, val_loader, ema_net, logger, epoch =epoch, best_cb_train=best_cb_train,  best_cb_val=best_cb_val, md = 'hdbscan', cb = config['training']['cb'])
            #         if cb_val > best_cb_val :
            #             best_cb_val = cb_val

            #         if cb_train > best_cb_train:
            #             best_cb_train = cb_train

            #             savepath = str(checkpoints_dir) + '/svm_best_model.pth'
            #             state = {
            #                 'best_loss': best_loss,
            #                 'model_state_dict': ema_net.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #             }
            #             torch.save(state, savepath)
            #         print(f'Epoch {epoch} predicts {num_clusters} clusters in the val set having {len(val_loader)} instances')
            #     else:
            #         silhouette_avg_train, db_index_train, silhouette_avg_val, db_index_val, num_clusters = validate(train_loader, val_loader, ema_net, logger, epoch = epoch, best_silhouette_avg_train=best_silhouette_avg_train, best_db_index_train=best_db_index_train, best_silhouette_avg_val=best_silhouette_avg_val, best_db_index_val=best_db_index_val, md = 'hdbscan', cb = config['training']['cb'])
            #         if silhouette_avg_val > best_silhouette_avg_val and db_index_val < best_db_index_val:
            #             best_silhouette_avg_val = silhouette_avg_val
            #             best_db_index_val = db_index_val
            #             savepath = str(checkpoints_dir) + '/svm_best_model.pth'
            #             state = {
            #                 'best_loss': best_loss,
            #                 'model_state_dict': ema_net.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #             }
            #             torch.save(state, savepath)
            #         if silhouette_avg_train > best_silhouette_avg_train and db_index_train < best_db_index_train:
            #             best_silhouette_avg_train = silhouette_avg_train
            #             best_db_index_train = db_index_train
            #         print(f'Epoch {epoch} predicts {num_clusters} clusters in the val set having {len(val_loader)} instances')
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


if __name__ == '__main__':
    config = parse_args()
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['util']['gpu'])
    '''CREATE DIR'''
    if config['training']['log_dir'] == 'DGCNN' and config['model']['md'] != 'dg':
        raise Exception(f"Model backbone {config['model']['md']} and log directory {config['training']['log_dir']} does not match")
    if config['training']['log_dir'] == 'PointNet' and config['model']['md'] != 'pn':
        raise Exception(f"Model backbone {config['model']['md']} and log directory {config['training']['log_dir']} does not match")
    times = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(f"./log/{config['training']['log_dir']}/" + times)
    os.makedirs(experiment_dir,exist_ok=True)
    # experiment_dir.mkdir(exist_ok=True)
    exp = 'sinkhorn'
    experiment_dir = experiment_dir.joinpath(exp)
    experiment_dir.mkdir(exist_ok=True)
    if config['training']['log_dir'] is None:
        experiment_dir = experiment_dir.joinpath(times)
    else:
        experiment_dir = experiment_dir.joinpath(config['training']['log_dir'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, config['model']['name']))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    '''DATA LOADING'''
    logger.info(config)
    logger.info('Load dataset ...')
    train_data, val_data = data_loader(config)
    train(config, logger, train_data, val_data, exp)
