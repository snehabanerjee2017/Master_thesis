#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:45 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : dataloader.py
# @Software: PyCharm
import torch
from dataset.abc import Abc


def data_loader(config):
    if config['dataset']['type'] == 'abc':
        if config['training']['train']:
            train_dataset = Abc(root=config['dataset']['path'], n_point=config['dataset']['num_point'], train=True, use_normal=config['training']['normal'], unsup=config['training']['unsup'], angle=config['model']['angle'], p_keep=config['model']['crop'], train_val_percent=config['dataset']['train_val_percent'], train_test_split = config['dataset']['train_test_split'])
            val_dataset = Abc(root=config['dataset']['path'], n_point=config['dataset']['num_point'], train=True, use_normal=config['training']['normal'], unsup=config['training']['unsup'], angle=config['model']['angle'], p_keep=config['model']['crop'], train_val_percent=config['dataset']['train_val_percent'],train_test_split = config['dataset']['train_test_split'], val = True)         
        else:
            test_dataset = Abc(root=config['dataset']['path'], n_point=config['dataset']['num_point'], train=False, unsup=False, angle=0.0,train_test_split=config['dataset']['train_test_split'],p_keep=config['model']['crop'])
    else:
        raise NotImplementedError
    
    if config['dataset']['type'] == 'abc':
        if config['training']['train']:
            train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                                                        shuffle=True, num_workers=config['util']['workers'])
            val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                                                    shuffle=False, num_workers=config['util']['workers'])
            return train_data_loader, val_data_loader

        else:
            test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                                                    shuffle=False, num_workers=config['util']['workers'])
            return test_data_loader


        
    else:
        raise NotImplementedError


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
