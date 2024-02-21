import time
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.is_available()
from tqdm import tqdm
from utils.parse import parse_args, calculate_similarity
from dataset.dataloader import data_loader
from multiprocessing import Pool,cpu_count
import gc   
from functools import partial
from statistics import mean 

config = parse_args()

test_data_loader = data_loader(config)
test_points = []
for batch_id, test_data in tqdm(enumerate(test_data_loader, 0), total=len(test_data_loader), smoothing=0.9,desc = 'test dataset'):
    test_points.append(test_data[0].detach().cpu().numpy())
del test_data_loader
gc.collect()
torch.cuda.empty_cache() 
test_points = np.concatenate(test_points,axis=0)
print(f'Dimension of dataset {test_points.shape}')

np.random.seed(42)
idx_ls = np.array(list(range(0,test_points.shape[0])))
test_points = np.take(test_points, np.random.choice(idx_ls, config['test']['num_point'], replace=False), axis=0, out=None, mode='raise')
print(f'Dimension of test points {test_points.shape}')

rep_objs = np.load(config['rep_objs_path'])
print(f'Dimension of representative objects {rep_objs.shape}')

processes = cpu_count()
chunk_size = len(test_points)//processes + 1
chunks = [test_points[i:i+chunk_size] for i in range(0,len(test_points),chunk_size)]

start = time.time()
g = partial(calculate_similarity,rep_objs = rep_objs)
with Pool(processes) as p:
    res = p.map(g,chunks)
end = time.time()

print(f"Calculating similarity for {config['test']['num_point']} test point took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours")

flat_list = [x[0] for xs in res for x in xs]
print(f"The average distance for {config['test']['num_point']} test points is {mean(flat_list)}")

