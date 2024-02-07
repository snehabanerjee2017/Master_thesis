import trimesh
import h5py
import numpy as np
import glob
import os
from tqdm import tqdm
import multiprocessing
import argparse
import yaml

def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def crop(points:np.ndarray, p_keep:float):
        
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument("-c", "--config", required=True,
                    help="JSON configuration string for this operation")
    
    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)
    return config

def create_hdf5(in_path:str, out_path:str, num_points:str=2048, p_keep:float=0.85):

    files = np.array(glob.glob(os.path.join(in_path, f"**/*.stl"), recursive=True))
    # print('Loaded files')

    hdf5_file = h5py.File(out_path, 'a')
    hdf5_file.create_dataset("data", (len(files), num_points, 3), dtype='f')
    hdf5_file.create_dataset("filenames", (len(files)), dtype=h5py.special_dtype(vlen=bytes))
    dataset = hdf5_file['data']
    dataset_filenames = hdf5_file['filenames']

    for i, file_ in tqdm(enumerate(files), total=len(files), desc="Creating dataset ..."):
        mesh = trimesh.load_mesh(file_)
        if hasattr(mesh, 'area_faces'):        
            points = trimesh.sample.sample_surface(trimesh.load_mesh(file_), num_points)[0]
            points_cropped = crop(points, p_keep)
            if points_cropped.size!=0:
                dataset[i] = points
                dataset_filenames[i] = file_



def create_file(file_chunk:list,in_dir:str,out_dir:str,num_points:int,p_keep:float):
    for file in tqdm(file_chunk):
        in_path = os.path.join(in_dir,file)
        os.makedirs(out_dir,exist_ok=True)
        out_path = os.path.join(out_dir,file+ '.hdf5')
        create_hdf5(in_path=in_path,out_path=out_path,num_points=num_points,p_keep=p_keep)
        print(f'Created file {file}')

if __name__ == '__main__':
    config = parse_args()
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(os.listdir(config['in_dir'])) // num_processes + 1
    file_chunks = [os.listdir(config['in_dir'])[i:i + chunk_size] for i in range(0, len(os.listdir(config['in_dir'])), chunk_size)]

    for file_chunk in file_chunks:
        process = multiprocessing.Process(target=create_file, args=(file_chunk,config['in_dir'], config['out_dir'], config['num_points'], config['p_keep']))
        process.start()