import os
import numpy as np
from tqdm import tqdm
import time
from DBCV.DBCV_multiproc import DBCV

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

import tensorflow as tf
from tensorflow import keras
from model1 import get_model
from utils.parse import parse_args, get_clusters, get_pca, get_tsne

def get_tfrecords_dataset(num_points=2048, train_test_split=0.95, train_val_percent  =0.85, batch_size=32, folder = '/mnt/data/mvoe/tfrecords'):
        
    def read(name):
        tfrecord_path = f"{folder}/stl2_{config['num_points']}_{train_test_split}_{train_val_percent}_{name}_999628_sampled.tfrecord"
        print(tfrecord_path)
        
        num_samples = int(tfrecord_path.split('_')[-2].split('.')[0])
        
        drop_remainder = False
        num_batches = num_samples // batch_size
        num_samples = num_batches * batch_size
        
        def _parse(serialized_example):
            features = {
                'points': tf.io.VarLenFeature(tf.float32),
            }
            example = tf.io.parse_single_example(serialized_example, features)
            return (
                tf.reshape(tf.sparse.to_dense(example['points']), (config['num_points'], -1)),
            )
    
        dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type='ZLIB')
        dataset = dataset.shuffle(32*32)
        dataset = dataset.map(_parse)
        if batch_size is not None:
            dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    # ds_train, ds_val, ds_test = read('train'), read('val'), read('test')
    ds_test = read('test')
    
    # return ds_train, ds_val, ds_test
    return ds_test


config = parse_args()
config['num_points'] = 2048

if config['num_points'] == 1024:
    num_features = 32
    #weights_path = './checkpoints/202208101426_model1_chamferx100000/weights_005.h5'
    weights_path = './checkpoints/202208111627_model1_newloss_chamfercontinued/weights_009.h5'

if config['num_points'] == 2048:   
    num_features = 64
    #weights_path = './checkpoints/202209071451_model1_2048_64_newloss_good/weights_012.h5'
    #weights_path = './checkpoints/202209221014_model1_2048_64_newloss_better_continued_k0.1/weights_006.h5'
    weights_path = config['weights_path']

model = get_model(config['num_points'], num_features)

encoder = model.layers[1]
decoder = model.layers[2]

for m in [model, encoder, decoder]:
    print('%-10s'%(m.name), m.input_shape, m.output_shape)

#model.summary(expand_nested=False)
#keras.utils.plot_model(model, dpi=50, show_shapes=True, expand_nested=True)
model.load_weights(weights_path)

# ds_train, ds_val, ds_test = get_tfrecords_dataset(batch_size = config['batch_size'])
ds_test = get_tfrecords_dataset(batch_size = config['batch_size'])

all_points = []
start = time.time()
for batch in tqdm(ds_test,desc = 'val dataset'):
    emb = tf.squeeze(model.predict(batch)[0],axis=1)
    all_points.append(emb)

all_points = np.concatenate(all_points,axis=0)

print(all_points.shape)

np.random.seed(42)
all_points = np.take(all_points, np.random.choice(np.array(list(range(0,all_points.shape[0]))), config['dbcv']['num_points'], replace=False), axis=0, out=None, mode='raise')
end = time.time()
print(f'Dimension of dataset {all_points.shape} and it takes {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours')

# all_points = get_pca(n_components=config['pca']['n_components'],data=all_points)

# all_points = get_pca(n_components=config['pca_2']['n_components'],data=all_points)

# all_points = get_tsne(n_components=config['tsne']['n_components'], data=all_points)

clf, pred, num_clusters = get_clusters(data=all_points,store_centers='medoid',classifier=config['results']['classifier'],min_samples=2,n_clusters=config['results']['n_clusters'],batch_size=config['results']['chunk_size'])

start = time.time()
dbcv_score = DBCV(all_points,pred)
print(f"DBCV score is {dbcv_score}")
end = time.time()
print(f'DBCV took {end-start} seconds or {(end-start)/60} minutes or {(end-start)/3600} hours for {all_points.shape[0]} datapoints')
        