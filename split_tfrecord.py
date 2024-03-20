import tensorflow as tf
import os
import numpy as np
import random

from dataset.abc import _load_data_file
from utils.parse import parse_args

def write_tfrecord(filename, points):
    options = tf.io.TFRecordOptions(
            compression_type='ZLIB',
            flush_mode=None,
            input_buffer_size=None,
            output_buffer_size=None,
            window_bits=None,
            compression_level=None,
            compression_method=None,
            mem_level=None,
            compression_strategy=None
        )
    with tf.io.TFRecordWriter(filename,options=options) as writer:
        for point in points:
            example = tf.train.Example(features=tf.train.Features(feature={
                'points': tf.train.Feature(float_list=tf.train.FloatList(value=point.reshape(-1)))
            }))
            writer.write(example.SerializeToString())

config = parse_args()

files = os.listdir(config['root'])
all_files = []
for file in files:
    file_path = os.path.join(config['root'],file)
    data = _load_data_file(file_path)
    all_files.append(data)
all_points = np.concatenate(all_files,axis=0)

indices = np.arange(all_points.shape[0], dtype=int)
random.seed(42)
random.shuffle(indices)
train_indices = np.sort(indices[:round(all_points.shape[0]*config['train_test_split'])])
test_indices = np.sort(indices[round(all_points.shape[0]*config['train_test_split']):])

random.seed(42)
random.shuffle(train_indices)

val_indices = np.sort(train_indices[round(len(train_indices)*config['train_val_percent']):])
train_indices = np.sort(train_indices[:round(len(train_indices)*config['train_val_percent'])])

train_points = all_points[train_indices]
val_points = all_points[val_indices]
test_points = all_points[test_indices]

print("Storing tfrecords for test points ")
write_tfrecord(f"{config['tfrecord_folder']}/stl2_{config['num_points']}_{config['train_test_split']}_{config['train_val_percent']}_test_{all_points.shape[0]}_sampled.tfrecord", test_points)
print(f"tfrecords for test points stored")

print("Storing tfrecords for train points ")
write_tfrecord(f"{config['tfrecord_folder']}/stl2_{config['num_points']}_{config['train_test_split']}_{config['train_val_percent']}_train_{all_points.shape[0]}_sampled.tfrecord", train_points)
print(f"tfrecords for train points stored")

print("Storing tfrecords for val points ")
write_tfrecord(f"{config['tfrecord_folder']}/stl2_{config['num_points']}_{config['train_test_split']}_{config['train_val_percent']}_val_{all_points.shape[0]}_sampled.tfrecord", val_points)
print(f"tfrecords for val points stored")