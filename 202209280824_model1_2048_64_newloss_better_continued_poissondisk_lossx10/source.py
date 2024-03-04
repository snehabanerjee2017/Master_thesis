# In[0]


# In[1]
import os, itertools, time
import numpy as np

from glob import glob
from tqdm.notebook import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # -1=CPU, 0,1,2,...=GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
np.set_printoptions(suppress=True)

import tensorflow as tf
import keras.backend as K
import keras

#%matplotlib widget

# In[2]
from model1 import get_model
#from model2 import get_model

#num_points = 1024
num_points = 2048
#num_points = 4096
#num_features = 32
num_features = 64

model = get_model(num_points, num_features)

weight_path = None
#weight_path = './checkpoints/202208101426_model1_chamferx100000/weights_005.h5'
#weight_path = './checkpoints/202208120940_model1/weights_020.h5'
#weight_path = './checkpoints/202209140956_model1_2048_64_newloss_better/weights_007.h5'
weight_path = './checkpoints/202209191848_model1_2048_64_newloss_better_continued/weights_012.h5'

# In[3]
from data import DataUtility, DataUtilitySampled

batch_size = 32

if 0:
    du = DataUtility()
    ds_train, ds_val = du.get_tfrecords_dataset(num_points=num_points, batch_size=batch_size, split=0.9, augmentation=True)
else:
    du = DataUtilitySampled()
    ds_train, ds_val = du.get_tfrecords_dataset(with_normals=False)

# In[4]
def chamfer_loss(pnts_a, pnts_b):
    # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of dimension D).
    # Somehow this solution is much more memory efficient compared to
    #ds = tf.reduce_sum((pnts_a[:,:,None,:] - pnts_b[:,None,:,:]) ** 2, axis=-1)
    ds = tf.reduce_sum(pnts_a*pnts_a, axis=-1)[:,:,None] - 2 * tf.matmul(pnts_a, pnts_b, transpose_b=True) + tf.reduce_sum(pnts_b*pnts_b, axis=-1)[:,None,:]
    loss = tf.reduce_mean(tf.reduce_min(ds, axis=-1), axis=-1) + tf.reduce_mean(tf.reduce_min(ds, axis=-2), axis=-1)
    return loss

def new_distance(pnts1, pnts2, k=0.1):
    
    b, n1, c = pnts1.shape
    b, n2, c = pnts2.shape
    
    ds = tf.reduce_sum((pnts1[:,:,None,:] - pnts2[:,None,:,:]) ** 2, axis=-1) ** 0.5
    
    dist1, dist2 = tf.reduce_min(ds, axis=-1), tf.reduce_min(ds, axis=-2)
    
    idx1, idx2 = tf.argmin(ds, axis=-1), tf.argmin(ds, axis=-2)
    
    count1 = tf.math.bincount(idx1, minlength=n2, maxlength=n2, axis=-1)
    count2 = tf.math.bincount(idx2, minlength=n1, maxlength=n1, axis=-1)
    
    count1 = tf.stop_gradient(tf.cast(count1, dist1.dtype))
    count2 = tf.stop_gradient(tf.cast(count2, dist2.dtype))
    
    count1_select = tf.gather(count1, idx1, axis=-1)
    count2_select = tf.gather(count2, idx2, axis=-1)
    
    
    num_assigned1 = tf.reduce_sum(tf.cast(tf.math.greater(count1, 0.0), dist1.dtype), axis=-1)
    num_assigned2 = tf.reduce_sum(tf.cast(tf.math.greater(count2, 0.0), dist2.dtype), axis=-1)
    max_count1 = tf.reduce_max(count1, axis=-1)
    max_count2 = tf.reduce_max(count2, axis=-1)
    mean_count1 = tf.reduce_sum(count1, axis=-1) / (num_assigned1 + 1e-8)
    mean_count2 = tf.reduce_sum(count2, axis=-1) / (num_assigned2 + 1e-8)
    
    
    #weight1 = (k + count1*n1**(-1))**(-1)
    #weight2 = (k + count2*n2**(-1))**(-1)
    
    #weight1 = tf.math.divide_no_nan(1.0,count1)
    #weight2 = tf.math.divide_no_nan(1.0,count2)
    
    #weight1 = (k + count1_select*n1**(-1))**(-1)
    #weight2 = (k + count2_select*n2**(-1))**(-1)
    
    #weight1 = tf.math.divide_no_nan(1.0,count1_select)
    #weight2 = tf.math.divide_no_nan(1.0,count2_select)
    
    #weight1 = tf.cast(tf.math.greater(count2, 0.0), dist2.dtype) * 10.0 + 1.0
    #weight2 = tf.cast(tf.math.greater(count1, 0.0), dist1.dtype) * 10.0 + 1.0
    
    #weight1 = tf.cast(tf.math.greater(count2, 0.0), dist2.dtype) * 1.0 + 1.0
    #weight2 = tf.cast(tf.math.greater(count1, 0.0), dist1.dtype) * 1.0 + 1.0
        
    #weight1 = tf.cast(tf.math.greater(count2, 0.0), dist2.dtype) * count1_select + 1.0
    #weight2 = tf.cast(tf.math.greater(count1, 0.0), dist1.dtype) * count2_select + 1.0
    
    # works
    #weight1 = tf.cast(tf.math.greater(count2, 0.0), dist2.dtype) * tf.math.divide_no_nan(1.0,count1_select) + 1.0
    #weight2 = tf.cast(tf.math.greater(count1, 0.0), dist1.dtype) * tf.math.divide_no_nan(1.0,count2_select) + 1.0
    
    # works better
    weight1 = tf.cast(tf.math.greater(count2, 0.0), dist2.dtype) * tf.divide(0.01, 1.0+count1_select) + 1.0
    weight2 = tf.cast(tf.math.greater(count1, 0.0), dist1.dtype) * tf.divide(0.01, 1.0+count2_select) + 1.0
    
    # not
    #weight1 = tf.cast(tf.math.equal(count2, 0.0), dist2.dtype) * tf.math.divide_no_nan(1.0,count1_select) + 1.0
    #weight2 = tf.cast(tf.math.equal(count1, 0.0), dist1.dtype) * tf.math.divide_no_nan(1.0,count2_select) + 1.0
    
    #weight1 = tf.cast(tf.math.equal(count1_select, 0.0), dist1.dtype) * tf.math.divide_no_nan(0.1,1.0+count2) + 1.0
    #weight2 = tf.cast(tf.math.equal(count2_select, 0.0), dist2.dtype) * tf.math.divide_no_nan(0.1,1.0+count1) + 1.0
    
    
    # TODO
    #weight1 = tf.cast(tf.math.equal(count2, 0.0) - tf.math.greater(count2, 1.0), dist2.dtype) * tf.math.divide_no_nan(1.0,count1_select) + 1.0

    
    #loss = tf.reduce_mean(dist1, axis=-1) + tf.reduce_mean(dist2, axis=-1)
    loss = tf.reduce_mean(weight1*dist1, axis=-1) + tf.reduce_mean(weight2*dist2, axis=-1)
    loss = loss * 10
    #loss = tf.reduce_mean(weight1*dist1, axis=-1)
    #loss = tf.reduce_mean(weight2*dist2, axis=-1)
    
    alpha = 1000.0
    #alpha = 1.0
    #loss = tf.reduce_mean(1-tf.math.divide_no_nan(1.0,count1)*tf.exp(-alpha*dist1)) + \
    #       tf.reduce_mean(1-tf.math.divide_no_nan(1.0,count2)*tf.exp(-alpha*dist2))
    
    #loss = tf.reduce_mean(1-tf.math.divide_no_nan(1.0,count1_select)*tf.exp(-alpha*dist1)) + \
    #       tf.reduce_mean(1-tf.math.divide_no_nan(1.0,count2_select)*tf.exp(-alpha*dist2))
    
    #loss = tf.reduce_mean(tf.math.divide_no_nan(1.0,count1_select)*(1-tf.exp(-alpha*dist1))) + \
    #       tf.reduce_mean(tf.math.divide_no_nan(1.0,count2_select)*(1-tf.exp(-alpha*dist2)))
    
    return loss, max_count1, max_count2, mean_count1, mean_count2, num_assigned1, num_assigned2

# In[5]
from utils.training import MetricUtility

num_epochs = 20

checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + model.name
checkdir += '_%i_%i' % (num_points, num_features)

metric_names = ['loss', 'chamfer_dist', 'new_dist', 
                'max_count1', 'max_count2', 'mean_count1', 'mean_count2', 'num_assigned1', 'num_assigned2',
                'latent_reg_mean', 'latent_reg_std', 'latent_mean', 'latent_std', 'R_reg']

metric_util = MetricUtility(metric_names, logdir=checkdir)

#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-5)

if weight_path is not None:
    model.load_weights(weight_path)
    sheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [(i*100)*1e3 for i in range(1,4)], [1/(2**i)*1e-3 for i in range(1,5)]
    )
else:
    sheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [(250+i*100)*1e3 for i in range(0,4)], [1/(2**i)*1e-3 for i in range(0,5)]
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=sheduler, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


@tf.function
def calc_metrics(x, y_true, y_pred, z, R):
    eps = K.epsilon()
    y_shape = K.shape(y_true)
    batch_size = y_shape[0]
    
    lambda_mean = 1e-4
    lambda_std = 1e-4

    chamfer_dist = K.mean(chamfer_loss(y_true, y_pred))
    
    latent_mean = K.mean(K.mean(z, axis=0))
    latent_std = K.mean(K.std(z, axis=0))

    latent_reg_mean = K.mean(K.abs(K.mean(z, axis=0)))
    latent_reg_std = K.mean(K.abs(K.std(z, axis=0) - 1))
    
    lambda_R = 1e-1
    R_norm_reg1 = tf.reduce_mean(tf.abs(1 - tf.norm(R, axis=-1)))
    R_norm_reg2 = tf.reduce_mean(tf.abs(1 - tf.norm(R, axis=-2)))
    R_reg = R_norm_reg1 + R_norm_reg2
    
    new_dist, max_count1, max_count2, mean_count1, mean_count2, num_assigned1, num_assigned2 = new_distance(y_true, y_pred, k=0.1)
    new_dist = K.mean(new_dist)
    max_count1 = K.mean(max_count1)
    max_count2 = K.mean(max_count2)
    mean_count1 = K.mean(mean_count1)
    mean_count2 = K.mean(mean_count2)
    num_assigned1 = K.mean(num_assigned1)
    num_assigned2 = K.mean(num_assigned2)
    
    #loss = 100000 * chamfer_dist + lambda_mean * latent_reg_mean + lambda_std * latent_reg_std + lambda_R * R_reg
    #loss = 1.0 * new_dist + lambda_mean * latent_reg_mean + lambda_std * latent_reg_std + lambda_R * R_reg
    loss = 10.0 * new_dist + lambda_mean * latent_reg_mean + lambda_std * latent_reg_std + lambda_R * R_reg
    #loss = 1.0 * tf.math.log(new_dist) + lambda_mean * latent_reg_mean + lambda_std * latent_reg_std + lambda_R * R_reg
    
    return {k:v for k,v in locals().items() if k in metric_names}


@tf.function
def step(batch):
    with tf.GradientTape() as tape:
        x = y_true = batch[0]
        z, R, y_pred = model(x, training=True)
        metric_values = calc_metrics(x, y_true, y_pred, z, R)
        total_loss = metric_values['loss']
        if len(model.losses) > 0:
            total_loss = total_loss + tf.add_n(model.losses)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return metric_values

@tf.function
def step_val(batch):
    x = y_true = batch[0]
    z, R, y_pred = model(x, training=False)
    metric_values = calc_metrics(x, y_true, y_pred, z, R)
    return metric_values


os.makedirs(checkdir, exist_ok=True)
with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
print(checkdir)

#names = ['batch_size', 'num_batches', 'num_batches_val', 'input_size', 'subsampling', 'num_epochs']
#names = ['batch_size', 'num_epochs', 'num_samples', 'num_samples_val', 'num_batches', 'num_batches_val']
names = ['batch_size', 'num_epochs', 'num_points', ]
print(*['%s %s ' % (n, eval(n)) for n in names])

for epoch in tqdm(range(num_epochs), 'total', leave=False):
    print('\nepoch %i/%i' % (epoch+1, num_epochs))
    
    metric_util.on_epoch_begin()
    
    for batch in tqdm(ds_train, 'training'):
        metric_values = step(batch)
        metric_util.update(metric_values, training=True)
    
    model.save_weights(checkdir+'/weights_%03i.h5' % (epoch+1,))
    
    for batch in tqdm(ds_val, 'validation'):
        
        metric_values = step_val(batch)
        metric_util.update(metric_values, training=False)

    metric_util.on_epoch_end(verbose=1)

