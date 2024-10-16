import os
import gc
import glob
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import random

import tensorflow as tf
import jax
import keras

from sklearn import metrics
from tqdm.notebook import tqdm

from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers.legacy import Adam


# build multi-worker environment from Slurm variables
cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=12345)           
 
# use NCCL communication protocol
implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation) 
 
# declare distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver,
                                                     communication_options=communication_options) 



print(tf.__version__)
print(keras.__version__)

#get the number of worker
n_workers = int(os.environ['SLURM_NTASKS'])
print("there are ", n_workers, " workers !")
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
print('GPUs list: ', physical_devices)
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


#Set Debug mode
DEBUG = False
num = 1
print('DEBUG?', DEBUG)

#for reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

#dataset path
path_for_glob = '/lustre/fsn1/projects/rech/psl/upu87pm/low_res_data/first_dataset/'

#hyperparameters
batch_size    = 2048
global_batch_size = batch_size*n_workers
output_length = 368
input_length  = 556
tds_shuffle_buffer = 384*30


#data loader
def load_py_dir_with_generator(filelist:list):
            def gen():
                for file in filelist:
                    # read inputs
                    ds = np.load(file)
                    # read outputs
                    dso = np.load(file.replace('input','target'))

                    ds =  (ds - col1_mean_GLOB)/col1_std_GLOB
                    dso = (dso - col1_mean_GLOB_o)/col1_std_GLOB_o

                    yield (ds, dso) # generating a tuple of (input, output)
           
            return tf.data.Dataset.from_generator(gen,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((None,input_length),(None,output_length))
                                            )


def create_dataset(f_mli):
    ds = load_py_dir_with_generator(f_mli)
    ds = ds.unbatch()
    ds = ds.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(global_batch_size)
    ds = ds.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/global_batch_size))) # in realtion to the batch size
    return ds




path_ = '/lustre/fswork/projects/rech/psl/upu87pm/Data_driven_DYNAMICO/saved/normalization_factors/lr/first_dataset/'

col1_min_GLOB = np.load(path_ + 'col1_min_GLOB.npy')
col1_mean_GLOB = np.load(path_ + 'col1_mean_GLOB.npy')
col1_std_GLOB = np.load(path_ + 'col1_std_GLOB.npy')

col2_mean_GLOB = np.load(path_ + 'col2_mean_GLOB.npy')
col2_std_GLOB = np.load(path_ + 'col2_std_GLOB.npy')

col1_mean_GLOB_o = np.load(path_ + 'col1_mean_GLOB_o.npy')
col1_std_GLOB_o = np.load(path_ + 'col1_std_GLOB_o.npy')

col2_mean_GLOB_o = np.load(path_ + 'col2_mean_GLOB_o.npy')
col2_std_GLOB_o = np.load(path_ + 'col2_std_GLOB_o.npy')


col1_std_GLOB[col1_std_GLOB==0.] = 1e-20
col2_std_GLOB[col2_std_GLOB==0.] = 1e-20
col1_std_GLOB_o[col1_std_GLOB_o==0.] = 1e-20
col2_std_GLOB_o[col2_std_GLOB_o==0.] = 1e-20




#get the list of files
f_mli = glob.glob(path_for_glob + 'input_*.npy')
f_mli = np.array(sorted(f_mli))    
#160 files in first_dataset


n_samples = np.load(f_mli[0]).shape[0]

#creation of the training set with the first 6 years
if DEBUG: 
    idx_f = list(range(4))
else:
    idx_f = list(range(138)) #approximately 6 years on the 7 availables

ds_train = create_dataset(f_mli[idx_f])
print("using ", len(idx_f), "files for training set, each of them contains ", n_samples, "samples ==> ", len(idx_f)*n_samples, "samples")


#creation of the valisation set with the year 7
if DEBUG: 
    idx_f = list(range(10,12))
else:
    idx_f = list(range(138, 160))

ds_val = create_dataset(f_mli[idx_f])
print("using ", len(idx_f), "files for validation set, each of them contains ", n_samples, "samples ==> ", len(idx_f)*n_samples, "samples")


#computing normalization factors
# norm_x = keras.layers.Normalization()
# norm_x.adapt(ds_train.map(lambda x, y: x).take(20 if DEBUG else 10000))

# np.save('norm_factors/mean_x_'+str(num), norm_x.mean)
# np.save('norm_factors/stdd_x_'+str(num), norm_x.variance ** 0.5)

# norm_y = keras.layers.Normalization()
# norm_y.adapt(ds_train.map(lambda x, y: y).take(20 if DEBUG else 10000))

# mean_output = norm_y.mean
# std_output = keras.ops.maximum(1e-10, norm_y.variance ** 0.5)
# np.save('norm_factors/mean_y_'+str(num), mean_output)
# np.save('norm_factors/stdd_y_'+str(num), std_output)


#some hyperparameters
epochs = 100

#cosine learning rate
learning_rate = 1e-3*float(n_workers)/2
epochs_warmup = 10
epochs_ending = 2
steps_per_epoch = int(np.ceil(n_samples/global_batch_size))

lr_scheduler = keras.optimizers.schedules.CosineDecay(
    1e-5*float(n_workers), 
    (epochs - epochs_warmup - epochs_ending) * steps_per_epoch, 
    warmup_target=learning_rate,
    warmup_steps=steps_per_epoch * epochs_warmup,
    alpha=0.1
)


plt.plot([lr_scheduler(it) for it in range(0, epochs * steps_per_epoch, steps_per_epoch)])
plt.xlabel('epochs')
plt.legend()
plt.savefig('Dense_lr'+str(num)+'.png')
plt.clf()

#definition of the model
with strategy.scope():

    model = keras.Sequential([
        #keras.layers.Normalization(mean=norm_x.mean, variance=norm_x.variance),
        keras.layers.Dense(1024, activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros'),
        keras.layers.Dense(512, activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros'),
        keras.layers.Dense(256, activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros'),
        keras.layers.Dense(512, activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros'),
        keras.layers.Dense(output_length, kernel_initializer='random_normal', bias_initializer='zeros')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr_scheduler), metrics=[keras.metrics.MeanSquaredError(), 
                    keras.metrics.R2Score(class_aggregation="variance_weighted_average"), 
    ])
    model.build(tuple(ds_train.element_spec[0].shape))

model.summary()


#training
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=epochs,
    verbose=1,
    callbacks=[keras.callbacks.ModelCheckpoint(filepath='../saved/models/Dense/Densemodel'+str(num)+'_epoch_{epoch:02d}.keras')]
)
###
#0 base => first_dataset + MSE
#1 => same than 0 but with lr divided by 2
#0 => elaborate splitting of the data t:1/7 sample 0-350 v: 1/3 370-400 s: 410-420
#2 => loss MAE
#3 => multiple data representation
#4 => confidence head 
#5 => mae + high_res


plt.plot(history.history['loss'], color='tab:blue', label='loss')
plt.plot(history.history['val_loss'], color='tab:red', label='validation loss')
plt.yscale('log')
plt.xlabel('epochs')
plt.legend()
plt.savefig('Dense_loss'+str(num)+'.png')