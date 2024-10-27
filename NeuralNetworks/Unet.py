import glob
import os
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
from tensorflow.keras.layers import GroupNormalization


print(tf.__version__)
print(keras.__version__)

#get the number of worker
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
num = 0
print('DEBUG?', DEBUG)

#for reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)
tf.random.set_seed(SEED)
# tf.config.experimental.enable_op_determinism()


#dataset path
path_for_glob = '/lustre/fsn1/projects/rech/psl/upu87pm/low_res_data/first_dataset/'

#hyperparameters
batch_size    = 512
global_batch_size = batch_size
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



if DEBUG:
    epochs = 4 # 
else:  
    epochs = 11 # 25  # 15  # 12 
    

#cosine learning rate
learning_rate = 1e-3/2
epochs_warmup = 1
epochs_ending = 1
steps_per_epoch = int(np.ceil(n_samples/global_batch_size))

lr_scheduler = keras.optimizers.schedules.CosineDecay(
    1e-5, 
    (epochs - epochs_warmup - epochs_ending) * steps_per_epoch, 
    warmup_target=learning_rate,
    warmup_steps=steps_per_epoch * epochs_warmup,
    alpha=0.1
)

plt.plot([lr_scheduler(it) for it in range(0, epochs * steps_per_epoch, steps_per_epoch)])
plt.xlabel('epochs')
plt.legend()
plt.savefig('Unet_lr'+str(num)+'.png')
plt.clf()



############## definition of the model #################################################################
@keras.saving.register_keras_serializable()
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.att = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(25)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@keras.saving.register_keras_serializable()
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    attention_output = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ff_output = tf.keras.layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

    return ff_output

# ResBlock function
@keras.saving.register_keras_serializable()
def res_block(x, filters, output_filters=None, groups=8):
    if output_filters is None:
        output_filters = filters
    norm1 = GroupNormalization(groups=groups, axis=-1)(x)
    silu1 = tf.keras.layers.Activation('swish')(norm1)
    conv1 = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same')(silu1)

    norm2 = GroupNormalization(groups=groups, axis=-1)(conv1)
    silu2 = tf.keras.layers.Activation('swish')(norm2)
    conv2 = tf.keras.layers.Conv1D(output_filters, kernel_size=3, padding='same')(silu2)

    if x.shape[-1] != conv2.shape[-1]:
        x = tf.keras.layers.Conv1D(output_filters, kernel_size=1, padding='same')(x)
    output = tf.keras.layers.Add()([conv2, x])
    return output

# Downsample block
@keras.saving.register_keras_serializable()
def repeat_block(x, filters, repeat):
    for _ in range(repeat):
        x = res_block(x, filters) 
    return x

# Upsample block
@keras.saving.register_keras_serializable()
def upsample_block(x, filters, repeat, concat_layer):
    x = tf.keras.layers.Conv1DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, concat_layer])
    for _ in range(repeat):
        x = res_block(x, filters)
    return x

# Define a custom Lambda function to print and remove specific data slices
# TPU does not support printing in this manner, for compatibility, direct value retrieval is used instead.
# This function was previously used but later removed. Hence, there might be inconsistencies.
@keras.saving.register_keras_serializable()
def slice_and_print(x):
    # Print the first 2 time steps that are being removed
    tf.print("Removed start:", x[:, :2, :], summarize=-1)
    # Print the last 2 time steps that are being removed
    tf.print("Removed end:", x[:, -2:, :], summarize=-1)
    # Return the data after removing the padding from the start and end
    return x[:, 2:-2, :]



def x_to_seq(x):
    x_seq0 = keras.ops.transpose(keras.ops.reshape(x[:, 0:60 * 6], (-1, 6, 60)), (0, 2, 1))
    x_seq1 = keras.ops.transpose(keras.ops.reshape(x[:, 60 * 6 + 16:60 * 9 + 16], (-1, 3, 60)), (0, 2, 1))
    x_flat = keras.ops.reshape(x[:, 60 * 6:60 * 6 + 16], (-1, 1, 16))
    x_flat = keras.ops.repeat(x_flat, 60, axis=1)
    return keras.ops.concatenate([x_seq0, x_seq1, x_flat], axis=-1) 


# Build 1D U-Net model
def create_unet(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    # Encoder
    encoder_1 = repeat_block(inputs, 128, 2)  # 64 x 128 x 2
    encoder_1_down = keras.layers.MaxPooling1D(pool_size=2, strides=2)(encoder_1)  # Downsample # 32 x 128

    encoder_2 = repeat_block(encoder_1_down, 256, 2)  # 32 x 256 x 2
    encoder_2_down = keras.layers.MaxPooling1D(pool_size=2, strides=2)(encoder_2)  # Downsample # 16 x 256

    encoder_3 = repeat_block(encoder_2_down, 256, 2)  # 16 x 256 x 2
    encoder_3_down = keras.layers.MaxPooling1D(pool_size=2, strides=2)(encoder_3)  # Downsample # 8 x 256

    encoder_4 = repeat_block(encoder_3_down, 256, 2)  # 8 x 256 x 2

    # Bottleneck (Transformer)
    bottleneck = transformer_block(encoder_4, head_size=4, num_heads=64, ff_dim=512)
    
    decoder_1 = keras.layers.Concatenate()([bottleneck, encoder_4])
    decoder_1_block = repeat_block(decoder_1, 256, 3)  # 8 x 256 x 3
    decoder_1_upsample = keras.layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(decoder_1_block)  # Upsample # 16 x 256

    decoder_2 = keras.layers.Concatenate()([decoder_1_upsample, encoder_3])
    decoder_2_block = repeat_block(decoder_2, 256, 3)  # 16 x 256 x 3
    decoder_2_upsample = keras.layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(decoder_2_block)  # Upsample # 32 x 256

    decoder_3 = keras.layers.Concatenate()([decoder_2_upsample, encoder_2])
    decoder_3_block = repeat_block(decoder_3, 256, 3)  # 32 x 256 x 3
    decoder_3_upsample = keras.layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(decoder_3_block)  # Upsample # 64 x 256

    decoder_4 = keras.layers.Concatenate()([decoder_3_upsample, encoder_1])
    decoder_4_block = repeat_block(decoder_4, 256, 3)  # 64 x 256 x 3

    model = keras.models.Model(inputs, decoder_4_block)
    return model
    
X_input = x = keras.layers.Input(ds_train.element_spec[0].shape[1:]) 
x = x_to_seq(x) 

# Zero-padding at the beginning and end of the sequence to extend the length from 60 to 64
x = keras.layers.ZeroPadding1D(padding=(2, 2))(x)
# 
e = keras.layers.Conv1D(48, 1, padding='same')(x)   
e = create_unet(e.shape[1:])(e)      
    
# Use a Lambda layer to remove the first and last 2 time steps 
e = e[:, 2:-2, :]

p_all = keras.layers.Conv1D(14, 1, padding='same')(e)
print(p_all.shape)

p_seq = p_all[:, :, :6]
p_seq = keras.ops.transpose(p_seq, (0, 2, 1))
p_seq = keras.layers.Flatten()(p_seq)
assert p_seq.shape[-1] == 360

p_flat = p_all[:, :, 6:6 + 8]
p_flat = keras.ops.mean(p_flat, axis=1)
assert p_flat.shape[-1] == 8

P = keras.ops.concatenate([p_seq, p_flat], axis=1)

# Build & compile the model
model = keras.Model(X_input, P)
model.compile(
    loss='mse', 
    optimizer=keras.optimizers.Adam(lr_scheduler),
    metrics=[keras.metrics.MeanSquaredError(), 
                    keras.metrics.R2Score(class_aggregation="variance_weighted_average"), 
    ]  # Updated R2Score
)
model.build(tuple(ds_train.element_spec[0].shape))
model.summary()    

# with kaggle like data
# 0 base => first_dataset + MSE + lr of Dense (cosine //2)
# 2 with mse high res
# 3 with my split lr mse
# 4 with my split lr mae


history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=epochs,
    verbose=1,
    callbacks=[keras.callbacks.ModelCheckpoint(filepath='../saved/models/Unet/Unetmodel'+str(num)+'_epoch_{epoch:02d}.keras')]
)


plt.plot(history.history['loss'], color='tab:blue', label='loss')
plt.plot(history.history['val_loss'], color='tab:red', label='validation loss')
plt.yscale('log')
plt.xlabel('epochs')
plt.legend()
plt.savefig('Unet_loss'+str(num)+'.png')