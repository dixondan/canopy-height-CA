from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, \
    GlobalAveragePooling2D, Reshape, Multiply
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tifffile as tiff
import json

import numpy as np
from keras.callbacks import LearningRateScheduler
import imgaug.augmenters as iaa


class DataGenerator(Sequence):
    def __init__(self, image_folder, mask_folder, batch_size, shuffle=True):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_files = sorted([i for i in os.listdir(image_folder) if i.endswith('.tif')])
        self.mask_files = sorted([i for i in os.listdir(mask_folder) if i.endswith('.tif')])

        self.indexes = np.arange(len(self.image_files))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        end_idx = min(end_idx, len(self.indexes))

        batch_indexes = self.indexes[start_idx:end_idx]

        X, y = self.__load_data(batch_indexes)

        return X, y

    def __load_data(self, batch_indexes):
        X = []
        y = []

        for i in batch_indexes:
            img_path = os.path.join(self.image_folder, self.image_files[i])

            img = tiff.imread(img_path) / 10000
            img = img[5:151, 5:151, :]  # at 128 , dims are 156 X 156
            # img = np.expanddims(img, 0)
            img = img.astype('float32')

            rows, cols = img.shape[0], img.shape[1]
            img_ps = img[:, :, 0:40]
            img_ps = img_ps.reshape((rows, cols, 10, 4))
            img_ps = np.moveaxis(img_ps, 2, 0)

            img_s1 = img[:, :, 40:44]

            # no augmentation to topo
            img_topo = img[:, :, 44:49]

            # applying augmentation to both planetscope and sentinel 1
            augmentation_seq_add = iaa.Sometimes(0.2,
                                                 iaa.Sequential([
                                                     iaa.Add((-0.1, 0.1))], random_order=True))

            augmentation_seq_noise = iaa.Sometimes(0.2,
                                                   iaa.Sequential([
                                                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05))],
                                                       random_order=True))

            # make sure dims are N, height, width, channels for augmentation
            augmented_ps = augmentation_seq_add(images=img_ps)
            augmented_ps = augmentation_seq_noise(images=augmented_ps)
            # have to reshape back to normal
            augmented_ps = np.moveaxis(augmented_ps, 0, 2)
            augmented_ps = augmented_ps.reshape(rows, cols, 40)

            augmented_s1 = augmentation_seq_add(images=img_s1)
            augmented_s1 = augmentation_seq_noise(images=augmented_s1)

            X_aug = np.concatenate([augmented_ps, augmented_s1, img_topo], axis=2)
            X_aug[X_aug < 0] = 0
            X_aug[X_aug > 1] = 1
            X_aug = np.expand_dims(X_aug, 0)

            mask_path = os.path.join(self.mask_folder, self.mask_files[i])
            mask = tiff.imread(mask_path)
            mask = np.expand_dims(mask, 0)

            X.append(X_aug)
            y.append(mask)

        X_stack = np.concatenate(X, axis=0)
        y_stack = np.concatenate(y, axis=0)
        y_stack = np.expand_dims(y_stack, axis=3)

        return X_stack, y_stack


class DataGenerator_testing(Sequence):
    def __init__(self, image_folder, mask_folder, batch_size, shuffle=False):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_files = sorted([i for i in os.listdir(image_folder) if i.endswith('.tif')])
        self.mask_files = sorted([i for i in os.listdir(mask_folder) if i.endswith('.tif')])

        self.indexes = np.arange(len(self.image_files))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        end_idx = min(end_idx, len(self.indexes))

        batch_indexes = self.indexes[start_idx:end_idx]

        X, y = self.__load_data(batch_indexes)

        return X, y



    def __load_data(self, batch_indexes):
        X = []
        y = []

        for i in batch_indexes:
            img_path = os.path.join(self.image_folder, self.image_files[i])
            img = tiff.imread(img_path) / 10000
            img = img[5:151, 5:151, :]  # at 128 , dims are 156 X 156
            img = np.expand_dims(img, 0)
            img = img.astype('float32')

            mask_path = os.path.join(self.mask_folder, self.mask_files[i])
            mask = tiff.imread(mask_path)
            mask = np.expand_dims(mask, 0)

            X.append(img)
            y.append(mask)

        X_stack = np.concatenate(X, axis=0)
        y_stack = np.concatenate(y, axis=0)
        y_stack = np.expand_dims(y_stack, axis=3)
        return X_stack, y_stack


def conv_2d(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = Conv2D(filters, 3, use_bias=False, padding='valid')(x)
    x = BatchNormalization()(x)
    x = swish(x)
    return x


def sep_conv_2d(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = Conv2D(filters, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish(x)
    x = DepthwiseConv2D(3, use_bias=False, padding='valid')(x)
    x = BatchNormalization()(x)
    x = swish(x)
    return x


def apply_2D_spatial(x_input_space, n_filters):
    x = conv_2d(x_input_space, 32)
    x = conv_2d(x, 32)
    x = conv_2d(x, 64)
    x = conv_2d(x, 64)
    x = sep_conv_2d(x, n_filters)
    x = sep_conv_2d(x, n_filters)
    x = sep_conv_2d(x, n_filters)
    x = sep_conv_2d(x, n_filters)
    x = sep_conv_2d(x, n_filters)
    return x

def apply_2D_spatial_topo(x_input_space):
    x = conv_2d(x_input_space, 32)
    x = conv_2d(x, 32)
    x = conv_2d(x, 64)
    x = conv_2d(x, 64)
    x = sep_conv_2d(x, 64)
    x = sep_conv_2d(x, 64)
    x = sep_conv_2d(x, 64)
    x = sep_conv_2d(x, 64)
    x = sep_conv_2d(x, 64)
    return x


def create_model_6(s1: bool, topo: bool, n_neurons: int, n_filters: int) -> tf.keras.Model:
    print(s1, 's1')
    print(topo, 'topo')
    # first defining the input shape
    x_input = tf.keras.layers.Input((None, None, 49), dtype=tf.float32)  # 52 bands stacked

    input_shape = tf.shape(x_input)
    batch_dim = input_shape[0]
    time_dim = 10
    height_dim = input_shape[1]
    width_dim = input_shape[2]

    ''' first, do planetscope time series '''
    x_ps = x_input[:, :, :, 0:40]

    # # Reshape
    x_ps = tf.reshape(x_ps, (batch_dim, height_dim, width_dim, 10, 4))
    x_ps = tf.transpose(x_ps, perm=[0, 3, 1, 2, 4])

    x_ps = tf.reshape(x_ps, (batch_dim * time_dim, height_dim, width_dim, 4))

    x_ps = apply_2D_spatial(x_ps, n_filters=n_filters)

    new_height_dim = height_dim - 18
    new_width_dim = width_dim - 18

    x_ps_time = tf.reshape(x_ps, (batch_dim, 10, new_height_dim, new_width_dim, n_filters))

    # bringing in the most fire power for the temporal convolutions

    x_ps = Conv3D(128, (3, 1, 1), activation='relu', use_bias=True, padding='same')(x_ps_time)
    x_ps = MaxPooling3D((2, 1, 1))(x_ps)

    x_ps = Conv3D(128, (3, 1, 1), activation='relu', use_bias=True, padding='same')(x_ps)
    x_ps = MaxPooling3D((2, 1, 1))(x_ps)

    x_ps = Conv3D(128, (3, 1, 1), activation='relu', use_bias=True, padding='same')(x_ps)
    x_ps = MaxPooling3D((2, 1, 1))(x_ps)

    x_ps = tf.reshape(x_ps, (batch_dim, new_height_dim, new_width_dim, 128))

    tensors_to_cat = []
    tensors_to_cat.append(x_ps)

    if s1 == True:
        x_s1 = x_input[:, :, :, 40:44]
        x_s1 = apply_2D_spatial(x_s1, n_filters=128)
        tensors_to_cat.append(x_s1)

    if topo == True:
        x_topo = x_input[:, :, :, 44:49]
        x_topo = apply_2D_spatial_topo(x_topo)
        tensors_to_cat.append(x_topo)

    # Concatenate the output feature maps along the channel axis

    if len(tensors_to_cat) > 1:
        x = Concatenate(axis=-1)(tensors_to_cat)
    else:
        x = x_ps

    x = Dense(n_neurons, activation='relu')(x)

    x = Dense(1)(x)
    return tf.keras.Model(inputs=x_input, outputs=x)


''' models running: 
    model 1: ps
    model 2: ps + s1
    model 3: ps + s1 + topo
    model 4: ps series
    model 5: ps series + s1
    model 6: ps series + s1 + topo
'''

mrun = 'stcnn'
error = 'mae'
opt = 'adam'

initial_learning_rate = 0.001

lr_schedule = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2000,
    decay_rate=0.85,
    staircase=True)

n_filters = 128
n_neurons = 128

for s1, topo in [(True, True), (False, False), (True, False)]:

    name = ('m7{}_s1{}_topo{}_e{}_opt{}adaptive_nfilters{}_neurons{}'
            .format(mrun, str(s1), str(topo), error, opt, n_filters, n_neurons))

    model_filepath = '{}'.format(name)
    EPOCHS = 60

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filepath + "/checkpoint_e{epoch:02d}",
        save_freq='epoch',
        verbose=1,
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model_6(s1=s1, topo=topo, n_neurons=n_neurons, n_filters=n_filters)
        model.compile(loss='mean_absolute_error',
                      metrics=['mean_absolute_error'],
                      optimizer=optimizer)

    # the input X image has the following dimensions:
    # 49 bands
    # bands 1 - 4 (first median planetscope image (b,g,r,nir) from June 1 to June 15) with scale factor  * 10000
    # bands 5 - 8 (secton median planetscope image from June 15 to June 30) with scale factor  * 10000
    # and so on to band 39
    # bands 41 - 44 are sentinel 1
    # bands 45 to 49 are solar layers

    image_folder = 'height_v9-128/train/X'
    mask_folder = 'height_v9-128/train/y'

    image_folder_test = 'height_v9-128/test/X'
    mask_folder_test = 'height_v9-128/test/y'

    train_generator = DataGenerator(image_folder, mask_folder,
                                    batch_size=12)

    validation_generator = DataGenerator_testing(image_folder_test, mask_folder_test,
                                                 batch_size=4,
                                                 shuffle=False)

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        validation_steps=250 * 4 / 4,
                        validation_freq=1,
                        callbacks=[model_checkpoint_callback])

    with open('results/{}.json'.format(name), 'w') as f:
        json.dump(history.history, f)

