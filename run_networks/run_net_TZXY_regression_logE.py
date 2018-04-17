# coding: utf-8
from __future__ import print_function

import numpy as np
import os
import pandas as pd
from network_models import train_neural_network
from network_models import TZXY_regression_logE_relu_psigmoid, TZXY_regression_logE_relu_tanh
from network_models import TZXY_regression_logE_vgg
from network_models import CHECKPOINT_FOLDER_PATH
from keras import backend as K
from data_loaders import data_generator, metadata_generator
from data_loaders import get_n_iterations
from pickle import dump
from tqdm import tqdm
from data_files import get_train_validation_test_files, get_multi_data_files
from sklearn.metrics import mean_squared_error, r2_score

N_FILES = 100
BATCH_SIZE = 64

model_build_func = TZXY_regression_logE_vgg
model = model_build_func()
model.summary()

TRAINING_WEIGHTS_FILEPATH = os.path.join(CHECKPOINT_FOLDER_PATH,
                                         '{}_net_weights_training.hdf5'.format(model.name))

HISTORY_FILEPATH = os.path.join(CHECKPOINT_FOLDER_PATH,
                                '{}_net_history.pkl'.format(model.name))

print('TRAINING_WEIGHTS: ', TRAINING_WEIGHTS_FILEPATH)
print('NET HISTORY: ', HISTORY_FILEPATH)

multi_data_folder = os.path.join('/', 'data', 'km3net', 'Xy_multi_data_files')
train_test_dir = os.path.join(multi_data_folder, 'train_test_files', 'log_energies_stratified')

fnames_train, fnames_val, fnames_test, index_filelist = get_train_validation_test_files(train_test_dir,
                                                                                        n_files=N_FILES)

steps_per_epoch, n_events = get_n_iterations(fnames_train[:N_FILES], batch_size=BATCH_SIZE)
print(steps_per_epoch, n_events)

validation_steps, n_evts_val = get_n_iterations(fnames_val[:N_FILES], batch_size=BATCH_SIZE)
print(validation_steps, n_evts_val)

prediction_steps, n_evts_test = get_n_iterations(fnames_test[:N_FILES], batch_size=BATCH_SIZE)
print(prediction_steps, n_evts_test)


def get_TZXY_data(X):
    TZ = np.sum(X, axis=(2, 3))
    XY = np.sum(X, axis=(1, 4))
    if K.image_data_format() == "channels_first":
        TZ = TZ[:, np.newaxis, ...]
        XY = XY[:, np.newaxis, ...]
    else:
        TZ = TZ[..., np.newaxis]
        XY = XY[..., np.newaxis]
    return [TZ, XY]


training_generator = data_generator(fnames_train[:N_FILES], batch_size=BATCH_SIZE,
                                    fdata=get_TZXY_data, ftarget=lambda y: y)

validation_generator = data_generator(fnames_val[:N_FILES], batch_size=BATCH_SIZE,
                                      fdata=get_TZXY_data, ftarget=lambda y: y)

training_history = train_neural_network(model, training_generator, steps_per_epoch, validation_generator,
                                        validation_steps, batch_size=BATCH_SIZE,
                                        log_suffix="regression_logE")

# Dump of Training History
print('Saving Training History & Weights...', end='')
history_filepath = HISTORY_FILEPATH
dump(training_history.history, open(history_filepath, 'wb'))

model.save_weights(TRAINING_WEIGHTS_FILEPATH)
print('...Done!')

# Inference
print('INFERENCE STEP')

xy_filelist = get_multi_data_files(multi_data_folder, n_files=N_FILES)
metadata_keylist = ["E", "dirx", "diry", "dirz", "posx", "posy", "posz", "dist"]

y_true = list()
y_pred = list()
metadata = None
predict_steps, n_test_events = get_n_iterations(fnames_test[:N_FILES], batch_size=64)
print(predict_steps, n_test_events)

metadata_gen = metadata_generator(index_filelist, xy_filelist, metadata_keylist)
data_gen = data_generator(fnames_test[:N_FILES], batch_size=BATCH_SIZE,
                          fdata=get_TZXY_data, ftarget=lambda y: y)

for i in tqdm(range(predict_steps)):
    [ZT_batch, XY_batch], y_batch_true = next(data_gen)
    metadata_batch = next(metadata_gen)
    if metadata is None:
        metadata = metadata_batch
    else:
        metadata = pd.concat((metadata, metadata_batch))
    y_batch_pred = model.predict_on_batch([ZT_batch, XY_batch])
    y_batch_pred = y_batch_pred.ravel()
    y_true.append(y_batch_true)
    y_pred.append(y_batch_pred)

y_true = np.hstack(np.asarray(y_true))
y_pred = np.hstack(np.asarray(y_pred))

print('MSE: ', mean_squared_error(y_true, y_pred))
print('R2 Score', r2_score(y_true, y_pred))
