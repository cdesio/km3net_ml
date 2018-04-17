# coding: utf-8
from __future__ import print_function

import numpy as np
import os
from network_models import train_neural_network, inference_step
from network_models import CHECKPOINT_FOLDER_PATH
from network_models import TZnet_regression_cosz
from data_loaders import data_generator, metadata_generator, get_n_iterations
from keras import backend as K
from data_files import get_train_validation_test_files, get_multi_data_files
from pickle import dump
from sklearn.metrics import mean_squared_error, r2_score

model = TZnet_regression_cosz()
model.summary()

TRAINING_WEIGHTS_FILEPATH = os.path.join(CHECKPOINT_FOLDER_PATH,
                                         '{}_net_weights_training_regression_crossz.hdf5'.format(model.name))

HISTORY_FILEPATH = os.path.join(CHECKPOINT_FOLDER_PATH,
                                '{}_net_history_regression_crossz.pkl'.format(model.name))

print('TRAINING_WEIGHTS: ', TRAINING_WEIGHTS_FILEPATH)
print('NET HISTORY: ', HISTORY_FILEPATH)

n_files = 100
batch_size = 64
train_test_dir = os.path.abspath("./cosz")

fnames_train, fnames_val, fnames_test, index_filelist = get_train_validation_test_files(train_test_dir, n_files=100)

steps_per_epoch, n_events = get_n_iterations(fnames_train[:n_files], batch_size=batch_size)
print(steps_per_epoch, n_events)

validation_steps, n_evts_val = get_n_iterations(fnames_val[:n_files], batch_size=batch_size)
print(validation_steps, n_evts_val)

prediction_steps, n_evts_test = get_n_iterations(fnames_test[:n_files], batch_size=batch_size)
print(prediction_steps, n_evts_test)


def get_TZ_only(X):
    TZ = np.sum(X, axis=(2, 3))
    if K.image_data_format() == "channels_first":
        TZ = TZ[:, np.newaxis, ...]
    else:
        TZ = TZ[..., np.newaxis]
    return TZ


training_generator = data_generator(fnames_train[:n_files], batch_size=batch_size,
                                    fdata=get_TZ_only, ftarget=lambda y: y)

validation_generator = data_generator(fnames_val[:n_files], batch_size=batch_size,
                                      fdata=get_TZ_only, ftarget=lambda y: y)

training_history = train_neural_network(model, training_generator, steps_per_epoch, validation_generator,
                                        validation_steps,
                                        batch_size=batch_size,
                                        log_suffix="regression_cosz.hdf5")

# Dump of Training History
print('Saving Training History & Weights...', end='')
history_filepath = HISTORY_FILEPATH
dump(training_history.history, open(history_filepath, 'wb'))

model.save_weights(TRAINING_WEIGHTS_FILEPATH)
print('...Done!')
# Inference
print('INFERENCE STEP')

multi_data_folder = os.path.join('/', 'data', 'km3net', 'Xy_multi_data_files')
xy_filelist = get_multi_data_files(multi_data_folder, n_files=n_files)
metadata_keylist = ["E", "dirx", "diry", "dirz", "posx", "posy", "posz", "dist"]

predict_steps, n_test_events = get_n_iterations(fnames_test[:n_files], batch_size=64)
print(predict_steps, n_test_events)

metadata_gen = metadata_generator(index_filelist, xy_filelist, metadata_keylist)
test_data_generator = data_generator(fnames_test[:n_files], batch_size=batch_size,
                                     fdata=get_TZ_only, ftarget=lambda y: y)

inference_res = inference_step(model, test_data_generator, predict_steps,
                               metadata_gen, categorical=False)

_, y_true, y_pred = inference_res

print('MSE: ', mean_squared_error(y_true, y_pred))
print('R2 Score', r2_score(y_true, y_pred))
