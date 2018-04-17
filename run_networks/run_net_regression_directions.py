# coding: utf-8
from __future__ import print_function

import numpy as np
import os
from network_models import train_neural_network
from network_models import CHECKPOINT_FOLDER_PATH
from network_models import TXnet_regression_cosx, TYnet_regression_cosy, TZnet_regression_cosz
from data_loaders import data_generator, metadata_generator, get_n_iterations
from keras import backend as K
from data_files import get_train_validation_test_files, get_multi_data_files
from pickle import dump
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm


TX_AXIS = (3, 4)
TZ_AXIS = (2, 3)
TY_AXIS = (2, 4)
SUM_AXIS = TX_AXIS


TZ_FILE = './model/regression_cosz_rerun/tz_net_regression_64_100_regression_cosz_rerun.hdf5'
TX_FILE = './model/regression_cosx/tx_net_regression_dirx_v1_64_100_regression_cosx.hdf5'
TY_FILE = './model/regression_cosy/ty_net_regression_diry_v1_64_100_regression_cosy.hdf5'
WEIGHT_FILE = TX_FILE


def get_Time_Coord(X):
    TC = np.sum(X, axis=SUM_AXIS)
    if K.image_data_format() == "channels_first":
        TC = TC[:, np.newaxis, ...]
    else:
        TC = TC[..., np.newaxis]
    return TC


model_build_function = TXnet_regression_cosx
model = model_build_function()
model.summary()

#refit
print('Loading WEIGHTS from {}'.format(WEIGHT_FILE))
model.load_weights(WEIGHT_FILE)

N_FILES = 100
BATCH_SIZE = 64

DATA_FOLDER_NAME = 'cosx_no_stratify'
TASK_NAME = 'regression_cosx'
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

TRAINING_WEIGHTS_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                         '{}_weights_training{}_REFIT.hdf5'.format(model.name, TASK_NAME))

HISTORY_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                '{}_history{}_REFIT.pkl'.format(model.name, TASK_NAME))

MODEL_JSON_FILEPATH = os.path.join(TASK_FOLDER_PATH, '{}_REFIT.json'.format(model.name))


print('TRAINING_WEIGHTS: ', TRAINING_WEIGHTS_FILEPATH)
print('NET HISTORY: ', HISTORY_FILEPATH)

multi_data_folder = os.path.join('/', 'data', 'km3net', 'Xy_multi_data_files')
train_test_dir = os.path.join(multi_data_folder, 'train_test_files', DATA_FOLDER_NAME)
fnames_train, fnames_val, fnames_test, index_filelist = get_train_validation_test_files(train_test_dir,
                                                                                        n_files=N_FILES)

steps_per_epoch, n_events = get_n_iterations(fnames_train[:N_FILES], batch_size=BATCH_SIZE)
print(steps_per_epoch, n_events)

validation_steps, n_evts_val = get_n_iterations(fnames_val[:N_FILES], batch_size=BATCH_SIZE)
print(validation_steps, n_evts_val)

prediction_steps, n_evts_test = get_n_iterations(fnames_test[:N_FILES], batch_size=BATCH_SIZE)
print(prediction_steps, n_evts_test)


training_generator = data_generator(fnames_train[:N_FILES], batch_size=BATCH_SIZE,
                                    fdata=get_Time_Coord, ftarget=lambda y: y)

validation_generator = data_generator(fnames_val[:N_FILES], batch_size=BATCH_SIZE,
                                      fdata=get_Time_Coord, ftarget=lambda y: y)

training_history = train_neural_network(model, training_generator, steps_per_epoch,
                                        validation_generator,
                                        validation_steps, batch_size=BATCH_SIZE,
                                        log_suffix="{}_REFIT".format(TASK_NAME),
                                        checkpoint_folder=TASK_FOLDER_PATH)

# Dump of Training History
print('Saving Model (JSON), Training History & Weights...', end='')
model_json_str = model.to_json()
with open(MODEL_JSON_FILEPATH, 'w') as model_json_f:
    model_json_f.write(model_json_str)

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

metadata_gen = metadata_generator(index_filelist, xy_filelist, metadata_keylist)
data_gen = data_generator(fnames_test[:N_FILES], batch_size=BATCH_SIZE,
                          fdata=get_Time_Coord, ftarget=lambda y: y)

for i in tqdm(range(prediction_steps)):
    TX_batch, y_batch_true = next(data_gen)
    metadata_batch = next(metadata_gen)
    if metadata is None:
        metadata = metadata_batch
    else:
        metadata = pd.concat((metadata, metadata_batch))
    y_batch_pred = model.predict_on_batch(TX_batch)
    y_batch_pred = y_batch_pred.ravel()
    y_true.append(y_batch_true)
    y_pred.append(y_batch_pred)

y_true = np.hstack(np.asarray(y_true))
y_pred = np.hstack(np.asarray(y_pred))

print('MSE: ', mean_squared_error(y_true, y_pred))
print('R2 Score', r2_score(y_true, y_pred))
