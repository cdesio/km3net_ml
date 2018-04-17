# coding: utf-8
from __future__ import print_function

import numpy as np
import os
from network_models import train_neural_network
from network_models import CHECKPOINT_FOLDER_PATH, DirectionNet, DirectionNetShared
from data_loaders import direction_net_data_generator, metadata_generator, get_n_iterations
from data_files import get_train_validation_test_files, get_multi_data_files
from pickle import dump
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm


# Paths to Weights Filess
TZ_FILE = './model/model_regression_directions/renamed/tz_net_regression_64_100_regression_cosz_renamed.hdf5'
TX_FILE = './model/model_regression_directions/renamed/tx_net_regression_dirx_v1_64_100_regression_cosx_renamed.hdf5'
TY_FILE = './model/model_regression_directions/renamed/ty_net_regression_diry_v1_64_100_regression_cosy_renamed.hdf5'

loss_weights = (1, 1, 1, 1, 1)
loss_weights_label = '_'.join(map(str, loss_weights))
#model = DirectionNet(loss_weights=loss_weights)
model = DirectionNetShared(loss_weights=loss_weights)
model.summary()

# Initialise Weights
#print('Pre-loading Weights for Three Branches')
#model.load_weights(TX_FILE, by_name=True)
#model.load_weights(TY_FILE, by_name=True)
#model.load_weights(TZ_FILE, by_name=True)

N_FILES = 100
BATCH_SIZE = 64
MAIN_DATA_FOLDER_NAME = 'km3net'
DATA_FOLDER_NAME = 'multi_target_directions'
TASK_NAME = 'regression_directions_shared_weights'
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

TRAINING_WEIGHTS_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                         '{}_weights_training_{}_lw_{}.hdf5'.format(model.name, TASK_NAME,
                                                                                    loss_weights_label))

HISTORY_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                '{}_history_{}_lw_{}.pkl'.format(model.name, TASK_NAME, loss_weights_label))

MODEL_JSON_FILEPATH = os.path.join(TASK_FOLDER_PATH, '{}_{}.json'.format(model.name, loss_weights_label))


print('TRAINING_WEIGHTS: ', TRAINING_WEIGHTS_FILEPATH)
print('NET HISTORY: ', HISTORY_FILEPATH)

multi_data_folder = os.path.join('/', 'data', MAIN_DATA_FOLDER_NAME, 'Xy_multi_data_files')
train_test_dir = os.path.join(multi_data_folder, 'train_test_files', DATA_FOLDER_NAME)
fnames_train, fnames_val, fnames_test, index_filelist = get_train_validation_test_files(train_test_dir,
                                                                                        n_files=N_FILES)

steps_per_epoch, n_events = get_n_iterations(fnames_train[:N_FILES], batch_size=BATCH_SIZE, target_key='dirz')
print(steps_per_epoch, n_events)

validation_steps, n_evts_val = get_n_iterations(fnames_val[:N_FILES], batch_size=BATCH_SIZE, target_key='dirz')
print(validation_steps, n_evts_val)

prediction_steps, n_evts_test = get_n_iterations(fnames_test[:N_FILES], batch_size=BATCH_SIZE, target_key='dirz')
print(prediction_steps, n_evts_test)


training_generator = direction_net_data_generator(fnames_train[:N_FILES], batch_size=BATCH_SIZE)

validation_generator = direction_net_data_generator(fnames_val[:N_FILES], batch_size=BATCH_SIZE)

training_history = train_neural_network(model, training_generator, steps_per_epoch,
                                        validation_generator,
                                        validation_steps, batch_size=BATCH_SIZE,
                                        log_suffix="{}_lw_{}".format(TASK_NAME, loss_weights_label),
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

dirx_true_l = list()
dirx_pred_l = list()
diry_true_l = list()
diry_pred_l = list()
dirz_true_l = list()
dirz_pred_l = list()
sumsq_pred_l = list()
metadata = None

metadata_gen = metadata_generator(index_filelist, xy_filelist, metadata_keylist)
data_gen = direction_net_data_generator(fnames_test[:N_FILES], batch_size=BATCH_SIZE)

for i in tqdm(range(prediction_steps)):
    X_batch, Y_batch_true = next(data_gen)
    metadata_batch = next(metadata_gen)
    if metadata is None:
        metadata = metadata_batch
    else:
        metadata = pd.concat((metadata, metadata_batch))
    Y_batch_pred = model.predict_on_batch(X_batch)
    dirx_true, diry_true, dirz_true, _, _, = Y_batch_true
    dirx_pred, diry_pred, dirz_pred, eu_pred, sumsq_pred = Y_batch_pred
    dirx_pred = dirx_pred.ravel()
    diry_pred = diry_pred.ravel()
    dirz_pred = dirz_pred.ravel()
    sumsq_pred = sumsq_pred.ravel()

    dirx_true_l.append(dirx_true)
    dirx_pred_l.append(dirx_pred)

    diry_true_l.append(diry_true)
    diry_pred_l.append(diry_pred)

    dirz_true_l.append(dirz_true)
    dirz_pred_l.append(dirz_pred)

    sumsq_pred_l.append(sumsq_pred)

dirx_true_l = np.hstack(np.asarray(dirx_true_l))
dirx_pred_l = np.hstack(np.asarray(dirx_pred_l))

diry_true_l = np.hstack(np.asarray(diry_true_l))
diry_pred_l = np.hstack(np.asarray(diry_pred_l))

dirz_true_l = np.hstack(np.asarray(dirz_true_l))
dirz_pred_l = np.hstack(np.asarray(dirz_pred_l))

sumsq_pred_l = np.hstack(np.asarray(sumsq_pred_l))

print('MSE (dirx): ', mean_squared_error(dirx_true_l, dirx_pred_l))
print('R2 Score (dirx): ', r2_score(dirx_true_l, dirx_pred_l))

print('MSE (diry): ', mean_squared_error(diry_true_l, diry_pred_l))
print('R2 Score (diry): ', r2_score(diry_true_l, diry_pred_l))

print('MSE (dirz): ', mean_squared_error(dirz_true_l, dirz_pred_l))
print('R2 Score (dirz): ', r2_score(dirz_true_l, dirz_pred_l))

print('MSE Sum of Squares: ', mean_squared_error(np.ones(shape=sumsq_pred_l.shape), sumsq_pred_l))
print('R2 Sum of Squares: : ', r2_score(np.ones(shape=sumsq_pred_l.shape), sumsq_pred_l))
