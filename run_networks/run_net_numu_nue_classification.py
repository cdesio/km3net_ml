# coding: utf-8
from __future__ import print_function

import numpy as np
import os
from network_models import train_neural_network, inference_step
from network_models import CHECKPOINT_FOLDER_PATH
from network_models import TZXY_numu_nue_classification
from data_loaders import data_generator, metadata_generator, get_n_iterations
from keras import backend as K
from data_files import get_train_validation_test_files, get_multi_data_files
from pickle import dump
from sklearn.metrics import accuracy_score, confusion_matrix

IRON_HIDE = os.path.join('/', 'data', 'km3net', 'Xy_multi_data_files')
UNISA = os.path.abspath("./Xy_multi_data_files_logE")

XY_AXIS = (1, 4)
TZ_AXIS = (2, 3)


def get_Time_Coord(X):
    TZ = np.sum(X, axis=TZ_AXIS)
    XY = np.sum(X, axis=XY_AXIS)
    if K.image_data_format() == "channels_first":
        TZ = TZ[:, np.newaxis, ...]
        XY = XY[:, np.newaxis, ...]
    else:
        TZ = TZ[..., np.newaxis]
        XY = XY[..., np.newaxis]
    return [TZ, XY]

model = TZXY_numu_nue_classification(2)
model.summary()

N_FILES = 100
BATCH_SIZE = 64

DATA_FOLDER_NAME = 'numu_nue_stratified_labels'
TASK_NAME = 'numu_nue_classification'
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

TRAINING_WEIGHTS_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                         '{}_weights_training_{}.hdf5'.format(model.name, TASK_NAME))

HISTORY_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                '{}_history_{}.pkl'.format(model.name, TASK_NAME))

MODEL_JSON_FILEPATH = os.path.join(TASK_FOLDER_PATH, '{}_{}.json'.format(model.name, TASK_NAME))


print('TRAINING_WEIGHTS: ', TRAINING_WEIGHTS_FILEPATH)
print('NET HISTORY: ', HISTORY_FILEPATH)

multi_data_folder = IRON_HIDE #Changed to re-run classification 26/01/2018
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
                                    fdata=get_Time_Coord)

validation_generator = data_generator(fnames_val[:N_FILES], batch_size=BATCH_SIZE,
                                      fdata=get_Time_Coord)

training_history = train_neural_network(model, training_generator, steps_per_epoch,
                                        validation_generator,
                                        validation_steps, batch_size=BATCH_SIZE,
                                        log_suffix="{}".format(TASK_NAME))

# Dump of Training History
print('Saving Model (JSON), Training History & Weights...', end='')
model_json_str = model.to_json()
with open(MODEL_JSON_FILEPATH, 'w') as model_json_f:
    model_json_f.write(model_json_str)

history_filepath = HISTORY_FILEPATH
dump(training_history.history, open(history_filepath, 'w'))

model.save_weights(TRAINING_WEIGHTS_FILEPATH)
print('...Done!')

# Inference
print('INFERENCE STEP')

xy_filelist = get_multi_data_files(multi_data_folder, n_files=N_FILES)
metadata_keylist = ["E", "dirx", "diry", "dirz", "posx", "posy", "posz", "dist"]

metadata_gen = metadata_generator(index_filelist, xy_filelist, metadata_keylist)
test_data_gen = data_generator(fnames_test[:N_FILES], batch_size=BATCH_SIZE,
                               fdata=get_Time_Coord)

metadata, y_true, y_pred, probs = inference_step(model, test_data_gen, prediction_steps, metadata_gen)

print('Accuracy: ', accuracy_score(y_true, y_pred))
print('Confusion Matrix', '\n', confusion_matrix(y_true, y_pred))
