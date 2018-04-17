import ROOT
import numpy as np
import root_numpy as rnp
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

NFLOORS = 18
NSTRING = 115
ndoms = NFLOORS * NSTRING

nuefile = "utilities/km3_v4_nuecc_1.evt.JTE.aa.root"
numufile = "utilities/km3_v4_numucc_1_B.evt.aa.root"

def import_trees(filename):
    ch_id = rnp.root2array(filename, treename="E", branches="Evt.hits.channel_id")
    dom_id = rnp.root2array(filename, treename="E", branches="Evt.hits.dom_id")
    trig = rnp.root2array(filename, treename= "E", branches="Evt.hits.trig")
    t = rnp.root2array(filename, treename="E", branches="Evt.hits.t")
    times = np.asarray([t[evt][trig[evt]==True] for evt in range(t.size)])
    
    return ch_id, dom_id, trig, times

def X_creation(timeslices, dom_id, trig, times):
    numu_events = dom_id.shape[0]
    n_timeslices = timeslices.shape[0] - 1

    X_nu = np.zeros((numu_events, n_timeslices, ndoms))

    # timeslices = np.arange.....

    # Iterate on events
    # Get Hit count for each timeslice
    for evt in range(numu_events):
        # Get all DOM ids for all triggered hits in current event
        triggered_dom_ids = (dom_id[evt][trig[evt] == True]) - 1
        #print("triggered_dom_ids", triggered_dom_ids)
        times_event_hits = times[evt] # select only hits for current event 
        #print("times_event_hits", times_event_hits, times_event_hits.shape[0])
        
        for ts, tslice in enumerate(zip(timeslices[:-1], timeslices[1:])):
            low, high = tslice
            #print(ts, "low", low, "high", high)
            # hits will hold indices of hits matching the condition of being in the selected timeslice
            hits = np.where((times_event_hits >= low) & (times_event_hits < high))[0] 
            #print('hits: ', hits)
            #continue
            if not len(hits):
                continue

            # Get all DOM ids associated to all hits in current time slice.
            dom_hit_in_slice = triggered_dom_ids[hits]
            #print("doms_hit_in_slice", dom_hit_in_slice)
            #print("hits", hits)
            #print(hits.shape)
            # Activate all DOMs for current event, timeslice.
            X_nu[evt, ts, dom_hit_in_slice] = 1
    return X_nu

if __name__ == '__main__':

    ch_id_numu, dom_id_numu, trig_numu, times_numu = import_trees(numufile)
    ch_id_nue, dom_id_nue, trig_nue, times_nue = import_trees(nuefile)

    ch_id_numu, dom_id_numu, trig_numu, times_numu = import_trees(numufile)
    ch_id_nue, dom_id_nue, trig_nue, times_nue = import_trees(nuefile)


    min_t_nue = np.min(np.hstack(times_nue))
    max_t_nue = np.max(np.hstack(times_nue))
    min_t_numu = np.min(np.hstack(times_numu))
    max_t_numu= np.max(np.hstack(times_numu))

    timeslices = np.arange(np.min((min_t_nue, min_t_numu)), np.max((max_t_nue, max_t_numu)), 150)

    X_numu = X_creation(timeslices, dom_id_numu, trig_numu, times_numu)
    X_nue = X_creation(timeslices, dom_id_nue, trig_nue, times_nue)

    Y_numu = np.ones(dom_id_numu.shape[0])
    Y_nue = np.zeros(dom_id_nue.shape[0])


    X = np.vstack((X_numu, X_nue))

    y_numu_categ = np_utils.to_categorical(Y_numu, 2)
    y_nue_categ = np_utils.to_categorical(Y_nue, 2)

    y = np.concatenate((y_numu_categ, y_nue_categ))
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, random_state=42)
import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D

img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
input_shape = (img_rows, img_cols, 1)
num_classes = 2
batch_size = 128

x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

network_history = model.fit(x_train, y_train, batch_size=128, 
                            epochs=100, verbose=1, validation_data=(x_test, y_test), 
                            callbacks=[early_stop])

np.save('history_cnn.npy', network_history.history) 



