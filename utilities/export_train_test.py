import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import Iterable


SPLITTING_SEED = 42  # fixed random seed for reproducibility
INDEX_TRAINING_KEY = 'train'
INDEX_TEST_KEY = 'test'
INDEX_VALIDATION_KEY = 'val'


def export_train_validation_test(i, fname_numu, fname_nue, out_dir="train_test_files", test_size=0.20,
                                 validation_size=0.20, data_key="x", target_key="y", multi_target_keys=None,
                                 stratify=True, stratify_key='y', fdata=lambda x: x, ftarget=lambda y: y,
                                 fstratify=lambda y: y):
    """
    Function to export Training, Validation, and Test sets for NuMu and NuE events

    Parameters
    ----------
    i: int
        Progressive index of the train-test split partition to save

    fname_numu: str
        Path to the numpy compressed (.npz) file containing data and labels for
        numu generated events

    fname_nue: str
        Path to the numpy compressed (.npz) file containing data and labels for
        nue generated events

    out_dir: str
        Path to the destination folder where files will be saved

    test_size: int (default: 0.2)
        Size of the test set to apply in train-test split

    validation_size: int (default: 0.2)
        Size of the valdation set to apply in train-validation split

    data_key: str (default: "x")
        Key in input files to use to get the data (i.e. features)

    target_key: str (default: "y")
        Key in input files to use to get the targets (i.e. labels)
        This parameter will be ignored if `multi_target_keys` parameter
        has values.

    multi_target_keys: list (default: None)
        List of multiple target keys to extract from files
        for each sample. Each of the key reported in the
        list must be contained in the target files.

    stratify: bool (default: True)
        Whether to apply stratification in splitting data or not

    stratify_key: str (default: "y")
        Key in input files to use to get the stratification array
        This is considered **only** if stratify is True

    fdata: function (default identity)
        Function to be applied on data (as extracted by `data_key`)
        as a pre-processing step **before** any `train_test_split`
        is applied.

    ftarget: function (default identity)
        Function to be applied on targets (as extracted by `target_key`)
        as a pre-processing step **before** any `train_test_split`
        is applied.
        In case of multiple target keyes (i.e.`multi_target_keys is not None`),
        this function is mapped to each target.

    fstratify: function (default identity)
        Function to be applied on stratification array
        for further processing. By default, the identity function
        is considered, thus **only** the `stratify_key` will be
        considered.
        The only constraint for the function is to return
        a numpy array as expected by the `sklearn.metrics.train_test_split`
        function.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if multi_target_keys:
        target_keys = multi_target_keys
    else:
        target_keys = [target_key]

    with np.load(fname_numu) as Xy_numu, np.load(fname_nue) as Xy_nue:
        X = np.vstack((Xy_numu[data_key], Xy_nue[data_key]))
        y = list()
        for target_key in target_keys:
            y_target = np.hstack((Xy_numu[target_key], Xy_nue[target_key]))
            y.append(y_target)

        X = fdata(X)
        y = list(map(ftarget, y))  # apply ftarget function to each collection of selected targets
        indices = np.arange(len(y[0]))  # all labels have the same number of elements.

        # Split Training Set and Test Set

        if stratify:
            strat_array = np.hstack((Xy_numu[stratify_key], Xy_nue[stratify_key]))
            strat_array = fstratify(strat_array)
        else:
            strat_array = None

        training_test_split = train_test_split(X, indices, *y, test_size=test_size,
                                               random_state=SPLITTING_SEED, stratify=strat_array)
        X_train, X_test, indx_train, indx_test, *train_test_labels = training_test_split
        y_train = [y for y in train_test_labels[::2]]
        y_test = [y for y in train_test_labels[1::2]]

        # Further split Training Set in Training and Validation Sets

        if stratify:
            strat_array = np.hstack((Xy_numu[stratify_key], Xy_nue[stratify_key]))
            strat_array = strat_array[indx_train]
            strat_array = fstratify(strat_array)
        else:
            strat_array = None

        training_validation_split = train_test_split(X_train, indx_train, *y_train, random_state=SPLITTING_SEED,
                                                     test_size=validation_size, stratify=strat_array)
        X_train, X_val, indx_train, indx_val, *train_validation_labels = training_validation_split
        y_train = [y for y in train_validation_labels[::2]]
        y_val = [y for y in train_validation_labels[1::2]]

        if len(target_keys) == 1:
            # If there is only one target key, it will be saved
            # in the output file as 'y' so to maintain backward-compatibility
            target_keys = ['y']

        y_train_dict = {k: l for k, l in zip(target_keys, y_train)}
        y_val_dict = {k: l for k, l in zip(target_keys, y_val)}
        y_test_dict = {k: l for k, l in zip(target_keys, y_test)}

        np.savez_compressed(os.path.join(out_dir, "Xy_train{}_sel5_doms.npz".format(i+1)),
                            x=X_train.astype(np.uint8), **y_train_dict)

        np.savez_compressed(os.path.join(out_dir, "Xy_val{}_sel5_doms.npz".format(i + 1)),
                            x=X_val.astype(np.uint8), **y_val_dict)

        np.savez_compressed(os.path.join(out_dir, "Xy_test{}_sel5_doms.npz".format(i+1)),
                            x=X_test.astype(np.uint8), **y_test_dict)

        np.savez(os.path.join(out_dir, "Xy_indx{}_sel5_doms.npz".format(i+1)),
                 train=indx_train.astype(np.uint), val=indx_val.astype(np.uint),
                 test=indx_test.astype(np.uint))


def export_train_test_updown_old(i, fname_numu_up, fname_nue_up, fname_numu_down, fname_nue_down, test_size=0.20):
    """
    Function to save to output file X_train, y_train and X_test, y_test
    
    Parameters:
    ------------
    
   """

    with np.load(fname_numu_up) as Xy_numu_up, np.load(fname_nue_up) as Xy_nue_up, np.load(
            fname_numu_down) as Xy_numu_down, np.load(fname_nue_down) as Xy_nue_down:
        X_up = np.vstack((Xy_numu_up["x"], Xy_nue_up["x"]))
        X_down = np.vstack((Xy_numu_down["x"], Xy_nue_down["x"]))

        y_up = np.hstack((Xy_numu_up["y"], Xy_nue_up["y"]))
        y_down = np.hstack((Xy_numu_down["y"], Xy_nue_down["y"]))

        X = np.vstack((X_up, X_down))
        y = np.hstack((y_up, y_down))

        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(X, y, indices,
                                                                                   test_size=test_size,
                                                                                   random_state=SPLITTING_SEED)
        np.savez_compressed("train_test_files/Xy_train" + str(i + 1) + "_sel5_updown" + ".npz",
                            x=X_train.astype(np.uint8), y=y_train.astype(np.uint8))
        np.savez_compressed("train_test_files/Xy_test" + str(i + 1) + "_sel5_updown" + ".npz",
                            x=X_test.astype(np.uint8), y=y_test.astype(np.uint8))
        np.savez("train_test_files/Xy_indx" + str(i + 1) + "_sel5_updown" + ".npz",
                 train=indx_train.astype(np.uint), test=indx_test.astype(np.uint))


def train_validation_split(train_test_folder="train_test_files", prefix="Xy_train", validation_size=0.20):
    """

    Parameters
    ----------
    train_test_folder
    prefix

    Returns
    -------

    """

    training_set_files = list(filter(lambda f: f.startswith(prefix),
                                     os.listdir(train_test_folder)))
    for fname in training_set_files:
        with np.load(fname) as Xy_train:
            X = Xy_train["x"]
            y = Xy_train["y"]

            X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SPLITTING_SEED,
                                                              test_size=validation_size, stratify=y)

            file_idx = int(fname.split('_')[0].replace(prefix, ''))
            np.savez_compressed(os.path.join(train_test_folder, "Xy_train{}_sel5_doms.npz".format(file_idx)),
                                x=X_train.astype(np.uint8), y=y_train.astype(np.int8))
            np.savez_compressed(os.path.join(train_test_folder, "Xy_validation{}_sel5_doms.npz".format(file_idx)),
                                x=X_val.astype(np.uint8), y=y_val.astype(np.int8))
