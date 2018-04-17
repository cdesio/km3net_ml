import numpy as np
from export_train_test import export_train_validation_test
import os

folder = os.path.join('/', 'data', 'km3net', 'Xy_multi_data_files')
main_output_dir = os.path.join(folder, 'train_test_files')
if not os.path.exists(main_output_dir):
    os.makedirs(main_output_dir)

output_dir = os.path.join(main_output_dir, 'log_energies_stratified')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fnames_numu = [os.path.join(folder, "Xy_numu_{}_multi_data.npz".format(str(i + 1))) for i in range(100)]
fnames_nue = [os.path.join(folder, "Xy_nue_{}_multi_data.npz".format(str(i + 1))) for i in range(100)]


def stratify_on_energies(E):
    logE = np.log10(E)
    minE, maxE = np.min(logE), np.max(logE)
    BIN_EDGES = sorted([minE, 1.0, 2.8, 3.5, 4.0, 7.0, maxE+0.1])
    hist, _ = np.histogram(logE, bins=np.asarray(BIN_EDGES))
    no_split_indx = (np.where(hist < 2)[0]) + 1  #
    BIN_EDGES = sorted([edge for idx, edge in enumerate(BIN_EDGES) if idx not in no_split_indx])
    assert np.all(np.histogram(logE, bins=np.asarray(BIN_EDGES))[0] > 1)
    return np.digitize(logE, np.asarray(BIN_EDGES))


for i in range(100):
    export_train_validation_test(i, fnames_numu[i], fnames_nue[i], out_dir=output_dir, target_key='E', stratify_key='E',
                                 ftarget=lambda E: np.log10(E), fstratify=stratify_on_energies)
    print('Export Complete {}'.format(i + 1))
