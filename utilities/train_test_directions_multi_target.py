from export_train_test import export_train_validation_test
import os

folder = os.path.join('/', 'data', 'km3net', 'Xy_multi_data_files')
main_output_dir = os.path.join(folder, 'train_test_files')
if not os.path.exists(main_output_dir):
    os.makedirs(main_output_dir)

output_dir = os.path.join(main_output_dir, 'multi_target_directions')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fnames_numu = [os.path.join(folder, "Xy_numu_{}_multi_data.npz".format(str(i + 1))) for i in range(100)]
fnames_nue = [os.path.join(folder, "Xy_nue_{}_multi_data.npz".format(str(i + 1))) for i in range(100)]
for i in range(100):
    export_train_validation_test(i, fnames_numu[i], fnames_nue[i], out_dir=output_dir,
                                 multi_target_keys=['dirx', 'diry', 'dirz'], stratify=False)
    print('Export Complete {}'.format(i + 1))
