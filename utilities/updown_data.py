import numpy as np
import os


def updown_data(i, fname_numu, fname_nue, fname_numu_updown, fname_nue_updown):
    with np.load(fname_numu) as Xy_numu, np.load(fname_nue) as Xy_nue, np.load(
            fname_numu_updown) as numu_updown, np.load(fname_nue_updown) as nue_updown:
        X_numu = Xy_numu["x"]
        X_nue = Xy_nue["x"]

        numu_up = numu_updown["up"]
        numu_down = numu_updown["down"]
        nue_up = nue_updown["up"]
        nue_down = nue_updown["down"]

        X_numu_up = X_numu[numu_up]
        X_numu_down = X_numu[numu_down]

        X_nue_up = X_nue[nue_up]
        X_nue_down = X_nue[nue_up]

        y_numu_up = np.ones(X_numu_up.shape[0])
        y_numu_down = np.zeros(X_numu_down.shape[0])

        y_nue_up = np.ones(X_nue_up.shape[0])
        y_nue_down = np.zeros(X_nue_down.shape[0])

        np.savez_compressed("Xy_files/Xy_numu_" + str(i + 1) + "_sel5_doms_up.npz",
                            x=X_numu_up.astype(np.uint8), y=y_numu_up.astype(np.uint8))
        np.savez_compressed("Xy_files/Xy_numu_" + str(i + 1) + "_sel5_doms_down.npz",
                            x=X_numu_down.astype(np.uint8), y=y_numu_down.astype(np.uint8))

        np.savez_compressed("Xy_files/Xy_nue_" + str(i + 1) + "_sel5_doms_up.npz",
                            x=X_nue_up.astype(np.uint8), y=y_nue_up.astype(np.uint8))
        np.savez_compressed("Xy_files/Xy_nue_" + str(i + 1) + "_sel5_doms_down.npz",
                            x=X_nue_down.astype(np.uint8), y=y_nue_down.astype(np.uint8))


UP_LABEL = 1
HORIZ_LABEL = 0
DOWN_LABEL = -1


def updown_z(i, fname_Xy, fname_dir_z, out_dir, prefix):
    '''

    Parameters
    ----------
    i
    fname_Xy
    fname_dir_z
    out_dir
    prefix

    Returns
    -------

    '''

    with np.load(fname_Xy) as Xy, np.load(fname_dir_z) as dir_z:

        X = Xy["x"]
        z = dir_z["z"]

        TZ = [np.nonzero(np.sum(X[evt], axis=(1, 2)))[1] for evt in range(X.shape[0])]

        def label_map(coords):
            '''
            Parameters
            ----------
            coords: array-like
                Set of coordinates of hits along z-azis

            '''
            if coords[-1] > coords[0]:
                return UP_LABEL
            elif coords[-1] == coords[0]:
                return HORIZ_LABEL
            else:
                return DOWN_LABEL

        labels = np.array(map(label_map, TZ), dtype=np.int8)

        np.savez_compressed(os.path.join(out_dir, "{}{}_sel5_doms_updown_z.npz".format(prefix, i+1)),
                            x=X.astype(np.uint8), y=labels, z=z)
