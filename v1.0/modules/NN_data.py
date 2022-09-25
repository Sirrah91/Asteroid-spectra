import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from copy import deepcopy
from scipy.interpolate import interp1d

from modules.NN_config import *
from modules.utilities import *

quiet = verb == 0


def load_data(filename_data: str, clean_dataset: bool = False, keep_all_labels: bool = False, return_meta: bool = False,
              reinterpolation: bool = True, subfolder_data: str = "") -> Tuple[np.ndarray, ...]:
    # not keeping all labels also mean to not do the normalisation
    # This function load a data from a dataset

    if not quiet:
        print('Loading train file')

    data_file = "".join((project_dir, '/Datasets/', subfolder_data, '/', filename_data))
    data = np.load(data_file, allow_pickle=True)

    # Select training data
    x_train, y_train = deepcopy(data["spectra"].astype(np.float32)), deepcopy(data["labels"].astype(np.float32))
    meta = deepcopy(data["metadata"].astype(np.object))

    if clean_dataset:
        x_train, y_train, inds = clean_data(x_train, y_train, return_indices=True)
        meta = meta[inds]

    # select pure or mixtures (potentially remove a lot of samples and makes the rest of these faster)
    if use_pure_only:
        x_train, y_train, inds = keep_pure_only(x_train, y_train, return_indices=True)
        meta = meta[inds]

    # possible re-interpolation of the data to different wavelength range
    if reinterpolation:
        x_train = reinterpolate_data(x_train, to_what_data=interpolate_to,
                                     wvl_old=data["wavelengths"], wvl_new=new_wvl_grid)

    if not keep_all_labels:
        x_train, y_train = remove_redundant_labels(x_train, y_train)

        stop = num_minerals
        # normalise modal
        if num_minerals > 0:
            start, stop = 0, num_minerals
            norm = np.sum(y_train[:, start:stop], axis=1)
            # normalise only where sum of numbers is non-zero (should not happen)
            non_zero = np.where(norm > 0)[0]
            y_train[non_zero, start:stop] = np.transpose(
                np.divide(np.transpose(y_train[non_zero, start:stop]), norm[non_zero]))

        # normalise chemical
        for k in range(len(subtypes)):
            start, stop = stop, stop + subtypes[k]
            # if subtypes[k] != len(subtypes_all_used[k]):
            norm = np.sum(y_train[:, start:stop], axis=1)
            # normalise only where sum of numbers is non-zero
            # IF IT IS ZERO, FILL IN WITH DUMMY DATA WHICH CAN BE NORMALISED (BEWARE OF FORBIDDEN REGIONS)
            zeros = np.where(norm == 0)[0]
            y_train[zeros, start:stop] = [0.2, 0.8, 0.0][:stop - start]
            norm = np.sum(y_train[:, start:stop], axis=1)
            # non_zero = np.where(norm > 0)[0]
            # y_train[non_zero, start:stop] = np.transpose(
            # np.divide(np.transpose(y_train[non_zero, start:stop]), norm[non_zero]))
            y_train[:, start:stop] = np.transpose(
                np.divide(np.transpose(y_train[:, start:stop]), norm))

    """
    # remove reflectances when these are the same for all spectra (probably due to normalisation)
    # THIS WILL AFFECT LENGTH OF THE INPUT
    ind_to_remove = np.where(np.std(x_train, axis=0) == 0)[0]
    x_train = np.delete(x_train, ind_to_remove, axis=1)
    """

    """
    # normalise the reflectances
    for i in range(np.shape(x_train)[1]):
        s = np.std(x_train[:, i])
        if s < 1e-5:
            x_train[:, i] = 0.
        else:
            x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i])) / s
    """

    if modal_in_wt_percent:  # vol% is default
        y_train = vol_to_wt_percent(y_train)  # this function is not fully working and may cause problems

    if return_meta:
        return x_train, y_train, meta

    return x_train, y_train


def split_data(x_train: np.ndarray, y_train: np.ndarray, rnd_seed: int = 0) -> Tuple[np.ndarray, ...]:
    # This function splits the training data

    if val_portion > 0:
        if not quiet:
            print('Creating validation data')
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_portion,
                                                          random_state=rnd_seed)
    else:
        x_val, y_val = np.array([]), np.array([])

    if test_portion > 0:
        if not quiet:
            print('Creating test data')
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=test_portion / (1 - val_portion),
                                                            random_state=rnd_seed)
    elif val_portion > 0:  # If test portion is zero then use validation data
        x_test, y_test = deepcopy(x_val), deepcopy(y_val)
    else:  # If even val portion is zero, use train data (just for visualisation purposes)
        x_test, y_test = deepcopy(x_train), deepcopy(y_train)

    return x_train, y_train, x_val, y_val, x_test, y_test


def split_data_proportional(x_train: np.ndarray, y_train: np.ndarray, rnd_seed: int = 0) -> Tuple[np.ndarray, ...]:
    # This function splits the training data
    if not quiet:
        print('Splitting data')

    def split_part(x_part: np.ndarray, y_part: np.ndarray) -> Tuple[np.ndarray, ...]:
        N = len(x_part)
        np.random.seed(rnd_seed)
        inds = np.random.choice(range(N), N, replace=False)  # unique `random' numbers
        x_tmp, y_tmp = x_part[inds], y_part[inds]

        # split points
        train_end = np.int(np.round(N * (1 - val_portion - test_portion)))
        val_end = train_end + np.int(np.round(N * val_portion))

        return (x_tmp[:train_end], y_tmp[:train_end], x_tmp[train_end:val_end],
                y_tmp[train_end:val_end], x_tmp[val_end:], y_tmp[val_end:])

    if num_minerals > 0:
        binary = (y_train[:, :num_minerals] > 0).astype(int)
        base = np.array([8, 4, 2, 1])[use_minerals]
        mixtures = np.sum(binary * base, axis=1)
        first = True  # to partially allocate the variables
        for i in range(1, 16):
            indices = mixtures == i

            if not np.any(indices):
                continue

            x_part, y_part = x_train[indices, :], y_train[indices, :]

            if first:
                x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p = split_part(x_part, y_part)
                first = False
            else:
                x_tr, y_tr, x_v, y_v, x_t, y_t = split_part(x_part, y_part)

                x_train_p, y_train_p = np.concatenate((x_train_p, x_tr)), np.concatenate((y_train_p, y_tr))
                x_val_p, y_val_p = np.concatenate((x_val_p, x_v)), np.concatenate((y_val_p, y_v))
                x_test_p, y_test_p = np.concatenate((x_test_p, x_t)), np.concatenate((y_test_p, y_t))

        x_train, y_train, x_val, y_val, x_test, y_test = x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = split_part(x_train, y_train)

    if test_portion == 0:
        if val_portion > 0:  # If test portion is zero then use validation data
            x_test, y_test = deepcopy(x_val), deepcopy(y_val)
        else:  # If even val portion is zero, use train data (just for visualisation purposes)
            x_test, y_test = deepcopy(x_train), deepcopy(y_train)

    return x_train, y_train, x_val, y_val, x_test, y_test


def remove_redundant_labels(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # remove unwanted minerals and their chemical compositions
    ind_to_remove = np.where(used_indices == False)[0]
    y_data = np.delete(y_data, ind_to_remove, axis=1)

    return x_data, y_data


def keep_pure_only(x_data: np.ndarray, y_data: np.ndarray, return_indices: bool = False) -> Tuple[np.ndarray, ...]:

    wanted = np.reshape(y_data[:, np.where(use_minerals_all)], (len(y_data), np.sum(use_minerals_all)))
    unwanted = np.reshape(y_data[:, np.where(~use_minerals_all)], (len(y_data), np.sum(~use_minerals_all)))

    if use_mix_of_the_pure_ones:  # mixtures if sum of unwanted == 0
        inds = np.where(np.sum(unwanted, axis=1) == 0)[0]
    else:  # pure if max of wanted == 1
        inds = np.where(np.max(wanted, axis=1) == 1)[0]

    if return_indices:
        return x_data[inds], y_data[inds], inds

    return x_data[inds], y_data[inds]


def reinterpolate_data(x_data: np.ndarray, wvl_old: np.ndarray, wvl_new: np.ndarray = None,
                       to_what_data: str = None) -> np.ndarray:
    # re-interpolate the spectra to ghe given wavelengths range and the given spacing
    # re-normalise it to the given wavelength

    # old resolution
    fun = interp1d(wvl_old, x_data, kind='cubic')

    if to_what_data is not None:
        if not quiet:
            print('Re-interpolating the data to ' + to_what_data + ' resolution')

        if to_what_data == 'ASPECT':  # roughly ASPECT imager
            wvl_new = np.linspace(500, 1650, 42)
            normalised_at_wvl = 550
        elif to_what_data == 'Itokawa':  # Itokawa
            data_file = "".join((project_dir, '/Datasets/Itokawa.npz'))
            data = np.load(data_file, allow_pickle=True)
            wvl_new = data["wavelengths"]
            normalised_at_wvl = wvl_new[70]  # roughly 1557 nm (done on this pixel in the Itokawa dataset)
        elif to_what_data == 'Eros':  # Eros
            data_file = "".join((project_dir, '/Datasets/Eros.npz'))
            data = np.load(data_file, allow_pickle=True)
            wvl_new = data["wavelengths"]
            normalised_at_wvl = wvl_new[23]  # roughly 1587 nm (done on this pixel in the Eros dataset)
        elif to_what_data == 'Didymos':
            wvl_new = np.arange(490, 2451, 10)
            normalised_at_wvl = 550
        else:
            raise ValueError('Unknown resolution of ' + to_what_data + '.')

        x_data = fun(wvl_new)
        v_norm = np.reshape(fun(normalised_at_wvl), (len(x_data), 1))

    elif wvl_new is not None:
        if not quiet:
            print('Re-interpolating the data to the given grid')
        x_data = fun(wvl_new)

        if new_wvl_grid_normalisation is not None:
            v_norm = np.reshape(fun(new_wvl_grid_normalisation), (len(x_data), 1))
        else:
            v_norm = 1
    else:
        return x_data

    return x_data / v_norm


def remove_nans(y_data: np.ndarray) -> np.ndarray:
    # NaNs in numbers
    inds = np.where(~np.isfinite(np.sum(y_data[:, used_indices], axis=1)))[0]
    inds = np.setdiff1d(np.arange(len(y_data)), inds)

    return inds


def remove_redundant_spectra(y_data: np.ndarray) -> np.ndarray:
    # remove spectra which do not contain much of the wanted minerals
    tmp = np.reshape(y_data[:, np.where(use_minerals_all)], (len(y_data), np.sum(use_minerals_all)))
    indices = np.where(np.sum(tmp, axis=1) >= lim_vol_part)[0]  # also no NaNs in used minerals

    # remove samples of other-than-wanted minerals  # is this part needed after adding of the lim_vol_part part?
    ind_to_remove = np.where(np.sum(y_data, axis=1) == 0)[0]

    inds = np.setdiff1d(indices, ind_to_remove)

    return inds


def remove_duplicities(x_data: np.ndarray) -> np.ndarray:
    indices = np.sort(np.unique(x_data, return_index=True, axis=0)[1])

    return indices


def remove_no_iron_samples(y_data: np.ndarray) -> np.ndarray:
    Fe_OL = np.where(~np.logical_and(y_data[:, 4] < Fa_lim / 100, y_data[:, 0] > 0))[0]
    Fe_OPX = np.where(~np.logical_and(y_data[:, 6] < Fs_lim / 100, y_data[:, 1] > 0))[0]
    Fe_CPX = np.where(~np.logical_and(y_data[:, 9] < Fs_lim / 100, y_data[:, 2] > 0))[0]

    indices = np.intersect1d(np.intersect1d(Fe_OL, Fe_OPX), Fe_CPX)

    return indices


def remove_too_red_spectra(x_data: np.ndarray) -> np.ndarray:
    # spectra with normalised reflectance > red_thresh are deleted
    indices = np.unique(np.where(np.all(x_data <= red_thresh, axis=1))[0])

    return indices


def delete_data(x_data: np.ndarray) -> np.ndarray:
    # this function filter out some samples from spectra catalogue
    # write rows, -2 is to skip headers and count from 0
    inds = np.array([]) - 2

    inds = np.setdiff1d(np.arange(len(x_data)), inds)

    return inds


def clean_data(x_data: np.ndarray, y_data: np.ndarray, return_indices: bool = False) -> Tuple[np.ndarray, ...]:
    if not quiet:
        print('Cleaning data')

    # delete unwanted spectra first (index dependent)
    inds_1 = delete_data(x_data)

    # remove olivine and pyroxene with low iron abundance (weak absorptions, must be before mineral deletion)
    inds_2 = remove_no_iron_samples(y_data)

    # remove spectra of other-than-wanted minerals and remove unwanted columns from data
    inds_3 = remove_redundant_spectra(y_data)

    # remove duplicities in data
    inds_4 = remove_duplicities(x_data)

    # remove spectra with too high reflectance
    inds_5 = remove_too_red_spectra(x_data)

    # Remove NaNs in labels
    inds_6 = remove_nans(y_data)

    # intersection of indices
    inds = np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(inds_1, inds_2), inds_3), inds_4),
                                         inds_5), inds_6)

    if return_indices:
        return x_data[inds], y_data[inds], inds

    return x_data[inds], y_data[inds]


def wt_to_vol_percent(y_data: np.ndarray) -> np.ndarray:
    # should be after the chemicals are filled with dummy data, otherwise you can divide by 0 here
    # zatim nefunguje:
    # pokud je nejaky mineral samotny bez chem slozeni

    # densities of Fa, Fo, Fs, En, Wo, Fs, En, Wo, An, Ab, Or
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])[flatten_list(subtypes_all)]

    # for not-pure samples only
    inds = np.max(y_data[:, :num_minerals], axis=1) != 1

    modals, chemical = deepcopy(y_data[:, :num_minerals]), deepcopy(y_data[:, num_minerals:])
    mineral_density = chemical * densities

    stop = 0
    for i in range(num_minerals):
        start, stop = stop, stop + subtypes[i]
        tmp = np.sum(mineral_density[inds, start:stop], axis=1)
        modals[inds, i] /= tmp

    norm = np.sum(modals, axis=1)
    tmp = np.transpose(np.divide(np.transpose(modals), norm))

    return np.concatenate((tmp, chemical), axis=1)


def vol_to_wt_percent(y_data: np.ndarray) -> np.ndarray:
    # should be after the chemicals are filled with dummy data, otherwise you can divide by 0 here
    # zatim nefunguje:
    # pokud je nejaky mineral samotny bez chem slozeni

    # densities of Fa, Fo, Fs, En, Wo, Fs, En, Wo, An, Ab, Or
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])[flatten_list(subtypes_all)]

    # for not-pure samples only
    inds = np.max(y_data[:, :num_minerals], axis=1) != 1

    modals, chemical = deepcopy(y_data[:, :num_minerals]), deepcopy(y_data[:, num_minerals:])
    mineral_density = chemical * densities

    stop = 0
    for i in range(num_minerals):
        start, stop = stop, stop + subtypes[i]
        tmp = np.sum(mineral_density[inds, start:stop], axis=1)
        modals[inds, i] *= tmp

    norm = np.sum(modals, axis=1)
    tmp = np.transpose(np.divide(np.transpose(modals), norm))

    return np.concatenate((tmp, chemical), axis=1)
