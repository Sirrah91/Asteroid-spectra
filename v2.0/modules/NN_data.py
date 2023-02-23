import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from functools import reduce
from scipy.interpolate import interp1d
from scipy.stats import norm
from keras.utils import np_utils
from typing import Literal
import warnings

from modules.utilities import normalise_in_rows, is_constant, stack, safe_arange
from modules.utilities_spectra import gimme_indices, used_indices, convert_classes, denoise_and_norm
from modules.utilities_spectra import if_no_test_data, vol_to_wt_percent

from modules.NN_config import num_minerals, minerals_all, minerals_used, endmembers_counts, endmembers_used, red_thresh
from modules.NN_config import Fa_lim, Fs_lim, lim_vol_part, use_pure_only, modal_in_wt_percent, use_mix_of_the_pure_ones

from modules.NN_config_taxonomy import classes, num_labels  # num_labels is not used by the composition model

from modules.NN_config_range_test import instrument

from modules.NN_config import verb as verb_comp
from modules.NN_config_taxonomy import verb as verb_tax

from modules._constants import _path_data


def load_composition_data(filename_data: str, clean_dataset: bool = True, keep_all_labels: bool = False,
                          return_meta: bool = False, reinterpolation: bool = True,
                          subfolder_data: str = "") -> tuple[np.ndarray, ...]:
    # not keeping all labels also mean to not do the normalisation
    # This function load a data from a dataset

    from modules.NN_config import interpolate_to, new_wvl_grid, new_wvl_grid_normalisation

    quiet = verb_comp == 0

    if not quiet:
        print("Loading train file")

    data_file = "".join((_path_data, subfolder_data, "/", filename_data))
    data = np.load(data_file, allow_pickle=True)

    # Select training data
    x_train = deepcopy(np.array(data["spectra"], dtype=np.float32))
    y_train = deepcopy(np.array(data["labels"], dtype=np.float32))
    meta = deepcopy(np.array(data["metadata"], dtype=object))

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
                                     wvl_old=data["wavelengths"], wvl_new=new_wvl_grid,
                                     new_normalisation=new_wvl_grid_normalisation)

    if not keep_all_labels:
        x_train, y_train = remove_redundant_labels(x_train, y_train)

        # normalisation to 1
        for start, stop in gimme_indices(num_minerals, endmembers_counts):
            norm = np.sum(y_train[:, start:stop], axis=1)

            # normalise only where sum of numbers is non-zero (should not happen)
            # IF IT IS ZERO, FILL IN WITH DUMMY DATA WHICH CAN BE NORMALISED (BEWARE OF FORBIDDEN REGIONS)
            zeros = np.where(norm == 0)[0]
            y_train[zeros, start:stop] = [0.2, 0.8, 0.0, 0.0][:stop - start]

            norm = np.sum(y_train[:, start:stop], axis=1)
            y_train[:, start:stop] = normalise_in_rows(y_train[:, start:stop], norm)

    if modal_in_wt_percent:  # vol% is default
        # this function is not fully working and may cause problems
        y_train = vol_to_wt_percent(y_train, minerals_all, endmembers_used)

    if return_meta:
        return x_train, y_train, meta

    return x_train, y_train


def load_taxonomy_data(filename_data: str, clean_dataset: bool = True, return_meta: bool = False,
                       reinterpolation: bool = True, subfolder_data: str = "") -> tuple[np.ndarray, ...]:
    # This function load a data from a dataset

    from modules.NN_config_taxonomy import interpolate_to, new_wvl_grid, new_wvl_grid_normalisation

    quiet = verb_tax == 0

    if not quiet:
        print("Loading train file")

    data_file = "".join((_path_data, subfolder_data, "/", filename_data))
    data = np.load(data_file, allow_pickle=True)

    # Select training data
    x_train = deepcopy(np.array(data["spectra"], dtype=np.float32))
    y_train = deepcopy(np.array(data["labels"], dtype=str))
    meta = deepcopy(np.array(data["metadata"], dtype=object))

    if clean_dataset:
        indices = filter_selected_classes(meta[:, 1])
        x_train, y_train, meta = x_train[indices], y_train[indices], meta[indices]

    # possible re-interpolation of the data to different wavelength range
    if reinterpolation:
        x_train = reinterpolate_data(x_train, wvl_old=data["wavelengths"], to_what_data=interpolate_to,
                                     wvl_new=new_wvl_grid, new_normalisation=new_wvl_grid_normalisation)

    y_train = convert_classes(y_train)

    # Set data into numpy arrays
    x_train, y_train = np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    if return_meta:
        return x_train, y_train, meta

    return x_train, y_train


def load_test_range_data(filename_data: str, index_of_range: int, clean_dataset: bool = True,
                         keep_all_labels: bool = False, return_meta: bool = False,
                         subfolder_data: str = "") -> tuple[np.ndarray, ...]:
    # not keeping all labels also mean to not do the normalisation
    # This function load a data from a dataset

    quiet = verb_comp == 0

    if not quiet:
        print("Loading train file")

    data_file = "".join((_path_data, subfolder_data, "/", filename_data))
    data = np.load(data_file, allow_pickle=True)

    # Select training data
    x_train = deepcopy(np.array(data["spectra"], dtype=np.float32))
    y_train = deepcopy(np.array(data["labels"], dtype=np.float32))
    meta = deepcopy(np.array(data["metadata"], dtype=object))

    if clean_dataset:
        x_train, y_train, inds = clean_data(x_train, y_train, return_indices=True)
        meta = meta[inds]

    # select pure or mixtures (potentially remove a lot of samples and makes the rest of these faster)
    if use_pure_only:
        x_train, y_train, inds = keep_pure_only(x_train, y_train, return_indices=True)
        meta = meta[inds]

    # re-interpolation of the data to different wavelength range
    if instrument is None:  # re-interpolate to the given grid
        from modules.NN_config_range_test import spacing, ranges
        x_train = reinterpolate_test_range_data(x_train, wvl_old=data["wavelengths"],
                                                wvl_spacing=spacing[index_of_range],
                                                from_wvl=ranges[index_of_range][0],
                                                to_wvl=ranges[index_of_range][1])
    elif instrument == "ASPECT":
        from modules.NN_config_range_test import wvl_all
        wavelengths = wvl_all[index_of_range]
        x_train = apply_aspect_like_filter(wavelengths, x_train)
    else:
        raise ValueError('instrument must be None or "ASPECT"')

    if not keep_all_labels:
        x_train, y_train = remove_redundant_labels(x_train, y_train)

        # normalisation to 1
        for start, stop in gimme_indices(num_minerals, endmembers_counts):
            norm = np.sum(y_train[:, start:stop], axis=1)

            # normalise only where sum of numbers is non-zero (should not happen)
            # IF IT IS ZERO, FILL IN WITH DUMMY DATA WHICH CAN BE NORMALISED (BEWARE OF FORBIDDEN REGIONS)
            zeros = np.where(norm == 0)[0]
            y_train[zeros, start:stop] = [0.2, 0.8, 0.0, 0.0][:stop - start]

            norm = np.sum(y_train[:, start:stop], axis=1)
            y_train[:, start:stop] = normalise_in_rows(y_train[:, start:stop], norm)

    if return_meta:
        return x_train, y_train, meta

    return x_train, y_train


def split_composition_data_proportional(x_train: np.ndarray, y_train: np.ndarray,
                                        rnd_seed: int = 42) -> tuple[np.ndarray, ...]:
    from modules.NN_config import val_portion, test_portion

    quiet = verb_comp == 0

    # This function splits the training data
    if not quiet:
        print("Splitting data")

    def split_part(x_part: np.ndarray, y_part: np.ndarray) -> tuple[np.ndarray, ...]:
        N = len(x_part)
        rng = np.random.default_rng(rnd_seed)
        inds = rng.permutation(N)

        x_tmp, y_tmp = x_part[inds], y_part[inds]

        # split points
        train_end = int(np.round(N * (1 - val_portion - test_portion)))
        val_end = train_end + int(np.round(N * val_portion))

        return (x_tmp[:train_end], y_tmp[:train_end], x_tmp[train_end:val_end],
                y_tmp[train_end:val_end], x_tmp[val_end:], y_tmp[val_end:])

    if num_minerals > 0:
        binary = np.array(y_train[:, :num_minerals] > 0, dtype=int)
        # this keeps the same order of mixtures even if a mineral is not present (each mineral has own base)
        base = np.array([2**i for i in reversed(range(len(minerals_used)))])[minerals_used]
        mixtures = np.sum(binary * base, axis=1)
        first = True  # to partially allocate the variables
        for i in range(1, np.sum(base) + 1):
            indices = mixtures == i

            if not np.any(indices):
                continue

            x_part, y_part = x_train[indices, :], y_train[indices, :]

            if first:
                x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p = split_part(x_part, y_part)
                first = False
            else:
                x_tr, y_tr, x_v, y_v, x_t, y_t = split_part(x_part, y_part)

                x_train_p, y_train_p = stack((x_train_p, x_tr), axis=0), stack((y_train_p, y_tr), axis=0)
                x_val_p, y_val_p = stack((x_val_p, x_v), axis=0), stack((y_val_p, y_v), axis=0)
                x_test_p, y_test_p = stack((x_test_p, x_t), axis=0), stack((y_test_p, y_t), axis=0)

        x_train, y_train, x_val, y_val, x_test, y_test = x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = split_part(x_train, y_train)

    if test_portion == 0:
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val, val_portion)

    return x_train, y_train, x_val, y_val, x_test, y_test


def split_taxonomy_data_proportional(x_train: np.ndarray, y_train: np.ndarray,
                                     rnd_seed: int = 42) -> tuple[np.ndarray, ...]:
    from modules.NN_config_taxonomy import val_portion, test_portion

    quiet = verb_tax == 0

    # This function splits the training data
    if not quiet:
        print("Splitting data")

    def split_part(x_part: np.ndarray, y_part: np.ndarray) -> tuple[np.ndarray, ...]:
        N = len(x_part)
        rng = np.random.default_rng(rnd_seed)
        inds = rng.permutation(N)

        x_tmp, y_tmp = x_part[inds], y_part[inds]

        # split points
        train_end = int(np.round(N * (1 - val_portion - test_portion)))
        val_end = train_end + int(np.round(N * val_portion))

        return (x_tmp[:train_end], y_tmp[:train_end], x_tmp[train_end:val_end],
                y_tmp[train_end:val_end], x_tmp[val_end:], y_tmp[val_end:])

    first = True  # to partially allocate the variables
    for i in range(num_labels):
        indices = y_train == i

        if not np.any(indices):
            continue

        x_part, y_part = x_train[indices, :], y_train[indices]

        if first:
            x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p = split_part(x_part, y_part)
            first = False
        else:
            x_tr, y_tr, x_v, y_v, x_t, y_t = split_part(x_part, y_part)

            x_train_p, y_train_p = stack((x_train_p, x_tr), axis=0), stack((y_train_p, y_tr))
            x_val_p, y_val_p = stack((x_val_p, x_v), axis=0), stack((y_val_p, y_v))
            x_test_p, y_test_p = stack((x_test_p, x_t), axis=0), stack((y_test_p, y_t))

    x_train, y_train, x_val, y_val, x_test, y_test = x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p

    if test_portion == 0:
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val, val_portion)

    return x_train, y_train, x_val, y_val, x_test, y_test


def split_meta_proportional(meta: np.ndarray, y_data: np.ndarray,
                            model_type: Literal["composition", "taxonomy"]) -> tuple[np.ndarray, ...]:

    # Might be inaccurate if you use different seed for splitting data and metadata

    if model_type == "composition":
        split_data_proportional = split_composition_data_proportional
    else:
        split_data_proportional = split_taxonomy_data_proportional

    meta_train, _, meta_val, _, meta_test, _ = split_data_proportional(meta, y_data)

    return meta_train, meta_val, meta_test


def split_data(x_train: np.ndarray, y_train: np.ndarray, rnd_seed: int = 42) -> tuple[np.ndarray, ...]:
    # This function splits the training data
    from modules.NN_config import val_portion, test_portion

    quiet = verb_comp == 0

    if val_portion > 0:
        if not quiet:
            print("Creating validation data")
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_portion,
                                                          random_state=rnd_seed)
    else:
        x_val, y_val = np.array([]), np.array([])

    if test_portion > 0:
        if not quiet:
            print("Creating test data")
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=test_portion / (1. - val_portion),
                                                            random_state=rnd_seed)
    else:
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val, val_portion)

    return x_train, y_train, x_val, y_val, x_test, y_test



def reinterpolate_data(x_data: np.ndarray, wvl_old: np.ndarray, wvl_new: np.ndarray | None = None,
                       new_normalisation: float | None = None,
                       to_what_data: Literal["Itokawa", "Eros", "Didymos", "ASPECT", "ASPECT_swir"] | None = None
                       ) -> np.ndarray:
    # re-interpolate the spectra to ghe given wavelengths range and the given spacing
    # re-normalise it to the given wavelength

    # old resolution
    fun = interp1d(wvl_old, x_data, kind="cubic")

    if to_what_data is not None:
        print("Re-interpolating the data to", to_what_data, "resolution")

        if to_what_data in ["Itokawa", "Eros", "Didymos"]:
            if to_what_data == "Didymos":
                to_what_data += "_2022"
            data_file = "".join((_path_data, to_what_data, "-denoised-norm.npz"))

            data = np.load(data_file, allow_pickle=True)
            wvl_new, reference_spectra = data["wavelengths"], data["spectra"]

            x_data = fun(wvl_new)

            normalised_at_wvl = wvl_new[is_constant(reference_spectra, axis=0, constant=1.0)]

            if np.size(normalised_at_wvl) == 1:
                v_norm = np.reshape(fun(normalised_at_wvl), (len(x_data), 1))
            else:
                warnings.warn("Normalisation can't be applied")
                v_norm = 1.

        elif "ASPECT" in to_what_data:  # roughly ASPECT imager
            keep_swir = "swir" in to_what_data
            wvl_new, x_data = apply_aspect_filter(wvl_old, x_data, keep_swir=keep_swir)
            v_norm = 1.  # normalised in the filtering function

        else:
            raise ValueError("Unknown resolution of", to_what_data)

    elif wvl_new is not None:
        print("Re-interpolating the data to the given grid")
        x_data = fun(wvl_new)

        if new_normalisation is not None:
            v_norm = np.reshape(fun(new_normalisation), (len(x_data), 1))
        else:
            v_norm = 1.
    else:
        return x_data

    return x_data / v_norm


def apply_aspect_filter(wvl_data: np.ndarray, x_data: np.ndarray, keep_swir: bool = False) -> tuple[np.ndarray, ...]:
    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))

    wvl_data = np.reshape(wvl_data, (len(wvl_data), 1))

    vis = np.linspace(500., 850., 13)[:-1]
    sigma_vis = np.polyval(np.polyfit([np.min(vis), np.max(vis)], (20., 20.), 1), vis) * fwhm_to_sigma

    nir = np.linspace(850., 1600., 26)
    sigma_nir = np.polyval(np.polyfit([np.min(nir), np.max(nir)], (40., 27.), 1), nir) * fwhm_to_sigma

    if keep_swir:
        swir = np.linspace(1650., 2500., 30)
        sigma_swir = np.polyval(np.polyfit([np.min(swir), np.max(swir)], (45., 45.), 1), swir) * fwhm_to_sigma

        wvl_new = stack((vis, nir, swir))
        sigma_new = stack((sigma_vis, sigma_nir, sigma_swir))
    else:
        wvl_new = stack((vis, nir))
        sigma_new = stack((sigma_vis, sigma_nir))

    gauss = norm.pdf(wvl_data, wvl_new, sigma_new)
    gauss /= np.sum(gauss, axis=0)
    filtered_data = x_data @ gauss

    # renormalisation
    normalised_at_wvl = wvl_data[is_constant(x_data, axis=0, constant=1.0)]

    if np.size(normalised_at_wvl) == 1:
        filtered_data = denoise_and_norm(data=filtered_data, wavelength=wvl_new, denoising=False, normalising=True,
                                         normalised_at_wvl=normalised_at_wvl)
    else:
        warnings.warn("Normalisation can't be applied")

    return wvl_new, filtered_data


def remove_redundant_labels(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # remove unwanted minerals and their chemical compositions
    ind_to_remove = np.where(~used_indices(minerals_used, endmembers_used))[0]
    y_data = np.delete(y_data, ind_to_remove, axis=1)

    return x_data, y_data


def keep_pure_only(x_data: np.ndarray, y_data: np.ndarray, return_indices: bool = False) -> tuple[np.ndarray, ...]:

    wanted = y_data[:, np.where(minerals_all)[0]]
    unwanted = y_data[:, np.where(~minerals_all)[0]]

    if use_mix_of_the_pure_ones:  # mixtures if sum of unwanted == 0
        inds = np.where(np.sum(unwanted, axis=1) == 0)[0]
    else:  # pure if max of wanted == 1
        inds = np.where(np.max(wanted, axis=1) == 1)[0]

    if return_indices:
        return x_data[inds], y_data[inds], inds

    return x_data[inds], y_data[inds]


def remove_nans(y_data: np.ndarray) -> np.ndarray:
    # NaNs in numbers
    inds = np.where(~np.isfinite(np.sum(y_data[:, used_indices(minerals_used, endmembers_used)], axis=1)))[0]
    inds = np.setdiff1d(np.arange(len(y_data)), inds)

    return inds


def remove_redundant_spectra(y_data: np.ndarray) -> np.ndarray:
    # remove spectra which do not contain much of the wanted minerals
    tmp = y_data[:, np.where(minerals_all)[0]]
    indices = np.where(np.sum(tmp, axis=1) >= lim_vol_part)[0]  # also no NaNs in used minerals

    # remove samples of other-than-wanted minerals  # is this part needed after adding of the lim_vol_part part?
    ind_to_remove = np.where(np.sum(y_data, axis=1) == 0)[0]

    inds = np.setdiff1d(indices, ind_to_remove)

    return inds


def remove_duplicities(x_data: np.ndarray) -> np.ndarray:
    indices = np.sort(np.unique(x_data, return_index=True, axis=0)[1])

    return indices


def remove_no_iron_samples(y_data: np.ndarray) -> np.ndarray:
    # this function is applied before removing unwanted labels
    iron_positions = [[0, 0], [1, 0], [2, 0]]
    minerals_total = len(minerals_all)
    endmembers_total = [len(endmember) for endmember in endmembers_used]

    inds = []

    for i, j in iron_positions:
        if i == 0:
            limit = Fa_lim
        elif i in [1, 2]:
            limit = Fs_lim
        else:
            limit = 0.

        if endmembers_used[i][j]:
            mineral_pos = i
            endmember_pos = int(minerals_total + np.sum(endmembers_total[:i]) + j)

            inds.append(np.where(~np.logical_and(y_data[:, endmember_pos] < limit / 100.,
                                                 y_data[:, mineral_pos] > 0.))[0])

    indices = reduce(np.intersect1d, inds)

    return indices


def remove_too_red_spectra(x_data: np.ndarray) -> np.ndarray:
    # spectra with normalised reflectance > red_thresh are deleted
    indices = np.unique(np.where(np.all(x_data <= red_thresh, axis=1))[0])

    return indices


def delete_data(x_data: np.ndarray) -> np.ndarray:
    # this function filter out some samples based on selected indices
    ind_to_remove = np.array([])

    inds = np.setdiff1d(np.arange(len(x_data)), ind_to_remove)

    return inds


def clean_data(x_data: np.ndarray, y_data: np.ndarray, return_indices: bool = False) -> tuple[np.ndarray, ...]:
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
    inds = reduce(np.intersect1d, [inds_1, inds_2, inds_3, inds_4, inds_5, inds_6])

    if return_indices:
        return x_data[inds], y_data[inds], inds

    return x_data[inds], y_data[inds]


def labels_to_categories(y_data: np.ndarray) -> np.ndarray:
    # Split test data into distinct class labels
    return np_utils.to_categorical(y_data, num_labels, dtype=np.float32)


def filter_selected_classes(taxonomy_classes: np.ndarray) -> np.ndarray:
    return np.array([ind for ind, taxonomy_class in enumerate(taxonomy_classes)
                     if taxonomy_class in list(classes.keys())])


def reinterpolate_test_range_data(x_data: np.ndarray, wvl_old: np.ndarray,
                                  from_wvl: float, to_wvl: float, wvl_spacing: float = 10) -> np.ndarray:
    # re-interpolate the spectra to ghe given wavelengths range and the given wavelength spacing
    # re-normalise it to the given wavelength
    quiet = verb_comp == 0

    if not quiet:
        print("Re-interpolating the data")

    # old resolution
    fun = interp1d(wvl_old, x_data, kind="cubic")

    wvl_new = safe_arange(from_wvl, to_wvl, wvl_spacing, endpoint=True)

    if wvl_new[0] <= 550 <= wvl_new[-1]:
        normalised_at_wvl = 550.
    elif wvl_new[0] <= 750 <= wvl_new[-1]:
        normalised_at_wvl = 750.
    elif wvl_new[0] <= 950 <= wvl_new[-1]:
        normalised_at_wvl = 950.
    elif wvl_new[0] <= 1150 <= wvl_new[-1]:
        normalised_at_wvl = 1150.
    elif wvl_new[0] <= 1350 <= wvl_new[-1]:
        normalised_at_wvl = 1350.
    elif wvl_new[0] <= 1550 <= wvl_new[-1]:
        normalised_at_wvl = 1550.
    elif wvl_new[0] <= 1750 <= wvl_new[-1]:
        normalised_at_wvl = 1750.
    elif wvl_new[0] <= 1950 <= wvl_new[-1]:
        normalised_at_wvl = 1950.
    elif wvl_new[0] <= 2150 <= wvl_new[-1]:
        normalised_at_wvl = 2150.
    elif wvl_new[0] <= 2350 <= wvl_new[-1]:
        normalised_at_wvl = 2350.
    else:
        normalised_at_wvl = wvl_new[0]

    x_data = fun(wvl_new)
    v_norm = np.reshape(fun(normalised_at_wvl), (len(x_data), 1))

    return x_data / v_norm


def apply_aspect_like_filter(wavelengths: np.ndarray, reflectances: np.ndarray) -> np.ndarray:
    wvl_old = safe_arange(450., 2450., 5., endpoint=True)  # this must follow resolution of your loaded data

    vis = np.linspace(500., 850., 13)[:-1]
    sigma_vis = np.polyfit((500., 850.), (20., 20.), 1)

    nir = np.linspace(850, 1600, 26)[:-1]
    sigma_nir = np.polyfit((850., 1600.), (40., 27.), 1)

    swir = np.linspace(1600., 2500., 30)
    sigma_swir = np.polyfit((1600., 2500.), (45., 45.), 1)

    filtered_data = np.zeros((len(reflectances), len(wavelengths)))

    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))

    for i, wvl in enumerate(wavelengths):
        if vis[0] <= wvl < nir[0]:
            sigma = np.polyval(sigma_vis, wvl) * fwhm_to_sigma
        elif nir[0] <= wvl < swir[0]:
            sigma = np.polyval(sigma_nir, wvl) * fwhm_to_sigma
        else:
            sigma = np.polyval(sigma_swir, wvl) * fwhm_to_sigma

        gauss = norm.pdf(wvl_old, wvl, sigma)
        gauss /= np.sum(gauss)
        filtered_data[:, i] = np.sum(reflectances * gauss, axis=1)

    return filtered_data
