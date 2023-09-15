import numpy as np
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import train_test_split
from functools import reduce
from scipy.interpolate import interp1d
from scipy.stats import norm
from keras.utils import np_utils
from typing import Literal
import warnings

from modules.utilities import normalise_in_rows, normalise_array, stack, safe_arange, my_polyfit, is_empty

from modules.utilities_spectra import gimme_indices, used_indices, normalise_spectra, if_no_test_data
from modules.utilities_spectra import join_data, load_npz, apply_transmission, normalise_spectrum_at_wvl

from modules.NN_config_parse import gimme_minerals_all, gimme_num_minerals

from modules.NN_config_composition import mineral_names_short

from modules._constants import _wp, _spectra_name, _wavelengths_name, _label_name
from modules._constants import _sep_in, _sep_out, _rnd_seed, _quiet

# defaults only
from modules.NN_config_composition import comp_grid, comp_filtering_setup, comp_data_split_setup
from modules.NN_config_composition import minerals_used, endmembers_used
from modules.NN_config_taxonomy import tax_grid, tax_data_split_setup, tax_filtering_setup, classes


def load_composition_data(filename_data: str, clean_dataset: bool = True, keep_all_labels: bool = False,
                          return_meta: bool = False, return_wavelengths: bool = False, reinterpolation: bool = True,
                          grid_setup: dict | None = None, filtering_setup: dict | None = None,
                          used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                          subfolder_data: str = "") -> (tuple[np.ndarray, ...] |
                                                        tuple[np.ndarray, np.ndarray, pd.DataFrame] |
                                                        tuple[np.ndarray, np.ndarray, pd.DataFrame], np.ndarray):
    # This function load a data from a dataset
    if grid_setup is None: grid_setup = comp_grid
    if filtering_setup is None: filtering_setup = comp_filtering_setup
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    if not _quiet:
        print("Loading train file")

    data = load_npz(filename_data, subfolder=subfolder_data)

    # Select training data
    wavelengths = deepcopy(np.array(data[_wavelengths_name], dtype=np.float32))
    x_train = deepcopy(np.array(data[_spectra_name], dtype=np.float32))
    y_train = join_data(data, "labels")  # DataFrame with header
    meta = join_data(data, "meta")

    if clean_dataset:  # original data are filtered; after normalisation, there can be some red spectra
        x_train, y_train, inds = clean_data(x_train, y_train, filtering_setup=filtering_setup,
                                            used_minerals=used_minerals, used_endmembers=used_endmembers, return_indices=True)
        meta = meta.iloc[inds]

    # possible re-interpolation of the data to different wavelength range
    if reinterpolation:
        instrument, wvl_grid, wvl_norm = grid_setup["instrument"], grid_setup["wvl_grid"], grid_setup["wvl_norm"]
        wavelengths, x_train = reinterpolate_data(x_train, wvl_old=wavelengths, wvl_new=wvl_grid, wvl_new_norm=wvl_norm,
                                                  instrument=instrument)

    if not keep_all_labels:  # not keeping all labels also mean to not do the normalisation
        y_train = remove_redundant_labels(y_train, used_minerals=used_minerals, used_endmembers=used_endmembers)

        # normalisation to 1
        for start, stop in gimme_indices(used_minerals=used_minerals, used_endmembers=used_endmembers):
            norm = np.sum(y_train[:, start:stop], axis=1)

            # normalise only where sum of numbers is non-zero (should not happen)
            # IF IT IS ZERO, FILL IN WITH DUMMY DATA WHICH CAN BE NORMALISED (BEWARE OF FORBIDDEN REGIONS)
            zeros = np.where(norm == 0.)[0]
            y_train[zeros, start:stop] = [0.2, 0.8, 0.0, 0.0][:stop - start]

            y_train[:, start:stop] = normalise_in_rows(y_train[:, start:stop])

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=_wp), np.array(y_train, dtype=_wp)

    if return_meta:
        if return_wavelengths:
            return x_train, y_train, meta, wavelengths
        else:
            return x_train, y_train, meta
    else:
        if return_wavelengths:
            return x_train, y_train, wavelengths
    return x_train, y_train


def load_taxonomy_data(filename_data: str, clean_dataset: bool = True, return_meta: bool = False,
                       return_wavelengths: bool = False, reinterpolation: bool = True,
                       grid_setup: dict | None = None, filtering_setup: dict | None = None,
                       used_classes: dict[str, int] | None = None,
                       subfolder_data: str = "") -> (tuple[np.ndarray, ...] |
                                                     tuple[np.ndarray, np.ndarray, pd.DataFrame] |
                                                     tuple[np.ndarray, np.ndarray, pd.DataFrame], np.ndarray):
    # This function load a data from a dataset
    if grid_setup is None: grid_setup = tax_grid
    if filtering_setup is None: filtering_setup = tax_filtering_setup
    if used_classes is None: used_classes = classes

    if not _quiet:
        print("Loading train file")

    data = load_npz(filename_data, subfolder=subfolder_data)

    # Select training data
    wavelengths = deepcopy(np.array(data[_wavelengths_name], dtype=np.float32))
    x_train = deepcopy(np.array(data[_spectra_name], dtype=np.float32))
    y_train = deepcopy(np.array(data[_label_name], dtype=str))
    meta = join_data(data, "meta")

    if clean_dataset:
        # metadata contains original taxonomy types even for "-reduced" data
        indices = filter_selected_classes(y_train, used_classes=used_classes)
        x_train, y_train, meta = x_train[indices], y_train[indices], meta.iloc[indices]

    # possible re-interpolation of the data to different wavelength range
    if reinterpolation:
        instrument, wvl_grid, wvl_norm = grid_setup["instrument"], grid_setup["wvl_grid"], grid_setup["wvl_norm"]
        wavelengths, x_train = reinterpolate_data(x_train, wvl_old=wavelengths, wvl_new=wvl_grid, wvl_new_norm=wvl_norm,
                                                  instrument=instrument)
    y_train = classes_to_numbers(y_train, used_classes=used_classes)

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=_wp), np.array(y_train, dtype=_wp)

    if return_meta:
        if return_wavelengths:
            return x_train, y_train, meta, wavelengths
        else:
            return x_train, y_train, meta
    else:
        if return_wavelengths:
            return x_train, y_train, wavelengths
    return x_train, y_train


def split_composition_data_proportional(x_data: np.ndarray, y_data: np.ndarray,
                                        metadata: np.ndarray | pd.DataFrame | None = None,
                                        val_portion: float | None = None, test_portion: float | None = None,
                                        rnd_seed: int | None = _rnd_seed,
                                        used_minerals: np.ndarray | None = None,
                                        out_type: type = _wp) -> (tuple[np.ndarray, ...] |
                                                                  tuple[np.ndarray, np.ndarray, pd.DataFrame,
                                                                  np.ndarray, np.ndarray, pd.DataFrame,
                                                                  np.ndarray, np.ndarray, pd.DataFrame]):
    if val_portion is None: val_portion = comp_data_split_setup["val_portion"]
    if test_portion is None: test_portion = comp_data_split_setup["test_portion"]
    if used_minerals is None: used_minerals = minerals_used

    # This function splits the training data
    if not _quiet:
        print("Splitting data")

    def return_indices(mask: np.ndarray) -> tuple[np.ndarray, ...]:
        N = len(mask)
        mask = mask[rng.permutation(N)]

        test_end = int(np.round(N * test_portion))
        val_end = test_end + int(np.round(N * val_portion))

        # minimum training 1 sample per mixture type
        if val_end == np.sum(mask):  # no training sample
            if val_end == test_end:  # no validation sample, move one test to train if train is empty
                test_end -= 1
                val_end = test_end
            else:  # move one validation to train if train is empty
                val_end -= 1

        return indices_all[mask][val_end:], indices_all[mask][test_end:val_end], indices_all[mask][:test_end]

    indices_all = np.arange(len(x_data), dtype=int)
    rng = np.random.default_rng(seed=rnd_seed)

    # num_minerals = int(np.round(np.sum(y_data[0]) - 1))
    num_minerals = gimme_num_minerals(used_minerals)

    if num_minerals > 0:
        binary = np.array(y_data[:, :num_minerals] > 0, dtype=int)
        # this keeps the same order of mixtures even if a mineral is not present (each mineral has own base)
        base = np.array([2. ** i for i in reversed(range(len(used_minerals)))], dtype=int)[used_minerals]
        mixtures = np.sum(binary * base, axis=1)

        ind_train, ind_val, ind_test = zip(*[return_indices(np.where(mixtures == i)[0]) for i in range(1, np.sum(base) + 1)])
        ind_train, ind_val, ind_test = stack(ind_train), stack(ind_val), stack(ind_test)

    else:
        ind_train, ind_val, ind_test = return_indices(np.full(len(x_data), True))

    x_train, y_train = x_data[ind_train], y_data[ind_train]
    x_val, y_val = x_data[ind_val], y_data[ind_val]
    x_test, y_test = x_data[ind_test], y_data[ind_test]

    # if test_portion == 0:
    if is_empty(x_test):
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val)

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=out_type), np.array(y_train, dtype=out_type)
    x_val, y_val = np.array(x_val, dtype=out_type), np.array(y_val, dtype=out_type)
    x_test, y_test = np.array(x_test, dtype=out_type), np.array(y_test, dtype=out_type)

    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            meta_train, meta_val, meta_test = metadata.iloc[ind_train], metadata.iloc[ind_val], metadata.iloc[ind_test]
        else:
            meta_train, meta_val, meta_test = metadata[ind_train], metadata[ind_val], metadata[ind_test]

        # if test_portion == 0:
        if is_empty(meta_test):
            _, meta_test = if_no_test_data(None, meta_train, None, meta_val)

        return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def split_taxonomy_data_proportional(x_data: np.ndarray, y_data: np.ndarray,
                                     metadata: np.ndarray | pd.DataFrame | None = None,
                                     val_portion: float | None = None, test_portion: float | None = None,
                                     rnd_seed: int | None = _rnd_seed,
                                     out_type: type = _wp
                                     ) -> (tuple[np.ndarray, ...] |
                                           tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray,
                                           pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]):

    if val_portion is None: val_portion = tax_data_split_setup["val_portion"]
    if test_portion is None: test_portion = tax_data_split_setup["test_portion"]

    # This function splits the training data
    if not _quiet:
        print("Splitting data")


    def return_indices(mask: np.ndarray) -> tuple[np.ndarray, ...]:
        N = len(mask)
        mask = mask[rng.permutation(N)]

        test_end = int(np.round(N * test_portion))
        val_end = test_end + int(np.round(N * val_portion))

        # minimum training 1 sample per taxonomy class
        if val_end == np.sum(mask):  # no training sample
            if val_end == test_end:  # no validation sample, move one test to train if train is empty
                test_end -= 1
                val_end = test_end
            else:  # move one validation to train if train is empty
                val_end -= 1

        return indices_all[mask][val_end:], indices_all[mask][test_end:val_end], indices_all[mask][:test_end]

    rng = np.random.default_rng(seed=rnd_seed)
    indices_all = np.arange(len(x_data), dtype=int)

    ind_train, ind_val, ind_test = zip(*[return_indices(np.where(y_data == i)[0]) for i in range(int(np.max(y_data)) + 1)])
    ind_train, ind_val, ind_test = stack(ind_train), stack(ind_val), stack(ind_test)

    x_train, y_train = x_data[ind_train], y_data[ind_train]
    x_val, y_val = x_data[ind_val], y_data[ind_val]
    x_test, y_test = x_data[ind_test], y_data[ind_test]

    # if test_portion == 0:
    if is_empty(x_test):
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val)

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=out_type), np.array(y_train, dtype=out_type)
    x_val, y_val = np.array(x_val, dtype=out_type), np.array(y_val, dtype=out_type)
    x_test, y_test = np.array(x_test, dtype=out_type), np.array(y_test, dtype=out_type)

    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            meta_train, meta_val, meta_test = metadata.iloc[ind_train], metadata.iloc[ind_val], metadata.iloc[ind_test]
        else:
            meta_train, meta_val, meta_test = metadata[ind_train], metadata[ind_val], metadata[ind_test]

        # if test_portion == 0:
        if is_empty(meta_test):
            _, meta_test = if_no_test_data(None, meta_train, None, meta_val)

        return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def split_data(x_data: np.ndarray, y_data: np.ndarray, model_type: Literal["composition", "taxonomy"],
               metadata: pd.DataFrame | np.ndarray | None = None,
               val_portion: float | None = None, test_portion: float | None = None,
               rnd_seed: int | None = _rnd_seed,
               used_minerals: np.ndarray | None = None,
               out_type: type = _wp) -> (tuple[np.ndarray, ...] |
                                         tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray,
                                         pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]):

    # This function splits the training data
    if model_type == "composition":
        data_split_setup = comp_data_split_setup

    elif model_type == "taxonomy":
        data_split_setup = tax_data_split_setup

    else:
        raise ValueError("Invalid input. Specify if the data contain composition or taxonomies.")

    if used_minerals is None: used_minerals = minerals_used
    if val_portion is None: val_portion = data_split_setup["val_portion"]
    if test_portion is None: test_portion = data_split_setup["test_portion"]

    def gimme_stratification(y_data: np.ndarray) -> np.ndarray | None:
        if model_type == "taxonomy":
            stratify = deepcopy(y_data)
        else:
            # num_minerals = int(np.round(np.sum(y_data[0]) - 1))
            num_minerals = gimme_num_minerals(used_minerals)

            if num_minerals > 0:
                binary = np.array(y_data[:, :num_minerals] > 0, dtype=int)
                # this keeps the same order of mixtures even if a mineral is not present (each mineral has own base)
                base = np.array([2. ** i for i in reversed(range(len(used_minerals)))], dtype=int)[used_minerals]
                stratify = np.sum(binary * base, axis=1)
            else:
                stratify = None

        return stratify

    ind_train = np.arange(len(x_data), dtype=int)
    ind_val = np.array([], dtype=int)
    ind_test = np.array([], dtype=int)

    if val_portion > 0.:
        if not _quiet:
            print("Creating validation data")
        ind_train, ind_val, _, _ = train_test_split(ind_train, y_data,
                                                    test_size=val_portion,
                                                    stratify=gimme_stratification(y_data),
                                                    random_state=rnd_seed)

    if test_portion > 0.:
        if not _quiet:
            print("Creating test data")
        ind_train, ind_test, _, _ = train_test_split(ind_train, y_data[ind_train],
                                                     test_size=test_portion / (1. - val_portion),
                                                     stratify=gimme_stratification(y_data[ind_train]),
                                                     random_state=rnd_seed)

    x_train, y_train = x_data[ind_train], y_data[ind_train]
    x_val, y_val = x_data[ind_val], y_data[ind_val]
    x_test, y_test = x_data[ind_test], y_data[ind_test]

    # if test_portion == 0:
    if is_empty(x_test):
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val)

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=out_type), np.array(y_train, dtype=out_type)
    x_val, y_val = np.array(x_val, dtype=out_type), np.array(y_val, dtype=out_type)
    x_test, y_test = np.array(x_test, dtype=out_type), np.array(y_test, dtype=out_type)

    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            meta_train, meta_val, meta_test = metadata.iloc[ind_train], metadata.iloc[ind_val], metadata.iloc[ind_test]
        else:
            meta_train, meta_val, meta_test = metadata[ind_train], metadata[ind_val], metadata[ind_test]

        # if test_portion == 0:
        if is_empty(meta_test):
            _, meta_test = if_no_test_data(None, meta_train, None, meta_val)

        return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def reinterpolate_data(x_data: np.ndarray, wvl_old: np.ndarray, wvl_new: np.ndarray | None = None,
                       wvl_new_norm: float | str | None = None,
                       instrument: str | None = None) -> tuple[np.ndarray, ...]:
    # re-interpolate the spectra to ghe given wavelengths range and the given spacing
    # re-normalise it to the given wavelength

    if instrument is not None:
        print(f"Re-interpolating the data to the {instrument} resolution.")
        if "ASPECT" in instrument:
            specifications = instrument.split(_sep_out)
            if not(len(specifications) == 3 and specifications[-1].isdigit()):
                raise ValueError(f'Invalid ASPECT specification ({instrument}). '
                                 f'Correct format is "ASPECT_channel1-channel2-..._resolution"')

            usable_channels = tuple(specifications[1].split(_sep_in))
            targeted_resolution = float(specifications[2])

            wvl_new, filtered_data = apply_aspect_filter(wvl_old, x_data,
                                                         targeted_resolution=targeted_resolution,
                                                         wvl_norm=wvl_new_norm,
                                                         usable_channels=usable_channels)

        elif "HS-H" in instrument:
            wvl_new, filtered_data = apply_hyperscout_filter(wvl_old, x_data, wvl_norm=wvl_new_norm)

        else:
            raise ValueError("Unknown instrument.")

        return np.array(wvl_new, dtype=np.float32), np.array(filtered_data, dtype=_wp)

    if wvl_new is not None:
        m, M, res = np.min(wvl_new), np.max(wvl_new), np.mean(np.diff(wvl_new))
        if wvl_new_norm is not None:
            model_grid = _sep_in.join(str(int(x)) for x in np.round([m, M, res, wvl_new_norm]))
        else:
            model_grid = _sep_in.join(str(int(x)) for x in np.round([m, M, res]))
        print(f"Re-interpolating the data to the {model_grid} grid.")

        # new resolution
        x_data = interp1d(wvl_old, x_data, kind="cubic")(wvl_new)

        if wvl_new_norm == "adaptive":
            wvl_new_norm = normalise_spectrum_at_wvl(wvl_new)

        if wvl_new_norm is not None:
            x_data = normalise_spectra(x_data, wvl_new, wvl_norm_nm=wvl_new_norm)

        return np.array(wvl_new, dtype=np.float32), np.array(x_data, dtype=_wp)


    return np.array(wvl_old, dtype=np.float32), np.array(x_data, dtype=_wp)


def apply_aspect_filter(wvl_data: np.ndarray, x_data: np.ndarray,
                        targeted_resolution: float = 30.,
                        wvl_norm: float | str | None = None,
                        usable_channels: tuple[str, ...] = ("vis", "nir1", "nir2", "swir"),
                        remove_overlaps: bool = False) -> tuple[np.ndarray, ...]:
    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))

    vis = safe_arange(650., 900., targeted_resolution, endpoint=True)
    nir1 = safe_arange(850., 1250., targeted_resolution, endpoint=True)
    nir2 = safe_arange(1200., 1600., targeted_resolution, endpoint=True)
    swir = safe_arange(1650., 2500., targeted_resolution, endpoint=True)

    sigma_vis = my_polyfit([np.min(vis), np.max(vis)], (20., 20.), 1,
                           return_fit_only=True, x_fit=vis) * fwhm_to_sigma
    sigma_nir1 = my_polyfit([np.min(nir1), np.max(nir2)], (40., 27.), 1,
                            return_fit_only=True, x_fit=nir1) * fwhm_to_sigma
    sigma_nir2 = my_polyfit([np.min(nir1), np.max(nir2)], (40., 27.), 1,
                            return_fit_only=True, x_fit=nir2) * fwhm_to_sigma
    sigma_swir = my_polyfit([np.min(swir), np.max(swir)], (45., 45.), 1,
                            return_fit_only=True, x_fit=swir) * fwhm_to_sigma

    wvl_new, sigma_new = np.array([]), np.array([])

    if "vis" in usable_channels:
        wvl_new, sigma_new = vis, sigma_vis

        if remove_overlaps and "nir1" in usable_channels:
            inds = wvl_new < np.min(nir1)
            wvl_new, sigma_new = wvl_new[inds], sigma_new[inds]

    if "nir1" in usable_channels:
        wvl_new, sigma_new = stack((wvl_new, nir1)), stack((sigma_new, sigma_nir1))

        if remove_overlaps and "nir2" in usable_channels:
            inds = wvl_new < np.min(nir2)
            wvl_new, sigma_new = wvl_new[inds], sigma_new[inds]

    if "nir2" in usable_channels:
        wvl_new, sigma_new = stack((wvl_new, nir2)), stack((sigma_new, sigma_nir2))

        if remove_overlaps and "swir" in usable_channels:
            inds = wvl_new < np.min(swir)
            wvl_new, sigma_new = wvl_new[inds], sigma_new[inds]

    if "swir" in usable_channels:
        wvl_new, sigma_new = stack((wvl_new, swir)), stack((sigma_new, sigma_swir))

    # this is to accept longer wavelength filter if it is overlap with previous
    # (e.g.resolution 10 nm, 850-900 from nir1, not from vis)
    wvl_new, sigma_new = wvl_new[::-1], sigma_new[::-1]

    wvl_new, sort_index = np.unique(wvl_new, return_index=True)
    sigma_new = sigma_new[sort_index]

    mask = np.logical_and.reduce((wvl_new >= np.min(wvl_data),
                                  wvl_new <= np.max(wvl_data)))

    wvl_new, sigma_new = wvl_new[mask], sigma_new[mask]

    gauss = np.transpose(norm.pdf(np.reshape(wvl_data, (len(wvl_data), 1)), wvl_new, sigma_new))  # one Gaussian per row

    wvl_new, filtered_data = apply_transmission(spectra=x_data, transmission=gauss,
                                                wvl_transmission=wvl_data, wvl_cen_method="argmax")

    if wvl_norm == "adaptive":
        wvl_norm = normalise_spectrum_at_wvl(wvl_new)

    if wvl_norm is not None:
        filtered_data = normalise_spectra(filtered_data, wvl_new, wvl_norm_nm=wvl_norm)

    return np.array(wvl_new, dtype=np.float32), np.array(filtered_data, dtype=_wp)


def apply_hyperscout_filter(wvl_data: np.ndarray, x_data: np.ndarray,
                            wvl_norm: float | str | None = None) -> tuple[np.ndarray, ...]:
    HS = load_npz(f"HS{_sep_in}H{_sep_out}transmission.npz", subfolder="HyperScout")
    wvl_raw, transmissions = HS["wavelengths"], HS["transmissions"]

    mask = np.logical_and.reduce((665. <= wvl_raw,  # or 650.?
                                  wvl_raw <= 975.,  # or 960.?
                                  wvl_raw >= np.min(wvl_data),
                                  wvl_raw <= np.max(wvl_data)))

    wvl_raw, transmissions = wvl_raw[mask], transmissions[:, mask]

    x_data = interp1d(wvl_data, x_data, kind="cubic")(wvl_raw)

    wvl_central, filtered_data = apply_transmission(spectra=x_data, transmission=transmissions,
                                                    wvl_transmission=wvl_raw, wvl_cen_method="argmax")

    if wvl_norm == "adaptive":
        wvl_norm = normalise_spectrum_at_wvl(wvl_central)

    if wvl_norm is not None:
        filtered_data = normalise_spectra(filtered_data, wvl_central, wvl_norm_nm=wvl_norm)

    return np.array(wvl_central, dtype=np.float32), np.array(filtered_data, dtype=_wp)


def remove_redundant_labels(y_data: np.ndarray,
                            used_minerals: np.ndarray | None = None,
                            used_endmembers: list[list[bool]] | None = None) -> np.ndarray:
    # remove unwanted minerals and their chemical compositions

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    y_data = y_data[:, used_indices(used_minerals, used_endmembers)]

    return y_data


def clean_data(x_data: np.ndarray, y_data: pd.DataFrame,
               return_indices: bool = False,
               filtering_setup: dict | None = None,
               used_minerals: np.ndarray | None = None,
               used_endmembers: list[list[bool]] | None = None
               ) -> tuple[np.ndarray, ...]:

    if filtering_setup is None: filtering_setup = comp_filtering_setup
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    # delete unwanted spectra first (index dependent)
    inds_1 = delete_data(x_data)

    # remove olivine and pyroxene with low iron abundance (weak absorptions, must be before mineral deletion)
    inds_2 = remove_no_iron_samples(y_data, filtering_setup["Fa_lim"], filtering_setup["Fs_lim"],
                                    used_minerals=used_minerals, used_endmembers=used_endmembers,
                                    remove_high_iron_unwanted=True, keep_if_not_used=False)

    # remove spectra of other-than-wanted minerals and remove unwanted columns from data
    inds_3 = remove_redundant_spectra(y_data,  filtering_setup["lim_vol_part"],
                                      used_minerals=used_minerals, used_endmembers=used_endmembers)

    # remove duplicities in data
    inds_4 = remove_duplicities(x_data)

    # remove spectra with too high reflectance
    inds_5 = remove_too_red_spectra(x_data, filtering_setup["red_thresh"])

    # remove NaNs in labels
    inds_6 = remove_nans(y_data, used_minerals=used_minerals, used_endmembers=used_endmembers)

    if filtering_setup["use_pure_only"]:
        # select pure or mixtures (potentially remove a lot of samples and makes the rest of these faster)
        inds_7 = keep_pure_only(y_data, filtering_setup["use_mix_of_the_pure_ones"],
                                used_minerals=used_minerals, used_endmembers=used_endmembers)
    else:
        inds_7 = np.arange(len(x_data))

    # intersection of indices
    inds = reduce(np.intersect1d, [inds_1, inds_2, inds_3, inds_4, inds_5, inds_6, inds_7])
    inds = np.array(inds, dtype=int)

    if return_indices:
        return np.array(x_data[inds], dtype=_wp), np.array(y_data.iloc[inds], dtype=_wp), inds

    return np.array(x_data[inds], dtype=_wp), np.array(y_data.iloc[inds], dtype=_wp)


def delete_data(x_data: np.ndarray) -> np.ndarray:
    # this function filter out some samples based on selected indices
    ind_to_remove = np.array([])

    inds = np.setdiff1d(np.arange(len(x_data)), ind_to_remove)

    return np.array(inds, dtype=int)


def remove_no_iron_samples(y_data: pd.DataFrame,
                           Fa_threshold: float = 0.,
                           Fs_threshold: float = 0.,
                           used_minerals: np.ndarray | None = None,
                           used_endmembers: list[list[bool]] | None = None,
                           remove_high_iron_unwanted: bool = True,
                           keep_if_not_used: bool = False) -> np.ndarray:
    # this function is applied before removing unwanted labels

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    all_minerals = gimme_minerals_all(used_minerals, used_endmembers)

    # iron limits are not important if the mineral is not present (0 modal and 0 iron should not be removed)
    header = np.array([c for c in y_data if c.startswith(tuple(mineral_names_short[:3]))])
    indices1 = np.array(y_data[header] > 0.)  # to remove

    # iron limits (if mineral is not in all_minerals, you actually prefer low-iron samples which are featureless)
    header = np.array(["Fa", "Fs (OPX)", "Fs (CPX)"])
    limits = np.array([Fa_threshold, Fs_threshold, Fs_threshold])

    if remove_high_iron_unwanted:
        indices2 = np.transpose([y_data[header[i]] < limits[i] / 100. if all_minerals[i] else
                                 y_data[header[i]] >= limits[i] / 100. for i in range(len(header))])  # to remove
        mask = np.logical_and(indices1, indices2)
    else:
        indices2 = np.array(y_data[header] < limits / 100.)  # to remove
        mask = np.logical_and(indices1, indices2)[:, all_minerals[:3]]

        # if you ask for modal but not in mineral composition -> you should remove low-iron samples anyway,
        # or you can possibly derive ol/opx from a flat spectra of OL and OPX
        # (you can mix them with any ratio and get just a flat spectrum...)

        # If you want to keep such samples (you ask for modal but do not want to delete low-iron chemical)
        if keep_if_not_used:
            # iron limits are not important if the end-member is not present
            endmember_mask = np.array([used_endmembers[0][0], used_endmembers[1][0],
                                       used_endmembers[2][0]])[all_minerals[:3]]
            mask = mask[:, endmember_mask]

    indices = np.where(~np.any(mask, axis=1))[0]  # to keep

    return np.array(indices, dtype=int)


def remove_redundant_spectra(y_data: pd.DataFrame,
                             vol_part_thresh: float = 0.,
                             used_minerals: np.ndarray | None = None,
                             used_endmembers: list[list[bool]] | None = None) -> np.ndarray:
    # remove spectra which do not contain much of the wanted minerals

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    all_minerals = gimme_minerals_all(used_minerals, used_endmembers)
    header = np.array([c for c in y_data if c.startswith(tuple(mineral_names_short))])[all_minerals]

    # np.array is needed here to remove NaNs in used minerals (which pandas does not do)
    inds = np.where(np.sum(np.array(y_data[header]), axis=1) >= vol_part_thresh)[0]

    return np.array(inds, dtype=int)


def remove_duplicities(x_data: np.ndarray) -> np.ndarray:
    indices = np.sort(np.unique(x_data, return_index=True, axis=0)[1])

    return np.array(indices, dtype=int)


def remove_too_red_spectra(x_data: np.ndarray, red_threshold: float = np.inf) -> np.ndarray:
    # spectra with normalised reflectance > red_thresh are deleted
    indices = np.unique(np.where(np.all(x_data <= red_threshold, axis=1))[0])

    return np.array(indices, dtype=int)


def remove_nans(y_data: pd.DataFrame,
                used_minerals: np.ndarray | None = None,
                used_endmembers: list[list[bool]] | None = None) -> np.ndarray:
    # NaNs in numbers

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    y_data = np.array(y_data)  # pandas does not remove NaNs, numpy does
    inds = np.where(np.isfinite(np.sum(y_data[:, used_indices(used_minerals, used_endmembers)], axis=1)))[0]

    return np.array(inds, dtype=int)


def keep_pure_only(y_data: pd.DataFrame,
                   use_mix_of_the_pure_ones: bool = False,
                   used_minerals: np.ndarray | None = None,
                   used_endmembers: list[list[bool]] | None = None) -> np.ndarray:

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    all_minerals = gimme_minerals_all(used_minerals, used_endmembers)
    header = np.array([c for c in y_data if c.startswith(tuple(mineral_names_short))])

    wanted = y_data[header[all_minerals]]
    unwanted = y_data[header[~all_minerals]]

    if use_mix_of_the_pure_ones:  # mixtures if sum of unwanted == 0
        inds = np.where(np.sum(unwanted, axis=1) == 0.)[0]
    else:  # pure if max of wanted == 1
        inds = np.where(np.max(wanted, axis=1) == 1.)[0]

    return np.array(inds, dtype=int)


def classes_to_numbers(args: np.ndarray, used_classes: dict[str, int] | None = None) -> list[float]:
    # This function convert classes to numbers which is required by the neural network
    if used_classes is None: used_classes = classes

    return [used_classes[x] for x in np.ravel(args)]


def numbers_to_classes(args: np.ndarray, used_classes: dict[str, int] | None = None) -> list[str] | str:
    # This function convert numbers to classes
    if used_classes is None: used_classes = classes

    classes_reverse = {value: key for key, value in used_classes.items()}
    if np.ndim(args) == 0:  # return only string, not list[str]
        return classes_reverse[args]

    return [classes_reverse[x] for x in np.ravel(args)]


def labels_to_categories(y_data: np.ndarray,
                         num_labels: int | None = None,
                         used_classes: dict[str, int] | None = None) -> np.ndarray:
    # Split test data into distinct class labels
    if num_labels is not None and used_classes is not None:
        raise ValueError('"num_labels" and "used_classes" cannot be both None.')

    check_classes = False

    if used_classes is None:
        used_classes = classes
    else:
        check_classes = True

    if num_labels is None:
        num_labels = len(used_classes)

    unique_labels = np.unique(y_data)

    if check_classes:
        use_unknown_class = "Other" in used_classes
        unused_labels = numbers_to_classes(np.setdiff1d(list(used_classes.values()), unique_labels), used_classes=used_classes)

        if use_unknown_class:
            unused_labels.remove("Other")
        if unused_labels:
            warnings.warn(f"Labels \"{', '.join(unused_labels)}\" are not used. Check the data and config.")

    try:
        return np_utils.to_categorical(y_data, num_labels, dtype=_wp)
    except IndexError:
        raise IndexError("There are more categories in your data than what is defined in the config or they have "
                         "higher indices. Check the data and config.")


def filter_selected_classes(taxonomy_classes: np.ndarray, used_classes: dict[str, int] | None = None) -> np.ndarray:
    if used_classes is None: used_classes = classes

    return np.array([ind for ind, taxonomy_class in enumerate(taxonomy_classes)
                     if taxonomy_class in list(used_classes.keys())])


def apply_aspect_fwhm(x_data: np.ndarray, wvl_old: np.ndarray, wvl_new: np.ndarray,
                      wvl_norm: float) -> tuple[np.ndarray, ...]:
    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))

    fwhm_vis = my_polyfit((400., 850.), (20., 20.), 1)
    fwhm_nir = my_polyfit((850., 1600.), (40., 27.), 1)
    fwhm_swir = my_polyfit((1600., 2500.), (45., 45.), 1)

    filtered_data = np.zeros((len(x_data), len(wvl_new)))

    for i, wvl in enumerate(wvl_new):
        if wvl < 850.:
            sigma = np.polyval(fwhm_vis, wvl) * fwhm_to_sigma
        elif wvl <= 1600.:
            sigma = np.polyval(fwhm_nir, wvl) * fwhm_to_sigma
        else:
            sigma = np.polyval(fwhm_swir, wvl) * fwhm_to_sigma

        gauss = norm.pdf(wvl_old, wvl, sigma)
        gauss = normalise_array(gauss)
        filtered_data[:, i] = np.sum(x_data * gauss, axis=1)

    return wvl_new, normalise_spectra(filtered_data, wvl_new, wvl_norm_nm=wvl_norm)
