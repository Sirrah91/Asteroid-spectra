from os import path, environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from copy import deepcopy
from typing import Callable, Literal
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential, Model
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.spatial import ConvexHull
from glob import glob
import h5py

from modules.utilities import (check_dir, flatten_list, normalise_in_rows, denoise_array, safe_arange, is_empty, stack,
                               split_path, argnearest, my_argmax, return_mean_std, my_polyfit, check_file, is_sorted)

from modules.NN_config_parse import (gimme_minerals_all, gimme_num_minerals, gimme_endmember_counts, bin_to_used,
                                     bin_to_cls)

from modules._constants import (_path_data, _path_model, _model_suffix, _spectra_name, _wavelengths_name, _wp,
                                _metadata_name, _metadata_key_name, _label_name, _label_key_name, _path_catalogues,
                                _sep_out, _sep_in)

# defaults only
from modules.CD_parameters import denoise, normalise
from modules.NN_config_composition import (minerals_used, endmembers_used, comp_grid, comp_filtering_setup,
                                           comp_model_setup, comp_output_setup)
from modules.NN_config_taxonomy import classes, tax_output_setup


def save_data(final_name: str, spectra: np.ndarray, wavelengths: np.ndarray, metadata: np.ndarray,
              labels: np.ndarray | None = None, labels_key: np.ndarray | None = None,
              metadata_key: np.ndarray | None = None, other_info: dict | None = None, subfolder: str = "",
              denoised: bool | None = None, normalised: bool | None = None) -> str:
    if len(spectra) != len(metadata):
        raise ValueError("Each spectrum must have its metadata. Length of spectra != length of metadata.")

    if denoised is None: denoised = denoise
    if normalised is None: normalised = normalise

    final_name = filename_adjustment(final_name, denoising=denoised, normalising=normalised)
    final_name = path.join(_path_data, subfolder, final_name)

    # collect data and metadata
    data_and_metadata = {_spectra_name: np.array(spectra, dtype=_wp),  # save spectra
                         _wavelengths_name: np.array(wavelengths, dtype=_wp),  # save wavelengths
                         _metadata_name: np.array(metadata, dtype=object)}  # save metadata

    if metadata_key is not None:
        data_and_metadata[_metadata_key_name] = np.array(metadata_key, dtype=str)

    if labels is not None:
        if len(spectra) != len(labels):
            raise ValueError("Each spectrum must have its label. Length of spectra != length of labels.")

        if np.shape(labels)[1] == 1:  # taxonomy class
            data_and_metadata[_label_name] = np.array(labels, dtype=str)  # save labels
        else:  # composition
            data_and_metadata[_label_name] = np.array(labels, dtype=_wp)  # save labels

        if labels_key is not None:
            data_and_metadata[_label_key_name] = np.array(labels_key, dtype=str)

    if other_info is not None:  # existing keys are not updated
        data_and_metadata = other_info | data_and_metadata

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, **data_and_metadata)

    return final_name


def load_npz(filename: str, subfolder: str = "", list_keys: list[str] | None = None,
             allow_pickle: bool = True, **kwargs):
    filename = check_file(filename, _path_data, subfolder)

    data = np.load(filename, allow_pickle=allow_pickle, **kwargs)

    if list_keys is None:
        return data

    return {key: data[key][()] for key in list_keys if key in data.files}


def load_h5(filename: str, subfolder: str = "", list_keys: list[str] | None = None) -> h5py.File | dict:
    filename = check_file(filename, _path_data, subfolder)

    if list_keys is None:
        print("Do not forget to close the file.")
        return h5py.File(filename, "r")

    with h5py.File(filename, "r") as f:
        return {key: np.array(f[key]) for key in list_keys if key in f.keys()}


def load_keras_model(filename: str, subfolder: str = "", custom_objects: dict | None = None,
                     compile: bool = True, **kwargs) -> Model:
    if custom_objects is None: custom_objects = gimme_custom_objects(model_name=filename)

    filename = check_file(filename, _path_model, subfolder)

    # compile=True is needed to get metrics names for composition vs. taxonomy check
    model = load_model(filename, custom_objects=custom_objects, compile=compile, **kwargs)

    return model


def load_xlsx(filename: str, subfolder: str = "", **kwargs) -> pd.DataFrame:
    filename = check_file(filename, _path_catalogues, subfolder)
    excel = pd.read_excel(filename, **kwargs)

    return excel


def load_txt(filename: str, subfolder: str = "", **kwargs) -> pd.DataFrame:
    filename = check_file(filename, _path_data, subfolder)
    data = pd.read_csv(filename, **kwargs)

    return data


def npz_to_dat(filename: str, data_in_columns: bool = True) -> None:
    data = load_npz(filename, subfolder="")

    # save wavelengths spectra
    spectra = stack((data[_wavelengths_name], data[_spectra_name]), axis=0)
    spectra_dat = filename.replace(".npz", ".dat")
    fmt = "%.5f"

    if data_in_columns:
        np.savetxt(spectra_dat, np.transpose(spectra), fmt=fmt, delimiter='\t')
    else:
        np.savetxt(spectra_dat, data, fmt=fmt, delimiter='\t')

    # save metadata
    if _metadata_name in data.files:
        if _metadata_key_name in data.files:
            meta = stack((data[_metadata_key_name], data[_metadata_name]), axis=0)
        else:
            meta = data[_metadata_name]

        meta_dat = filename.replace(".npz", f"{_sep_out}metadata.dat")
        fmt = "%s"
        np.savetxt(meta_dat, meta, fmt=fmt, delimiter='\t')

    # save labels
    if _label_name in data.files:
        if _label_key_name in data.files:
            labels = stack((data[_label_key_name], data[_label_name]), axis=0)
        else:
            labels = data[_label_name]

        labels_dat = filename.replace(".npz", f"{_sep_out}labels.dat")
        fmt = "%.5f" if np.issubdtype(np.result_type(labels), np.number) else "%s"
        np.savetxt(labels_dat, labels, fmt=fmt, delimiter='\t')


def combine_files(filenames: tuple[str, ...], final_name: str, subfolder: str = "") -> str:

    denoised = np.all(["denoised" in filename for filename in filenames])
    normalised = np.all(["norm" in filename for filename in filenames])

    final_name = filename_adjustment(final_name, denoising=denoised, normalising=normalised)
    final_name = path.join(_path_data, subfolder, final_name)

    combined_file = dict(load_npz(filenames[0]))
    for filename in filenames[1:]:
        file_to_merge = dict(load_npz(filename))

        if np.all(combined_file[_wavelengths_name] == file_to_merge[_wavelengths_name]):
            combined_file[_spectra_name] = stack((combined_file[_spectra_name], file_to_merge[_spectra_name]), axis=0)
            combined_file[_label_name] = stack((combined_file[_label_name], file_to_merge[_label_name]), axis=0)
            combined_file[_metadata_name] = stack((combined_file[_metadata_name], file_to_merge[_metadata_name]), axis=0)

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, **combined_file)

    return final_name


def filename_adjustment(filename: str, denoising: bool, normalising: bool, always_add_suffix: bool = False) -> str:
    tmp = Path(filename)
    final_name = tmp.name.replace(tmp.suffix, "")  # remove suffix

    if denoising:
        if always_add_suffix or "_denoised" not in final_name:
            final_name += f"{_sep_out}denoised"
    if normalising:
        if always_add_suffix or "_norm" not in final_name:
            final_name += f"{_sep_out}norm"

    return f"{final_name}.npz"


def gimme_predicted_class(y_pred: np.ndarray, used_classes: dict | list | np.ndarray | None = None,
                          return_index: bool = False) -> np.ndarray:
    if used_classes is None: used_classes = classes

    if isinstance(used_classes, dict):
        list_of_classes = np.array(list(used_classes.keys()))
    else:
        list_of_classes = np.array(used_classes)

    most_probable = np.argmax(y_pred, axis=1)

    if return_index:
        return most_probable

    return list_of_classes[most_probable]


def if_no_test_data(x_train: np.ndarray | None, y_train: np.ndarray,
                    x_val: np.ndarray | None, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not is_empty(y_val):  # If the test portion is zero then use validation data
        x_test, y_test = deepcopy(x_val), deepcopy(y_val)
    else:  # If even the val portion is zero, use train data (just for visualisation purposes)
        x_test, y_test = deepcopy(x_train), deepcopy(y_train)

    return x_test, y_test


def join_data(data: np.lib.npyio.NpzFile | np.ndarray, header: str | np.ndarray) -> pd.DataFrame:
    if isinstance(header, str):
        if "meta" in header:
            return pd.DataFrame(data=data[_metadata_name], columns=data[_metadata_key_name])
        else:
            return pd.DataFrame(data=data[_label_name], columns=data[_label_key_name])
    else:
        return pd.DataFrame(data=data, columns=header)


def denoise_spectra(data: np.ndarray, wavelength: np.ndarray, sigma_nm: float | None = 7.) -> np.ndarray:
    if sigma_nm is None:
        return deepcopy(data)

    if sigma_nm <= 0.:
        raise ValueError(f'"sigma_nm" must be positive float but equals {sigma_nm}')

    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    nm_to_px = 1. / (wavelength[1] - wavelength[0])  # conversion from nm to px
    sigma_px = sigma_nm * nm_to_px

    return denoise_array(data, sigma_px=sigma_px)


def normalise_spectra(data: np.ndarray, wavelength: np.ndarray, wvl_norm_nm: float | None = 550.,
                      on_pixel: bool = True, fun: Callable[[float], float] | None = None) -> np.ndarray:
    if wvl_norm_nm is None:
        return deepcopy(data)

    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    if wvl_norm_nm in wavelength:
        v_norm = data[:, wavelength == wvl_norm_nm]

    elif on_pixel:
        v_norm = data[:, argnearest(wavelength, wvl_norm_nm)]

    elif fun is not None:
        v_norm = fun(wvl_norm_nm)

    else:
        v_norm = interp1d(wavelength, data, kind="cubic")(wvl_norm_nm)

    return normalise_in_rows(data, norm_vector=v_norm, norm_constant=1.)


def denoise_and_norm(data: np.ndarray, wavelength: np.ndarray,
                     denoising: bool = True, normalising: bool = True,
                     sigma_nm: float = 7.,
                     wvl_norm_nm: float = 550.,
                     on_pixel: bool = True) -> np.ndarray:
    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    if denoising:
        data = denoise_spectra(data, wavelength, sigma_nm=sigma_nm)

    # Normalised reflectance
    if normalising:
        return normalise_spectra(data, wavelength, wvl_norm_nm=wvl_norm_nm, on_pixel=on_pixel)

    return deepcopy(data)


def denoise_and_norm_file(file: str, denoising: bool, normalising: bool, sigma_nm: float = 7.,
                     normalised_at_wvl: float = 550., subfolder: str = "") -> None:
    if not denoising and not normalising:
        return

    # load the data
    data = load_npz(file)

    xq, spectra = data[_wavelengths_name], data[_spectra_name]

    spectra = denoise_and_norm(spectra, xq, denoising=denoising, normalising=normalising, sigma_nm=sigma_nm,
                               wvl_norm_nm=normalised_at_wvl)

    final_name = filename_adjustment(file, denoising=denoising, normalising=normalising, always_add_suffix=False)

    save_data(final_name, spectra=spectra, wavelengths=xq, labels=data[_label_name],
              metadata=data[_metadata_name], labels_key=data[_label_key_name], metadata_key=data[_metadata_key_name],
              denoised=denoising, normalised=normalise, subfolder=subfolder)


def clean_and_resave(filename: str, reinterpolate: bool = False, used_minerals: np.ndarray | None = None,
                     used_endmembers: list[list[bool]] | None = None, grid_setup: dict | None = None,
                     filtering_setup: dict | None = None,) -> None:
    from modules.NN_data import load_composition_data as load_data

    if grid_setup is None: grid_setup = comp_grid
    if filtering_setup is None: filtering_setup = comp_filtering_setup
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    denoised = "denoise" in filename
    normalised = "norm" in filename

    tmp = Path(filename)
    final_name = f"{tmp.stem}{_sep_out}clean"  # ".npz" is added in save_data

    # Load data for keys and wavelengths
    data = load_npz(filename, subfolder="")

    # load cleaned data
    spectra, labels, meta = load_data(filename, return_meta=True, keep_all_labels=True, clean_dataset=True,
                                      reinterpolation=reinterpolate, used_minerals=used_minerals,
                                      used_endmembers=used_endmembers, grid_setup=grid_setup,
                                      filtering_setup=filtering_setup)

    # re-save it
    save_data(final_name, spectra=spectra, wavelengths=data[_wavelengths_name], labels=labels, metadata=meta,
              labels_key=data[_label_key_name], metadata_key=data[_metadata_key_name],
              denoised=denoised, normalised=normalised)


def apply_transmission(spectra: np.ndarray,
                       transmission: np.ndarray,
                       wvl_transmission: np.ndarray,
                       wvl_cen_method: Literal["argmax", "dot"] = "argmax") -> tuple[np.ndarray, ...]:

    if np.ndim(transmission) == 1:  # need num_transmissions x num_wavelengths
        transmission = np.reshape(transmission, (1, -1))

    # sort wavelengths first
    idx = np.argsort(wvl_transmission)
    wvl_transmission, transmission = wvl_transmission[idx], transmission[:, idx]

    transmission = normalise_in_rows(transmission, simps(y=transmission, x=wvl_transmission))
    # transmission = normalise_in_rows(transmission)

    if wvl_cen_method == "argmax":
        wvl_central = np.array([my_argmax(wvl_transmission, transm, fit_method="ransac") for transm in transmission])

    elif wvl_cen_method == "dot":
        wvl_central = simps(y=transmission * wvl_transmission, x=wvl_transmission)
        # wvl_central = np.dot(transmission, wvl_transmission)

    else:
        raise ValueError('Unknown method how to estimate central wavelengths. Available methods are "argmax" and "dot".')

    index_sort = np.argsort(wvl_central)
    wvl_central, transmission = wvl_central[index_sort], transmission[index_sort]

    final_spectra = simps(y=np.einsum('ij, kj -> ikj', spectra, transmission), x=wvl_transmission)
    # final_spectra = spectra @ np.transpose(transmission)

    wvl_central, final_spectra = np.array(wvl_central, dtype=_wp), np.array(final_spectra, dtype=_wp)

    return wvl_central, final_spectra


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                    cleaning: bool = True, all_to_one: bool = False,
                    return_r2: bool = False, return_sam: bool = False, return_mae: bool = False,
                    remove_px_outliers: bool = False,
                    return_dict: bool = False) -> tuple[np.ndarray, ...] | dict[str, np.ndarray]:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_mae, my_rmse, my_r2, my_sam

    if np.any(y_true > 1.) or np.any(y_pred > 1.):
        y_true, y_pred = y_true / 100., y_pred / 100.

    if remove_px_outliers:
        ind_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                                     px_only=True)
        y_true, y_pred = np.delete(y_true, ind_outliers, axis=0), np.delete(y_pred, ind_outliers, axis=0)

    metric = my_rmse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                     cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred).numpy()
    results = {"RMSE (pp)": metric}

    if return_r2:
        metric = my_r2(used_minerals=used_minerals, used_endmembers=used_endmembers,
                       cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred).numpy()
        results["R2"] = metric

    if return_sam:
        metric = np.rad2deg(my_sam(used_minerals=used_minerals, used_endmembers=used_endmembers,
                                   cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred).numpy())
        results["SAM (deg)"] = metric

    if return_mae:
        metric = my_mae(used_minerals=used_minerals, used_endmembers=used_endmembers,
                        cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred).numpy()
        results["MAE (pp)"] = metric

    if return_dict:
        return results

    return tuple([*results.values()])


def compute_within(y_true: np.ndarray, y_pred: np.ndarray, error_limit: tuple[float, ...],
                   used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                   cleaning: bool = True, all_to_one: bool = False,
                   step_percentile: float | None = 0.1,
                   remove_px_outliers: bool = False,
                   return_dict: bool = False) -> tuple[np.ndarray, ...] | dict[str, np.ndarray]:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_quantile, my_ae

    if np.any(y_true > 1.) or np.any(y_pred > 1.):
        y_true, y_pred = y_true / 100., y_pred / 100.

    if remove_px_outliers:
        ind_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                                     px_only=True)
        y_true, y_pred = np.delete(y_true, ind_outliers, axis=0), np.delete(y_pred, ind_outliers, axis=0)

    if step_percentile is None:
        ae = my_ae(used_minerals=used_minerals, used_endmembers=used_endmembers,
                   cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred)
        within = np.array([100. * np.sum(ae <= x, axis=0) / np.sum(np.isfinite(ae), axis=0) for x in error_limit])

    else:
        percentile = safe_arange(0., 100., step_percentile, endpoint=True)
        quantile = my_quantile(percentile=percentile, used_minerals=used_minerals, used_endmembers=used_endmembers,
                               cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred).numpy()

        # quantiles are always sorted
        within = np.transpose([np.interp(error_limit, quantile[:, i], percentile, left=0., right=100.)
                               for i in range(np.shape(quantile)[1])])

    if return_dict:
        return {f"within {limit} pp": within_limit for limit, within_limit in zip(error_limit, within)}

    return tuple(within)


def compute_one_sigma(y_true: np.ndarray, y_pred: np.ndarray,
                      used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                      cleaning: bool = True, all_to_one: bool = False,
                      remove_px_outliers: bool = False) -> tuple[np.ndarray, ...]:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_quantile

    if np.any(y_true > 1.) or np.any(y_pred > 1.):
        y_true, y_pred = y_true / 100., y_pred / 100.

    if remove_px_outliers:
        ind_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                                     px_only=True)
        y_true, y_pred = np.delete(y_true, ind_outliers, axis=0), np.delete(y_pred, ind_outliers, axis=0)

    one_sigma = my_quantile(percentile=68.27, used_minerals=used_minerals, used_endmembers=used_endmembers,
                            cleaning=cleaning, all_to_one=all_to_one)(y_true, y_pred).numpy()

    return one_sigma


def gimme_model_specification(model_name: str) -> str:
    bare_name = split_path(model_name)[1]

    name_parts = np.array(bare_name.split(_sep_out))

    # dt_string is made of 14 decimals
    dt_string_index = np.where([part.isdecimal() and len(part) == 14 for part in name_parts])[0]

    if np.size(dt_string_index) > 0:
        return _sep_out.join(name_parts[1:dt_string_index[0]])  # cut model_type, dt_string and following parts
    return _sep_out.join(name_parts[1:])  # cut model_type


def gimme_bin_code_from_name(model_name: str) -> str:
    specification = gimme_model_specification(model_name=model_name)
    return specification.split(_sep_out)[-1]


def gimme_model_grid_from_name(model_name: str) -> str:
    specification = gimme_model_specification(model_name=model_name)
    return _sep_out.join(specification.split(_sep_out)[:-1])


def gimme_grid_setup_from_name(model_name: str) -> dict:
    model_grid = gimme_model_grid_from_name(model_name)
    if "ASPECT" in model_grid or "HS-H" in model_grid:
        instrument = model_grid
    else:
        instrument = None

    if instrument is not None:
        new_wvl_grid = None
        new_wvl_grid_normalisation = "adaptive"
    else:
        new_wvl_grid = safe_arange(*model_grid.split(_sep_in)[:-1], endpoint=True, dtype=_wp)
        new_wvl_grid_normalisation = model_grid.split(_sep_in)[-1]

        if new_wvl_grid_normalisation == "None":
            new_wvl_grid_normalisation = None
        elif new_wvl_grid_normalisation == "adaptive":
            pass
        else:
            new_wvl_grid_normalisation = float(new_wvl_grid_normalisation)

    return {"model_grid": model_grid,
            "instrument": instrument,
            "wvl_grid": new_wvl_grid,
            "wvl_norm": new_wvl_grid_normalisation}


def gimme_used_from_name(model_name: str) -> tuple[np.ndarray, list[list[bool]]]:
    return bin_to_used(bin_code=gimme_bin_code_from_name(model_name=model_name))


def is_taxonomical(model: str | Model | Sequential | None = None, bin_code: str | None = None) -> bool:
    if isinstance(model, str):
        bare_name = split_path(model)[1]
        bin_code = gimme_bin_code_from_name(bare_name)

    if bin_code is not None:
        return len(bin_code.split(_sep_in)) == 2

    if isinstance(model, Model | Sequential):  # model was compiled when loaded
        if model.metrics_names:
            possible_taxonomy_metrics = ["categorical_accuracy", "f1_score"]
            return np.any([metric in model.metrics_names for metric in possible_taxonomy_metrics])

        # The last layer is output activation or dense (if activation is set as a parameter of Dense)
        if model.get_config()["layers"][-1]["class_name"] in ["Activation", "Dense"]:
            possible_composition_activatins = ["sigmoid_norm", "softmax_norm", "relu_norm", "plu_norm",
                                               "my_sigmoid", "my_softmax", "my_relu", "my_plu"]
            return model.get_config()["layers"][-1]["config"]["activation"] not in possible_composition_activatins

    raise ValueError("Unable to distinguish between composition and taxonomy models.")


def gimme_custom_objects(model_name: str, **kwargs) -> dict:
    from modules.NN_losses_metrics_activations import create_custom_objects
    if is_taxonomical(model_name):
        used_minerals, used_endmembers = None, None
    else:
        used_minerals, used_endmembers = gimme_used_from_name(model_name)

    return create_custom_objects(used_minerals=used_minerals, used_endmembers=used_endmembers, **kwargs)


def remove_continuum(filename: str, subfolder: str = "", saving: bool = False) -> tuple[np.ndarray, ...]:
    control_plot = False

    input_file = path.join(_path_data, subfolder, filename)
    data = load_npz(input_file)

    denoised = "denoise" in filename
    normalised = "norm" in filename

    output_file = input_file.replace(".npz", f"{_sep_out}CH.npz")

    xq, spectra = data[_wavelengths_name], data[_spectra_name]

    n_data, len_data = np.shape(spectra)
    rectified_spectra = np.zeros((n_data, len_data))

    # 2D data for convex hull
    ch_data = np.zeros((len_data, 2))
    ch_data[:, 0] = xq

    for i in range(n_data):
        spectrum = spectra[i]
        ch_data[:, 1] = spectrum

        hull = ConvexHull(ch_data).vertices

        # remove the lower branch from vertices (delete all vertices between 0 and len_data - 1
        hull = np.roll(hull, -np.where(hull == 0)[0][0] - 1)  # move 0 to the end of the list
        hull = np.sort(hull[np.where(hull == len_data - 1)[0][0]:])

        """
        # keep the UV bands
        x0 = my_argmax(xq, spectrum, x0=650.)
        hull = hull[np.argmin(np.abs(xq[hull] - x0)):]
        """

        continuum = np.zeros(np.shape(xq))  # necessary since the UVs start at different positions

        # linear fit to the convex hull
        for j in range(len(hull) - 1):
            x_fit, y_fit = xq[[hull[j], hull[j + 1]]], spectrum[[hull[j], hull[j + 1]]]
            if j == 0 and hull[j] != 0:
                x_new = xq[:hull[j + 1] + 1]
                continuum[:hull[j + 1] + 1] = np.polval(my_polyfit(x_fit, y_fit, 1), x_new)
            else:
                x_new = xq[hull[j]:hull[j + 1] + 1]
                continuum[hull[j]:hull[j + 1] + 1] = np.polyval(my_polyfit(x_fit, y_fit, 1), x_new)

        rectified_spectra[i] = spectrum / continuum
        rectified_spectra = np.round(rectified_spectra, 5)

        if control_plot:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            mpl.use("TkAgg")

            fig, ax = plt.subplots()
            ax.plot(xq, spectrum / continuum)
            ax.plot(xq, spectrum)
            ax.plot(xq, continuum)

    if saving:
        save_data(output_file, spectra=rectified_spectra, wavelengths=xq, labels=data[_label_name],
                  metadata=data[_metadata_name], labels_key=data[_label_key_name], metadata_key=data[_metadata_key_name],
                  denoised=denoised, normalised=normalised)

    return xq, rectified_spectra


def combine_same_range_models(indices: np.ndarray, ranges_all_or_spacing_all: np.ndarray, what_rmse_all: np.ndarray,
                              applied_function: Callable):
    #  combine different models

    ranges = len(np.unique(indices)) * ["str"]
    what_rmse = np.zeros(len(np.unique(indices)))

    for ind, unique_index in enumerate(np.unique(indices)):
        ranges[ind] = ranges_all_or_spacing_all[np.where(unique_index == indices)[0]][0]
        what_rmse[ind] = applied_function(what_rmse_all[np.where(unique_index == indices)[0]])

    return np.array(ranges).ravel(), what_rmse


def cut_error_bars(y_true: np.ndarray, y_true_error: np.ndarray | float, y_pred: np.ndarray, y_pred_error: np.ndarray,
                   lim_min: float = 0., lim_max: float = 100.) -> tuple[np.ndarray, np.ndarray]:
    # equivalently
    # lower_error, upper_error = y_true - y_true_error, y_true_error + y_true
    # lower_error[lower_error < lim_min], upper_error[upper_error > lim_max] = lim_min, lim_max
    # lower_error, upper_error = y_true - lower_error, upper_error - y_true

    lower_error = y_true - np.clip(y_true - y_true_error, lim_min, lim_max)
    upper_error = np.clip(y_true_error + y_true, lim_min, lim_max) - y_true
    axis = np.ndim(lower_error)  # to create a new axes using stack
    actual_errorbar_reduced = np.moveaxis(stack((lower_error, upper_error), axis=axis),
                                          source=0, destination=-1)  # to shift the last axes to the beginning

    lower_error = y_pred - np.clip(y_pred - y_pred_error, lim_min, lim_max)
    upper_error = np.clip(y_pred_error + y_pred, lim_min, lim_max) - y_pred
    predicted_errorbar_reduced = np.moveaxis(stack((lower_error, upper_error), axis=axis),
                                             source=0, destination=-1)  # to shift the last axes to the beginning

    predicted_errorbar_reduced[predicted_errorbar_reduced < 0.] = 0.
    actual_errorbar_reduced[actual_errorbar_reduced < 0.] = 0.

    return predicted_errorbar_reduced, actual_errorbar_reduced


def error_estimation_overall(y_true: np.ndarray, y_pred: np.ndarray, actual_error: np.ndarray | float = 3.,
                             used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_rmse

    if np.all(y_true <= 1.):  # to percents
        y_true = y_true[:] * 100.
        y_pred = y_pred[:] * 100.

    RMSE = my_rmse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                   cleaning=True, all_to_one=False)(y_true, y_pred).numpy() / 100.  # is multiplied by 100 in the code

    return cut_error_bars(y_true, actual_error, y_pred, RMSE)


def error_estimation_bin_like(y_true: np.ndarray, y_pred: np.ndarray, actual_error: np.ndarray | float = 3.,
                              used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_rmse, clean_ytrue_ypred

    num_minerals = gimme_num_minerals(used_minerals)
    num_labels = np.shape(y_true)[1]

    # multiplied by 100 here
    y_true_clean, y_pred_clean = clean_ytrue_ypred(y_true, y_pred, used_minerals=used_minerals,
                                                   used_endmembers=used_endmembers, cleaning=True, all_to_one=False)
    y_true_clean, y_pred_clean = y_true_clean.numpy(), y_pred_clean.numpy()

    if np.any(y_true_clean > 100.):  # to percents
        y_true_clean /= 100.
        y_pred_clean /= 100.

    # N bins (step 100 / N)
    N = 10

    predicted_error = np.zeros((N, np.shape(y_pred)[1]))  # errors for each bin; for info only
    predicted_error_no = np.zeros((N, np.shape(y_pred)[1]))  # number of points for each bin; for info only

    errors = np.zeros((len(y_pred), num_labels))  # final errors for each point

    for i in range(N):
        mask = np.logical_and(100. / N * i <= y_pred_clean, y_pred_clean <= 100. / N * (i + 1.))

        # Modal and chemical must be done separately (to keep correct information about w_true)
        # filtered modal
        y_p = np.where(mask[:, :num_minerals], y_pred_clean[:, :num_minerals], np.nan)
        y_t = np.where(mask[:, :num_minerals], y_true_clean[:, :num_minerals], np.nan)

        rmse_modal = my_rmse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                             cleaning=False, all_to_one=False)(y_t, y_p).numpy()

        # filtered chemical
        y_p = np.where(mask[:, num_minerals:], y_pred_clean[:, num_minerals:], np.nan)
        y_t = np.where(mask[:, num_minerals:], y_true_clean[:, num_minerals:], np.nan)

        # Add true modal to filter modeless samples
        y_p = stack((y_pred_clean[:, :num_minerals], y_p), axis=1)
        y_t = stack((y_true_clean[:, :num_minerals], y_t), axis=1)

        rmse_chem = my_rmse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                            cleaning=True, all_to_one=False)(y_t, y_p).numpy()[num_minerals:]

        rmse = stack((rmse_modal, rmse_chem))

        predicted_error_no[i], predicted_error[i] = np.sum(mask, axis=0), rmse

        rmse = np.reshape(rmse, (1, -1))
        rmse = np.repeat(rmse, repeats=len(y_pred), axis=0)
        errors = np.where(mask, rmse, errors)

    errors /= 100.

    """
    # This can be printed as info to table
    predicted_error /= 100.
    predicted_error = np.round(predicted_error, 1)
    np.transpose(stack((predicted_error, predicted_error_no), axis=2))
    """

    return cut_error_bars(y_true_clean, actual_error, y_pred_clean, errors)


def gimme_indices(used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                  reduced: bool = True, return_mineral_indices: bool = False) -> np.ndarray:
    # This function returns the first and last indices of modal/mineral groups

    count_endmembers = gimme_endmember_counts(used_endmembers)
    all_minerals = gimme_minerals_all(used_minerals, used_endmembers)
    num_minerals = gimme_num_minerals(all_minerals)

    indices = np.zeros((len(count_endmembers) + 1, 3), dtype=int)

    indices[0, 0], indices[1:, 0] = -1, np.cumsum(all_minerals) - 1  # cumsum - 1 to get indices

    indices[0, 1:] = 0, num_minerals

    for k, counts in enumerate(count_endmembers):
        indices[k + 1, 1:] = indices[k, 2], indices[k, 2] + counts

    indices = indices[stack(([True], all_minerals))]

    if reduced:
        indices = np.array([[ind_of_mineral, start, stop] for ind_of_mineral, start, stop in indices if start != stop])

    if return_mineral_indices:
        return indices

    return indices[:, 1:]


def used_indices(used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                 return_digits: bool = False) -> np.ndarray:
    indices = stack((used_minerals, flatten_list(used_endmembers)))
    if return_digits:
        return np.where(indices)[0]
    return indices


def unique_indices(used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                   all_minerals: bool = False, return_digits: bool = False) -> np.ndarray:
    # modification of used indices (if there are two labels, their absolute errors are the same; one is enough)
    unique_used_inds = deepcopy([list(used_minerals)] + used_endmembers)
    for i, unique_inds in enumerate(unique_used_inds):
        if np.sum(unique_inds) == 2:
            unique_used_inds[i][np.where(unique_inds)[0][-1]] = False

    used_inds = used_indices(used_minerals, used_endmembers)
    unique_used_inds = flatten_list(unique_used_inds) * used_inds  # is this multiplication necessary?

    if all_minerals:
        indices = unique_used_inds
    else:
        # keep indices which are both used and unique (unused are removed, so this only shifts the unique)
        indices = unique_used_inds[used_inds]

    if return_digits:
        return np.where(indices)[0]
    return indices

        # equivalently
        # return np.searchsorted(np.where(used_inds)[0], np.where(unique_used_inds)[0], side="left")
        # return np.digitize(np.where(unique_used_inds)[0], np.where(used_inds)[0], right=True)
        # return np.array([c for c, ind in enumerate(np.where(used_inds)[0]) if ind in np.where(unique_used_inds)[0]])


def print_comp_accuracy_header(used_minerals: np.ndarray | None = None,
                               used_endmembers: list[list[bool]] | None = None,
                               all_types_to_one: bool = False, all_to_one: bool = False) -> None:
    # Function to print header of the vector accuracy

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    if all_to_one:
        print(f"{'':22} Total")

    elif all_types_to_one:
        header = np.array(["Modal", "OL", "OPX", "CPX", "PLG"])
        all_minerals = gimme_minerals_all(used_minerals, used_endmembers)
        indices = stack((np.any(used_minerals), all_minerals))
        print(f"{'':23} {''.join(f'{head:7}' for head in header[indices])}")

    else:
        header = np.array(["OL", "OPX", "CPX", "PLG",
                           "Fa", "Fo",
                           "Fs", "En", "Wo",
                           "Fs", "En", "Wo",
                           "An", "Ab", "Or"])

        indices = unique_indices(used_minerals, used_endmembers, all_minerals=True)
        print(f"{'':23} {''.join(f'{head:6}' for head in header[indices])}")

    return


def print_tax_accuracy_header(used_classes: dict[str, int] | list | np.ndarray | None = None,
                              all_to_one: bool = False) -> None:
    # Function to print header of the F1 accuracy
    if used_classes is None: used_classes = classes

    if all_to_one:
        print(f"{'':27} Total")
    else:
        print(f"{'':29} {''.join(f'{cls:7}' for cls in used_classes)}")

    return


def print_header(bin_code: str):
    taxonomical = is_taxonomical(bin_code=bin_code)

    if taxonomical:
        print_tax_accuracy_header(used_classes=bin_to_cls(bin_code))

    else:
        print_comp_accuracy_header(*bin_to_used(bin_code=bin_code))


def print_comp_accuracy(rmse_accuracy: np.ndarray, what: str,
                        used_minerals: np.ndarray | None = None,
                        used_endmembers: list[list[bool]] | None = None,
                        all_types_to_one: bool = False, all_to_one: bool = False) -> None:
    # Function to print vector accuracy

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    pref = f"Mean {what.lower()} RMSE:"
    rmse_accuracy = np.array(rmse_accuracy)

    if all_to_one:
        print(f"{pref:21} {np.round(np.mean(rmse_accuracy), 3):6.3f}")

    elif all_types_to_one:
        mse_accuracy = np.square(rmse_accuracy)
        indices = gimme_indices(used_minerals, used_endmembers)

        accuracies = np.array([np.sqrt(np.mean(mse_accuracy[range(inds[0], inds[1])])) for inds in indices])
        print(f"{pref:21} [{', '.join(f'{acc:5.1f}' for acc in np.round(accuracies, 1))}]")

    else:
        indices = unique_indices(used_minerals, used_endmembers)

        print(f"{pref:21} [{', '.join(f'{acc:4.1f}' for acc in np.round(rmse_accuracy[indices], 1))}]")

    return



def print_tax_accuracy(f1_accuracy: np.ndarray, what: str, all_to_one: bool = False) -> None:
    # Function to print F1 accuracy

    pref = f"Mean {what.lower()} F-1 score:"
    f1_accuracy = np.array(f1_accuracy)

    if all_to_one:
        print(f"{pref:26} {np.round(np.mean(f1_accuracy), 3):6.3f}")
    else:
        print(f"{pref:26} [{', '.join(f'{acc:5.3f}' for acc in np.round(f1_accuracy, 2))}]")

    return


def print_info(y_true: np.ndarray, y_pred: np.ndarray, bin_code: str, which: str = "test") -> np.ndarray:
    from modules.NN_losses_metrics_activations import my_rmse, my_f1_score

    taxonomical = is_taxonomical(bin_code=bin_code)

    if taxonomical:
        acc = my_f1_score(all_to_one=False)(y_true, y_pred).numpy()
        print_tax_accuracy(acc, which)

    else:
        used_minerals, used_endmembers = bin_to_used(bin_code=bin_code)

        acc = my_rmse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                      cleaning=True, all_to_one=False)(y_true, y_pred).numpy()
        print_comp_accuracy(acc, which, used_minerals=used_minerals, used_endmembers=used_endmembers)

    return acc


def collect_all_models(subfolder_model: str, prefix: str | None = None, suffix: str | None = None,
                       regex: str | None = None, file_suffix: str = _model_suffix, full_path: bool = True) -> list[str]:

    final_suffix = "" if file_suffix == "SavedModel" else f".{file_suffix}"

    if prefix is not None:
        model_str = path.join(_path_model, subfolder_model, f"{prefix}*{final_suffix}")
    elif suffix is not None:
        model_str = path.join(_path_model, subfolder_model, f"*{suffix}{final_suffix}")
    elif regex is not None:
        model_str = path.join(_path_model, subfolder_model, f"{regex}{final_suffix}")
    else:
        model_str = path.join(_path_model, subfolder_model, f"*{final_suffix}")

    if full_path:
        return glob(model_str)
    else:
        return [path.basename(x) for x in glob(model_str)]


def remove_jumps_in_spectra(wavelengths: np.ndarray, reflectance: np.ndarray, jump_index: int,
                            n_points: int = 3, shift: int = 0, deg: int = 1) -> float:
    # fit n_points behind and after the jump
    # You can shift the points with "shift" if the values around the jump are damaged
    if n_points < 0:
        raise ValueError(f'"n_points" must be non-negative but is {n_points}.')
    if shift < 0:
        raise ValueError(f'"shift" must be non-negative but is {shift}.')

    # to not have poorly conditioned polyfit
    n_points = np.max((n_points, deg + 1))

    start = jump_index - n_points - shift
    stop = jump_index - shift
    if start < 0:  # to prevent starting from negative
        start = 0

    wvl_before, refl_before = wavelengths[start:stop], reflectance[start:stop]
    wvl_after, refl_after = wavelengths[jump_index:jump_index + n_points], reflectance[jump_index:jump_index + n_points]

    wvl_fit = stack((wvl_before, wvl_after))

    fitted_no_shift = np.polyval(my_polyfit(wvl_before, refl_before, deg), wvl_fit)
    fitted_shift = np.polyval(my_polyfit(wvl_after, refl_after, deg), wvl_fit)

    return np.mean(fitted_no_shift / fitted_shift)


def match_spectra(wavelengths: tuple[np.ndarray, ...], reflectance: tuple[np.ndarray, ...],
                  min_points: int = 3, deg: int = 1) -> tuple[np.ndarray, ...]:

    if min_points < 0:
        raise ValueError(f'"min_points" must be non-negative but is {min_points}.')

    # sort wavelength tuple first
    index, minimum = zip(*[(i, np.min(wvl)) for i, wvl in enumerate(wavelengths)])
    index, minimum = np.array(index), np.array(minimum)
    inds = np.argsort(minimum)
    index = index[inds]

    wavelengths = [wavelengths[i] for i in index]
    reflectance = [reflectance[i] for i in index]

    wvl_joined, refl_joined = wavelengths[0], reflectance[0]

    for i in range(1, len(wavelengths)):
        # common wavelengths
        start = np.max([np.min(wavelength) for wavelength in [wvl_joined, wavelengths[i]]])
        stop = np.min([np.max(wavelength) for wavelength in [wvl_joined, wavelengths[i]]])

        if stop < start:
            print("The spectra do not overlap.")

            # stack spectra and remove possible jump
            jump_index = len(wvl_joined)

            wvl_joined = stack((wvl_joined, wavelengths[i]))
            refl_joined = stack((refl_joined, reflectance[i]))

            factor = remove_jumps_in_spectra(wvl_joined, refl_joined, jump_index)
            refl_joined[jump_index:] *= factor

        else:
            mask1 = np.logical_and(start <= wvl_joined, wvl_joined <= stop)
            mask2 = np.logical_and(start <= wavelengths[i], wavelengths[i] <= stop)

            # ensure there are enough points
            mask1[-min_points:] = True
            mask2[:min_points] = True

            wvl1, refl1 = wvl_joined[mask1], refl_joined[mask1]
            wvl2, refl2 = wavelengths[i][mask2], reflectance[i][mask2]

            wvl_fit = stack((wvl1, wvl2))

            fit1 = np.polyval(my_polyfit(wvl1, refl1, deg), wvl_fit)
            fit2 = np.polyval(my_polyfit(wvl2, refl2, deg), wvl_fit)

            factor = np.mean(fit1 / fit2)

            # stack spectra and remove possible jump
            jump_index = len(wvl_joined)

            wvl_joined = stack((wvl_joined, wavelengths[i][~mask2]))
            refl_joined = stack((refl_joined, reflectance[i][~mask2] * factor))

            factor = remove_jumps_in_spectra(wvl_joined, refl_joined, jump_index)
            refl_joined[jump_index:] *= factor

    return wvl_joined, refl_joined


def find_outliers(y_true: np.ndarray, y_pred: np.ndarray,
                  used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                  threshold: float = 40.0, px_only: bool = False,
                  meta: pd.DataFrame | np.ndarray | None = None) -> np.ndarray:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_ae

    minimum_PX_fraction = 0.95  # minimum 95 vol% of OPX + CPX

    if not np.all(used_minerals[1:3]):
        return np.array([], dtype=int)

    pos_opx_cpx = np.cumsum(used_minerals) - 1  # -1 to get indices
    pos_opx_cpx = pos_opx_cpx[[1, 2]]

    absolute_error = my_ae(used_minerals=used_minerals, used_endmembers=used_endmembers,
                           cleaning=True, all_to_one=False)(y_true, y_pred).numpy()

    inds_samples, inds_quantities = np.where(absolute_error > threshold)

    if px_only:
        # error in OPX or CPX
        # 95+ vol% of OPX + CPX
        mask = np.logical_and.reduce([np.logical_or.reduce([inds_quantities == pos_px for pos_px in pos_opx_cpx]),
                                      np.sum(y_true[inds_samples][:, pos_opx_cpx], axis=1) >= minimum_PX_fraction])

        samples = np.unique(inds_samples[mask])
    else:
        samples = np.unique(inds_samples)

    samples = np.array(samples, dtype=int)

    if meta is None:
        return samples
    else:
        meta = np.array(meta)
        return np.array(list(zip(meta[samples], samples)), dtype=object)


def outliers_frequency(y_true: np.ndarray, y_pred: np.ndarray,
                       used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                       threshold: float = 40.0) -> np.ndarray:

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    from modules.NN_losses_metrics_activations import my_ae

    absolute_error = my_ae(used_minerals=used_minerals, used_endmembers=used_endmembers,
                           cleaning=True, all_to_one=False)(y_true, y_pred).numpy()

    _, inds_quantities = np.where(absolute_error > threshold)

    return np.array([np.sum(inds_quantities == ind) for ind in range(np.shape(y_true)[1])])


def wt_vol_conversion(conversion_direction: Literal["wt_to_vol", "vol_to_wt"], y_data: np.ndarray,
                      used_minerals: np.ndarray, used_endmembers: list[list[bool]]) -> np.ndarray:
    # should be after the chemicals are filled with dummy data, otherwise you can divide by 0 here
    # WAS NOT PROPERLY TESTED
    # zatim nefunguje:
    # pokud je nejaky mineral samotny bez chem slozeni

    if conversion_direction not in ["wt_to_vol", "vol_to_wt"]:
        raise ValueError('conversion_direction" must be "wt_to_vol" or "vol_to_wt".')

    # densities of Fa, Fo, Fs, En, Wo, Fs, En, Wo, An, Ab, Or
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56]
                         )[flatten_list(used_endmembers)]

    num_minerals = gimme_num_minerals(used_minerals)

    # for not-pure samples only
    inds = np.max(y_data[:, :num_minerals], axis=1) != 1

    modals, chemical = deepcopy(y_data[:, :num_minerals]), deepcopy(y_data[:, num_minerals:])
    mineral_density = chemical * densities

    #[:1] to avoid mineral position indices but keep mineral indices together with end-member indices
    for i, start, stop in gimme_indices(used_minerals, used_endmembers, return_mineral_indices=True)[1:]:
        norm_density = np.sum(mineral_density[inds, start - num_minerals:stop - num_minerals], axis=1)
        if np.all(norm_density) > 0:
            if conversion_direction == "vol_to_wt":
                modals[inds, i] *= norm_density
            else:  # must be "wt_to_vol"
                modals[inds, i] /= norm_density

    modals = normalise_in_rows(modals)

    return stack((modals, chemical), axis=1)


def vol_to_wt_percent(y_data: np.ndarray, used_minerals_all: np.ndarray,
                      used_endmembers: list[list[bool]]) -> np.ndarray:
    return wt_vol_conversion("vol_to_wt", y_data, used_minerals_all, used_endmembers)


def wt_to_vol_percent(y_data: np.ndarray, used_minerals_all: np.ndarray,
                      used_endmembers: list[list[bool]]) -> np.ndarray:
    return wt_vol_conversion("wt_to_vol", y_data, used_minerals_all, used_endmembers)


def return_mineral_position(what_type: str, metadata: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    if what_type.lower() == "ordinary chondrite":
        return np.where(metadata["Type1"] == "Ordinary Chondrite")[0]

    elif what_type.lower() == "hed":
        return np.where(metadata["SubType"].str.contains("HED"))[0]

    elif what_type.lower() in ["ol", "olivine"]:
        return np.where(labels[:, 0] == 1)[0]

    elif what_type.lower() in ["px", "pyroxene"]:
        return np.where(np.sum(labels[:, 1:3], axis=1) == 1)[0]

    else:  # asteroid
        return np.where(what_type == metadata["taxonomy class"])[0]


def combine_composition_and_taxonomy_predictions(filename: str, bin_code_comp: str | None = None,
                                                 bin_code_tax: str | None = None,
                                                 grid_model: str | None = None,
                                                 proportiontocut: float | None = None) -> tuple[np.ndarray, ...]:
    from modules.NN_evaluate import evaluate

    if bin_code_comp is None: bin_code_comp = comp_output_setup["bin_code"]
    if bin_code_tax is None: bin_code_tax = tax_output_setup["bin_code"]
    if grid_model is None: grid_model = comp_grid["model_grid"]
    if proportiontocut is None: proportiontocut = comp_model_setup["trim_mean_cut"]

    comp_subfolder_model = path.join("composition", grid_model)
    comp_models = collect_all_models(subfolder_model=comp_subfolder_model, regex=f"*{bin_code_comp}*", full_path=False)

    tax_subfolder_model = path.join("taxonomy", grid_model)
    tax_models = collect_all_models(subfolder_model=tax_subfolder_model, regex=f"*{bin_code_tax}*", full_path=False)

    pred_comp = evaluate(comp_models, filename, proportiontocut=proportiontocut, subfolder_model=comp_subfolder_model)
    pred_tax = evaluate(tax_models, filename, proportiontocut=proportiontocut, subfolder_model=tax_subfolder_model)

    return pred_comp, pred_tax


def compute_mean_predictions(y_pred: np.ndarray) -> tuple[np.ndarray, ...]:
    return return_mean_std(y_pred * 100., axis=0)


def print_conversion_chart(used_classes: dict[str, int] | None) -> None:
    if used_classes is None:used_classes = classes

    print("Conversion chart:")
    print("".join(f"{key}\t=\t{value}\t\t"
                  if (value % 5 > 0 or value == 0) else f"\n{key}\t=\t{value}\t\t"
                  for key, value in used_classes.items()))
