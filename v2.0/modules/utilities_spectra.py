from copy import deepcopy
from typing import Callable, Literal
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
from glob import glob
import os

from modules.utilities import check_dir, flatten_list, normalise_in_rows, stack, safe_arange
from modules.CD_parameters import denoise, normalise, normalised_at
from modules.NN_config_taxonomy import classes, classes2

from modules._constants import _path_data, _path_model


def save_data(final_name: str, spectra: np.ndarray, wavelengths: np.ndarray, metadata: np.ndarray,
              labels: np.ndarray | None = None, labels_key: np.ndarray | None = None,
              metadata_key: np.ndarray | None = None, subfolder: str = "") -> str:
    if len(spectra) != len(metadata):
        raise ValueError("Each spectrum must have its metadata. Length of spectra != length of metadata.")

    final_name = filename_adjustment(final_name)
    final_name = "".join((_path_data, subfolder, "/", final_name))

    # collect data and metadata
    data_and_metadata = {"spectra": np.array(spectra, dtype=np.float32),  # save spectra
                         "wavelengths": np.array(wavelengths, dtype=np.float32),  # save wavelengths
                         "metadata": np.array(metadata, dtype=object)}  # save metadata

    if metadata_key is not None:
        data_and_metadata["metadata key"] = np.array(metadata_key, dtype=object)

    if labels is not None:
        if len(spectra) != len(labels):
            raise ValueError("Each spectrum must have its label. Length of spectra != length of labels.")

        if np.shape(labels)[1] == 1:  # taxonomy class
            data_and_metadata["labels"] = np.array(labels, dtype=str)  # save labels
        else:  # composition
            data_and_metadata["labels"] = np.array(labels, dtype=np.float32)  # save labels

        if labels_key is not None:
            data_and_metadata["label metadata"] = np.array(labels_key, dtype=str)

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez(f, **data_and_metadata)

    return final_name


def combine_files(filenames: tuple[str, ...], final_name: str, subfolder: str = "") -> str:
    final_name = filename_adjustment(final_name)
    final_name = "".join((_path_data, subfolder, "/", final_name))

    combined_file = dict(np.load(filenames[0], allow_pickle=True))
    for filename in filenames[1:]:
        file_to_merge = dict(np.load(filename, allow_pickle=True))

        if np.all(combined_file["wavelengths"] == file_to_merge["wavelengths"]):
            combined_file["spectra"] = stack((combined_file["spectra"], file_to_merge["spectra"]), axis=0)
            combined_file["labels"] = stack((combined_file["labels"], file_to_merge["labels"]), axis=0)
            combined_file["metadata"] = stack((combined_file["metadata"], file_to_merge["metadata"]), axis=0)

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez(f, **combined_file)

    return final_name


def filename_adjustment(filename: str) -> str:
    tmp = Path(filename)
    final_name = tmp.name.replace(tmp.suffix, "")  # remove suffix

    if denoise:
        if "-denoised" not in final_name:
            final_name += "-denoised"
    if normalise:
        if "-norm" not in final_name:
            final_name += "-norm"

    return "".join((final_name, ".npz"))


def if_no_test_data(x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction > 0:  # If test portion is zero then use validation data
        x_test, y_test = deepcopy(x_val), deepcopy(y_val)
    else:  # If even val portion is zero, use train data (just for visualisation purposes)
        x_test, y_test = deepcopy(x_train), deepcopy(y_train)

    return x_test, y_test


def denoise_and_norm(data: np.ndarray, wavelength: np.ndarray, denoising: bool, normalising: bool, sigma_nm: float = 7.,
                     normalised_at_wvl: float = 550.) -> np.ndarray:
    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    if denoising:
        nm_to_px = 1. / (wavelength[1] - wavelength[0])  # conversion from nm to px
        correction = gaussian_filter1d(np.ones(len(wavelength)), sigma=sigma_nm * nm_to_px, mode="constant")
        data_denoised = gaussian_filter1d(data, sigma=sigma_nm * nm_to_px, mode="constant") / correction
    else:
        data_denoised = data

    # Normalised reflectance
    if normalising:
        fun = interp1d(wavelength, data_denoised, kind="cubic")  # v_final differs from v
        v_norm = np.reshape(fun(normalised_at_wvl), (len(data_denoised), 1))
    else:
        v_norm = 1.

    return data_denoised / v_norm


def clean_and_resave(filename: str, reinterpolate: bool = False) -> None:
    from modules.NN_data import load_compositional_data as load_data

    tmp = Path(filename)
    final_name = "".join((tmp.stem, "-clean"))  # ".npz" is added in save_data

    # load data for keys and wavelengths
    data = np.load("".join((_path_data, filename)), allow_pickle=True)

    # load cleaned data
    spectra, labels, meta = load_data(filename, return_meta=True, keep_all_labels=True,
                                      clean_dataset=True, reinterpolation=reinterpolate)

    # re-save it
    save_data(final_name, spectra=spectra, wavelengths=data["wavelengths"], labels=labels, metadata=meta,
              labels_key=data["label metadata"], metadata_key=data["metadata key"])


def normalize_spectra(file: str, save_it: bool = False, subfolder: str = "") -> None:
    path_to_data = "".join((_path_data, "/", subfolder, "/"))

    # load the data
    data_file = "".join((path_to_data, "/", file, ".npz"))
    data = np.load(data_file, allow_pickle=True)

    xq, spectra = data["spectra"], data["wavelengths"]

    fun = interp1d(xq, spectra, kind="cubic")
    v_norm = fun(normalised_at)
    spectra_final = normalise_in_rows(spectra, v_norm)

    if save_it:
        save_data("".join((file, "-normalised")), spectra=spectra_final, wavelengths=xq, labels=data["labels"],
                  metadata=data["metadata"], labels_key=data["label metadata"], metadata_key=data["metadata key"])


def remove_continuum(filename: str, saving: bool = False, subfolder: str = "") -> tuple[np.ndarray, ...]:
    control_plot = False

    input_file = "".join((_path_data, subfolder, "/", filename))
    output_file = input_file.replace(".npz", "_CH.npz")

    data = np.load(input_file, allow_pickle=True)
    xq, spectra = data["wavelengths"], data["spectra"]

    n_data, len_data = np.shape(spectra)
    rectified_spectra = np.zeros((n_data, len_data))

    # 2D data for convex hull
    ch_data = np.zeros((len_data, 2))
    ch_data[:, 0] = xq

    for i in range(n_data):
        spectrum = spectra[i]
        ch_data[:, 1] = spectrum

        hull = ConvexHull(ch_data).vertices

        # remove lower branch from vertices (delete all vertices between 0 and len_data - 1
        hull = np.roll(hull, -np.where(hull == 0)[0][0] - 1)  # move 0 to the end of the list
        hull = np.sort(hull[np.where(hull == len_data - 1)[0][0]:])

        """
        # keep the UV bands
        x0 = my_argmax(xq, spectrum, x0=650)
        hull = hull[np.argmin(np.abs(xq[hull] - x0)):]
        """

        continuum = np.zeros(np.shape(xq))  # necessary since the UVs start at different positions

        # linear fit to the convex hull
        for j in range(len(hull) - 1):
            x_fit, y_fit = xq[[hull[j], hull[j + 1]]], spectrum[[hull[j], hull[j + 1]]]
            if j == 0 and hull[j] != 0:
                x_new = xq[:hull[j + 1] + 1]
                continuum[:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)
            else:
                x_new = xq[hull[j]:hull[j + 1] + 1]
                continuum[hull[j]:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)

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
        save_data(output_file, spectra=rectified_spectra, wavelengths=xq, labels=data["labels"],
                  metadata=data["metadata"], labels_key=data["label metadata"], metadata_key=data["metadata key"])

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
    lower_error, upper_error = y_true - y_true_error, y_true_error + y_true
    lower_error[lower_error < lim_min], upper_error[upper_error > lim_max] = lim_min, lim_max
    lower_error, upper_error = y_true - lower_error, upper_error - y_true
    actual_errorbar_reduced = np.transpose(np.array(list(zip(lower_error, upper_error))))

    lower_error, upper_error = y_pred - y_pred_error, y_pred_error + y_pred
    lower_error[lower_error < lim_min], upper_error[upper_error > lim_max] = lim_min, lim_max
    lower_error, upper_error = y_pred - lower_error, upper_error - y_pred
    predicted_errorbar_reduced = np.transpose(np.array(list(zip(lower_error, upper_error))))

    return predicted_errorbar_reduced, actual_errorbar_reduced


def error_estimation_overall(y_true: np.ndarray, y_pred: np.ndarray, num_minerals: int,
                             actual_error: np.ndarray | float = 3.) -> tuple[np.ndarray, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_rmse

    if np.all(y_true < 1.1):  # to percents
        y_true = y_true[:] * 100.
        y_pred = y_pred[:] * 100.

    RMSE = my_rmse(num_minerals)(y_true, y_pred).numpy() / 100.  # is multiplied with 100 in the code

    return cut_error_bars(y_true, actual_error, y_pred, RMSE)


def error_estimation_bin_like(y_true: np.ndarray, y_pred: np.ndarray, num_minerals: int, num_labels: int,
                              actual_error: np.ndarray | float = 3.) -> tuple[np.ndarray, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_rmse, clean_ytrue_ypred

    y_true_clean, y_pred_clean = clean_ytrue_ypred(y_true, y_pred, num_minerals)
    y_true_clean, y_pred_clean = y_true_clean.numpy(), y_pred_clean.numpy()

    if np.any(y_true_clean > 101.):  # to percents
        y_true_clean /= 100.
        y_pred_clean /= 100.

    # N bins (step 100 / N)
    N = 10

    predicted_error = np.zeros((N, np.shape(y_pred)[1]))  # errors for each bin
    predicted_error_no = np.zeros((N, np.shape(y_pred)[1]))  # number of points for each bin
    errors_mod = np.zeros((len(y_pred), num_minerals))  # final errors for each point
    errors_chem = np.zeros((len(y_pred), num_labels - num_minerals))  # final errors for each point

    for i in range(N):
        mask = np.logical_and(100. / N * i <= y_pred_clean, y_pred_clean <= 100. / N * (i + 1))

        predicted_error_no[i] = np.sum(mask, axis=0)

        # modal and chemical must be done separately
        mask_modal, mask_chemical = mask[:, :num_minerals], mask[:, num_minerals:]

        y_pred_mod, y_pred_chem = y_pred_clean[:, :num_minerals], y_pred_clean[:, num_minerals:]
        y_true_mod, y_true_chem = y_true_clean[:, :num_minerals], y_true_clean[:, num_minerals:]

        # modal first
        y_p = np.where(mask_modal, y_pred_mod, np.nan)
        y_t = np.where(mask_modal, y_true_mod, np.nan)

        y_p = stack((y_p, y_pred_chem), axis=1)
        y_t = stack((y_t, y_true_chem), axis=1)

        # must be /100
        tmp_rmse = my_rmse(num_minerals)(y_t, y_p).numpy()[:num_minerals]
        predicted_error[i, :num_minerals] = tmp_rmse

        # easier to replicate it and then copy
        tmp_rmse = np.reshape(tmp_rmse, (1, np.size(tmp_rmse)))
        tmp_rmse = np.repeat(tmp_rmse, repeats=len(y_pred), axis=0)
        errors_mod = np.where(mask_modal, tmp_rmse, errors_mod)

        # chemical second
        y_p = np.where(mask_chemical, y_pred_chem, np.nan)
        y_t = np.where(mask_chemical, y_true_chem, np.nan)

        y_p = stack((y_pred_mod, y_p), axis=1)
        y_t = stack((y_true_mod, y_t), axis=1)

        tmp_rmse = my_rmse(num_minerals)(y_t, y_p).numpy()[num_minerals:]
        predicted_error[i, num_minerals:] = tmp_rmse

        # easier to replicate it and then copy
        tmp_rmse = np.reshape(tmp_rmse, (1, np.size(tmp_rmse)))
        tmp_rmse = np.repeat(tmp_rmse, repeats=len(y_pred), axis=0)
        errors_chem = np.where(mask_chemical, tmp_rmse, errors_chem)

    errors = stack((errors_mod, errors_chem), axis=1)
    errors /= 100.

    """
    # this can be printed as info to table
    predicted_error /= 100.
    predicted_error = np.round(predicted_error, 1)
    np.transpose(stack((predicted_error, predicted_error_no), axis=2))
    """

    return cut_error_bars(y_true_clean, actual_error, y_pred_clean, errors)


def gimme_indices(num_minerals: int, endmembers_counts: np.ndarray, reduced: bool = True,
                  return_mineral_indices: bool = False) -> np.ndarray:
    # This function returns first and last indices of modal/mineral groups

    indices = np.zeros((len(endmembers_counts) + 1, 3), dtype=int)

    indices[:, 0] = np.arange(-1, len(endmembers_counts))
    # indices[0, 0] = np.nan

    indices[0, 1:] = 0, num_minerals

    for k, counts in enumerate(endmembers_counts):
        indices[k + 1, 1:] = indices[k, 2], indices[k, 2] + counts

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
    unique_used_inds = flatten_list(unique_used_inds) * used_inds  # is this necessary?

    if all_minerals:
        indices = unique_used_inds
    else:
        # keep indices which are both used and unique (unused are removed, so this only shift the unique)
        indices = unique_used_inds[used_inds]
    if return_digits:
        return np.where(indices)[0]
    return indices

        # equivalently
        # return np.searchsorted(np.where(used_inds)[0], np.where(unique_used_inds)[0], side="left")
        # return np.digitize(np.where(unique_used_inds)[0], np.where(used_inds)[0], right=True)
        # return np.array([c for c, ind in enumerate(np.where(used_inds)[0]) if ind in np.where(unique_used_inds)[0]])


def print_accuracy(accuracy: np.ndarray, what: str, used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                   counts_endmembers: np.ndarray, all_types_to_one: bool = False) -> None:
    # Function to print vector accuracy

    pref = " ".join(("Mean", what, "RMSE:"))

    if all_types_to_one:
        indices = gimme_indices(int(np.sum(used_minerals)), counts_endmembers)
        tmp = np.array([np.mean(accuracy[range(inds[0], inds[1])]) for inds in indices])
        print("Mean", what, "RMSE:", "[" + ", ".join("{:.1f}".format(k) for k in tmp) + "]")
    else:
        indices = unique_indices(used_minerals, used_endmembers)
        print("{:21s}".format(pref), "[" + ", ".join("{:4.1f}".format(k) for k in accuracy[indices]) + "]")

    return


def print_accuracy_header(used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                          all_types_to_one: bool = False) -> None:
    # Function to print header of the vector accuracy

    header = np.array(["OL", "OPX", "CPX", "PLG",
                       "Fa", "Fo",
                       "Fs", "En", "Wo",
                       "Fs", "En", "Wo",
                       "An", "Ab", "Or"])

    if all_types_to_one:
        pass
    else:
        indices = unique_indices(used_minerals, used_endmembers, all_minerals=True)
        print("{:23s}".format(""), "".join("{:6s}".format(k) for k in header[indices]))

    return


def collect_all_models(suffix: str, subfolder_model: str, full_path: bool = True) -> list[str]:
    model_str = "".join((_path_model, subfolder_model, "/", "*", suffix, ".h5"))

    if full_path:
        return glob(model_str)
    else:
        return [os.path.basename(x) for x in glob(model_str)]


def find_outliers(y_true: np.ndarray, y_pred: np.ndarray, num_minerals: int, threshold: float = 40.0,
                  px_only: bool = True, pos_opx_cpx: np.ndarray = np.array([1, 2]),
                  meta: np.ndarray | None = None) -> np.ndarray:
    from modules.NN_losses_metrics_activations import my_ae

    absolute_error = my_ae(num_minerals=num_minerals, all_to_one=False)(y_true, y_pred).numpy()

    inds_samples, inds_quantities = np.where(absolute_error > threshold)

    unique_samples = np.unique(inds_samples)

    if px_only:
        # 95+ vol% of OPX + CPX
        samples = unique_samples[np.sum(y_true[unique_samples][:, pos_opx_cpx], axis=1) >= 0.95]
    else:
        samples = unique_samples

    if meta is None:
        return samples
    else:
        return np.array(list(zip(meta[samples], samples)), dtype=object)


def wt_vol_conversion(conversion_direction: Literal["wt_to_vol", "vol_to_wt"], y_data: np.ndarray,
                      used_minerals_all: np.ndarray, used_endmembers: list[list[bool]]) -> np.ndarray:
    # should be after the chemicals are filled with dummy data, otherwise you can divide by 0 here
    # WAS NOT PROPERLY TESTED
    # zatim nefunguje:
    # pokud je nejaky mineral samotny bez chem slozeni

    if conversion_direction not in ["wt_to_vol", "vol_to_wt"]:
        raise ValueError('conversion_direction" must be "wt_to_vol" or "vol_to_wt".')

    # densities of Fa, Fo, Fs, En, Wo, Fs, En, Wo, An, Ab, Or
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56]
                         )[flatten_list(used_endmembers)]

    num_minerals = int(np.sum(used_minerals_all))
    num_minerals = 0 if num_minerals == 1 else num_minerals

    # for not-pure samples only
    inds = np.max(y_data[:, :num_minerals], axis=1) != 1

    modals, chemical = deepcopy(y_data[:, :num_minerals]), deepcopy(y_data[:, num_minerals:])
    mineral_density = chemical * densities

    end_member_counts = np.array([np.sum(endmember) for endmember in used_endmembers])

    #[:1] to avoid mineral position indices but keep mineral indices together with end-member indices
    for i, start, stop in gimme_indices(num_minerals, end_member_counts, return_mineral_indices=True)[1:]:
        norm_density = np.sum(mineral_density[inds, start - num_minerals:stop - num_minerals], axis=1)
        if np.all(norm_density) > 0:
            if conversion_direction == "vol_to_wt":
                modals[inds, i] *= norm_density
            else:  # must be "wt_to_vol"
                modals[inds, i] /= norm_density

    norm = np.sum(modals, axis=1)
    modals = normalise_in_rows(modals, norm)

    return stack((modals, chemical), axis=1)


def vol_to_wt_percent(y_data: np.ndarray, used_minerals_all: np.ndarray,
                      used_endmembers: list[list[bool]]) -> np.ndarray:
    return wt_vol_conversion("vol_to_wt", y_data, used_minerals_all, used_endmembers)


def wt_to_vol_percent(y_data: np.ndarray, used_minerals_all: np.ndarray,
                      used_endmembers: list[list[bool]]) -> np.ndarray:
    return wt_vol_conversion("wt_to_vol", y_data, used_minerals_all, used_endmembers)


def return_mineral_position(what_type: str, metadata: np.ndarray, labels: np.ndarray,
                            inds_in_meta: list[int]) -> np.ndarray:
    sample_type = metadata[:, inds_in_meta]

    if what_type.lower() == "ordinary chondrite":
        return np.where(sample_type[:, 0] == "Ordinary Chondrite")[0]

    elif what_type.lower() == "hed":
        return np.where(["HED" in a[1] for a in sample_type])[0]

    elif what_type.lower() in ["ol", "olivine"]:
        return np.where(labels[:, 0] == 1)[0]

    elif what_type.lower() in ["px", "pyroxene"]:
        return np.where(np.sum(labels[:, 1:3], axis=1) == 1)[0]

    else:  # asteroid
        return np.where(sample_type == what_type)[0]


def combine_compositional_and_taxonomical_predictions(filename: str, comp_models: list[str] | None = None,
                                                      tax_models: list[str] | None = None) -> np.ndarray:
    from modules.NN_evaluate import evaluate

    if comp_models is None:
        asteroid_name = filename[:filename.find("-")]
        comp_models = collect_all_models(suffix="".join(("CNN_", asteroid_name)), subfolder_model="compositional",
                                         full_path=False)

    if tax_models is None:
        asteroid_name = filename[:filename.find("-")]
        tax_models = collect_all_models(suffix="".join(("CNN_", asteroid_name)), subfolder_model="taxonomical",
                                        full_path=False)

    pred_comp = evaluate(comp_models, filename, subfolder_model="compositional")
    pred_tax = evaluate(tax_models, filename, subfolder_model="taxonomical")

    return stack((pred_comp, pred_tax), axis=1)


def compute_mean_predictions(y_pred: np.ndarray) -> tuple[np.ndarray, ...]:
    mean_value = np.mean(y_pred * 100., axis=0)
    std_value = np.std(y_pred * 100., axis=0, ddof=1)

    return mean_value, std_value


def convert_classes(args: np.ndarray) -> list[float]:
    # This function convert classes to numbers which is required by the neural network

    return [classes[x] for x in args.ravel()]


def convert_classes2(args: np.ndarray) -> list[str]:
    # This function convert numbers to classes

    return [classes2[x] for x in args.ravel()]


def print_conversion_chart() -> None:
    print("Conversion chart:")
    print("".join("{key}\t=\t{value}\t\t".format(key=k, value=v)
                  if (v % 5 > 0 or v == 0) else "\n{key}\t=\t{value}\t\t".format(key=k, value=v)
                  for k, v in classes.items()))
