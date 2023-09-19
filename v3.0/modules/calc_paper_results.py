from os import environ, path
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

from modules.NN_data import load_composition_data as load_data
from modules.NN_data import split_composition_data_proportional as split_data_proportional
from modules.NN_evaluate import evaluate_test_data, evaluate

from modules.tables import mean_asteroid_type, accuracy_table, quantile_table, mean_S_asteroid_type
from modules.tables import taxonomy_metrics, taxonomy_class_of_mineral_types, chelyabinsk_sw, kachr_sw, kachr_sw_laser

from modules.control_plots import plot_corr_matrix, plot_error_density_plots, result_plots

from modules.paper_plots import plot_PC1_PC2_NN, plot_Fa_vs_Fs_ast_only, plot_EI_type_hist
from modules.paper_plots import plot_PC1_PC2_BAR, plot_scatter_NN_BC, plot_ast_type_histogram, plot_Sq_histogram

from modules.paper_plots import plot_surface_spectra, plot_surface_spectra_shapeViewer, plot_Fa_vs_Fs
from modules.collect_data import resave_data_for_shapeViewer

from modules.utilities_spectra import collect_all_models, combine_composition_and_taxonomy_predictions
from modules.utilities_spectra import used_indices, compute_mean_predictions, join_data, load_npz
from modules.utilities_spectra import gimme_bin_code_from_name, gimme_grid_setup_from_name
from modules.utilities import flatten_list, stack

from modules.NN_config_parse import bin_to_used

from modules.NN_config_composition import mineral_names_short, endmember_names

from modules._constants import _path_accuracy_tests, _sep_out, _sep_in


def paper_1(what_model: str = "final") -> None:
    if what_model == "OL":  # OL only
        model_names = ["CNN_450-2450-5-550_0000-11-000-000-000_20220325105832.h5"]

    elif what_model == "OPX":  # OPX only, no Wo_OPX
        model_names = ["CNN_450-2450-5-550_0000-00-110-000-000_20220325114608.h5"]

    elif what_model == "OL-OPX":  # OL + OPX + OL_OPX_mix only, no low-iron OL, no Wo_OPX
        model_names = ["CNN_450-2450-5-550_1100-11-110-000-000_20220404141225.h5"]

    elif what_model == "all-1":  # all, no low-iron, no Wo_OPX, no PLG chemical
        model_names = ["CNN_450-2450-5-550_1111-11-110-111-000_20220329232107.h5"]

    elif what_model == "all-2": # all, no low-iron
        model_names = ["CNN_450-2450-5-550_1111-11-111-111-111_20220331112738.h5"]

    elif what_model == "final":  # all, no low-iron, no PLG, no Wo_OPX -- final model
        model_names = ["CNN_450-2450-5-550_1110-11-110-111-000_20220330113805.h5"]

    else:
        raise ValueError("Unknown model.")

    used_minerals, used_endmembers = bin_to_used(gimme_bin_code_from_name(model_names[0]))

    grid_setup = gimme_grid_setup_from_name(model_names[0])

    filtering_setup = {"use_pure_only": False if what_model in ["all-1", "all-2", "final"] else True,
                       "use_mix_of_the_pure_ones": False if what_model not in ["OL-OPX"] else True,
                       "lim_vol_part":  0.65,
                       "chem_limits": {"OL": {"Fa": 3. if what_model not in ["OL"] else 0.},
                                       "OPX": {"Fs (OPX)": 5. if what_model not in ["OPX", "OL-OPX"] else 0.},
                                       "CPX": {"Fs (CPX)": 5. if what_model not in ["OPX", "OL-OPX"] else 0.}},
                       "remove_high_iron_unwanted": True,
                       "keep_if_not_used": False,
                       "red_thresh": 5.
                       }

    filename_train_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"

    x_train, y_train, meta, wavelengths = load_data(filename_train_data, clean_dataset=True,
                                                    return_meta=True, return_wavelengths=True,
                                                    used_minerals=used_minerals, used_endmembers=used_endmembers,
                                                    grid_setup=grid_setup, filtering_setup=filtering_setup)

    x_train, y_train, _, x_val, y_val, _, x_test, y_true, meta_test = split_data_proportional(x_train, y_train, meta,
                                                                                              val_portion=0.2,
                                                                                              test_portion=0.2,
                                                                                              used_minerals=used_minerals)

    subfolder_model = path.join("composition", f"450{_sep_in}2450{_sep_in}5{_sep_in}550")

    y_pred, accuracy = evaluate_test_data(model_names,
                                          x_test, y_true,
                                          x_val, y_val,
                                          x_train, y_train,
                                          proportiontocut=0.2,
                                          subfolder_model=subfolder_model)

    accuracy_table(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers)
    quantile_table(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers)

    if what_model in ["final", "OL-OPX"]:
        filename_data = path.join("taxonomy", f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz")
        y_pred_asteroids = evaluate(model_names, filename_data, proportiontocut=0.2, subfolder_model=subfolder_model)

        plot_PC1_PC2_NN(y_pred_asteroids)

        if what_model == "final":
            plot_error_density_plots(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers)

            plot_Fa_vs_Fs(y_true, y_true, meta_test, used_minerals=used_minerals, used_endmembers=used_endmembers)

            plot_scatter_NN_BC(x_test, y_true, meta_test, y_pred, wavelengths,
                               used_minerals=used_minerals, used_endmembers=used_endmembers)


            mean_asteroid_type(y_pred_asteroids, used_minerals=used_minerals, used_endmembers=used_endmembers)
            mean_S_asteroid_type(y_pred_asteroids, used_minerals=used_minerals, used_endmembers=used_endmembers)

            plot_Fa_vs_Fs_ast_only()
            plot_PC1_PC2_BAR()

            filename_data = f"Chelyabinsk{_sep_out}denoised{_sep_out}norm.npz"
            y_pred_chelyabinsk = evaluate(model_names, filename_data, proportiontocut=0.2,
                                          subfolder_model=subfolder_model)

            chelyabinsk_sw(y_pred_chelyabinsk, used_minerals=used_minerals, used_endmembers=used_endmembers)

            filename_data = f"ol{_sep_in}opx{_sep_in}pure{_sep_out}denoised{_sep_out}norm.npz"
            y_pred_kachr = evaluate(model_names, filename_data, proportiontocut=0.2, subfolder_model=subfolder_model)

            kachr_sw(y_pred_kachr)
            kachr_sw_laser(y_pred_kachr)


def paper_2_composition(asteroid_name: str = "full") -> None:
    if asteroid_name == "full":
        model_grid = "450-2450-5-550"
        filename = "composition_450-2450-5-550_1110-11-110-111-000_20221014103548.npz"

    elif asteroid_name == "Eros":
        model_grid = "820-2360-20-1300"
        filename = "composition_820-2360-20-1300_1110-11-110-111-000_20230423134006.npz"

    elif asteroid_name == "Itokawa":
        model_grid = "820-2080-20-1500"
        filename = "composition_820-2080-20-1500_1110-11-110-111-000_20221026162734.npz"

    else:
        raise ValueError("Unknown grid.")

    model_subdir = path.join("composition", model_grid)

    bin_code = gimme_bin_code_from_name(filename)
    used_minerals, used_endmembers = bin_to_used(bin_code=bin_code)

    full_path = path.join(_path_accuracy_tests, filename)
    data = load_npz(full_path)

    y_true, y_pred = data["labels_true"], data["labels_predicted"]

    accuracy_table(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers)
    quantile_table(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers)

    suf = f"_{model_grid}_accuracy_test"

    result_plots(y_true, y_pred, bin_code=bin_code, density_plot=True, suf=suf)

    if asteroid_name in ["Itokawa", "Eros"]:
        filename_data = f"{asteroid_name}{_sep_out}denoised{_sep_out}norm.npz"
        model_names = collect_all_models(subfolder_model=model_subdir, full_path=False)

        y_pred_asteroid = evaluate(model_names, filename_data, proportiontocut=0.2, subfolder_model=model_subdir)

        resave_data_for_shapeViewer(y_pred_asteroid, asteroid_name, "composition", 
                                    used_minerals=used_minerals, used_endmembers=used_endmembers, used_classes=None)
        plot_surface_spectra(y_pred_asteroid, filename_data, "composition", used_minerals=used_minerals, 
                             used_endmembers=used_endmembers, used_classes=None)
        plot_surface_spectra_shapeViewer(asteroid_name, "composition", used_minerals=used_minerals,
                                         used_endmembers=used_endmembers, used_classes=None)

        labels = [mineral_names_short] + endmember_names
        labels = flatten_list(labels)[used_indices(used_minerals, used_endmembers)]
        labels = np.reshape(labels, (len(labels), 1))

        labels_tax = np.array(list(load_npz(path.join(_path_accuracy_tests,
                                                      "taxonomy_820-2360-20-1300_16-0_20230423070712.npz"))
                                   ["model_info"][()]["used classes"].keys()))

        stat = np.round(stack((compute_mean_predictions(y_pred_asteroid)), axis=1), 1)
        print(stack((labels, stat), axis=1))

        # compute and plot correlation matrix
        combined_predictions = combine_composition_and_taxonomy_predictions(filename_data, bin_code_comp=bin_code,
                                                                            bin_code_tax="16-0", grid_model=model_grid,
                                                                            proportiontocut=0.2)
        combined_predictions = stack(combined_predictions, axis=1)
        corr_mat = pd.DataFrame(combined_predictions).corr()
        corr_labels = stack((labels.ravel(), labels_tax))

        plot_corr_matrix(corr_labels, pd.DataFrame(corr_mat), suf=f"_{model_grid}_full")

        index_to_delete = np.array([2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25])
        corr_labels = np.delete(corr_labels, index_to_delete)
        corr_mat = np.delete(np.array(corr_mat), index_to_delete, axis=0)
        corr_mat = np.delete(np.array(corr_mat), index_to_delete, axis=1)

        plot_corr_matrix(corr_labels, pd.DataFrame(corr_mat), suf=f"_{model_grid}")


def paper_2_taxonomy(asteroid_name: str = "full") -> None:
    if asteroid_name == "full":
        model_grid = "450-2450-5-550"
        filename = "taxonomy_450-2450-5-550_16-0_20221026114416.npz"  # old notation of classes (S, not S+ etc.)

    elif asteroid_name == "Eros":
        model_grid = "820-2360-20-1300"
        filename = "taxonomy_820-2360-20-1300_16-0_20230423070712.npz"

    elif asteroid_name == "Itokawa":
        model_grid = "820-2080-20-1500"
        filename = "taxonomy_820-2080-20-1500_16-0_20221026100327.npz"  # old notation of classes (S, not S+ etc.)

    else:
        raise ValueError("Unknown grid.")

    model_subdir = path.join("taxonomy", model_grid)

    filtering_setup = {"use_pure_only": False,
                       "use_mix_of_the_pure_ones": False,
                       "lim_vol_part": 0.65,
                       "chem_limits": {"OL": {"Fa": 3.},
                                       "OPX": {"Fs (OPX)": 5.},
                                       "CPX": {"Fs (CPX)": 5.}},
                       "remove_high_iron_unwanted": True,
                       "keep_if_not_used": False,
                       "red_thresh": 5.
                       }

    full_path = path.join(_path_accuracy_tests, filename)
    data = load_npz(full_path)

    y_true, y_pred, model_info = data["labels_true"], data["labels_predicted"], data["model_info"][()]

    # This is needed to be able to load old data
    try:
        labels = np.array(list(model_info["used_classes"].keys()))
    except KeyError:
        labels = np.array(list(model_info["used classes"].keys()))

    labels = np.array(labels, dtype="<U4")
    labels[labels == "A"] = "A+"
    labels[labels == "C"] = "C+"
    labels[labels == "Cgh"] = "Cgh+"
    labels[labels == "S"] = "S+"
    labels[labels == "Sr"] = "Sr+"
    labels[labels == "V"] = "V+"
    labels[labels == "X"] = "X+"

    used_classes = {label: i for i, label in enumerate(labels)}

    if asteroid_name == "Eros":
        plot_ast_type_histogram(y_true, y_pred, type1="S+", type2="Sr+", used_classes=used_classes)

    taxonomy_metrics(y_true, y_pred, used_classes=used_classes, latex_output=False)

    suf = f"_{model_grid}_accuracy_test"

    result_plots(y_true, y_pred, bin_code=gimme_bin_code_from_name(filename), suf=suf)

    model_names = collect_all_models(subfolder_model=model_subdir, full_path=False)

    if asteroid_name == "full":  # assign taxonomies to "chemical" spectra and evaluate Sq asteroids
        # meteorites and minerals
        filename_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"

        filename_comp = "composition_450-2450-5-550_1110-11-110-111-000_20221014103548.npz"
        used_minerals, used_endmembers = bin_to_used(gimme_bin_code_from_name(filename_comp))
        grid_setup = gimme_grid_setup_from_name(filename_comp)

        spectra, y_true, meta = load_data(filename_data, clean_dataset=True, return_meta=True, reinterpolation=False,
                                          used_minerals=used_minerals, used_endmembers=used_endmembers,
                                          grid_setup=grid_setup, filtering_setup=filtering_setup)
        y_pred_min_to_tax = evaluate(model_names, spectra, proportiontocut=0.2, subfolder_model=model_subdir)

        types = ["Ordinary Chondrite", "HED", "pyroxene", "olivine"]
        mean_classes_met, winning_classes_met = taxonomy_class_of_mineral_types(types, y_true, y_pred_min_to_tax, meta,
                                                                                used_classes=used_classes)

        # asteroids
        data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}16{_sep_out}deleted{_sep_out}denoised_norm.npz",
                        subfolder="taxonomy")

        spectra, y_true, meta = data["spectra"], data["labels"], join_data(data, "meta")
        y_pred_tax_to_tax = evaluate(model_names, spectra, proportiontocut=0.2, subfolder_model=model_subdir)

        types = np.unique(data["labels"])
        mean_classes_ast, winning_classes_ast = taxonomy_class_of_mineral_types(types, y_true, y_pred_tax_to_tax, meta,
                                                                                used_classes=used_classes)

        plot_Sq_histogram(y_true, y_pred_tax_to_tax, used_classes=used_classes)

    elif asteroid_name in ["Itokawa", "Eros"]:

        filename_data = f"{asteroid_name}{_sep_out}denoised{_sep_out}norm.npz"

        y_pred_taxonomy = evaluate(model_names, filename_data, proportiontocut=0.2, subfolder_model=model_subdir)

        resave_data_for_shapeViewer(y_pred_taxonomy, asteroid_name, "taxonomy",  used_minerals=None, 
                                    used_endmembers=None, used_classes=used_classes)
        plot_surface_spectra(y_pred_taxonomy, filename_data, "taxonomy", used_minerals=None, used_endmembers=None, 
                             used_classes=used_classes)
        plot_surface_spectra_shapeViewer(asteroid_name, "taxonomy", used_minerals=None, used_endmembers=None,
                                         used_classes=used_classes)

        thresh = 1.
        mean_value, std_value = compute_mean_predictions(y_pred_taxonomy)
        labels = labels[mean_value > thresh]
        labels = np.reshape(labels, (len(labels), 1))

        stat = np.round(stack((mean_value[mean_value > thresh], std_value[mean_value > thresh]), axis=1), 1)
        print(stack((labels, stat), axis=1))

        if asteroid_name == "Eros":  # calc predictions for the other asteroid
            subfolder_model = path.join("taxonomy", f"820{_sep_in}2080{_sep_in}20{_sep_in}1500")
            model_names = collect_all_models(subfolder_model=subfolder_model, full_path=False)
            filename_data = f"Itokawa{_sep_out}denoised{_sep_out}norm.npz"
            y_pred_I = evaluate(model_names, filename_data, proportiontocut=0.2, subfolder_model=subfolder_model)

            plot_EI_type_hist(y_pred_taxonomy, y_pred_I, tax_type="S+", used_classes=used_classes)  # Eros first

        else:
            subfolder_model = path.join("taxonomy", f"820{_sep_in}2360{_sep_in}20{_sep_in}1300")
            model_names = collect_all_models(subfolder_model=subfolder_model, full_path=False)
            filename_data = f"Eros{_sep_out}denoised{_sep_out}norm.npz"
            y_pred_E = evaluate(model_names, filename_data, proportiontocut=0.2, subfolder_model=subfolder_model)

            plot_EI_type_hist(y_pred_E, y_pred_taxonomy, tax_type="S+", used_classes=used_classes)  # Eros first
