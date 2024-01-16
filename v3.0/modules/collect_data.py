from os import environ, path
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from urllib.request import urlopen

import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.interpolate import interp1d
from itertools import product
from typing import Literal

# import matplotlib.pyplot as plt
# import pywavefront
# from sklearn.metrics import pairwise_distances

from modules.asteroid_spectra_averaging import dlat, dlon
from modules.NN_data_grids import data_grids

from modules.utilities_spectra import denoise_and_norm, save_data, combine_files, used_indices, unique_indices
from modules.utilities_spectra import join_data, load_npz, load_xlsx, load_txt, load_h5, normalise_spectra
from modules.utilities_spectra import collect_all_models, remove_jumps_in_spectra, match_spectra
from modules.utilities import flatten_list, stack, my_mv, check_dir, safe_arange, normalise_in_rows, find_all
from modules.utilities import is_empty, remove_outliers, find_outliers

from modules.NN_config_composition import mineral_names, endmember_names, mineral_names_short

from modules.CD_parameters import lambda_max, lambda_min, resolution_max, resolution_final, usecols
from modules.CD_parameters import num_labels_CD, num_minerals_CD, minerals_CD, endmembers_CD
from modules.CD_parameters import denoise, denoising_sigma, normalise, wvl_norm

from modules._constants import _path_data, _path_relab_spectra, _relab_web_page
from modules._constants import _spectra_name, _wavelengths_name, _label_name, _coordinates_name, _sep_in, _sep_out

# defaults only
from modules.NN_config_composition import minerals_used, endmembers_used, comp_model_setup
from modules.NN_config_taxonomy import classes
from modules._constants import _num_eps


def collect_data_RELAB(start_line_number: tuple[int, ...] | int, end_line_number: tuple[int, ...] | int | None = None,
                       final_names: tuple[str, ...] = ("PLG", "CPX", "OPX", "OL")) -> list[str]:

    # The input tuples are numbers of lines in the Excel file (counting from 1)
    def load_excel_data(start_line_number: tuple[int, ...] | int,
                        end_line_number: tuple[int, ...] | int | None = None) -> tuple[pd.DataFrame, ...]:
        # This function reads the data from the Excel files

        # 2 is the offset; 1 due to header, 1 due to counting rows in Excel from 1, not from 0
        # rows are of the Spectra_Catalogue.xlsx
        rows = np.arange(np.min(start_line_number), np.max(end_line_number) + 1) - 2

        # Real the files (one spectrum per sample)
        Spectra_catalogue = load_xlsx("Spectra_Catalogue.xlsx", usecols="A, B, F:H").iloc[rows]
        Sample_catalogue = load_xlsx("Sample_Catalogue.xlsx", usecols=usecols, header=2, sheet_name="RELAB")

        return Spectra_catalogue, Sample_catalogue

    def split_and_filter_data(spectra_catalogue: pd.DataFrame,
                              sample_catalogue: pd.DataFrame) -> tuple[np.ndarray, list[float], np.ndarray]:
        start = np.array(spectra_catalogue["Start"], dtype=float)
        stop = np.array(spectra_catalogue["Stop"], dtype=float)
        step = np.array(spectra_catalogue["Resolution"], dtype=float)

        spectrumIDs = np.array(spectra_catalogue["SpectrumID"], dtype=str)
        sampleIDs = np.array(spectra_catalogue["SampleID"], dtype=str)

        sampleID_sample = np.array(sample_catalogue["SampleID"], dtype=str)

        # Take only these spectra
        mask = np.where((start <= lambda_min) & (stop >= lambda_max) & (step <= resolution_max))[0]

        if is_empty(mask):
            return np.array([]), [], np.array([])

        # Use only these spectra and samples
        spectrumIDs = spectrumIDs[mask]
        sampleIDs = sampleIDs[mask]

        lines_in_sample_catalogue = flatten_list([np.where(sampleID_sample == sampleID)[0]
                                                  for sampleID in sampleIDs])

        # Find corresponding PIs and filenames
        PIs = np.array(sample_catalogue["PI"].iloc[lines_in_sample_catalogue], dtype=str)
        filenames = np.array([path.join(_path_relab_spectra, PI.lower(), sampleID[:2].lower(), f"{spectrumID.lower()}.asc")
                              for PI, sampleID, spectrumID in zip(PIs, sampleIDs, spectrumIDs)])

        # Metadata
        sample_catalogue = sample_catalogue.iloc[lines_in_sample_catalogue]
        sample_catalogue = sample_catalogue.astype({"Chem#": "int"})
        metadata = np.array(sample_catalogue.iloc[:, :-num_labels_CD], dtype=object)
        sample_catalogue = np.array(sample_catalogue.iloc[:, -num_labels_CD:], dtype=float)

        # Find corresponding numbers
        num_labels_individual = stack((num_minerals_CD, endmembers_CD))
        last_index = 0
        numbers = [0] * len(num_labels_individual)

        for i, num in enumerate(num_labels_individual):
            first_index, last_index = last_index, last_index + num
            numbers[i] = sample_catalogue[:, first_index:last_index] / 100.

        return filenames, numbers, metadata

    def select_numbers(numbers: list[float], num_eps: float = _num_eps) -> np.ndarray:
        # modals
        modal = np.array(numbers[0], dtype=float)
        modal = modal[:, np.where(minerals_CD)[0]]

        # normalise to unit sum
        norm = np.array(np.nansum(modal, axis=1), dtype=float)
        mask = norm > num_eps
        modal[mask] = normalise_in_rows(modal[mask])

        # chemical
        indices = np.where(minerals_CD)[0] + 1  # +1 for modals
        chemical = np.array(stack([numbers[index] for index in indices], axis=1), dtype=float)

        numbers = stack((modal, chemical), axis=1)

        return numbers

    def collect_spectra(xq: np.ndarray, fnames: np.ndarray) -> np.ndarray:
        # This function collects spectra from the database, denoises and normalises them

        N = len(fnames)
        vq = np.zeros((N, len(xq)))

        # List of missing spectra (rows)
        irow = []

        # A loop through spectra
        for i, filename in enumerate(fnames):
            if path.exists(filename):  # load the data from the file
                if filename.endswith(".txt"):
                    # skip the first two lines (header + 1)
                    data = load_txt(filename, sep="\t", skiprows=1).to_numpy()
                    to_nm = 1000.

                elif filename.endswith(".asc"):
                    with open(filename, "r") as f:
                        n_lines = int(f.readline())  # First line contains info about the length of the data
                        data = np.array([np.array(f.readline().split()[:2], dtype=float) for _ in range(n_lines)])
                        to_nm = 1.

            else:  # download the data
                try:  # THE WEB PAGE DOES NOT WORK ANYMORE
                    pos_slash = find_all(filename, path.sep)
                    url = path.join(_relab_web_page, filename[pos_slash[-3]:])
                    print(f"Downloading spectrum from {url}")

                    if url.endswith(".txt"):
                        spectrum = urlopen(url).read().decode("utf-8").split("\n")  # splitlines() can be better

                        # Remove the header and the blank line at the end
                        spectrum = spectrum[2:-1]

                        data = np.array([np.array(line.split("\t")[:2], dtype=float) for line in spectrum])
                        to_nm = 1000.
                    elif filename.endswith(".asc"):
                        spectrum = urlopen(url).read().decode("utf-8").split("\r\n")  # splitlines() can be better

                        nlines = int(spectrum[0])
                        spectrum = spectrum[1:nlines + 1]

                        data = np.array([np.array(line.split()[:2], dtype=float) for line in spectrum])
                        to_nm = 1.
                except:
                    print(f"Spectrum {filename} does not exist and cannot be downloaded.")
                    irow.append(i)
                    continue

            x = data[:, 0] * to_nm  # to nm
            v = data[:, 1]

            # This has to be done due to some spectra
            x, idx = np.unique(x, return_index=True)
            v = v[idx]
            idx = np.logical_and(lambda_min - 50. <= x, x <= lambda_max + 50.)  # 50. to cover the edges for interp1d
            x, v = x[idx], v[idx]
            v, x = remove_outliers(y=v, x=x, z_thresh=0.5)

            vq[i, :] = interp1d(x, v, kind="cubic")(xq)

        # Remove missing data and normalise
        vq = np.delete(vq, irow, axis=0)
        vq = denoise_and_norm(data=vq, wavelength=xq, denoising=denoise, normalising=normalise,
                              sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

        return vq

    # skip header
    if end_line_number is None: start_line_number, end_line_number = 2, start_line_number

    print("Collecting data from RELAB...")
    subfolder = "RELAB"

    # read the data
    spectra_catalogue, sample_catalogue = load_excel_data(start_line_number, end_line_number)

    # label and metadata keys
    keys = np.array(sample_catalogue.keys().to_numpy(), dtype=str)
    metadata_key, labels_key = keys[:-num_labels_CD], keys[-num_labels_CD:]

    # The new axis
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)
    filenames = [""] * len(final_names)

    for i, (start, stop, final_name) in enumerate(zip(start_line_number, end_line_number, final_names)):
        # start_line_number[0] is offset
        rows = np.arange(start, stop + 1) - start_line_number[0]

        # split the data and filter them
        fnames, numbers, metadata = split_and_filter_data(spectra_catalogue.iloc[rows], sample_catalogue)

        if is_empty(fnames):
            continue

        # select the numbers according to the config file
        numbers = select_numbers(numbers, num_eps=_num_eps)

        # Collecting and normalising the spectra
        spectra = collect_spectra(xq, fnames)

        # Save the interpolated spectra
        filenames[i] = save_data(final_name, spectra=spectra, wavelengths=xq, labels=numbers, metadata=metadata,
                                 labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)

    return filenames


def collect_data_CTAPE() -> list[str]:
    print("Collecting data from C-Tape...")
    subfolder = f"C{_sep_in}Tape"

    # number of files
    N_files = 7, 1, 1, 4  # OL_OPX_num, OL_OPX_CPX_num, OL_CPX_num, OPX_CPX_num
    names = ("OL_OPX", "OL_OPX_CPX", "OL_CPX", "OPX_CPX")

    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    final_mat = np.zeros((45 + 6 + 4 + 22, len(xq)))
    start, stop = 0, 0

    for i, name in enumerate(names):
        for j in range(N_files[i]):
            data = np.transpose(load_txt(f"{name}_{j}.dat", subfolder=subfolder, sep="\t", header=None).to_numpy())
            x, v = data[0], data[1:]

            # delete pure spectra
            if i == 0 and j <= 3:
                v = np.delete(v, [0, 10], axis=0)
            elif i == 0 and j == 4:
                v = np.delete(v, [0, 4], axis=0)
            elif i == 0 and (j == 5 or j == 6):
                v = np.delete(v, [0, 1], axis=0)
            ####################
            elif i == 1 and j == 0:
                v = np.delete(v, [0], axis=0)
            ####################
            elif i == 2 and j == 0:
                v = np.delete(v, [0, 1], axis=0)
            ####################
            elif i == 3 and j == 0:
                v = np.delete(v, [0], axis=0)  # keep the OPX too (denoised here)
            elif i == 3 and j == 1:
                v = np.delete(v, [0, 5], axis=0)
            elif i == 3 and j == 2:
                v = np.delete(v, [0, 5], axis=0)
            elif i == 3 and j == 3:
                v = np.delete(v, [0], axis=0)  # keep the CPX too

            start, stop = stop, stop + len(v)

            # can be done before deleting the incorrect wavelengths, because the interpolation is just slicing here
            vq = interp1d(x, v, kind="cubic")(xq)

            # incorrect wavelengths (not noise-like)
            inds = np.array([505, 510, 520, 525, 535, 540, 545, 550, 555, 560, 565, 615, 625, 630, 635, 645, 650, 655,
                             675, 680, 685, 1385, 1390, 1395, 1400, 1405, 1410, 1415, 2285, 2290, 2305, 2310, 2315,
                             2320, 2325, 2390, 1485, 2280,
                             *safe_arange(1365, 1415, 5, endpoint=True, dtype=int),
                             *safe_arange(2250, 2415, 5, endpoint=True, dtype=int)])
            inds = np.where(np.in1d(xq, inds))[0]

            xq_clean = np.delete(xq, inds)
            vq_clean = np.delete(vq, inds, axis=1)

            # interpolate the spectra
            inds_to_delete = [find_outliers(spectrum, xq_clean, z_thresh=0.5) for spectrum in vq_clean]
            X = [np.delete(xq_clean, ind) for ind in inds_to_delete]
            Y = [np.delete(vq_clean[i], ind) for i, ind in enumerate(inds_to_delete)]

            vq_clean = np.array([interp1d(x, y, kind="cubic")(xq) for x, y in zip(X, Y)])

            vq_c = denoise_and_norm(data=vq_clean, wavelength=xq, denoising=denoise, normalising=normalise,
                                    sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

            """
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(xq, np.transpose(vq))
            ax[1].plot(xq, np.transpose(vq_clean))
            ax[2].plot(xq, np.transpose(vq_c))
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.show()
            """

            final_mat[start:stop, :] = vq_c

    sample_catalogue = load_xlsx("Sample_Catalogue.xlsx", usecols=usecols, header=1, sheet_name="C-Tape").to_numpy()

    labels = sample_catalogue[1:, -num_labels_CD:] / 100.
    metadata = sample_catalogue[1:, :-num_labels_CD]

    labels_key = sample_catalogue[0, -num_labels_CD:]
    metadata_key = sample_catalogue[0, :-num_labels_CD]

    filename = save_data(f"C{_sep_in}Tape{_sep_in}spectra", spectra=final_mat, wavelengths=xq, labels=labels,
                         metadata=metadata, labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)

    return [filename]


def resave_Tomas_OL_OPX_mixtures() -> list[str]:
    print("Collecting Tomas' OL-OPX mixtures...")

    subfoder = f"ol{_sep_in}opx{_sep_in}mix"
    dirin = path.join(_path_data, subfoder)
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # overlap in x1[130:] and x2[:6]
    x1 = safe_arange(400, 1100, 5, endpoint=True)
    x2 = safe_arange(1050, 2500, 10, endpoint=True)

    # olivine
    with open(path.join(dirin, "02_0px.dat")) as f:
        f.readline()
        data1 = np.array([f.readline() for _ in range(len(x1))], dtype=float)

    with open(path.join(dirin, "03_0px.dat")) as f:
        f.readline()
        data2 = np.array([f.readline() for _ in range(len(x2))], dtype=float)

    # match the two spectra, remove jumps at 850 [:91] and 1170 [147:] and remove outliers
    x, data = match_spectra((x1, x2), (data1, data2))
    data[:91] /= remove_jumps_in_spectra(x, data, jump_index=91)
    data[147:] *= remove_jumps_in_spectra(x, data, jump_index=147)
    data, x = remove_outliers(y=data, x=x, z_thresh=1.)

    OL = interp1d(x, data, kind="cubic")(xq)

    # pyroxene
    with open(path.join(dirin, "02_100px.dat")) as f:
        f.readline()
        data1 = np.array([f.readline() for _ in range(len(x1))], dtype=float)

    with open(path.join(dirin, "03_100px.dat")) as f:
        f.readline()
        data2 = np.array([f.readline() for _ in range(len(x2))], dtype=float)

    # match the two spectra, remove jumps at 850 [:91] and 1170 [147:] and remove outliers
    x, data = match_spectra((x1, x2), (data1, data2))
    data[:91] /= remove_jumps_in_spectra(x, data, jump_index=91)
    data[147:] *= remove_jumps_in_spectra(x, data, jump_index=147)
    data, x = remove_outliers(y=data, x=x, z_thresh=1.)

    OPX = interp1d(x, data, kind="cubic")(xq)

    # mixtures
    C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) * 100.
    C = np.array(C, dtype=int)

    C = np.array([f"OL {100 - s}, OPX {s}" for s in C])

    spectra_msm = np.zeros((len(xq), len(C)))
    spectra_msm[:, 0] = OL
    spectra_msm[:, 6] = OPX

    # Remove jumps at 850 [:91] and 1170/1160 [147:]/[146:] and remove outliers
    for i, filename in enumerate(["10", "25", "50", "75", "90"]):
        data = load_txt(f"{filename}px.dat", subfolder=subfoder, sep="\t", header=None).to_numpy()
        data[:91, 1] /= remove_jumps_in_spectra(data[:, 0], data[:, 1], jump_index=91)
        if i != 1:
            data[147:, 1] *= remove_jumps_in_spectra(data[:, 0], data[:, 1], jump_index=147)
        else:
            data[146:, 1] *= remove_jumps_in_spectra(data[:, 0], data[:, 1], jump_index=146)
        data = np.transpose(remove_outliers(y=data[:, 1], x=data[:, 0], z_thresh=1.)[::-1])  # wavelengths first
        spectra_msm[:, i + 1] = interp1d(data[:, 0], data[:, 1], kind="cubic")(xq)

    spectra = denoise_and_norm(data=np.transpose(spectra_msm), wavelength=xq, denoising=denoise, normalising=normalise,
                               sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

    sample_catalogue = load_xlsx("Sample_Catalogue.xlsx", usecols=usecols, header=1, sheet_name="TK").to_numpy()

    labels = sample_catalogue[1:, -num_labels_CD:] / 100.
    metadata = sample_catalogue[1:, :-num_labels_CD]

    labels_key = sample_catalogue[0, -num_labels_CD:]
    metadata_key = sample_catalogue[0, :-num_labels_CD]

    filename = save_data(f"ol{_sep_in}opx{_sep_in}mix", spectra=spectra, wavelengths=xq, labels=labels,
                         metadata=metadata, labels_key=labels_key, metadata_key=metadata_key, subfolder=subfoder)

    return [filename]


def resave_Chelyabinsk() -> list[str]:
    print("Re-saving Chelyabinsk data...")

    subfolder = "Chelyabinsk"

    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # SD spectra
    SD_names = ["SD0.txt", "SD5.txt", "SD10.txt", "SD20.txt", "SD30.txt", "SD40.txt", "SD50.txt", "SD60.txt",
                "SD70.txt", "SD80.txt", "SD90.txt", "SD95.txt", "SD100.txt"]
    SD_spectra = np.zeros((len(SD_names), len(xq)))

    for i, SD_name in enumerate(SD_names):
        tmp = load_txt(SD_name, subfolder=subfolder, sep="\t", header=None).to_numpy()
        x, y = tmp[:, 0], tmp[:, 1]
        y, x = remove_outliers(y=y, x=x, z_thresh=0.3)

        tmp_spec = interp1d(x, y, kind="cubic")(xq)

        SD_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                            sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

    # IM spectra
    IM_names = ["100LL.txt", "10IM.txt", "20IM.txt", "30IM.txt", "40IM.txt", "50IM.txt", "60IM.txt", "70IM.txt",
                "80IM.txt", "90IM.txt", "95IM.txt", "100IM.txt"]
    IM_spectra = np.zeros((len(IM_names), len(xq)))

    for i, IM_name in enumerate(IM_names):
        tmp = load_txt(IM_name, subfolder=subfolder, sep="\t", header=None).to_numpy()
        x, y = tmp[:, 0] * 1000., tmp[:, 1]
        y, x = remove_outliers(y=y, x=x, z_thresh=0.3)

        tmp_spec = interp1d(x, y, kind="cubic")(xq)

        IM_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                            sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

    # SW spectra
    SW_names = ["SW0.txt", "SW400.txt", "SW500.txt", "SW600.txt", "SW700.txt"]
    SW_spectra = np.zeros((len(SW_names), len(xq)))

    for i, SW_name in enumerate(SW_names):
        tmp = load_txt(SW_name, subfolder=subfolder, sep="\t", header=None).to_numpy()
        x, y = tmp[:, 0], tmp[:, 1]
        y, x = remove_outliers(y=y, x=x, z_thresh=0.7)

        tmp_spec = interp1d(x, y, kind="cubic")(xq)

        SW_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                            sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

    combined_spectra = stack((SD_spectra, IM_spectra, SW_spectra), axis=0)

    # names for metadata
    SD_names = ["SD 0%", "SD 5%", "SD 10%", "SD 20%", "SD 30%", "SD 40%", "SD 50%",
                "SD 60%", "SD 70%", "SD 80%", "SD 90%", "SD 95%", "SD 100%"]
    IM_names = ["IM 0%", "IM 10%", "IM 20%", "IM 30%", "IM 40%", "IM 50%",
                "IM 60%", "IM 70%", "IM 80%", "IM 90%", "IM 95%", "IM 100%"]
    SW_names = ["fresh", "SW 400", "SW 500", "SW 600", "SW 700"]

    metadata = np.array(SD_names + IM_names + SW_names)
    metadata = np.reshape(metadata, (len(metadata), 1))

    metadata_key = np.array(["observation type"])

    labels = np.array([[66.2, 33.8, 0.0, 0.0, 28.6, 71.4, 23.9, 76.1, 1.6, 0.0, 0.0, 0.0, 8.5, 84.5, 7.0]])
    labels = np.repeat(labels, len(metadata), axis=0)

    sample_catalogue = load_xlsx("Sample_Catalogue.xlsx", usecols=usecols, header=1, sheet_name="TK").to_numpy()

    labels_key = sample_catalogue[0, -num_labels_CD:]

    filename = save_data("Chelyabinsk", spectra=combined_spectra, wavelengths=xq, labels=labels,
                         metadata=metadata, metadata_key=metadata_key, labels_key=labels_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    return [filename]


def resave_asteroid_taxonomy_data(grouping_options: list[str]) -> None:
    # This function interpolates B-D + MITHNEOS data to the given grid
    print("Re-sampling asteroid data...")

    if not normalise:
        print("I don't have non-normalised spectra of asteroids. Skipping it.")
        return

    subfolder = "taxonomy"

    # old grid
    x_old = safe_arange(450, 2450, 10, endpoint=True)

    # new grid
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # load the data
    raw_data = "asteroid_pca-spectra-combined.dat"
    data = load_txt(raw_data, subfolder=subfolder, sep="\t", header=None).to_numpy()

    labels_key = np.array(["taxonomy class"])
    metadata_key = np.array(["asteroid number", "taxonomy class", "slope", "PC1", "PC2", "PC3", "PC4", "PC5"])

    # asteroid number, taxonomy, slope, PC1--PC5, spectra
    metadata_raw, spectra_raw = data[:, :len(metadata_key)], np.array(data[:, len(metadata_key):], dtype=float)

    # labels data (B-DM class)
    labels_raw = metadata_raw[:, [1]]

    final_name = f"asteroid{_sep_in}spectra"

    # interpolate the spectra
    inds_to_delete = [find_outliers(s, x_old, z_thresh=0.5) for s in spectra_raw]
    X = [np.delete(x_old, ind) for ind in inds_to_delete]
    Y = [np.delete(spectra_raw[i], ind) for i, ind in enumerate(inds_to_delete)]

    spectra_raw = np.array([interp1d(x, y, kind="cubic")(xq) for x, y in zip(X, Y)])

    spectra_raw = denoise_and_norm(data=spectra_raw, wavelength=xq, denoising=denoise, normalising=normalise,
                                   sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

    filename = save_data(final_name, spectra=spectra_raw, wavelengths=xq, labels=labels_raw, metadata=metadata_raw,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    for option in grouping_options:
        final_name_option = f"{final_name}{_sep_out}{option}"

        if option == "15":
            classes_to_delete = ["O", "Qw", "Sq", "Sq:", "Sv", "Svw", "U", "Xn"]

            classes_to_combine = [("A", "A+"), ("Sa", "A+"),
                                  ("C", "C+"), ("Cb", "C+"), ("Cg", "C+"),
                                  ("Ch", "Ch+"), ("Cgh", "Ch+"),
                                  ("S", "S+"), ("Sqw", "S+"), ("Sw", "S+"),
                                  ("Sr", "Sr+"), ("Srw", "Sr+"), ("R", "Sr+"),
                                  ("V", "V+"), ("Vw", "V+"),
                                  ("X", "X+"), ("Xc", "X+")]

        elif option == "9":
            classes_to_delete = ["K", "O", "Qw", "R", "Sq", "Sq:", "Sv", "Svw", "T", "U", "Xn"]

            classes_to_combine = [("A", "A+"), ("Sa", "A+"),
                                  ("C", "C+"), ("Cb", "C+"), ("Cg", "C+"), ("B", "C+"),
                                  ("Ch", "Ch+"), ("Cgh", "Ch+"),
                                  ("S", "S+"), ("Sqw", "S+"), ("Sr", "S+"), ("Srw", "S+"), ("Sw", "S+"),
                                  ("V", "V+"), ("Vw", "V+"),
                                  ("X", "X+"), ("Xc", "X+"), ("Xe", "X+"), ("Xk", "X+")]
        elif option == "16":
            classes_to_delete = ["O", "Qw", "Sq", "Sq:", "Sv", "Svw", "U", "Xn"]

            classes_to_combine = [("A", "A+"), ("Sa", "A+"),
                                  ("C", "C+"), ("Cb", "C+"),
                                  ("Cg", "Cgh+"), ("Cgh", "Cgh+"),
                                  ("S", "S+"), ("Sqw", "S+"), ("Sw", "S+"),
                                  ("Sr", "Sr+"), ("Srw", "Sr+"), ("R", "Sr+"),
                                  ("V", "V+"), ("Vw", "V+"),
                                  ("X", "X+"), ("Xc", "X+")]

        elif option == "C":
            classes_to_delete = ["A", "Cgh", "Ch", "D", "K", "L", "O", "Q", "Qw", "R", "S", "Sa", "Sq", "Sq:", "Sqw",
                                 "Sr", "Srw", "Sv", "Svw", "Sw", "T", "U", "V", "Vw", "X", "Xc", "Xe", "Xk", "Xn"]

            classes_to_combine = [("C", "C+"), ("Cb", "C+"), ("Cg", "C+")]

        elif option == "X":
            classes_to_delete = ["A", "B", "C", "Cb", "Cg", "Cgh", "Ch", "D", "K", "L", "O", "Q", "Qw", "R", "S", "Sa",
                                 "Sq", "Sq:", "Sqw", "Sr", "Srw", "Sv", "Svw", "Sw", "T", "U", "V", "Vw", "Xn"]

            classes_to_combine = [("X", "X+"), ("Xc", "X+"), ("Xe", "X+")]

        # DELETED CLASSES #
        inds_deleted = np.array([ind for ind, taxonomy_class in enumerate(labels_raw.ravel())
                                 if taxonomy_class in classes_to_delete])

        filename = save_data(f"{final_name_option}{_sep_out}deleted", spectra=spectra_raw[inds_deleted], wavelengths=xq,
                             labels=labels_raw[inds_deleted], labels_key=labels_key,
                             metadata=metadata_raw[inds_deleted], metadata_key=metadata_key, subfolder=subfolder)
        # my_mv(filename, filename.replace(subfolder, "", 1), "cp")

        # REDUCED TAXONOMY #
        # !!! METADATA CONTAINS ORIGINAL TAXONOMY CLASSES !!!
        labels_reduced = deepcopy(labels_raw)
        for old, new in classes_to_combine:
            inds = np.array([old == ast_type for ast_type in labels_raw.ravel()])
            labels_reduced[inds] = new

        # delete the classes
        labels_reduced = np.delete(labels_reduced, inds_deleted, axis=0)
        spectra_reduced = np.delete(spectra_raw, inds_deleted, axis=0)
        metadata_reduced = np.delete(metadata_raw, inds_deleted, axis=0)

        filename = save_data(f"{final_name_option}{_sep_out}reduced", spectra=spectra_reduced, wavelengths=xq,
                             labels=labels_reduced, metadata=metadata_reduced, labels_key=labels_key,
                             metadata_key=metadata_key, subfolder=subfolder)
        my_mv(filename, filename.replace(subfolder, "", 1), "cp")

        # REDUCED TAXONOMY WITH METEORITES #
        # add OC from RELAB to Q-type
        if "Q" not in classes_to_delete:
            data = load_npz(f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz")  # to read the file

            metadata_min = join_data(data, header="meta")
            types = np.array(metadata_min[["Type1"]], dtype=str)
            inds = np.where(np.array(["Ordinary Chondrite" == ast_type for ast_type in types]))[0]
            types = types[inds]

            # combine Q with some others?
            Q_combine_mask = ["Q" in combine[0] for combine in classes_to_combine]
            if np.any(Q_combine_mask):
                types[:] = classes_to_combine[np.where(Q_combine_mask)[0][0]][1]
            else:  # Q is not combined and is just Q
                types[:] = "Q"

            OC = data[_spectra_name][inds]
            x_oc = data[_wavelengths_name]
            OC = interp1d(x_oc, OC, kind="cubic")(xq)

            OC = denoise_and_norm(data=OC, wavelength=xq, denoising=False, normalising=True, sigma_nm=denoising_sigma,
                                  wvl_norm_nm=wvl_norm)

            OC_meta = np.array([np.shape(metadata_reduced)[1] * [np.nan]], dtype=object)
            OC_meta[0, 1] = "Q"
            OC_meta = np.repeat(OC_meta, len(OC), axis=0)

            labels_reduced = stack((labels_reduced, types), axis=0)
            spectra_reduced = stack((spectra_reduced, OC), axis=0)
            metadata_reduced = stack((metadata_reduced, OC_meta), axis=0)

            filename = save_data(f"{final_name_option}{_sep_out}reduced{_sep_in}met", spectra=spectra_reduced,
                                 wavelengths=xq, labels=labels_reduced, metadata=metadata_reduced, labels_key=labels_key,
                                 metadata_key=metadata_key, subfolder=subfolder)
            # my_mv(filename, filename.replace(subfolder, "", 1), "cp")


def resave_Itokawa_Eros() -> None:
    print("Re-saving Itokawa and Eros data...")

    if not normalise:
        print("I don't have non-normalised spectra of Itokawa and Eros. Skipping it.")
        return

    metadata_key = np.array(["longitude", "latitude", "asteroid name", "instrument"])

    # load the data
    norm_at = 1500.
    filename = f"Itokawa{_sep_out}averaged.h5"  # Itokawa
    subfolder = path.join("asteroids", "Itokawa")

    data, coordinates, wavelengths = load_h5(filename, subfolder=subfolder,
                                             list_keys=[_spectra_name, _coordinates_name, _wavelengths_name]).values()

    inds_to_delete = [find_outliers(d, wavelengths, z_thresh=0.5) for d in data]
    X = [np.delete(wavelengths, ind) for ind in inds_to_delete]
    Y = [np.delete(data[i], ind) for i, ind in enumerate(inds_to_delete)]

    data = np.array([interp1d(x, y, kind="cubic")(wavelengths) for x, y in zip(X, Y)])

    metadata = np.array([["Itokawa", "Hayabusa Near Infrared Spectrometer"]])
    metadata = np.repeat(metadata, len(data), axis=0)
    metadata = stack((np.array(coordinates, dtype=object), metadata), axis=1)

    data = denoise_and_norm(data=data, wavelength=wavelengths, denoising=denoise, normalising=normalise,
                            sigma_nm=denoising_sigma, wvl_norm_nm=norm_at)

    filename = save_data("Itokawa", spectra=data, wavelengths=wavelengths, metadata=metadata,
                         metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    # load the data
    norm_at = 1300.
    filename = f"Eros{_sep_out}averaged.h5"  # Eros
    subfolder = path.join("asteroids", "Eros")

    data, coordinates, wavelengths = load_h5(filename, subfolder=subfolder,
                                             list_keys=[_spectra_name, _coordinates_name, _wavelengths_name]).values()

    metadata = np.array([["Eros", "NEAR Shoemaker Near-Infrared Spectrometer"]])
    metadata = np.repeat(metadata, len(data), axis=0)
    metadata = stack((np.array(coordinates, dtype=object), metadata), axis=1)

    # keep only data from 820 to 2360 nm
    mask = wavelengths <= 2360.
    wavelengths = wavelengths[mask]
    data = data[:, mask]

    inds_to_delete = [find_outliers(d, wavelengths, z_thresh=0.5) for d in data]
    X = [np.delete(wavelengths, ind) for ind in inds_to_delete]
    Y = [np.delete(data[i], ind) for i, ind in enumerate(inds_to_delete)]

    data = np.array([interp1d(x, y, kind="cubic")(wavelengths) for x, y in zip(X, Y)])

    data = denoise_and_norm(data=data, wavelength=wavelengths, denoising=denoise, normalising=normalise,
                            sigma_nm=denoising_sigma, wvl_norm_nm=norm_at)

    # NIS reflectances (damaged at 1500+)
    save_data(f"Eros{_sep_in}NIS", spectra=data, wavelengths=wavelengths, metadata=metadata,
              metadata_key=metadata_key, subfolder=subfolder)

    # keep only data from 820 to 1480 nm
    mask = wavelengths <= 1480.
    wavelengths_1480 = wavelengths[mask]
    data_1480 = data[:, mask]

    save_data(f"Eros{_sep_in}1480", spectra=data_1480, wavelengths=wavelengths_1480, metadata=metadata,
              metadata_key=metadata_key, subfolder=subfolder)

    # combine NIS and telescope data
    ast = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    meta = join_data(ast, "meta")
    eros = "433" == np.array(meta["asteroid number"], dtype=str)

    spectrum, wvl = ast[_spectra_name][eros], ast[_wavelengths_name]

    spectra = interp1d(wvl, spectrum, kind="cubic")(wavelengths)
    spectrum = normalise_spectra(spectra, wavelengths, wvl_norm_nm=norm_at)

    data_1, data_2 = data[:, mask], data[:, ~mask]

    # correction = spectrum[:, ~mask] - np.mean(data_2, axis=0)
    # data_2_corr = data_2 + correction

    correction = spectrum[:, ~mask] / np.mean(data_2, axis=0)
    align = np.mean(data[:, np.where(mask)[0][-1]], axis=0) / spectrum[:, np.where(mask)[0][-1]]
    data_2_corr = data_2 * correction * align

    data_corr = stack((data_1, data_2_corr), axis=1)

    filename = save_data("Eros", spectra=data_corr, wavelengths=wavelengths, metadata=metadata,
                         metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")


def resave_kachr_ol_opx() -> list[str]:
    print("Re-saving Katka's data...")

    subfolder = f"ol{_sep_in}opx{_sep_in}pure"

    metadata_key = np.array(["weathering type", "area density of damage"])

    metadata = np.array([["Olivine-Ar+", "fresh"], ["Olivine-Ar+", "1e15 Ar+/cm2"],
                         ["Olivine-Ar+", "3e15 Ar+/cm2"], ["Olivine-Ar+", "6e15 Ar+/cm2"],
                         ["Olivine-Ar+", "1e16 Ar+/cm2"], ["Olivine-Ar+", "2e16 Ar+/cm2"],
                         ["Olivine-Ar+", "6e16 Ar+/cm2"], ["Olivine-Ar+", "1e17 Ar+/cm2"],
                         #
                         ["Olivine-H+", "fresh"], ["Olivine-H+", "1e14 H+/cm2"], ["Olivine-H+", "1e15 H+/cm2"],
                         ["Olivine-H+", "1e16 H+/cm2"], ["Olivine-H+", "1e17 H+/cm2"],
                         ["Olivine-H+", "2e17 H+/cm2"], ["Olivine-H+", "5e17 H+/cm2"],
                         ["Olivine-H+", "1e18 H+/cm2"],
                         #
                         ["Olivine-He+", "fresh"], ["Olivine-He+", "1e16 He+/cm2"], ["Olivine-He+", "3e16 He+/cm2"],
                         ["Olivine-He+", "6e16 He+/cm2"], ["Olivine-He+", "1e17 He+/cm2"],
                         #
                         ["Olivine-laser", "fresh"], ["Olivine-laser", "1.7 J/cm2"], ["Olivine-laser", "2.4 J/cm2"],
                         ["Olivine-laser", "3.8 J/cm2"], ["Olivine-laser", "4.6 J/cm2"],
                         ["Olivine-laser", "6.7 J/cm2"], ["Olivine-laser", "10.4 J/cm2"],
                         ["Olivine-laser", "15.0 J/cm2"], ["Olivine-laser", "23.4 J/cm2"],
                         ["Olivine-laser", "30.6 J/cm2"], ["Olivine-laser", "60.0 J/cm2"],
                         ["Olivine-laser", "93.8 J/cm2"], ["Olivine-laser", "375.0 J/cm2"],
                         #
                         #
                         ["Pyroxene-Ar+", "fresh"], ["Pyroxene-Ar+", "1e15 Ar+/cm2"],
                         ["Pyroxene-Ar+", "3e15 Ar+/cm2"], ["Pyroxene-Ar+", "6e15 Ar+/cm2"],
                         ["Pyroxene-Ar+", "1e16 Ar+/cm2"], ["Pyroxene-Ar+", "2e16 Ar+/cm2"],
                         ["Pyroxene-Ar+", "6e16 Ar+/cm2"], ["Pyroxene-Ar+", "1e17 Ar+/cm2"],
                         #
                         ["Pyroxene-H+", "fresh"], ["Pyroxene-H+", "1e16 H+/cm2"], ["Pyroxene-H+", "1e17 H+/cm2"],
                         ["Pyroxene-H+", "2e17 H+/cm2"], ["Pyroxene-H+", "5e17 H+/cm2"],
                         ["Pyroxene-H+", "1e18 H+/cm2"],
                         #
                         ["Pyroxene-He+", "fresh"], ["Pyroxene-He+", "1e16 He+/cm2"],
                         ["Pyroxene-He+", "3e16 He+/cm2"], ["Pyroxene-He+", "6e16 He+/cm2"],
                         ["Pyroxene-He+", "1e17 He+/cm2"],
                         #
                         ["Pyroxene-laser", "fresh"], ["Pyroxene-laser", "4.5 J/cm2"],
                         ["Pyroxene-laser", "5.6 J/cm2"], ["Pyroxene-laser", "12.5 J/cm2"],
                         ["Pyroxene-laser", "18.0 J/cm2"], ["Pyroxene-laser", "28.1 J/cm2"],
                         ["Pyroxene-laser", "36.7 J/cm2"], ["Pyroxene-laser", "50.0 J/cm2"],
                         ["Pyroxene-laser", "72.0 J/cm2"], ["Pyroxene-laser", "112.5 J/cm2"],
                         ["Pyroxene-laser", "200.0 J/cm2"], ["Pyroxene-laser", "450.0 J/cm2"],
                         ["Pyroxene-laser", "1800.0 J/cm2"]
                         ])

    x_new_part = safe_arange(545, 2415, 5, endpoint=True)
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # file names and suffixes
    names = ["ol", "py"]
    suffixes = ["Ar", "H", "He", "laser"]
    spectra = np.zeros((len(metadata), len(xq)))

    stop = 0  # indices in spectra

    for name, suffix in product(names, suffixes):
        tmp = load_txt(f"{name}-{suffix}.csv", subfolder=subfolder, sep="\t", header=None).to_numpy()

        x, y = tmp[:, 0], tmp[:, 1:]

        inds_to_delete = [find_outliers(spectrum, x, z_thresh=1.) for spectrum in np.transpose(y)]
        X = [np.delete(x, ind) for ind in inds_to_delete]
        Y = [np.delete(y[:, i], ind) for i, ind in enumerate(inds_to_delete)]

        tmp_spec = np.array([interp1d(x, y, kind="cubic")(x_new_part) for x, y in zip(X, Y)])

        # Linear extrapolation if needed
        tmp_spec = interp1d(x_new_part, tmp_spec, kind="linear", fill_value="extrapolate")(xq)

        start, stop = stop, stop + len(tmp_spec)

        """
        plt.figure()
        plt.plot(xq, np.transpose(tmp_spec))
        """

        spectra[start:stop] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                               sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

    labels_ol = np.array([[1, 0, 0, 0, 9.9, 90.1, 0, 0, 0.0, 0, 0, 0, 0, 0, 0]])
    labels_ol = np.repeat(labels_ol, 34, axis=0)
    labels_px = np.array([[0, 84/(84+5) * 100, 5/(84+5) * 100, 0, 0, 0, 32.9, 67.1, 0, 0, 0, 0, 0, 0, 0]])
    labels_px = np.repeat(labels_px, 32, axis=0)

    labels = stack((labels_ol, labels_px), axis=0)

    sample_catalogue = load_xlsx("Sample_Catalogue.xlsx", usecols=usecols, header=1, sheet_name="TK").to_numpy()

    labels_key = sample_catalogue[0, -num_labels_CD:]
    labels_key[1] = labels_key[1].replace("vol", "wt")
    labels_key[2] = labels_key[2].replace("vol", "wt")

    filename = save_data(f"ol{_sep_in}opx{_sep_in}pure", spectra=spectra, wavelengths=xq, metadata=metadata,
                         labels=labels, metadata_key=metadata_key, labels_key=labels_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    return [filename]


def resave_didymos_2004() -> list[str]:
    print("Re-saving Didymos data...")

    subfolder = "Didymos"

    xq = safe_arange(490, lambda_max, resolution_final, endpoint=True)

    file = "Didymos_vnir_albedo.dat"
    tmp = load_txt(file, subfolder=subfolder, sep="\t").to_numpy()
    x, y = tmp[:, 0], np.transpose(tmp[:, 1:])
    y, x = remove_outliers(y=y[0], x=x, z_thresh=0.4)

    spectrum = interp1d(x, y, kind="cubic")(xq)  # one spectrum per row
    spectrum = denoise_and_norm(data=spectrum, wavelength=xq, denoising=denoise, normalising=normalise, sigma_nm=30.,
                                wvl_norm_nm=wvl_norm)

    metadata = np.array([["Didymos spectrum", "NOT_TNG_20040116", "10.1051/0004-6361/200913852"]])

    filename = save_data(f"Didymos{_sep_in}2004", spectra=spectrum, wavelengths=xq, metadata=metadata,
                         subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    return [filename]


def resave_didymos_2022(add_blue_part: bool = False) -> list[str]:
    print("Re-saving Didymos data...")

    if not normalise:
        print("I don't have non-normalised spectra of Didymos. Skipping it.")
        return [""]

    files = ["a65803_IR_Spec_IRTF_20220926_Polishook.dat", "a65803_IR_Spec_IRTF_20220927_Polishook.dat"]

    # xq = safe_arange(650, 2450, 5, endpoint=True)
    xq = safe_arange(800, lambda_max, resolution_final, endpoint=True)

    if add_blue_part:  # the original spectrum starts at 650. I added mean S type here
        asteroid_data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
        S_types = asteroid_data[_label_name].ravel() == "S"
        S_part = asteroid_data[_wavelengths_name] < xq[0]
        S_spectra_mean = np.mean(asteroid_data[_spectra_name][S_types], axis=0)[S_part]
        S_wavelengths = asteroid_data[_wavelengths_name][S_part]

        xq_full = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

        final_spectra = np.zeros((len(files), len(xq_full)))
    else:
        final_spectra = np.zeros((len(files), len(xq)))

    subfolder = "Didymos"

    for i, file in enumerate(files):
        tmp = load_txt(file, subfolder=subfolder, sep="\s+", header=None).iloc[:, :2].to_numpy()

        x, y = tmp[:, 0] * 1000., tmp[:, 1]
        y, x = remove_outliers(y=y, x=x, z_thresh=0.2)  # spectra are very noisy...

        spectrum = interp1d(x, np.transpose(y), kind="linear")(xq)  # too noisy to do cubic interpolation

        spectrum = denoise_and_norm(data=spectrum, wavelength=xq, denoising=denoise, normalising=normalise,
                                    sigma_nm=20., wvl_norm_nm=1500.)

        if add_blue_part:
            x_old_full, spectrum = match_spectra((S_wavelengths, xq), (S_spectra_mean, spectrum[0]))
            spectrum = interp1d(x_old_full, spectrum, kind="cubic")(xq_full)

            spectrum = denoise_and_norm(data=spectrum, wavelength=xq_full, denoising=False, normalising=normalise,
                                        sigma_nm=denoising_sigma, wvl_norm_nm=wvl_norm)

            # to decrease a slope of the mean S type
            spectrum *= 1. + (xq[0] - xq_full + np.abs(xq[0] - xq_full)) / 4000.
            spectrum = denoise_and_norm(data=spectrum, wavelength=xq_full, denoising=True, normalising=normalise,
                                        sigma_nm=5., wvl_norm_nm=wvl_norm)

        final_spectra[i] = spectrum

    if add_blue_part:
        xq = xq_full

    metadata = np.array([["Didymos spectrum", "IR_Spec_IRTF_20220926"],
                         ["Didymos spectrum", "IR_Spec_IRTF_20220927"]])

    filename = save_data(f"Didymos{_sep_in}2022", spectra=final_spectra, wavelengths=xq,
                         metadata=metadata, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    return [filename]


def resave_data_for_shapeViewer(y_pred: np.ndarray, asteroid_name: str,
                                taxonomy_or_composition: Literal["composition", "taxonomy"],
                                used_minerals: np.ndarray | None = None,
                                used_endmembers: list[list[bool]] | None = None,
                                used_classes: dict[str, int] | list | np.ndarray | None = None,
                                indices_to_save: list[int] | None = None) -> None:

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used
    if used_classes is None: used_classes = classes

    if isinstance(used_classes, dict):
        labels = np.array(list(used_classes.keys()))
    else:
        labels = np.array(used_classes)

    # bg white (white background)
    # L (turn off legend)

    if indices_to_save is None: indices_to_save = range(np.shape(y_pred)[1])

    num_digits = 5

    if taxonomy_or_composition == "composition":
        quantity_names = flatten_list([mineral_names] + endmember_names)[used_indices(used_minerals,
                                                                                      used_endmembers)]
    elif taxonomy_or_composition == "taxonomy":
        quantity_names = labels
    else:
        raise ValueError('"taxonomy_or_composition" must be either "taxonomy" or "composition"')

    """
    if asteroid_name == "Itokawa":  # save position of craters if Itokawa
        scene = pywavefront.Wavefront(path.join(_path_data, "shapeViewer", f"{asteroid_name}.obj"),
                                      collect_faces=True, create_materials=True)
        vertices = np.array(scene.vertices)
        x, y, z = np.transpose(vertices)

        r = np.sqrt(x * x + y * y + z * z)  # km
        phi = np.arctan2(y, x) + np.pi
        theta = np.arccos(z / r) - np.pi / 2.

        # list of craters from "A survey of possible impact structures on 25143 Itokawa"
        # https://doi.org/10.1016/j.icarus.2008.10.027
        # id (index of array); lon, lat, confidence
        craters = np.zeros((38, 3))
        craters[0], craters[1], craters[2], craters[3] = [348, 25, 4], [175, -10, 2], [275, -2, 4], [128, 0, 3]
        craters[4], craters[5], craters[6], craters[7] = [112, 40, 2], [8, 8, 2], [17, -8, 4], [172, 15, 4]
        craters[8], craters[9], craters[10], craters[11] = [134, 20, 4], [244, -40, 3], [151, -6, 3], [215, 17, 3]
        craters[12], craters[13], craters[14], craters[15] = [269, 34, 1], [145, 4, 3], [102, -10, 1], [205, -18, 2]
        craters[16], craters[17], craters[18], craters[19] = [216, -26, 4], [221, -36, 4], [212, -33, 3], [254, -15, 4]
        craters[20], craters[21], craters[22], craters[23] = [7, -18, 4], [162, 1, 2], [14, -17, 2], [52, 12, 3]
        craters[24], craters[25], craters[26], craters[27] = [183, 17, 4], [169, 24, 4], [345, -17, 3], [277, -13, 4]
        craters[28], craters[29], craters[30], craters[31] = [45, 19, 3], [117, -39, 3], [202, 28, 4], [207, 33, 4]
        craters[32], craters[33], craters[34], craters[35] = [232, -40, 4], [45, -28, 1], [244, 6, 2], [111, -33, 3]
        craters[36], craters[37] = [319, -28, 1], [205, -76, 1]

        craters = stack((np.arange(1, len(craters) + 1), craters), axis=1)  # add IDs

        lon, lat, ID = craters[:, [1]], craters[:, [2]], np.array(craters[:, [0]], dtype=int)

        # find which vertex is closest to the data point
        radians_vertices = stack((theta, phi), axis=1)
        radians_target = stack((np.radians(lat), np.radians(lon)), axis=1)
        distance = pairwise_distances(radians_vertices, radians_target, metric="haversine")

        inds_vertices = np.argmin(distance, axis=0)
        altitude = np.transpose(r[[inds_vertices]]) + 0.01 # km; 0.01 to keep points above surface

        data_type = np.repeat([["c"]], len(lon), axis=0)

        results = stack((data_type, lat, lon, altitude, ID), axis=1)

        # delete some craters that are not visible
        inds_to_delete = []  # ID - 1
        results = np.delete(results, inds_to_delete, axis=0)

        filename = path.join(_path_data, "shapeViewer", f"{asteroid_name}_craters.dat")
        check_dir(filename)
        np.savetxt(filename, results, fmt="%s", delimiter=" ")
    """

    data = load_npz(f"{asteroid_name}{_sep_out}denoised{_sep_out}norm.npz")
    metadata = join_data(data, "meta")

    lon = np.array(metadata[["longitude"]], dtype=float)
    lat = np.array(metadata[["latitude"]], dtype=float)

    # This is to match the shapeViewer notation
    data_type = np.repeat([["g"]], len(lon), axis=0)

    # shapeViewer uses angular radius
    # pi r^2 = area -> r = sqrt(area / pi), but "/ pi" will do gaps in figures
    # radius = np.full(np.shape(lon), np.sqrt(dlon * dlat / np.pi))

    # radius of a circumscribed circle
    radius = np.full(np.shape(lon), np.sqrt(dlon * dlon + dlat * dlat)) * 0.5

    lat, lon = np.round(lat, 1), np.round(lon, 1)
    radius = np.ceil(radius * 100.) / 100.  # ceil to 2 decimals

    for index_to_save in indices_to_save:
        quantity_name = quantity_names[index_to_save].replace(" ", _sep_in)

        reverse_caxis = quantity_name in ["Q", "olivine"]

        prediction_to_save = np.round(y_pred[:, [index_to_save]] * 100., num_digits)

        if reverse_caxis:
            prediction_to_save = 100. - prediction_to_save

        prediction_to_save = np.array([f"{p:.{num_digits}f}" for p in np.round(prediction_to_save.ravel(), num_digits)])
        prediction_to_save = np.reshape(prediction_to_save, (len(prediction_to_save), 1))

        results = stack((data_type, lat, lon, radius, prediction_to_save), axis=1)

        if reverse_caxis:
            filename = path.join(_path_data, "shapeViewer", f"min{_sep_out}max{_sep_out}cbar",
                                 f"{asteroid_name}{_sep_out}{quantity_name}{_sep_out}reversed.dat")
        else:
            filename = path.join(_path_data, "shapeViewer", f"min{_sep_out}max{_sep_out}cbar",
                                 f"{asteroid_name}{_sep_out}{quantity_name}.dat")

        check_dir(filename)
        np.savetxt(filename, results, fmt="%s", delimiter=" ")

        # to make the shapeViewer colorbar be from 0 to 100
        # these points are not visible
        tmp = np.array([["g", "-89.9", "0.0", "0.1", "0.0"],
                       ["g", "-89.9", "1.0", "0.1", "100.0"]])
        results = stack((tmp, results), axis=0)

        if reverse_caxis:
            filename = path.join(_path_data, "shapeViewer", f"0{_sep_in}100{_sep_out}cbar",
                                 f"{asteroid_name}{_sep_out}{quantity_name}{_sep_out}reversed.dat")
        else:
            filename = path.join(_path_data, "shapeViewer", f"0{_sep_in}100{_sep_out}cbar",
                                 f"{asteroid_name}{_sep_out}{quantity_name}.dat")


        check_dir(filename)
        np.savetxt(filename, results, fmt="%s", delimiter=" ")


def resave_Itokawa_Eros_results(used_minerals: np.ndarray | None = None,
                                used_endmembers: list[list[bool]] | None = None,
                                used_classes: dict[str, int] | None = None,
                                proportiontocut: float | None = None):
    from modules.NN_evaluate import evaluate

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used
    if used_classes is None: used_classes = classes
    if proportiontocut is None: proportiontocut = comp_model_setup["trim_mean_cut"]

    labels = [[f"{name} (vol%)" for name in mineral_names_short]] + endmember_names
    labels = flatten_list(labels)[used_indices(used_minerals, used_endmembers)]

    header_coords, header_tax, header_comp = ["lon (deg)", "lat (deg)"], list(used_classes.keys()), labels
    header = stack((header_coords, header_tax, header_comp))

    nC, nT = len(header_coords), len(header_coords) + len(header_tax)

    keep_coords = [0, 1]

    s_class = "S" if "S" in used_classes else "S+"
    q_class = "Q" if "Q" in used_classes else "Q+"
    l_class = "L" if "L" in used_classes else "L+"

    keep_tax_E = np.array([used_classes[s_class], used_classes[q_class], used_classes[l_class]]) + nC
    keep_tax_I = np.array([used_classes[s_class], used_classes[q_class]]) + nC
    keep_comp = unique_indices(used_minerals, used_endmembers, all_minerals=False, return_digits=True) + nT

    keep_inds_I = stack((keep_coords, keep_tax_I, keep_comp))
    keep_inds_E = stack((keep_coords, keep_tax_E, keep_comp))

    filenames = ["Itokawa", "Eros"]

    for filename in filenames:
        model_grid = "_".join(str(int(x)) for x in np.round(data_grids[filename]))
        data = load_npz(f"{filename}{_sep_out}denoised{_sep_out}norm.npz")
        metadata = join_data(data, "meta")

        lon = np.array(metadata["longitude"], dtype=float)
        lat = np.array(metadata["latitude"], dtype=float)

        results = stack((lon, lat), axis=1)

        for model_type in ["taxonomy", "composition"]:
            subfolder_model = path.join(model_type, model_grid)
            model_names = collect_all_models(subfolder_model=subfolder_model, full_path=False)
            results = stack((results, 100. * evaluate(model_names, data[_spectra_name],
                                                      proportiontocut=proportiontocut,
                                                      subfolder_model=model_type)), axis=1)

        results = np.array(np.round(results, 1), dtype=str)

        results = stack((header, results), axis=0)

        if filename == "Itokawa":
            results = results[:, keep_inds_I]
        else:
            results = results[:, keep_inds_E]

        np.savetxt(path.join(_path_data, f"{filename}{_sep_out}results.dat"), results, fmt="%s", delimiter="\t")


def resave_HyperScout_transmission() -> None:
    print("Re-saving HyperScout's transmission...")

    filename = path.join(_path_data, "HyperScout", "HS-H_transmission.xlsx")
    transmissions = load_xlsx(filename, sheet_name="Sheet1").to_numpy()

    wavelengths, transmissions = transmissions[:, 0], np.transpose(transmissions[:, 1:])

    # sort wavelengths
    idx = np.argsort(wavelengths)
    wavelengths, transmissions = wavelengths[idx], transmissions[:, idx]

    transmissions = {"wavelengths": wavelengths,
                     "transmissions": transmissions}

    filename = path.join(_path_data, "HyperScout", f"HS{_sep_in}H{_sep_out}transmission.npz")

    check_dir(filename)
    with open(filename, "wb") as f:
        np.savez_compressed(f, **transmissions)


if __name__ == "__main__":
    start_line_number, end_line_number, final_names = (2,), (591,), (f"RELAB{_sep_in}spectra",)
    # start_line_number, end_line_number, final_names = (227, 271, 387, 493), (270, 386, 492, 591), ("PLG", "CPX", "OPX", "OL")
    names_relab = collect_data_RELAB(start_line_number, end_line_number, final_names)

    names_ctape = collect_data_CTAPE()
    names_Tomas = resave_Tomas_OL_OPX_mixtures()

    # combine data
    save_names = names_relab + names_ctape + names_Tomas
    final_name = f"mineral{_sep_in}spectra"
    combine_files(tuple(save_names), final_name)

    resave_asteroid_taxonomy_data(["16", "15", "9", "C", "X"])

    resave_Chelyabinsk()
    resave_Itokawa_Eros()
    resave_kachr_ol_opx()

    resave_didymos_2004()
    resave_didymos_2022(add_blue_part=True)
    
    resave_HyperScout_transmission()
