from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from os import path
from urllib.request import urlopen

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from itertools import product
import pywavefront
from sklearn.metrics import pairwise_distances
from typing import Literal

from modules.asteroid_spectra_averaging import dlat, dlon

from modules.utilities_spectra import denoise_and_norm, save_data, combine_files, used_indices, unique_indices
from modules.utilities import flatten_list, stack, my_mv, check_dir, safe_arange, normalise_in_rows

from modules.NN_config import mineral_names, endmember_names, minerals_used, endmembers_used, mineral_names_short
from modules.NN_config_taxonomy import classes
from modules.CD_parameters import lambda_max, lambda_min, resolution_max, resolution_final
from modules.CD_parameters import num_labels_CD, num_minerals_CD, minerals_CD, endmembers_CD
from modules.CD_parameters import denoising_kernel_width, denoise, normalise, normalised_at

from modules._constants import _path_data, _path_catalogues, _path_relab_spectra, _relab_web_page, _num_eps


def collect_data_RELAB(start_line_number: tuple[int, ...] | int, end_line_number: tuple[int, ...] | int | None = None,
                       final_names: tuple[str, ...] = ("PLG", "CPX", "OPX", "OL")) -> list[str]:

    # The input tuples are numbers of line IN Excel file (counting from 1)
    def load_excel_data(start_line_number: tuple[int, ...] | int,
                        end_line_number: tuple[int, ...] | int | None = None) -> tuple[np.ndarray, ...]:
        # This function read the data from the Excel files

        # 2 is the offset; 1 due to header, 1 due to counting rows in Excel from 1, not from 0
        # rows are of the Spectra_Catalogue.xlsx
        rows = np.arange(np.min(start_line_number), np.max(end_line_number) + 1) - 2

        # Real the files (one spectrum per sample)
        Spectra_catalogue = pd.read_excel("".join((_path_catalogues, "Spectra_Catalogue.xlsx")), index_col=None,
                                          na_values=["NA"], usecols="A, B, F:H", engine="openpyxl").to_numpy()[rows]

        Sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                         na_values=["NA"], sheet_name="RELAB",
                                         usecols="A:AG", engine="openpyxl").to_numpy()

        return Spectra_catalogue, Sample_catalogue

    def split_and_filter_data(spectra_catalogue: np.ndarray, sample_catalogue: np.ndarray) -> tuple[np.ndarray, ...]:
        start = np.array(spectra_catalogue[:, 2])
        stop = np.array(spectra_catalogue[:, 3])
        step = np.array(spectra_catalogue[:, 4])

        spectrumIDs = spectra_catalogue[:, 0]
        sampleIDs = spectra_catalogue[:, 1]

        # Take only these spectra
        mask = np.array(np.where((start <= lambda_min) & (stop >= lambda_max) & (step <= resolution_max))).ravel()

        if mask.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # Use only these spectra and samples
        spectrumIDs = spectrumIDs[mask]
        sampleIDs = sampleIDs[mask]

        lines_in_sample_catalogue = flatten_list([np.where(sample_catalogue[:, 0] == sampleID)[0]
                                                  for sampleID in sampleIDs])

        # Find corresponding PIs and filenames
        PIs = sample_catalogue[lines_in_sample_catalogue, 2]
        filenames = np.array(["".join((_path_relab_spectra, PI.lower(), "/", sampleID[:2].lower(), "/",
                                       spectrumID.lower(), ".asc"))
                              for PI, sampleID, spectrumID in zip(PIs, sampleIDs, spectrumIDs)])

        # Find corresponding numbers
        offset = 17  # the values start at this column
        first_index, last_index = offset, offset + num_minerals_CD
        modal_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100.

        first_index, last_index = last_index, last_index + endmembers_CD[0]
        OL_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100.

        first_index, last_index = last_index, last_index + endmembers_CD[1]
        OPX_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100.

        first_index, last_index = last_index, last_index + endmembers_CD[2]
        CPX_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100.

        first_index, last_index = last_index, last_index + endmembers_CD[3]
        PLG_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100.

        # Metadata; +1 because the last one is weathering
        metadata = sample_catalogue[lines_in_sample_catalogue, :-(num_labels_CD + 1)]

        return filenames, modal_numbers, OL_numbers, OPX_numbers, CPX_numbers, PLG_numbers, metadata

    def select_numbers(numbers: np.ndarray) -> np.ndarray:
        # modals
        modal = np.array(numbers[0], dtype=float)
        modal = modal[:, np.where(minerals_CD)[0]]

        # renormalise to unity
        norm = np.array(np.nansum(modal, axis=1), dtype=np.float64)
        mask = norm > _num_eps
        modal[mask] = normalise_in_rows(modal[mask], norm[mask])

        # chemical
        indices = np.where(minerals_CD)[0] + 1  # +1 for modals
        chemical = np.array(stack([numbers[index] for index in indices], axis=1), dtype=float)

        numbers = stack((modal, chemical), axis=1)

        return numbers

    def collect_spectra(xq: np.ndarray, fnames: np.ndarray) -> np.ndarray:
        # This function collects spectra from the database, denoises and normalises them

        N = len(fnames)
        vq = np.zeros((N, len(xq)))

        # list of missing spectra (rows)
        irow = []

        # A loop through spectra
        for i, filename in enumerate(fnames):
            if path.exists(filename):  # load the data from the file
                if filename.endswith(".txt"):
                    # skip the first line and the second line as header
                    data = pd.read_csv(filename, sep="\t", index_col=None, header=[0], skiprows=[0],
                                       engine="python").to_numpy()
                    to_nm = 1000
                elif filename.endswith(".asc"):
                    with open(filename, "r") as f:
                        n_lines = int(f.readline())  # First line contains info about the length of the data
                        data = np.array([np.array(f.readline().split()[:2], dtype=np.float64) for _ in range(n_lines)])
                        to_nm = 1
            else:  # download the data
                try:
                    pos_slash = [i for i in range(len(filename)) if filename.startswith("/", i)]
                    url = "".join((_relab_web_page, filename[pos_slash[-3]:]))
                    print("".join(("Downloading spectrum ", url)))

                    if url.endswith(".txt"):
                        spectrum = urlopen(url).read().decode("utf-8").split("\n")

                        # Remove header and the blank line at the end
                        spectrum = spectrum[2:-1]

                        data = np.array([np.array(line.split("\t")[:2], dtype=np.float64) for line in spectrum])
                        to_nm = 1000.
                    elif filename.endswith(".asc"):
                        spectrum = urlopen(url).read().decode("utf-8").split("\r\n")

                        nlines = int(spectrum[0])
                        spectrum = spectrum[1:nlines + 1]

                        data = np.array([np.array(line.split()[:2], dtype=np.float64) for line in spectrum])
                        to_nm = 1.
                except:
                    print("".join(("Spectrum ", filename, " does not exist and cannot be downloaded.")))
                    irow.append(i)
                    continue

            x = data[:, 0] * to_nm  # to nm
            v = data[:, 1]

            # This have to be done due to some spectra
            x, idx = np.unique(x, return_index=True)
            v = v[idx]

            fun = interp1d(x, v, kind="cubic")

            vq[i, :] = fun(xq)

        # Remove missing data and normalise
        vq = np.delete(vq, irow, axis=0)
        vq = denoise_and_norm(data=vq, wavelength=xq, denoising=denoise, normalising=normalise,
                              normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

        return vq

    if end_line_number is None:
        # skip header
        start_line_number, end_line_number = 2, start_line_number

    print("Collecting data from RELAB...")
    subfolder = "RELAB/"

    # read the data
    spectra_catalogue, sample_catalogue = load_excel_data(start_line_number, end_line_number)

    # label and metadata keys
    labels_key = sample_catalogue[1, -num_labels_CD - 1:-1]  # -1 for weathering which is included in the catalogue
    metadata_key = sample_catalogue[1, :-num_labels_CD - 1]  # -1 for weathering which is included in the catalogue

    # The new axis
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)
    filenames = [""] * len(final_names)

    for i, (start, stop, final_name) in enumerate(zip(start_line_number, end_line_number, final_names)):
        # start_line_number[0] is offset
        rows = np.arange(start, stop + 1) - start_line_number[0]

        # split the data and filter them
        fnames, *numbers, metadata = split_and_filter_data(spectra_catalogue[rows, :], sample_catalogue)

        if np.size(fnames) == 0:
            continue

        # select the numbers according to config file
        numbers = select_numbers(numbers)

        # collecting and normalising the spectra
        spectra = collect_spectra(xq, fnames)

        # Save the interpolated spectra
        filenames[i] = save_data(final_name, spectra=spectra, wavelengths=xq, labels=numbers, metadata=metadata,
                                 labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)

    return filenames


def collect_data_CTAPE() -> list[str]:

    print("Collecting data from C-Tape...")
    subfolder = "C-Tape/"

    # number of files
    N_files = 7, 1, 1, 4  # OL_OPX_num, OL_OPX_CPX_num, OL_CPX_num, OPX_CPX_num
    names = ("OL_OPX", "OL_OPX_CPX", "OL_CPX", "OPX_CPX")
    N = len(names)

    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    final_mat = np.zeros((45 + 6 + 4 + 22, len(xq)))
    start, stop = 0, 0

    for i, name in enumerate(names):
        for j in range(N_files[i]):
            data_file = "".join((_path_data, subfolder, name, "_", str(j), ".dat"))
            data = np.transpose(np.loadtxt(data_file, delimiter="\t"))
            x = data[0]
            v = data[1:]

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

            # can be done before deleting the wrong wl, because the interpolation is just slicing here
            fun = interp1d(x, v, kind="cubic")
            vq = fun(xq)

            # wrong wl
            inds = np.array([505, 510, 520, 525, 535, 540, 545, 550, 555, 560, 565, 615, 625, 630, 635, 645, 650, 655,
                             675, 680, 685, 1385, 1390, 1395, 1400, 1405, 1410, 1415, 2285, 2290, 2305, 2310, 2315,
                             2320, 2325, 2390, 1485, 2280,
                             *safe_arange(1365, 1415, 5, endpoint=True, dtype=int),
                             *safe_arange(2250, 2415, 5, endpoint=True, dtype=int)])
            inds = np.where(np.in1d(xq, inds))[0]

            xq_clean = np.delete(xq, inds)
            vq_clean = np.delete(vq, inds, axis=1)

            fun = interp1d(xq_clean, vq_clean, kind="cubic")
            vq_clean = fun(xq)
            vq_c = denoise_and_norm(data=vq_clean, wavelength=xq, denoising=denoise, normalising=normalise,
                                    normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

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

    sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A:AF", header=1,
                                     sheet_name="C-Tape", engine="openpyxl").to_numpy()

    labels = sample_catalogue[1:, -num_labels_CD:] / 100.
    metadata = sample_catalogue[1:, :-num_labels_CD]

    labels_key = sample_catalogue[0, -num_labels_CD:]
    metadata_key = sample_catalogue[0, :-num_labels_CD]

    filename = save_data("CTape_spectra", spectra=final_mat, wavelengths=xq, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)

    return [filename]


def resave_Tomas_OL_OPX_mixtures() -> list[str]:

    print("Collecting Tomas' OL-OPX mixtures...")

    subfoder = "ol_opx_mix/"
    dirin = "".join((_path_data, subfoder))
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # prekryv x1[130:] a x2[:6]
    x1 = safe_arange(400, 1100, 5, endpoint=True)
    x2 = safe_arange(1050, 2500, 10, endpoint=True)

    with open("".join((dirin, "02_0px.dat"))) as f:
        f.readline()
        data1 = np.array([f.readline() for _ in range(len(x1))], dtype=float)

    with open("".join((dirin, "03_0px.dat"))) as f:
        f.readline()
        data2 = np.array([f.readline() for _ in range(len(x2))], dtype=float)

    # plt.plot(x1, data1)
    # plt.plot(x2, data2)

    # delete the outlier
    x2_new = stack((x2[:54], x2[55:]))
    data2 = stack((data2[:54], data2[55:]))

    # 400:5:1080 + 1080:10:2500
    data = np.zeros(136 + 142)
    data11, data12 = data1[:91], data1[91:]
    data[:91] = data11 * 0.995
    data[91:136] = data12[:-5]
    data[136:] = data2[3:] * 0.99

    x = np.zeros(136 + 142)
    x[:136] = x1[:136]
    x[136:] = x2_new[3:]

    # plt.plot(x, data)

    fun = interp1d(x, data, kind="cubic")
    OL = fun(xq)
    # plt.plot(xq, OL)

    with open("".join((dirin + "02_100px.dat"))) as f:
        f.readline()
        data1 = np.array([f.readline() for _ in range(len(x1))], dtype=float)

    with open("".join((dirin + "03_100px.dat"))) as f:
        f.readline()
        data2 = np.array([f.readline() for _ in range(len(x2))], dtype=float)

    # plt.plot(x1, data1)
    # plt.plot(x2, data2*0.993)

    data11, data12 = data1[:91], data1[91:]

    # 400:5:1080 + 1080:10:2500
    data = np.zeros(136 + 143)
    data[:91] = data11 * 0.965
    data[91:136] = data12[:-5]
    data[136:] = data2[3:] * 0.993

    x = np.zeros(136 + 143)
    x[:136] = x1[:136]
    x[136:] = x2[3:]

    # plt.plot(x, data)

    fun = interp1d(x, data, kind="cubic")
    OPX = fun(xq)
    # plt.plot(xq, OPX)

    C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) * 100.
    C = np.array(C, dtype=int)

    C = np.array(["".join(("OL ", str(100 - s), ", OPX ", str(s))) for s in C])

    px10 = np.loadtxt("".join((dirin, "10px.dat")))
    px25 = np.loadtxt("".join((dirin, "25px.dat")))
    px50 = np.loadtxt("".join((dirin, "50px.dat")))
    px75 = np.loadtxt("".join((dirin, "75px.dat")))
    px90 = np.loadtxt("".join((dirin, "90px.dat")))

    spectra_msm = np.zeros((len(xq), len(C)))
    spectra_msm[:, 0] = OL
    spectra_msm[:, 6] = OPX

    # jumps at 850 [:91] and 1170 [147:] and remove outliers
    px10[:91, 1] *= 0.98
    px10[147:, 1] *= 0.995
    px10 = np.delete(px10, 260, axis=0)
    spectra_msm[:, 1] = interp1d(px10[:, 0], px10[:, 1], kind="cubic")(xq)

    px25[:91, 1] *= 0.985
    px25[147:, 1] *= 0.995
    px25 = np.delete(px25, 146, axis=0)
    spectra_msm[:, 2] = interp1d(px25[:, 0], px25[:, 1], kind="cubic")(xq)

    px50[:91, 1] *= 0.995
    px50[147:, 1] *= 0.997
    px50 = np.delete(px50, 35, axis=0)
    spectra_msm[:, 3] = interp1d(px50[:, 0], px50[:, 1], kind="cubic")(xq)

    px75[:91, 1] *= 0.98
    px75[147:, 1] *= 0.997
    spectra_msm[:, 4] = interp1d(px75[:, 0], px75[:, 1], kind="cubic")(xq)

    px90[:91, 1] *= 0.988
    px90[147:, 1] *= 0.9985
    spectra_msm[:, 5] = interp1d(px90[:, 0], px90[:, 1], kind="cubic")(xq)

    spectra = denoise_and_norm(data=np.transpose(spectra_msm), wavelength=xq, denoising=denoise,
                               normalising=normalise, normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

    sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A:AF", header=1,
                                     sheet_name="TK", engine="openpyxl").to_numpy()

    labels = sample_catalogue[1:, -num_labels_CD:] / 100.
    metadata = sample_catalogue[1:, :-num_labels_CD]

    labels_key = sample_catalogue[0, -num_labels_CD:]
    metadata_key = sample_catalogue[0, :-num_labels_CD]

    filename = save_data("OL_OPX", spectra=spectra, wavelengths=xq, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder=subfoder)

    return [filename]


def resave_Chelyabinsk() -> list[str]:

    print("Re-saving Chelyabinsk data...")

    subfolder = "chelyabinsk/"
    dirin = "".join((_path_data, subfolder))

    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # SD spektra
    SD_names = ["SD0.txt", "SD5.txt", "SD10.txt", "SD20.txt", "SD30.txt", "SD40.txt", "SD50.txt", "SD60.txt",
                "SD70.txt", "SD80.txt", "SD90.txt", "SD95.txt", "SD100.txt"]
    SD_spectra = np.zeros((len(SD_names), len(xq)))

    for i, SD_name in enumerate(SD_names):
        tmp = np.loadtxt("".join((dirin, SD_name)))
        x, y = tmp[:, 0], tmp[:, 1]

        fun = interp1d(x, y, kind="cubic")
        tmp_spec = fun(xq)
        SD_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                            normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

    # IM spektra
    IM_names = ["100LL.txt", "10IM.txt", "20IM.txt", "30IM.txt", "40IM.txt", "50IM.txt", "60IM.txt", "70IM.txt",
                "80IM.txt", "90IM.txt", "95IM.txt", "100IM.txt"]
    IM_spectra = np.zeros((len(IM_names), len(xq)))

    for i, IM_name in enumerate(IM_names):
        tmp = np.loadtxt("".join((dirin, IM_name)))
        x, y = tmp[:, 0] * 1000., tmp[:, 1]

        fun = interp1d(x, y, kind="cubic")
        tmp_spec = fun(xq)

        IM_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                            normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

    # SW spektra
    SW_names = ["SW0.txt", "SW400.txt", "SW500.txt", "SW600.txt", "SW700.txt"]
    SW_spectra = np.zeros((len(SW_names), len(xq)))

    for i, SW_name in enumerate(SW_names):
        tmp = np.loadtxt("".join((dirin, SW_name)))
        x, y = tmp[:, 0], tmp[:, 1]

        fun = interp1d(x, y, kind="cubic")
        tmp_spec = fun(xq)

        SW_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                            normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

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

    sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A:AF", header=1,
                                     sheet_name="TK", engine="openpyxl").to_numpy()

    labels_key = sample_catalogue[0, -num_labels_CD:]

    filename = save_data("Chelyabinsk", spectra=combined_spectra, wavelengths=xq, labels=labels,
                         metadata=metadata, metadata_key=metadata_key, labels_key=labels_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    return [filename]


def resave_asteroid_taxonomy_data() -> None:
    # this function interpolate B-D + MINTHEOS data to the given grid
    print("Re-samving asteroid data...")

    subfolder = "taxonomy/"
    dirin = "".join((_path_data, subfolder))

    # final name
    final_name = "asteroid_spectra"

    # new grid
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)
    # old grid
    x_old = safe_arange(450, 2450, 10, endpoint=True)

    # load the data
    raw_data = "asteroid_pca-spectra-combined.dat"
    data = pd.read_csv("".join((dirin, raw_data)), sep="\t", header=None).to_numpy()

    # asteroid number, taxonomy, slope, PC1--PC5, spectra
    metadata, spectra = data[:, :8], np.array(data[:, 8:], dtype=float)

    # labels data (B-DM class)
    labels = metadata[:, [1]]

    # interpolate the spectra
    fun = interp1d(x_old, spectra, kind="cubic")
    spectra = fun(xq)

    labels_key = np.array(["taxonomy class"])
    metadata_key = np.array(["asteroid number", "taxonomy class", "slope", "PC1", "PC2", "PC3", "PC4", "PC5"])

    spectra = denoise_and_norm(data=spectra, wavelength=xq, denoising=denoise, normalising=normalise,
                               normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

    filename = save_data(final_name, spectra=spectra, wavelengths=xq, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    # DELETED CLASSES #
    classes_to_delete = ["O", "Qw", "Sq", "Sq:", "Sv", "Svw", "Xn", "U"]
    inds_deleted = np.array([ind for ind, taxonomy_class in enumerate(labels.ravel())
                             if taxonomy_class in classes_to_delete])

    filename = save_data("".join((final_name, "-deleted")), spectra=spectra[inds_deleted], wavelengths=xq,
                         labels=labels[inds_deleted], metadata=metadata[inds_deleted], labels_key=labels_key,
                         metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    # REDUCED TAXONOMY #
    # !!! METADATA CONTAINS ORIGINAL TAXONOMY CLASSES !!!
    # combine some classes
    classes = [("Sqw", "S"), ("Sw", "S"), ("Srw", "Sr"), ("R", "Sr"), ("Vw", "V"), ("Sa", "A"),
               ("Cg", "Cgh"), ("Cb", "C"), ("Xc", "X")]
    for old, new in classes:
        inds = np.array([old == ast_type for ast_type in labels.ravel()])
        labels[inds] = new

    # delete the classes
    labels = np.delete(labels, inds_deleted, axis=0)
    spectra = np.delete(spectra, inds_deleted, axis=0)
    metadata = np.delete(metadata, inds_deleted, axis=0)

    filename = save_data("".join((final_name, "-reduced")), spectra=spectra, wavelengths=xq, labels=labels,
                         metadata=metadata, labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    # REDUCED TAXONOMY WITH METEORITES#
    # add OC from RELAB to Q type
    filename = "".join((_path_data, "combined-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = np.array(data["metadata"][:, [6]], dtype=str)
    inds = np.where(np.array(["Ordinary Chondrite" == ast_type for ast_type in types]))[0]
    types = types[inds]

    types[:] = "Q"
    OC = data["spectra"][inds]
    x_oc = data["wavelengths"]
    fun = interp1d(x_oc, OC, kind="cubic")
    OC = fun(xq)

    OC_meta = np.array([np.shape(metadata)[1] * [np.nan]], dtype=object)
    OC_meta[0, 1] = "Q"
    OC_meta = np.repeat(OC_meta, len(OC), axis=0)

    labels = stack((labels, types), axis=0)
    spectra = stack((spectra, OC), axis=0)
    metadata = stack((metadata, OC_meta), axis=0)

    filename = save_data("".join((final_name, "-reduced_met")), spectra=spectra, wavelengths=xq, labels=labels,
                         metadata=metadata, labels_key=labels_key, metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")


def resave_Itokawa_Eros() -> None:
    print("Re-saving Itokawa and Eros data...")

    metadata_key = np.array(["longitude", "latitude", "asteroid name", "instrument"])

    # load the data
    norm_at = 1500.
    filename = "Itokawa_averaged.h5"  # Itokawa
    subfolder = "asteroids/Itokawa/"

    with h5py.File("".join((_path_data, subfolder, filename)), "r") as f:
        data, coordinates, wavelengths = np.array(f["spectra"]), np.array(f["coordinates"]), np.array(f["wavelengths"])

    metadata = np.array([["Itokawa", "Hayabusa Near Infrared Spectrometer"]])
    metadata = np.repeat(metadata, len(data), axis=0)
    metadata = stack((np.array(coordinates, dtype=object), metadata), axis=1)

    data = denoise_and_norm(data=data, wavelength=wavelengths, denoising=denoise,
                            normalising=normalise, normalised_at_wvl=norm_at, sigma_nm=denoising_kernel_width)

    filename = save_data("Itokawa", spectra=data, wavelengths=wavelengths, metadata=metadata,
                         metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    # load the data
    norm_at = 1300.
    filename = "Eros_averaged.h5"  # Eros
    subfolder = "asteroids/Eros/"

    with h5py.File("".join((_path_data, subfolder, filename)), "r") as f:
        data, coordinates, wavelengths = np.array(f["spectra"]), np.array(f["coordinates"]), np.array(f["wavelengths"])

    metadata = np.array([["Eros", "NEAR Shoemaker Near-Infrared Spectrometer"]])
    metadata = np.repeat(metadata, len(data), axis=0)
    metadata = stack((np.array(coordinates, dtype=object), metadata), axis=1)

    # keep only data from 820 to 2360 nm
    mask = wavelengths <= 2360
    wavelengths = wavelengths[mask]
    data = data[:, mask]

    data = denoise_and_norm(data=data, wavelength=wavelengths, denoising=denoise,
                            normalising=normalise, normalised_at_wvl=norm_at, sigma_nm=denoising_kernel_width)

    # NIS reflectances (damaged at 1500+)
    save_data("Eros_NIS", spectra=data, wavelengths=wavelengths, metadata=metadata,
              metadata_key=metadata_key, subfolder=subfolder)

    # keep only data from 820 to 1480 nm
    mask = wavelengths <= 1480
    wavelengths_1480 = wavelengths[mask]
    data_1480 = data[:, mask]

    save_data("Eros_1480", spectra=data_1480, wavelengths=wavelengths_1480, metadata=metadata,
              metadata_key=metadata_key, subfolder=subfolder)

    # combine NIS and telescope data
    ast = np.load("".join((_path_data, "asteroid_spectra-denoised-norm.npz")), allow_pickle=True)
    eros = "433" == ast["metadata"][:, 0]
    spectrum, wvl = ast["spectra"][eros], ast["wavelengths"]

    fun = interp1d(wvl, spectrum, kind="cubic")
    spectrum = fun(wavelengths) / fun(norm_at)

    data_1, data_2 = data[:, mask], data[:, ~mask]

    # correction = spectrum[:, ~mask] - np.mean(data_2, axis=0)
    # data_2_corr = data_2 + correction

    correction = spectrum[:, ~mask] / np.mean(data_2, axis=0)
    align = np.mean(data[:, np.where(mask)[0][-1]], axis=0) / spectrum[:, np.where(mask)[0][-1]]
    data_2_corr = data_2 * correction * align

    data_corr = stack((data_1, data_2_corr), axis=1)

    filename = save_data("Eros", spectra=data_corr, wavelengths=wavelengths, metadata=metadata,
                         metadata_key=metadata_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")


def resave_kachr_ol_opx() -> list[str]:
    print("Re-saving Katka's data...")

    subfolder = "kachr_ol_opx/"
    dirin = "".join((_path_data, subfolder))

    metadata_key = np.array(["weathering type", "area density of damage"])

    x_new_part = safe_arange(545, 2415, 5, endpoint=True)
    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

    # file names and suffixes
    names = ["ol", "py"]
    suffixes = ["Ar", "H", "He", "laser"]
    first = True  # allocate some space during first iteration

    for name, suffix in product(names, suffixes):
        tmp = np.loadtxt("".join((dirin, name, "-", suffix, ".csv")))
        x, y = tmp[:, 0], tmp[:, 1:]

        fun = interp1d(x, np.transpose(y), kind="cubic")  # do all at once
        tmp_spec = fun(x_new_part)  # one spectrum per row

        fun = interp1d(x_new_part, tmp_spec, kind="linear", fill_value="extrapolate")  # do all at once
        tmp_spec = fun(xq)  # one spectrum per row

        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xq, np.transpose(tmp_spec))
        """

        if first:
            spectra = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                       normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

            first = False
        else:
            another_spectra = denoise_and_norm(data=tmp_spec, wavelength=xq, denoising=denoise, normalising=normalise,
                                               normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)
            spectra = stack((spectra, another_spectra), axis=0)

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

    labels_ol = np.array([[1, 0, 0, 0, 9.9, 90.1, 0, 0, 0.0, 0, 0, 0, 0, 0, 0]])
    labels_ol = np.repeat(labels_ol, 34, axis=0)
    labels_px = np.array([[0, 84/(84+5) * 100, 5/(84+5) * 100, 0, 0, 0, 32.9, 67.1, 0, 0, 0, 0, 0, 0, 0]])
    labels_px = np.repeat(labels_px, 32, axis=0)

    labels = stack((labels_ol, labels_px), axis=0)

    sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A:AF", header=1,
                                     sheet_name="TK", engine="openpyxl").to_numpy()

    labels_key = sample_catalogue[0, -num_labels_CD:]
    labels_key[1] = labels_key[1].replace("vol", "wt")
    labels_key[2] = labels_key[2].replace("vol", "wt")

    filename = save_data("Kachr_ol_opx", spectra=spectra, wavelengths=xq, metadata=metadata, labels=labels,
                         metadata_key=metadata_key, labels_key=labels_key, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    return [filename]


def resave_didymos_2004() -> list[str]:

    print("Re-saving Didymos' data...")

    subfolder = "didymos/"
    dirin = "".join((_path_data, subfolder))

    xq = safe_arange(490, 2450, 5, endpoint=True)

    file = "Didymos_vnir_albedo.dat"
    tmp = pd.read_csv("".join((dirin, file)), sep="\t", index_col=None, header=[0], engine="python").to_numpy()
    x, y = tmp[:, 0], tmp[:, 1:]

    fun = interp1d(x, np.transpose(y), kind="cubic")  # do all at once
    spectrum = fun(xq)  # one spectrum per row
    spectrum = denoise_and_norm(data=spectrum, wavelength=xq, denoising=denoise, normalising=normalise,
                                normalised_at_wvl=normalised_at, sigma_nm=30)

    metadata = np.array([["Didymos spectrum", "NOT_TNG_20040116", "10.1051/0004-6361/200913852"]])

    filename = save_data("Didymos_2004", spectra=spectrum, wavelengths=xq, metadata=metadata, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    return [filename]


def resave_didymos_2022(add_blue_part: bool = False) -> list[str]:

    print("Re-saving Didymos' data...")

    files = ["a65803_IR_Spec_IRTF_20220926_Polishook.dat", "a65803_IR_Spec_IRTF_20220927_Polishook.dat"]

    # xq = safe_arange(650, 2450, 5, endpoint=True)
    xq = safe_arange(800, 2450, 5, endpoint=True)

    if add_blue_part:  # the original spectrum starts at 650. I added mean S type here
        asteroid_data = np.load("".join((_path_data, "asteroid_spectra-denoised-norm.npz")),
                                allow_pickle=True)
        S_types = asteroid_data["metadata"][:, 1] == "S"
        S_part = asteroid_data["wavelengths"] < xq[0]
        S_spectra_mean = np.mean(asteroid_data["spectra"][S_types], axis=0)[S_part]
        S_wavelengths = asteroid_data["wavelengths"][S_part]

        x_old_full = stack((S_wavelengths, xq))

        xq_full = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

        final_spectra = np.zeros((len(files), len(xq_full)))
    else:
        final_spectra = np.zeros((len(files), len(xq)))

    subfolder = "didymos/"
    dirin = "".join((_path_data, subfolder))

    for i, file in enumerate(files):
        tmp = np.loadtxt("".join((dirin, file)))[:, :2]
        x, y = tmp[:, 0] * 1000, tmp[:, 1]

        if i == 0:
            wvl_to_delete = np.array([750, 765, 1875, 1895, 1900, 2430, 2440
                                      *safe_arange(1340, 1375, 5, endpoint=True),
                                      *safe_arange(1385, 1395, 5, endpoint=True),
                                      *safe_arange(1990, 2030, 5, endpoint=True)])
            inds = np.array([np.where(wvl == np.round(x))[0] for wvl in wvl_to_delete]).ravel()

            x = np.delete(x, inds)
            y = np.delete(y, inds)

        fun = interp1d(x, np.transpose(y), kind="linear")  # do all at once
        spectrum = fun(xq)  # one spectrum per row

        spectrum = denoise_and_norm(data=spectrum, wavelength=xq, denoising=denoise, normalising=normalise,
                                    normalised_at_wvl=1500, sigma_nm=20)

        if add_blue_part:
            spectrum = stack((S_spectra_mean * spectrum[0, 0] / S_spectra_mean[-1], spectrum[0]))
            spectrum = interp1d(x_old_full, spectrum, kind="cubic")(xq_full)

            spectrum = denoise_and_norm(data=spectrum, wavelength=xq, denoising=False, normalising=normalise,
                                        normalised_at_wvl=normalised_at, sigma_nm=denoising_kernel_width)

            # to decrease a slope of the mean S type
            spectrum *= 1. + (xq[0] - xq_full + np.abs(xq[0] - xq_full)) / 4000.
            spectrum = denoise_and_norm(data=spectrum, wavelength=xq_full, denoising=True,
                                        normalising=normalise, normalised_at_wvl=normalised_at, sigma_nm=5)

        final_spectra[i] = spectrum

    if add_blue_part:
        xq = xq_full

    metadata = np.array([["Didymos spectrum", "IR_Spec_IRTF_20220926"],
                         ["Didymos spectrum", "IR_Spec_IRTF_20220927"]])

    filename = save_data("Didymos_2022", spectra=final_spectra, wavelengths=xq,
                         metadata=metadata, subfolder=subfolder)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    return [filename]


def resave_data_for_shapeViewer(y_pred: np.ndarray, asteroid_name: str,
                                taxonomy_or_composition: Literal["composition", "taxonomy"],
                                indices_to_save: list[int] | None = None) -> None:

    # bg white (white background)
    # L (turn off legend)

    if indices_to_save is None:
        indices_to_save = range(np.shape(y_pred)[1])

    num_digits = 5

    if taxonomy_or_composition == "composition":
        quantity_names = flatten_list([mineral_names] + endmember_names)[used_indices(minerals_used,
                                                                                            endmembers_used)]
    elif taxonomy_or_composition == "taxonomy":
        quantity_names = list(classes.keys())
    else:
        raise ValueError('"taxonomy_or_composition" must be either "taxonomy" or "composition"')

    """
    if asteroid_name == "Itokawa":  # save position of craters if Itokawa
        scene = pywavefront.Wavefront("".join((_path_data, "shapeViewer/", asteroid_name, ".obj")),
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

        # delete some craters which are not visible
        inds_to_delete = []  # ID - 1
        results = np.delete(results, inds_to_delete, axis=0)

        filename = "".join((_path_data, "/shapeViewer/", asteroid_name, "_craters.dat"))
        check_dir(filename)
        np.savetxt(filename, results, fmt="%s", delimiter=" ")
    """

    filename_data = "".join((asteroid_name, "-denoised-norm.npz"))

    data = np.load("".join((_path_data, filename_data)), allow_pickle=True)
    lon, lat = np.array(data["metadata"][:, [0]], dtype=float), np.array(data["metadata"][:, [1]], dtype=float)

    # this is to match shapeViewer notation
    data_type = np.repeat([["g"]], len(lon), axis=0)

    # shapeViewer use angular radius
    # pi r^2 = area -> r = sqrt(area / pi), but "/ pi" will do gaps in figures
    angle = np.full(np.shape(lon), np.sqrt(dlon * dlat))

    lat, lon, angle = np.round(lat, 1), np.round(lon, 1), np.round(angle, 2)

    for index_to_save in indices_to_save:
        quantity_name = quantity_names[index_to_save].replace(" ", "_")

        reverse_caxis = quantity_name in ["Q", "olivine"]

        prediction_to_save = np.round(y_pred[:, [index_to_save]] * 100., num_digits)

        if reverse_caxis:
            prediction_to_save = 100. - prediction_to_save

        prediction_to_save = np.array(["{:.{prec}f}".format(p, prec=num_digits) for p in prediction_to_save.ravel()])
        prediction_to_save = np.reshape(prediction_to_save, (len(prediction_to_save), 1))

        results = stack((data_type, lat, lon, angle, prediction_to_save), axis=1)

        if reverse_caxis:
            filename = "".join((_path_data, "/shapeViewer/min_max_cbar/",
                                asteroid_name, "_", quantity_name, "_reversed.dat"))
        else:
            filename = "".join((_path_data, "/shapeViewer/min_max_cbar/", asteroid_name, "_", quantity_name, ".dat"))

        check_dir(filename)
        np.savetxt(filename, results, fmt="%s", delimiter=" ")

        # to make the shapeViewer colorbar be from 0 to 100
        # these points are not visible
        tmp = np.array([["g", "-89.9", "0.0", "0.1", "0.0"],
                       ["g", "-89.9", "1.0", "0.1", "100.0"]])
        results = stack((tmp, results), axis=0)

        if reverse_caxis:
            filename = "".join((_path_data, "/shapeViewer/0-100_cbar/",
                                asteroid_name, "_", quantity_name, "_reversed.dat"))
        else:
            filename = "".join((_path_data, "/shapeViewer/0-100_cbar/", asteroid_name, "_", quantity_name, ".dat"))


        check_dir(filename)
        np.savetxt(filename, results, fmt="%s", delimiter=" ")


def resave_Itokawa_Eros_results():
    from modules.NN_evaluate import evaluate
    from modules.utilities_spectra import collect_all_models

    labels = [[" ".join((name, "(vol%)")) for name in mineral_names_short]] + endmember_names
    labels = flatten_list(labels)[used_indices(minerals_used, endmembers_used)]

    header_coords, header_tax, header_comp = ["lon (deg)", "lat (deg)"], list(classes.keys()), labels
    header = stack((header_coords, header_tax, header_comp))

    nC, nT = len(header_coords), len(header_coords) + len(header_tax)

    keep_coords = [0, 1]
    keep_tax_E = np.array([classes["S"], classes["Q"], classes["L"]]) + nC
    keep_tax_I = np.array([classes["S"], classes["Q"]]) + nC
    keep_comp = unique_indices(minerals_used, endmembers_used, all_minerals=False, return_digits=True) + nT

    keep_inds_I = stack((keep_coords, keep_tax_I, keep_comp))
    keep_inds_E = stack((keep_coords, keep_tax_E, keep_comp))

    filenames = ["Itokawa", "Eros"]

    for filename in filenames:
        data = np.load("".join((_path_data, filename, "-denoised-norm.npz")), allow_pickle=True)

        lon, lat = np.array(data["metadata"][:, 0], dtype=float), np.array(data["metadata"][:, 1], dtype=float)

        results = stack((lon, lat), axis=1)

        for model_type in ["taxonomy", "composition"]:

            model_names = collect_all_models(suffix=filename, subfolder_model=model_type, full_path=False)
            results = stack((results, 100. * evaluate(model_names, data["spectra"], model_type)), axis=1)

        results = np.array(np.round(results, 1), dtype=str)

        results = stack((header, results), axis=0)

        if filename == "Itokawa":
            results = results[:, keep_inds_I]
        else:
            results = results[:, keep_inds_E]

        np.savetxt("".join((_path_data, filename, "-results.dat")), results, fmt='%s', delimiter='\t')


if __name__ == "__main__":
    start_line_number, end_line_number, final_names = (2,), (591,), ("relab",)
    # start_line_number, end_line_number, final_names = (227, 271, 387, 493), (270, 386, 492, 591), ("PLG", "CPX", "OPX", "OL")
    names_relab = collect_data_RELAB(start_line_number, end_line_number, final_names)

    names_ctape = collect_data_CTAPE()
    names_Tomas = resave_Tomas_OL_OPX_mixtures()

    # combine data
    save_names = names_relab + names_ctape + names_Tomas
    final_name = "combined"
    combine_files(tuple(save_names), final_name)

    resave_asteroid_taxonomy_data()

    resave_Chelyabinsk()
    resave_Itokawa_Eros()
    resave_kachr_ol_opx()

    resave_didymos_2004()
    resave_didymos_2022()
