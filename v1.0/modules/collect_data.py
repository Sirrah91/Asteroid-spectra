import pandas as pd
import numpy as np
import random
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import ConvexHull
from urllib.request import urlopen
from os import path
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from pathlib import Path
import matplotlib as mpl
import itertools
import os
from modules.mixing_models import *

mpl.use('TkAgg')

from modules.NN_data import load_data, clean_data
from modules.utilities import *
from modules.CD_parameters import *
from modules.NN_config import num_labels_all


def combine_files(filenames: Tuple[str, ...], final_name: str) -> str:
    outfile_name = "".join((project_dir, '/Datasets/', final_name))
    tmp = Path(outfile_name)
    if tmp.suffix == '':
        outfile_name += '.dat'
    with open(outfile_name, 'w') as outfile:
        for fname in filenames:
            tmp = Path(fname)
            if tmp.suffix == '':
                fname += '.dat'
            if Path(fname).parent.__str__() == '.':  # no directory; relab is default
                with open("".join((project_dir, '/Datasets/RELAB/', fname)), 'r') as infile:
                    for line in infile:
                        outfile.write(line)

            else:
                with open(fname, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
    return outfile_name


def clean_and_resave(filename: str, meta: bool = False) -> None:
    # meta = True for resaving metadata
    tmp = Path(filename)
    if not meta:
        final_name = tmp.stem + '-clean'  # '.dat' is added in save_data
    else:
        final_name = tmp.stem + '-clean-meta'

    if not meta:
        # open data file and clean it
        spectra, labels = load_data(filename, clean_it=True, keep_all_labels=True)
        # resave it (it is saved to relab folder...)
        file = save_data(final_name, spectra=spectra, labels=labels)
    else:
        file = project_dir + '/Datasets/' + final_name + '.dat'
        metadata = pd.read_csv(project_dir + '/Datasets/' + tmp.stem + '-meta.dat', sep='\t', header=None).to_numpy()

        data_file = "".join((project_dir, '/Datasets/', filename))
        data = np.loadtxt(data_file, delimiter='\t')

        # Select training data
        x_train, y_train = deepcopy(data[:, :-num_labels_all].astype(np.float32)), deepcopy(
            data[:, -num_labels_all:].astype(np.float32))

        *_, inds = clean_data(x_train, y_train, return_indices=True)

        # filter metadata
        metadata = metadata[inds]

        with open(file, 'w') as f:
            for element in metadata:
                final_string = '\t'.join([str(e) for e in element]) + '\n'

                for e in final_string:
                    f.write(e)

    # move it to Datasets folder
    os.rename(file, project_dir + '/Datasets/' + final_name + '.dat')


def load_excel_data(start_line_number: Union[Tuple[int, ...], int],
                    end_line_number: Union[Tuple[int, ...], int] = None) -> Tuple[np.ndarray, ...]:
    # This function read the data from the excel files

    # 2 is the offset; 1 due to header, 1 due to counting rows in excel from 1, not from 0
    rows = np.arange(np.min(start_line_number), np.max(end_line_number) + 1) - 2

    # Real the files (one spectrum per sample)
    Spectra_catalogue = pd.read_excel("".join((path_relab, 'Spectra_Catalogue.xlsx')), index_col=None, na_values=['NA'],
                                      usecols="A, B, F:H", engine='openpyxl').to_numpy()[rows]

    Sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None, na_values=['NA'],
                                     usecols="A:AG", engine='openpyxl').to_numpy()

    return Spectra_catalogue, Sample_catalogue


def split_and_filter_data(spectra_catalogue: np.ndarray, sample_catalogue: np.ndarray) -> Tuple[np.ndarray, ...]:
    start = np.array(spectra_catalogue[:, 2])
    stop = np.array(spectra_catalogue[:, 3])
    step = np.array(spectra_catalogue[:, 4])

    spectrumIDs = spectra_catalogue[:, 0]
    sampleIDs = spectra_catalogue[:, 1]

    # Take only these spectra
    if not mixtures:
        mask = np.array(np.where((start <= lambda_min) & (stop >= lambda_max) & (step <= resolution_max))).ravel()
    else:

        mask = np.array(np.where(
            (np.array([(sample_catalogue[sample_catalogue[:, 0] == sampleID, 3] == 'Synthetic')[0] for sampleID in
                       sampleIDs]) |
             np.array([(sample_catalogue[sample_catalogue[:, 0] == sampleID, 10] <= 50)[0] for sampleID in sampleIDs]) &
             np.array([(sample_catalogue[sample_catalogue[:, 0] == sampleID, 15] > 0)[0] for sampleID in sampleIDs]))
            & (start <= lambda_min) & (stop >= lambda_max) & (step <= resolution_max))).ravel()
    """
    mask = np.array(np.where(
        (np.array(
            [(sample_catalogue[sample_catalogue[:, 0] == sampleID, 6] == 'Ordinary Chondrite')[0] for sampleID in
             sampleIDs])) &
        (start <= lambda_min) & (stop >= lambda_max) & (step <= resolution_max))).ravel()
    """

    if mask.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Use only these spectra and samples
    spectrumIDs = spectrumIDs[mask]
    sampleIDs = sampleIDs[mask]

    lines_in_sample_catalogue = flatten_list([np.where(sample_catalogue[:, 0] == sampleID)[0]
                                              for sampleID in sampleIDs])

    # Find corresponding PIs and filenames
    PIs = sample_catalogue[lines_in_sample_catalogue, 2]
    filenames = np.array(["".join(
        (path_relab, 'data/', PIs[i].lower(), '/', sampleIDs[i][:2].lower(), '/', spectrumIDs[i].lower(), '.asc'))
        for i in range(len(spectrumIDs))])

    # Find corresponding numbers
    offset = 17  # the values start at this column
    first_index, last_index = offset, offset + num_minerals_CD
    modal_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes_CD[0]
    OL_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes_CD[1]
    OPX_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes_CD[2]
    CPX_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes_CD[3]
    PLG_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    # Metadata
    metadata = sample_catalogue[lines_in_sample_catalogue, :-(num_labels_CD + 1)]  # +1 because the last one is weathering

    return filenames, modal_numbers, OL_numbers, OPX_numbers, CPX_numbers, PLG_numbers, metadata


def select_numbers(numbers: np.ndarray) -> np.ndarray:
    # modals
    modal = numbers[0]
    modal = modal[:, np.where(use_minerals_CD)[0]]

    # renormalise to unity
    norm = np.sum(modal, axis=1).astype(np.float64)
    # if an of the values is missing, norm is nan
    # modal = np.transpose(np.divide(np.transpose(modal), norm))
    # no_nan = np.where(1 - np.isnan(norm))

    for i in range(np.shape(modal)[0]):
        if not np.isnan(norm[i]):
            modal[i] = modal[i] / norm[i]

    # chemical
    indices = np.where(use_minerals_CD)[0] + 1  # +1 for modals
    chemical = np.hstack([numbers[index] for index in indices])

    numbers = np.hstack((modal, chemical))

    return numbers


def denoise_and_norm(data: np.ndarray, denoising: bool, normalising: bool) -> np.ndarray:
    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    if denoising:
        width = 9
        cent = int(np.round(width / 2))
        kernel = np.zeros(width)

        for ii in range(int(np.floor(width / 2))):
            kernel[ii] = 1 / (np.abs(ii - cent) + 1)
            kernel[-ii - 1] = 1 / (np.abs(ii - cent) + 1)
        kernel[cent] = 1

        kernel = kernel / np.sum(kernel)
        correction = np.convolve(np.ones(len(xq)), kernel, 'same')

        if data.ndim == 1:
            data = np.reshape(data, (1, len(data)))
        data_denoised = np.apply_along_axis(lambda m: np.convolve(m, kernel, 'same'), axis=1, arr=data) / correction
    else:
        data_denoised = data

    # Normalised reflectance
    if normalising:
        fun = interp1d(xq, data_denoised, kind='cubic')  # v_final differs from v
        v_norm = np.reshape(fun(normalised_at), (len(data_denoised), 1))
    else:
        v_norm = 1

    return data_denoised / v_norm


def collect_spectra(xq: np.ndarray, fnames: np.ndarray) -> np.ndarray:
    # This function collects spectra from the database, denoises and normalises them

    N = len(fnames)
    vq = np.zeros((N, len(xq)))

    # List of missing spectra (rows)
    irow = []

    # A loop through spectra
    for i in range(N):
        filename = fnames[i]

        if path.exists(filename):  # load the data from the file
            if filename.endswith('.txt'):
                # skip the first line and the second line as header
                data = pd.read_csv(filename, sep='\t', index_col=None, header=[0], skiprows=[0],
                                   engine='python').to_numpy()
                to_nm = 1000
            elif filename.endswith('.asc'):
                with open(filename, 'r') as f:
                    n_lines = int(f.readline())  # First line contains info about the length of the data
                    data = np.array([np.array(f.readline().split()[:2], dtype=np.float64) for _ in range(n_lines)])
                    to_nm = 1
        else:  # download the data
            try:
                pos_slash = [i for i in range(len(filename)) if filename.startswith('/', i)]
                url = "".join((web_page, filename[pos_slash[-3]:]))
                print("".join(("Downloading spectrum ", url)))

                if url.endswith('.txt'):
                    spectrum = urlopen(url).read().decode('utf-8').split('\n')

                    # Remove header and the blank line at the end
                    spectrum = spectrum[2:-1]

                    data = np.array([np.array(line.split('\t')[:2], dtype=np.float64) for line in spectrum])
                    to_nm = 1000
                elif filename.endswith('.asc'):
                    spectrum = urlopen(url).read().decode('utf-8').split('\r\n')

                    nlines = int(spectrum[0])
                    spectrum = spectrum[1:nlines + 1]

                    data = np.array([np.array(line.split()[:2], dtype=np.float64) for line in spectrum])
                    to_nm = 1
            except:
                print("".join(("Spectrum ", filename, " does not exist and cannot be downloaded.")))
                irow.append(i)
                continue

        x = data[:, 0] * to_nm  # to nm
        v = data[:, 1]

        # This have to be done due to some spectra
        x, idx = np.unique(x, return_index=True)
        v = v[idx]

        fun = interp1d(x, v, kind='cubic')

        vq[i, :] = fun(xq)

    # Remove missing data and normalise
    vq = np.delete(vq, irow, axis=0)
    vq = denoise_and_norm(vq, denoise, normalise)

    return vq


def save_data(final_name: str, spectra: np.ndarray, labels: np.ndarray = None, meta: np.ndarray = None,
              Tuomas_data: bool = False) -> str:
    check_dir("".join((path_relab, final_name, '.dat')))
    if mixtures:
        final_name += '_for_mixing'
    else:
        if denoise:
            if '-denoised' not in final_name:
                final_name += '-denoised'
        if normalise:
            if '-norm' not in final_name:
                final_name += '-norm'

        if labels is None:
            # Save data without labels (not needed for mixtures)
            if not Tuomas_data:
                filename = "".join((path_relab, final_name, '-nolabel.dat'))
            else:
                filename = "".join((path_tuomas, final_name, '-nolabel.dat'))

            np.savetxt(filename, spectra, fmt='%.5f', delimiter='\t')

    # save metadata if these exist
    # data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    if meta is not None:
        filename = "".join((path_relab, final_name, '-meta.dat'))
        with open(filename, 'w') as f:
            for element in meta:
                final_string = '\t'.join([str(e) for e in element]) + '\n'

                for e in final_string:
                    f.write(e)

    # Save data with labels
    if labels is not None:
        if mixtures:
            filename = "".join((path_MGM, final_name, '.dat'))
        elif not Tuomas_data:
            filename = "".join((path_relab, final_name, '.dat'))
        else:
            filename = "".join((path_tuomas, final_name, '.dat'))

        if not Tuomas_data:
            spectra = np.hstack((spectra, labels))
            np.savetxt(filename, spectra, fmt='%.5f', delimiter='\t')
        else:
            spectra = np.hstack((labels, spectra))  # he stored labels in the first coordinate
            np.savetxt(filename, spectra, fmt='%s', delimiter='\t')

    return filename


def collect_data_RELAB(start_line_number: Union[Tuple[int, ...], int],
                       end_line_number: Union[Tuple[int, ...], int] = None,
                       final_names: Tuple[str, ...] = ('PLG', 'CPX', 'OPX', 'OL')) -> List[str]:
    # The input Tuples are numbers of line IN excel file (counting from 1)

    if end_line_number is None:
        # skip header
        start_line_number, end_line_number = 2, start_line_number

    print("Collecting data from RELAB...")
    # read the data
    spectra_catalogue, sample_catalogue = load_excel_data(start_line_number, end_line_number)

    # The new axis
    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    filenames = [''] * len(final_names)

    for i_modal in range(len(start_line_number)):
        # start_line_number[0] is offset
        rows = np.arange(start_line_number[i_modal], end_line_number[i_modal] + 1) - start_line_number[0]

        # split the data and filter them
        fnames, *numbers, metadata = split_and_filter_data(spectra_catalogue[rows, :], sample_catalogue)

        if fnames.size == 0:
            continue

        # select the numbers according to config file
        numbers = select_numbers(numbers)

        # collecting and normalising the spectra
        spectra = collect_spectra(xq, fnames)

        # Save the interpolate spectra
        filenames[i_modal] = save_data(final_names[i_modal], spectra, numbers, meta=metadata)

    return filenames


def collect_data_CTAPE() -> List[str]:
    # number of files
    N_files = 7, 1, 1, 4  # OL_OPX_num, OL_OPX_CPX_num, OL_CPX_num, OPX_CPX_num
    names = ('OL_OPX', 'OL_OPX_CPX', 'OL_CPX', 'OPX_CPX')
    N = len(names)

    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)

    final_mat = np.zeros((45 + 6 + 4 + 22, len(xq)))
    start, stop = 0, 0

    for i in range(N):
        for j in range(N_files[i]):
            data_file = "".join((path_ctape, '/', names[i], '_', str(j), '.dat'))
            data = np.transpose(np.loadtxt(data_file, delimiter='\t'))
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

            start = stop
            stop += len(v)

            # can be done before deleting the wrong wl, because the interpolation is just slicing here
            fun = interp1d(x, v, kind='cubic')
            vq = fun(xq)

            # wrong wl
            inds = np.array([505, 510, 520, 525, 535, 540, 545, 550, 555, 560, 565, 615, 625, 630, 635, 645, 650,
                             655, 675, 680, 685, 1385, 1390, 1395, 1400, 1405, 1410, 1415, 2285, 2290, 2305, 2310, 2315,
                             2320, 2325, 2390, 1485, 2280, *np.arange(2250, 2420, 5), *np.arange(1365, 1420, 5)])
            inds = np.where(np.in1d(xq, inds))[0]

            xq_clean = np.delete(xq, inds)
            vq_clean = np.delete(vq, inds, axis=1)

            fun = interp1d(xq_clean, vq_clean, kind='cubic')
            vq_clean = fun(xq)
            vq_c = denoise_and_norm(vq_clean, denoise, normalise)

            '''
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(xq, np.transpose(vq))
            ax[1].plot(xq, np.transpose(vq_clean))
            ax[2].plot(xq, np.transpose(vq_c))
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.show()
            '''

            final_mat[start:stop, :] = vq_c

    labels = np.array([[0.9, 0.1, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.8, 0.2, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.7, 0.3, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.6, 0.4, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.4, 0.6, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.3, 0.7, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.2, 0.8, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.1, 0.9, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       #####
                       [0.9, 0.1, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.8, 0.2, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.7, 0.3, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.6, 0.4, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.4, 0.6, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.3, 0.7, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.2, 0.8, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.1, 0.9, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       #####
                       [0.9, 0.1, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.8, 0.2, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.7, 0.3, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.6, 0.4, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.4, 0.6, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.3, 0.7, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.2, 0.8, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.1, 0.9, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       #####
                       [0.1, 0.9, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.8, 0.2, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.7, 0.3, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.6, 0.4, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.4, 0.6, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.3, 0.7, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.2, 0.8, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.9, 0.1, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.05, 0.95, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.15, 0.85, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       #####
                       [0.9, 0.1, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.1, 0.9, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.75, 0.25, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.25, 0.75, 0, 0, 0.10, 0.90, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       #####
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.41, 0.55, 0.04, 0, 0, 0, 0, 0, 0],
                       #####
                       [0.5, 0.5, 0, 0, 0.40, 0.60, 0.41, 0.55, 0.04, 0, 0, 0, 0, 0, 0],
                       ################################
                       [0.75, 0.25, 0, 0, 0.1, 0.9, 0.13, 0.87, 0, 0, 0, 0, 0, 0, 0],
                       [0.675, 0.225, 0.1, 0, 0.1, 0.9, 0.13, 0.87, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.60, 0.20, 0.2, 0, 0.1, 0.9, 0.13, 0.87, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.525, 0.175, 0.3, 0, 0.1, 0.9, 0.13, 0.87, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.45, 0.15, 0.4, 0, 0.1, 0.9, 0.13, 0.87, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.375, 0.125, 0.5, 0, 0.1, 0.9, 0.13, 0.87, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       ################################
                       [0.8, 0, 0.2, 0, 0.10, 0.90, 0, 0, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.6, 0, 0.4, 0, 0.10, 0.90, 0, 0, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.4, 0, 0.6, 0, 0.10, 0.90, 0, 0, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       [0.2, 0, 0.8, 0, 0.10, 0.90, 0, 0, 0, 0.026, 0.485, 0.489, 0, 0, 0],
                       ################################
                       [0, 0.1, 0.9, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.2, 0.8, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.3, 0.7, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.4, 0.6, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.5, 0.5, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.6, 0.4, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.7, 0.3, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.8, 0.2, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.9, 0.1, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0.094, 0.894, 0.012, 0, 0, 0, 0, 0, 0],
                       #####
                       [0, 0.2, 0.8, 0, 0, 0, 0.41, 0.55, 0.04, 0.08, 0.48, 0.44, 0, 0, 0],
                       [0, 0.4, 0.6, 0, 0, 0, 0.41, 0.55, 0.04, 0.08, 0.48, 0.44, 0, 0, 0],
                       [0, 0.6, 0.4, 0, 0, 0, 0.41, 0.55, 0.04, 0.08, 0.48, 0.44, 0, 0, 0],
                       [0, 0.8, 0.2, 0, 0, 0, 0.41, 0.55, 0.04, 0.08, 0.48, 0.44, 0, 0, 0],
                       #####
                       [0, 0.2, 0.8, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.4, 0.4, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.6, 0.6, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       [0, 0.8, 0.2, 0, 0, 0, 0.094, 0.894, 0.012, 0.1, 0.41, 0.49, 0, 0, 0],
                       #####
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0.085, 0.428, 0.487, 0, 0, 0],
                       [0, 0.75, 0.25, 0, 0, 0, 0.41, 0.55, 0.04, 0.085, 0.428, 0.487, 0, 0, 0],
                       [0, 0.50, 0.50, 0, 0, 0, 0.094, 0.894, 0.012, 0.085, 0.428, 0.487, 0, 0, 0],
                       [0, 0.25, 0.75, 0, 0, 0, 0.094, 0.894, 0.012, 0.085, 0.428, 0.487, 0, 0, 0],
                       ])

    # wt% to vol%
    densities = np.array([4.49, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])
    modal, chemical = labels[:, :4], labels[:, 4:]

    modal_densities = chemical * densities
    ol_den = np.sum(modal_densities[:, :2], axis=1, keepdims=True)
    opx_den = np.sum(modal_densities[:, 2:5], axis=1, keepdims=True)
    cpx_den = np.sum(modal_densities[:, 5:8], axis=1, keepdims=True)
    plg_den = np.sum(modal_densities[:, 8:11], axis=1, keepdims=True)

    # if mineral is not present, set density to 1 (to prevent 0/density = 0/0)
    ol_den[ol_den == 0] = 1
    opx_den[opx_den == 0] = 1
    cpx_den[cpx_den == 0] = 1
    plg_den[plg_den == 0] = 1

    # combine densities
    modal_densities = np.concatenate((ol_den, opx_den, cpx_den, plg_den), axis=1)
    modal_vol = modal / modal_densities
    # normalise it to percents
    modal_vol /= np.sum(modal_vol, axis=1, keepdims=True)

    labels[:, :4] = modal_vol

    metadata = np.array(len(final_mat) * ['C-Tape'])

    filename = save_data('CTape_spectra', final_mat, labels, meta=metadata)

    return [filename]


def resave_Tomas_OL_OPX_mixtures() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dirin = '/home/dakorda/Python/NN/Datasets/mix_test/'

    data1 = np.zeros(141)  # 400:5:1100
    data2 = np.zeros(146)  # 1050:10:2500

    with open(dirin + '02_0px.dat') as f:
        f.readline()
        for i in range(141):
            data1[i] = f.readline()

    with open(dirin + '03_0px.dat') as f:
        f.readline()
        for i in range(146):
            data2[i] = f.readline()

    # prekryv x1[130:] a x2[:6]
    x1 = np.arange(400, 1101, 5)
    x2 = np.arange(1050, 2501, 10)

    # plt.plot(x1, data1)
    # plt.plot(x2, data2)

    # delete the outlier
    x2 = np.concatenate((x2[:54], x2[55:]))
    data2 = np.concatenate((data2[:54], data2[55:]))

    # 400:5:1080 + 1080:10:2500
    data = np.zeros(136 + 142)
    data11, data12 = data1[:91], data1[91:]
    data[:91] = data11 * 0.995
    data[91:136] = data12[:-5]
    data[136:] = data2[3:] * 0.99

    x = np.zeros(136 + 142)
    x[:136] = x1[:136]
    x[136:] = x2[3:]

    # plt.plot(x, data)

    fun = interp1d(x, data, kind='cubic')
    x_new = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    OL = fun(x_new)
    # plt.plot(x_new, OL)

    data1 = np.zeros(141)  # 400:5:1100
    data2 = np.zeros(146)  # 1050:10:2500

    with open(dirin + '02_100px.dat') as f:
        f.readline()
        for i in range(141):
            data1[i] = f.readline()

    with open(dirin + '03_100px.dat') as f:
        f.readline()
        for i in range(146):
            data2[i] = f.readline()

    # prekryv x1[130:] a x2[:6]
    x1 = np.arange(400, 1101, 5)
    x2 = np.arange(1050, 2501, 10)

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

    fun = interp1d(x, data, kind='cubic')
    OPX = fun(x_new)
    # plt.plot(x_new, OPX)

    # C = np.arange(0, 1.1, 0.1)  # OPC number
    C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    spectra_syn = np.zeros((len(x_new), len(C)))

    for i in range(len(C)):
        spectra_syn[:, i] = mixing_function_hapke_based(OPX, OL, C[i])
    C *= 100
    C = C.astype(int)

    C = np.array(['OL ' + str(100 - s) + ', OPX ' + str(s) for s in C])

    px10 = np.loadtxt(dirin + '10px.dat')
    px25 = np.loadtxt(dirin + '25px.dat')
    px50 = np.loadtxt(dirin + '50px.dat')
    px75 = np.loadtxt(dirin + '75px.dat')
    px90 = np.loadtxt(dirin + '90px.dat')

    spectra_msm = np.zeros((len(x_new), len(C)))
    spectra_msm[:, 0] = OL
    spectra_msm[:, 6] = OPX

    # jumps at 850 [:91] and 1170 [147:] and remove outliers
    px10[:91, 1] *= 0.98
    px10[147:, 1] *= 0.995
    px10 = np.delete(px10, 260, axis=0)
    spectra_msm[:, 1] = interp1d(px10[:, 0], px10[:, 1], kind='cubic')(x_new)

    px25[:91, 1] *= 0.985
    px25[147:, 1] *= 0.995
    px25 = np.delete(px25, 146, axis=0)
    spectra_msm[:, 2] = interp1d(px25[:, 0], px25[:, 1], kind='cubic')(x_new)

    px50[:91, 1] *= 0.995
    px50[147:, 1] *= 0.997
    px50 = np.delete(px50, 35, axis=0)
    spectra_msm[:, 3] = interp1d(px50[:, 0], px50[:, 1], kind='cubic')(x_new)

    px75[:91, 1] *= 0.98
    px75[147:, 1] *= 0.997
    spectra_msm[:, 4] = interp1d(px75[:, 0], px75[:, 1], kind='cubic')(x_new)

    px90[:91, 1] *= 0.988
    px90[147:, 1] *= 0.9985
    spectra_msm[:, 5] = interp1d(px90[:, 0], px90[:, 1], kind='cubic')(x_new)

    # En67, Fo90 podle Katky
    labels = np.array([[1, 0, 0, 0, 0.10, 0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0.9, 0.1, 0, 0, 0.10, 0.90, 0.33, 0.67, 0, 0, 0, 0, 0, 0, 0],
                       [0.75, 0.25, 0, 0, 0.10, 0.90, 0.33, 0.67, 0, 0, 0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0.10, 0.90, 0.33, 0.67, 0, 0, 0, 0, 0, 0, 0],
                       [0.25, 0.75, 0, 0, 0.10, 0.90, 0.33, 0.67, 0, 0, 0, 0, 0, 0, 0],
                       [0.1, 0.9, 0, 0, 0.10, 0.90, 0.33, 0.67, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0.33, 0.67, 0, 0, 0, 0, 0, 0, 0],
                       ])

    # wt% to vol%
    densities = np.array([4.49, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])
    modal, chemical = labels[:, :4], labels[:, 4:]

    modal_densities = chemical * densities
    ol_den = np.sum(modal_densities[:, :2], axis=1, keepdims=True)
    opx_den = np.sum(modal_densities[:, 2:5], axis=1, keepdims=True)
    cpx_den = np.sum(modal_densities[:, 5:8], axis=1, keepdims=True)
    plg_den = np.sum(modal_densities[:, 8:11], axis=1, keepdims=True)

    # if mineral is not present, set density to 1 (to prevent 0/density = 0/0)
    ol_den[ol_den == 0] = 1
    opx_den[opx_den == 0] = 1
    cpx_den[cpx_den == 0] = 1
    plg_den[plg_den == 0] = 1

    # combine densities
    modal_densities = np.concatenate((ol_den, opx_den, cpx_den, plg_den), axis=1)
    modal_vol = modal / modal_densities
    # normalise it to percents
    modal_vol /= np.sum(modal_vol, axis=1, keepdims=True)

    labels[:, :4] = modal_vol

    metadata = np.array(len(labels) * ['Tomas mixtures'])

    filename = save_data('Tomas_OL_OPX', denoise_and_norm(np.transpose(spectra_msm), denoise, normalise),
                         labels=labels, meta=metadata)

    return x_new, spectra_msm, spectra_syn, C, [filename]


def resave_Tomas_Chelyabinsk() -> List[str]:
    dirin = '/home/dakorda/Python/NN/Datasets/met_test/'

    x_new = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)

    # SD spektra
    SD_names = ['SD0.txt', 'SD5.txt', 'SD10.txt', 'SD20.txt', 'SD30.txt', 'SD40.txt', 'SD50.txt', 'SD60.txt',
                'SD70.txt', 'SD80.txt', 'SD90.txt', 'SD95.txt', 'SD100.txt']
    N_SD = len(SD_names)
    SD_spectra = np.zeros((N_SD, len(x_new)))

    for i in range(N_SD):
        tmp = np.loadtxt(dirin + SD_names[i])
        x, y = tmp[:, 0], tmp[:, 1]

        fun = interp1d(x, y, kind='cubic')
        tmp_spec = fun(x_new)

        SD_spectra[i, :] = denoise_and_norm(tmp_spec, denoising=True, normalising=True)

    # IM spektra
    IM_names = ['100LL.txt', '10IM.txt', '20IM.txt', '30IM.txt', '40IM.txt', '50IM.txt', '60IM.txt', '70IM.txt',
                '80IM.txt', '90IM.txt', '95IM.txt', '100IM.txt']
    N_IM = len(IM_names)
    IM_spectra = np.zeros((N_IM, len(x_new)))

    for i in range(N_IM):
        tmp = np.loadtxt(dirin + IM_names[i])
        x, y = tmp[:, 0] * 1000, tmp[:, 1]

        fun = interp1d(x, y, kind='cubic')
        tmp_spec = fun(x_new)

        IM_spectra[i, :] = denoise_and_norm(tmp_spec, denoising=True, normalising=True)

    # SW spektra
    SW_names = ['SW0.txt', 'SW400.txt', 'SW500.txt', 'SW600.txt', 'SW700.txt']
    N_SW = len(SW_names)
    SW_spectra = np.zeros((N_SW, len(x_new)))

    for i in range(N_SW):
        tmp = np.loadtxt(dirin + SW_names[i])
        x, y = tmp[:, 0], tmp[:, 1]

        fun = interp1d(x, y, kind='cubic')
        tmp_spec = fun(x_new)

        SW_spectra[i, :] = denoise_and_norm(tmp_spec, denoising=True, normalising=True)

    combined_spectra = np.concatenate((SD_spectra, IM_spectra, SW_spectra))
    filename = save_data('Tomas_Chelyabinsk', combined_spectra)

    # here can be moving it from relab to ../

    return [filename]


@timing
def mixing_among_minerals(what_minerals: Tuple[str, ...], nmax: int = 10000, type: str = 'GEN') -> None:
    # mixing functions are in modules.mixing_models.py

    same_grain_size = True
    mixing_function = mixing_function_hapke_general

    def load_spectra(what: str):
        data_file = "".join((path_to_data, '/', what, data_suffix, '.dat'))
        data = np.loadtxt(data_file, delimiter='\t')
        data_file = "".join((path_to_data, '/', what, data_suffix, '_meta.dat'))
        meta = pd.read_csv(data_file, sep='\t', header=None).to_numpy()  # to read the file
        grain_size = meta[:, 9:11].astype(np.float32)

        return data, grain_size

    def mixing_coefficients(type_of_mixture: str = None) -> np.ndarray:
        coefs = np.zeros(num_minerals_CD)

        if type_of_mixture == 'OC':  # ORDINARY CHONDRITES
            fact = 2.5  # maximum deviation from sigma

            mu, sigma = np.array([0.508, 0.305, 0.082, 0.105]), np.array([0.058, 0.063, 0.019, 0.025])
            coefs = sigma * np.random.randn() + mu
            coefs /= np.sum(coefs)
            while np.any(np.abs(coefs - mu) > fact * sigma):
                coefs = sigma * np.random.randn() + mu
                coefs /= np.sum(coefs)

        if type_of_mixture == 'GEN':  # GENERAL
            coefs = np.random.random(num_minerals_CD)

            # dropout (otherwise there are no pure minerals and partial mixtures)
            dropout = 0.50
            prob_dropout = np.random.random(num_minerals_CD)
            # keep at least one endmember
            prob_dropout[prob_dropout == np.max(prob_dropout)] = 1
            # apply dropout
            coefs[prob_dropout <= dropout] = 0

            # heuristic (more uniform distribution of modals)
            for _ in range(1):
                coefs *= np.random.random(num_minerals_CD)

            # normalise
            coefs /= np.sum(coefs)

        return coefs / np.sum(coefs)

    def give_me_indices_unique_spectra(N, nmax):
        # randomly selected unique indices (no two sets of mixed spectra are the same)
        n = np.prod(N)
        nmax_gs = np.min((n, nmax))
        indices = random.sample(range(n), nmax_gs)

        inds = list(np.unravel_index(indices, N))
        if used_minerals[0]:
            iol = inds.pop(0)
        else:
            iol = np.zeros(nmax_gs).astype(np.int8)

        if used_minerals[1]:
            iopx = inds.pop(0)
        else:
            iopx = np.zeros(nmax_gs).astype(np.int8)

        if used_minerals[2]:
            icpx = inds.pop(0)
        else:
            icpx = np.zeros(nmax_gs).astype(np.int8)

        if used_minerals[3]:
            iplg = inds.pop(0)
        else:
            iplg = np.zeros(nmax_gs).astype(np.int8)

        return iol, iopx, icpx, iplg

    def give_me_indices_nonunique_spectra(N, nmax):
        # randomly selected unique indices (two same sets spectra can be mixed -- different modal composition)
        # TOHLE ALE NENI PRO UNIQUE INDEXY!! NP.RANDOM.CHOICE(REPEAT=FALSE) NEBO TAK NECO
        iol = np.random.randint(N[0], size=nmax)
        iopx = np.random.randint(N[1], size=nmax)
        icpx = np.random.randint(N[2], size=nmax)
        iplg = np.random.randint(N[3], size=nmax)

        return iol, iopx, icpx, iplg

    path_to_data = path_relab
    # path_to_data = '/home/local/dakorda/MGM/david_mgm/results/'

    # which data
    data_suffix = '-denoised'  # _mixtures

    used_minerals = 4 * [False]

    # load the data
    OL, OL_gs = load_spectra('OL')
    if 'OL' in what_minerals:
        used_minerals[0] = True

    OPX, OPX_gs = load_spectra('OPX')
    if 'OPX' in what_minerals:
        used_minerals[1] = True

    CPX, CPX_gs = load_spectra('CPX')
    if 'CPX' in what_minerals:
        used_minerals[2] = True

    PLG, PLG_gs = load_spectra('PLG')
    if 'PLG' in what_minerals:
        used_minerals[3] = True

    n_labels = np.sum(subtypes_CD)

    N_values = np.shape(OL)[1]

    # if you don't want to mix the same grain sizes, comment the part below
    grain_sizes = np.array([x for x in set(tuple(x) for x in OL_gs) & set(tuple(x) for x in OPX_gs) &
                            set(tuple(x) for x in CPX_gs) & set(tuple(x) for x in PLG_gs)])
    # delete [0, 0] grain size
    grain_sizes = grain_sizes[np.sum(grain_sizes, 1) > 0]

    for index, grain_size in enumerate(grain_sizes):
        ol = OL[np.sum((np.equal(OL_gs, grain_size)), 1) == 2]
        opx = OPX[np.sum((np.equal(OPX_gs, grain_size)), 1) == 2]
        cpx = CPX[np.sum((np.equal(CPX_gs, grain_size)), 1) == 2]
        plg = PLG[np.sum((np.equal(PLG_gs, grain_size)), 1) == 2]
        #####

        # write into a file
        filename = 'synthetic_' + type + '_' + "_".join(what_minerals) + data_suffix

        # indices of the spectra
        N = np.array([len(ol), len(opx), len(cpx), len(plg)])[used_minerals]

        # iol, iopx, icpx, iplg = give_me_indices_unique_spectra(N, nmax)
        iol, iopx, icpx, iplg = give_me_indices_nonunique_spectra(N, nmax)

        synthetic_spectra = np.zeros((len(iol), N_values))

        # This can be done even without the for loop. For loop is probably slower but needs less memory
        for i in range(len(iol)):
            # each combination has unique coefficients in order to cover modal space
            coefs = mixing_coefficients(type)
            # only used minerals
            coefs *= used_minerals  # this delete only spectra and modals, chemicals are done below
            # normalise it to 1
            coefs /= np.sum(coefs)

            gs = np.array([np.mean(OL_gs[iol[i]]), np.mean(OPX_gs[iopx[i]]),
                           np.mean(CPX_gs[icpx[i]]), np.mean(PLG_gs[iplg[i]])])
            gs[gs < 10] = 10

            spectra = np.row_stack((ol[iol[i], :-num_labels_CD], opx[iopx[i], :-num_labels_CD],
                                    cpx[icpx[i], :-num_labels_CD], plg[iplg[i], :-num_labels_CD]))

            # mixing spectra
            synthetic_spectra[i, :-num_labels_CD] = mixing_function(spectra, coefs, gs)

            # mixing modals (linear)
            synthetic_spectra[i, -num_labels_CD:-n_labels] = (coefs[0] * ol[iol[i], -num_labels_CD:-n_labels] +
                                                           coefs[1] * opx[iopx[i], -num_labels_CD:-n_labels] +
                                                           coefs[2] * cpx[icpx[i], -num_labels_CD:-n_labels] +
                                                           coefs[3] * plg[iplg[i], -num_labels_CD:-n_labels]
                                                           )

            # mixing chemicals (linear)
            synthetic_spectra[i, -n_labels:] = (ol[iol[i], -n_labels:] * used_minerals[0] +
                                                opx[iopx[i], -n_labels:] * used_minerals[1] +
                                                cpx[icpx[i], -n_labels:] * used_minerals[2] +
                                                plg[iplg[i], -n_labels:] * used_minerals[3]
                                                )
        if index == 0:
            final_synthetic = synthetic_spectra
        else:
            final_synthetic = np.concatenate((final_synthetic, synthetic_spectra))

    # spectra normalisation
    final_synthetic[:, :-num_labels_CD] = denoise_and_norm(final_synthetic[:, :-num_labels_CD],
                                                        denoising=False, normalising=normalise)

    save_data(filename, final_synthetic[:, :-num_labels_CD], final_synthetic[:, -num_labels_CD:])

    return


def resample_Tuomas_data() -> None:
    # this function interpolate Tuomas data to the given grid

    # old grid
    x_old = np.linspace(450, 2450, 200)
    # new grid
    x_new = np.arange(450, 2450.1, 5)
    # final name
    final_name = 'Tuomas_HB_spectra'

    # load the data
    tuomas_data = 'MyVISNIR-simulated-HB-simplified-taxonomy.dat'
    data_file = "".join((project_dir, '/Datasets/Tuomas/', tuomas_data))
    data = pd.read_csv(data_file, sep='\t', header=None).to_numpy()

    # select spectra
    spectra = data[:, 1:].astype(np.float32)

    # labels
    labels = data[:, 0]
    labels = np.reshape(labels, (np.shape(labels)[0], 1))

    # interpolate the spectra
    fun = interp1d(x_old, spectra, kind='cubic')
    spectra = fun(x_new)

    spectra = denoise_and_norm(spectra, denoising=True, normalising=True)

    # to save spectra with 5 decimals only (needed for data with string labels)
    spectra = np.round(spectra, 5)

    save_data(final_name, spectra, labels=labels, Tuomas_data=True)


def resample_AP_raw_data() -> None:
    # this function interpolate AP (BD + MINTHEOS) data to the given grid

    # final name
    final_name = 'AP_spectra_v2'

    # new grid
    x_new = np.arange(450, 2450.1, 5)
    # old grid
    x_old = np.arange(450, 2450.1, 10)

    # load the data
    raw_data = 'AP_pca-spectra-combined.dat'
    data_file = "".join((project_dir, '/Datasets/Tuomas/', raw_data))
    data = pd.read_csv(data_file, sep='\t', header=None).to_numpy()

    metadata = np.zeros((len(data), 8)).astype(np.object)  # asteroid number, taxonomy, slope, PC1--PC5

    metadata[:, 0] = data[:, 0]
    metadata[:, 1] = data[:, 1]
    metadata[:, 2] = data[:, 2]
    metadata[:, 3:] = data[:, 3:8]

    # select spectra
    spectra = data[:, 8:].astype(np.float64)

    # labels data (B-DM class)
    labels = metadata[:, 1]
    labels = np.reshape(labels, (np.shape(labels)[0], 1))

    # interpolate the spectra
    fun = interp1d(x_old, spectra, kind='cubic')
    spectra = fun(x_new)

    spectra = denoise_and_norm(spectra, denoising=True, normalising=True)

    # to save spectra with 5 decimals only (needed for data with string labels)
    spectra = np.round(spectra, 5)

    save_data(final_name, spectra, labels=labels, Tuomas_data=True, meta=metadata)
    save_data(final_name, spectra, Tuomas_data=True)


def normalize_spectra(file: str, save_it: bool = False) -> None:
    path_to_data = path_relab

    # load the data
    data_file = "".join((path_to_data, '/', file, '.dat'))

    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    data = np.loadtxt(data_file, delimiter='\t')

    spectra, numbers = data[:, :-num_labels_CD], data[:, -num_labels_CD:]

    fun = interp1d(xq, spectra, kind='cubic')
    v_norm = fun(normalised_at)
    spectra_final = np.transpose(np.divide(np.transpose(spectra), v_norm))

    if save_it:
        save_data(file, spectra_final, labels=numbers, Tuomas_data=False)


def remove_continuum(modal: str) -> np.ndarray:
    control_plot = False
    n_labels = 15

    input_file = "".join((path_relab, '/', modal, '-denoised.dat'))
    output_file = "".join((path_relab, '/', modal, '-denoised-nocont_CH.dat'))

    data = np.loadtxt(input_file, delimiter='\t')
    spectra, numbers = data[:, :-n_labels], data[:, -n_labels:]

    n_data, len_data = np.shape(spectra)
    x = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    rectified_spectra = np.zeros((n_data, len_data))

    # 2D data for convex hull
    ch_data = np.zeros((len_data, 2))
    ch_data[:, 0] = x

    for i in range(n_data):
        spectrum = spectra[i]
        ch_data[:, 1] = spectrum

        hull = ConvexHull(ch_data).vertices

        # remove lower branch from vertices (delete all vertices between 0 and len0data - 1
        hull = np.roll(hull, -np.where(hull == 0)[0][0] - 1)  # move 0 to the end of the list
        hull = np.sort(hull[np.where(hull == len_data - 1)[0][0]:])

        # keep the UV bands
        x0 = my_argmin(x, spectrum, x0=650, minimum=False)
        hull = hull[np.argmin(np.abs(x[hull] - x0)):]
        continuum = np.zeros(np.shape(x))  # necessary since the UVs start at different positions

        # linear fit to the convex hull
        for j in range(len(hull) - 1):
            x_fit, y_fit = x[[hull[j], hull[j + 1]]], spectrum[[hull[j], hull[j + 1]]]
            if j == 0 and hull[j] != 0:
                x_new = x[:hull[j + 1] + 1]
                continuum[:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)
            else:
                x_new = x[hull[j]:hull[j + 1] + 1]
                continuum[hull[j]:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)

        rectified_spectra[i] = spectrum / continuum
        rectified_spectra = np.round(rectified_spectra, 5)

        if control_plot:
            fig, ax = plt.subplots()
            ax.plot(x, spectrum / continuum)
            ax.plot(x, spectrum)
            ax.plot(x, continuum)

    rectified_spectra = np.hstack((rectified_spectra, numbers))
    np.savetxt(output_file, rectified_spectra, fmt='%.5f', delimiter='\t')

    return rectified_spectra


if __name__ == '__main__':
    # synthetic sampler or samples with chemical analysis only?
    # mixtures probably should not be normalised
    mixtures = False

    # the numbers come from SPECTRAL CATALOGUE
    if mixtures:
        start_line_number, end_line_number, final_names = (161, 207, 348, 454), \
                                                          (206, 347, 453, 555), \
                                                          ('PLG', 'CPX', 'OPX', 'OL')
    else:
        start_line_number, end_line_number, final_names = (2, 19, 38, 74, 130, 161, 207, 348, 454), (
            18, 37, 73, 129, 160, 206, 347, 453, 555), ('OC_ALL', 'OC_OL_OPX_CPX', 'OC_OL_OPX', 'achondrites',
                                                        'mixtures', 'PLG', 'CPX', 'OPX', 'OL')

    start_line_number, end_line_number, final_names = (2,), (591,), ('relab',)

    names_relab = collect_data_RELAB(start_line_number, end_line_number, final_names)
    names_ctape = collect_data_CTAPE()
    *_, names_Tomas = resave_Tomas_OL_OPX_mixtures()

    # resave data
    save_names = names_relab + names_ctape + names_Tomas
    final_name = 'combined-denoised-norm.dat'
    combine_files(tuple(save_names), final_name)
    clean_and_resave(final_name)

    # resave metadata
    names_relab[0] = names_relab[0][:-4] + '-meta.dat'
    names_ctape[0] = names_ctape[0][:-4] + '-meta.dat'
    names_Tomas[0] = names_Tomas[0][:-4] + '-meta.dat'

    save_names = names_relab + names_ctape + names_Tomas
    final_name_meta = 'combined-denoised-norm-meta.dat'
    combine_files(tuple(save_names), final_name_meta)
    clean_and_resave(final_name, meta=True)

    """
    full_final_data_file = combine_files(('OL-denoised-nocont.dat', 'OPX-denoised-nocont.dat', 'CPX-denoised-nocont.dat',
                                          'PLG-denoised-nocont.dat', 'mixtures-denoised-nocont.dat'),
                                         'min_and_mix-denoised-nocont.dat')
    """
    # mixing_among_minerals(('OL', 'OPX', 'CPX', 'PLG'), type='GEN', nmax=1000)

    # mixing_function_test()

    # resample_Tuomas_data()
    # resample_AP_raw_data()

    # normalize_spectra('OL_mixtures')
    # normalize_spectra('OPX_mixtures')
    # normalize_spectra('CPX_mixtures')
    # normalize_spectra('PLG_mixtures')

    # remove_continuum('mixtures')
