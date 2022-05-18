from os import path
from typing import Union
from urllib.request import urlopen

import h5py
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d

from modules.spectra_module import denoise_and_norm, save_data, clean_and_resave, combine_files

mpl.use('TkAgg')

from modules.utilities import *
from modules.CD_parameters import *


def collect_data_RELAB(start_line_number: Union[Tuple[int, ...], int],
                       end_line_number: Union[Tuple[int, ...], int] = None,
                       final_names: Tuple[str, ...] = ('PLG', 'CPX', 'OPX', 'OL')) -> List[str]:
    # The input Tuples are numbers of line IN excel file (counting from 1)
    def load_excel_data(start_line_number: Union[Tuple[int, ...], int],
                        end_line_number: Union[Tuple[int, ...], int] = None) -> Tuple[np.ndarray, ...]:
        # This function read the data from the excel files

        # 2 is the offset; 1 due to header, 1 due to counting rows in excel from 1, not from 0
        rows = np.arange(np.min(start_line_number), np.max(end_line_number) + 1) - 2

        # Real the files (one spectrum per sample)
        Spectra_catalogue = pd.read_excel("".join((path_relab, 'Spectra_Catalogue.xlsx')), index_col=None,
                                          na_values=['NA'], usecols="A, B, F:H", engine='openpyxl').to_numpy()[rows]

        Sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None,
                                         na_values=['NA'], usecols="A:AG", engine='openpyxl').to_numpy()

        return Spectra_catalogue, Sample_catalogue

    def split_and_filter_data(spectra_catalogue: np.ndarray, sample_catalogue: np.ndarray) -> Tuple[np.ndarray, ...]:
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
        filenames = np.array(["".join((path_relab_raw, 'data/', PIs[i].lower(), '/', sampleIDs[i][:2].lower(), '/',
                                       spectrumIDs[i].lower(), '.asc')) for i in range(len(spectrumIDs))])

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
        filenames[i_modal] = save_data(final_names[i_modal], spectra, numbers, meta=metadata, subfolder='RELAB')

    return filenames


def collect_data_CTAPE() -> List[str]:

    print('Collecting data from C-Tape...')

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

    filename = save_data('CTape_spectra', final_mat, labels, meta=metadata, subfolder='C-Tape')

    return [filename]


def resave_Tomas_OL_OPX_mixtures() -> str:

    print("Collecting Tomas' OL-OPX mixtures...")

    dirin = '/home/dakorda/Python/NN/Datasets/ol_opx_mix/'

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

    C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

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

    filename = save_data('OL_OPX', denoise_and_norm(np.transpose(spectra_msm), denoise, normalise),
                         labels=labels, meta=metadata, subfolder='ol_opx_mix')

    return [filename]


def resave_Tomas_Chelyabinsk() -> List[str]:

    print('Re-saving Chelyabinsk data...')

    dirin = '/home/dakorda/Python/NN/Datasets/chelyabinsk/'

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

    filename = save_data('Chelyabinsk', combined_spectra, subfolder='chelyabinsk')
    my_mv(filename, filename.replace('chelyabinsk', ''), 'cp')

    return [filename]


def resample_AP_data() -> None:
    # this function interpolate AP (BD + MINTHEOS) data to the given grid
    print('Resampling asteroid data...')

    # final name
    final_name = 'asteroid_spectra'

    # new grid
    x_new = np.arange(450, 2450.1, 5)
    # old grid
    x_old = np.arange(450, 2450.1, 10)

    # load the data
    raw_data = 'asteroid_pca-spectra-combined.dat'
    data_file = "".join((project_dir, '/Datasets/taxonomy/', raw_data))
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

    filename = save_data(final_name, spectra, labels=labels, meta=metadata, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')

    filename = save_data(final_name, spectra, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')

    # combine some classes
    classes = [('Sqw', 'S'), ('Sw', 'S'), ('Srw', 'Sr'), ('R', 'Sr'), ('Vw', 'V'), ('Sa', 'A'),
               ('Cg', 'Cgh'), ('Cb', 'C'), ('Xc', 'X')]
    for old, new in classes:
        inds = np.array([old == ast_type for ast_type in labels.ravel()])
        labels[inds] = new

    # Sq for predictions (should be between S and Q)
    inds = np.where(np.array(['Sq' == ast_type for ast_type in labels.ravel()]))[0]
    spectra_Sq = spectra[inds]
    filename = save_data(final_name + '_Sq', spectra_Sq, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')

    # delete some classes
    classes = ['O', 'Qw', 'Sq', 'Sq:', 'Sv', 'Svw', 'Xn', 'U']
    for old in classes:
        inds = np.where(np.array([old == ast_type for ast_type in labels.ravel()]))[0]
        labels = np.delete(labels, inds, axis=0)
        spectra = np.delete(spectra, inds, axis=0)

    # add OC from RELAB to Q type
    filename = project_dir + '/Datasets/combined-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 6]
    types = np.reshape(types, (len(types), 1))
    inds = np.where(np.array(['Ordinary Chondrite' == ast_type for ast_type in types]))[0]

    filename = project_dir + '/Datasets/combined-denoised-norm.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = types[inds]

    types[:] = 'Q'
    OC = data[inds, :-15]  # there are 15 labels
    x_oc = np.arange(450, 2451, 5)
    fun = interp1d(x_oc, OC, kind='cubic')
    OC = fun(x_new)

    labels = np.concatenate((labels, types))
    spectra = np.concatenate((spectra, OC))

    filename = save_data(final_name + '-reduced', spectra, labels=labels, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')


def resave_Tuomas_Itokawa_Eros() -> None:

    print('Re-saving Itokawa and Eros data...')

    # load the data
    filename = 'polysum.h5'  # Itokawa
    with h5py.File("".join((project_dir, 'Datasets/taxonomy/', filename)), 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Removing the coordinate info from the beginning of each sample
        data = np.array(f['d'][:, 2:])

    filename = save_data('Itokawa', data)
    my_mv(filename, filename.replace('-denoised', '').replace('-norm', ''), 'mv')

    filename = 'polysumeros1000.h5'  # Eros
    with h5py.File("".join((project_dir, 'Datasets/taxonomy/', filename)), 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Removing the coordinate info from the beginning of each sample
        data = np.array(f['d'][:, 2:])

    filename = save_data('Eros', data)
    my_mv(filename, filename.replace('-denoised', '').replace('-norm', ''), 'mv')


if __name__ == '__main__':
    start_line_number, end_line_number, final_names = (2,), (591,), ('relab',)

    names_relab = collect_data_RELAB(start_line_number, end_line_number, final_names)
    names_ctape = collect_data_CTAPE()
    names_Tomas = resave_Tomas_OL_OPX_mixtures()

    # resave data
    save_names = names_relab + names_ctape + names_Tomas
    final_name = 'combined-denoised-norm.dat'
    combine_files(tuple(save_names), final_name)
    clean_and_resave(final_name)

    # resave metadata
    names_relab[0] = names_relab[0].replace('.dat', '-meta.dat')
    names_ctape[0] = names_ctape[0].replace('.dat', '-meta.dat')
    names_Tomas[0] = names_Tomas[0].replace('.dat', '-meta.dat')

    save_names = names_relab + names_ctape + names_Tomas
    final_name_meta = final_name.replace('.dat', '-meta.dat')
    combine_files(tuple(save_names), final_name_meta)
    clean_and_resave(final_name, meta=True)

    resample_AP_data()
    resave_Tomas_Chelyabinsk()
    resave_Tuomas_Itokawa_Eros()
