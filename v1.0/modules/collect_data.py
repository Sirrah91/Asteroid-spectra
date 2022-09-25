from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from os import path
from typing import Union
from urllib.request import urlopen

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from itertools import product

from modules.utilities_spectra import denoise_and_norm, save_data, clean_and_resave, combine_files

from modules.utilities import *
from modules.CD_parameters import *


def collect_data_RELAB(start_line_number: Union[Tuple[int, ...], int],
                       end_line_number: Union[Tuple[int, ...], int] = None,
                       final_names: Tuple[str, ...] = ('PLG', 'CPX', 'OPX', 'OL')) -> List[str]:
    # The input Tuples are numbers of line IN Excel file (counting from 1)
    def load_excel_data(start_line_number: Union[Tuple[int, ...], int],
                        end_line_number: Union[Tuple[int, ...], int] = None) -> Tuple[np.ndarray, ...]:
        # This function read the data from the Excel files

        # 2 is the offset; 1 due to header, 1 due to counting rows in Excel from 1, not from 0
        rows = np.arange(np.min(start_line_number), np.max(end_line_number) + 1) - 2

        # Real the files (one spectrum per sample)
        Spectra_catalogue = pd.read_excel("".join((path_relab, 'Spectra_Catalogue.xlsx')), index_col=None,
                                          na_values=['NA'], usecols="A, B, F:H", engine='openpyxl').to_numpy()[rows]

        Sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None,
                                         na_values=['NA'], sheet_name='RELAB',
                                         usecols="A:AG", engine='openpyxl').to_numpy()

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

        # Metadata; +1 because the last one is weathering
        metadata = sample_catalogue[lines_in_sample_catalogue, :-(num_labels_CD + 1)]

        return filenames, modal_numbers, OL_numbers, OPX_numbers, CPX_numbers, PLG_numbers, metadata

    def select_numbers(numbers: np.ndarray) -> np.ndarray:
        # modals
        modal = numbers[0]
        modal = modal[:, np.where(use_minerals_CD)[0]]

        # renormalise to unity
        norm = np.sum(modal, axis=1).astype(np.float64)
        # if any of the values is missing, norm is nan
        # modal = np.transpose(np.divide(np.transpose(modal), norm))
        # no_nan = np.where(1 - np.isnan(norm))

        for i in range(np.shape(modal)[0]):
            if not np.isnan(norm[i]):
                modal[i] = modal[i] / norm[i]

        # chemical
        indices = np.where(use_minerals_CD)[0] + 1  # +1 for modals
        chemical = np.hstack([numbers[index] for index in indices])

        numbers = np.hstack((modal, chemical)).astype(np.float32)

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
        vq = denoise_and_norm(data=vq, wavelength=xq, denoising=denoise, normalising=normalise)

        return vq

    if end_line_number is None:
        # skip header
        start_line_number, end_line_number = 2, start_line_number

    print("Collecting data from RELAB...")
    # read the data
    spectra_catalogue, sample_catalogue = load_excel_data(start_line_number, end_line_number)

    # label and metadata keys
    labels_key = sample_catalogue[1, -num_labels_CD - 1:-1]  # -1 for weathering which is included in the catalogue
    metadata_key = sample_catalogue[1, :-num_labels_CD - 1]  # -1 for weathering which is included in the catalogue

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

        # Save the interpolated spectra
        filenames[i_modal] = save_data(final_names[i_modal], spectra=spectra, wavelengths=xq, labels=numbers,
                                       metadata=metadata, labels_key=labels_key, metadata_key=metadata_key,
                                       subfolder='RELAB')

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
            vq_c = denoise_and_norm(data=vq_clean, wavelength=xq, denoising=denoise, normalising=normalise)

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

    sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None, na_values=['NA'],
                                     usecols="A:AF", header=1, sheet_name='C-Tape', engine='openpyxl').to_numpy()

    labels = sample_catalogue[1:, -num_labels_CD:] / 100
    metadata = sample_catalogue[1:, :-num_labels_CD]

    labels_key = sample_catalogue[0, -num_labels_CD:]
    metadata_key = sample_catalogue[0, :-num_labels_CD]

    filename = save_data('CTape_spectra', spectra=final_mat, wavelengths=xq, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder='C-Tape')

    return [filename]


def resave_Tomas_OL_OPX_mixtures() -> List[str]:

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

    spectra = denoise_and_norm(data=np.transpose(spectra_msm), wavelength=x_new,
                               denoising=denoise, normalising=normalise)

    sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None, na_values=['NA'],
                                     usecols="A:AF", header=1, sheet_name='TK', engine='openpyxl').to_numpy()

    labels = sample_catalogue[1:, -num_labels_CD:] / 100
    metadata = sample_catalogue[1:, :-num_labels_CD]

    labels_key = sample_catalogue[0, -num_labels_CD:]
    metadata_key = sample_catalogue[0, :-num_labels_CD]

    filename = save_data('OL_OPX', spectra=spectra, wavelengths=x_new, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder='ol_opx_mix')

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
        SD_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=x_new, denoising=denoise, normalising=normalise)

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

        IM_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=x_new, denoising=denoise, normalising=normalise)

    # SW spektra
    SW_names = ['SW0.txt', 'SW400.txt', 'SW500.txt', 'SW600.txt', 'SW700.txt']
    N_SW = len(SW_names)
    SW_spectra = np.zeros((N_SW, len(x_new)))

    for i in range(N_SW):
        tmp = np.loadtxt(dirin + SW_names[i])
        x, y = tmp[:, 0], tmp[:, 1]

        fun = interp1d(x, y, kind='cubic')
        tmp_spec = fun(x_new)

        SW_spectra[i, :] = denoise_and_norm(data=tmp_spec, wavelength=x_new, denoising=denoise, normalising=normalise)

    combined_spectra = np.concatenate((SD_spectra, IM_spectra, SW_spectra))
    metadata = np.array(SD_names + IM_names + SW_names)
    metadata = np.reshape(metadata, (len(metadata), 1))

    labels = np.array([[66.2, 33.8, 0.0, 0.0, 28.6, 71.4, 23.9, 76.1, 1.6, 0.0, 0.0, 0.0, 8.5, 84.5, 7.0]])
    labels = np.repeat(labels, len(metadata), axis=0)

    sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None, na_values=['NA'],
                                     usecols="A:AF", header=1, sheet_name='TK', engine='openpyxl').to_numpy()

    labels_key = sample_catalogue[0, -num_labels_CD:]

    filename = save_data('Chelyabinsk', spectra=combined_spectra, wavelengths=x_new, labels=labels,
                         metadata=metadata, labels_key=labels_key, subfolder='chelyabinsk')
    my_mv(filename, filename.replace('chelyabinsk', ''), 'cp')

    return [filename]


def resample_asteroid_taxonomy_data() -> None:
    # this function interpolate BD + MINTHEOS data to the given grid
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

    labels_key = np.array(['taxonomy class'])
    metadata_key = np.array(['asteroid number', 'taxonomy class', 'slope', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    spectra = denoise_and_norm(data=spectra, wavelength=x_new, denoising=denoise, normalising=normalise)

    filename = save_data(final_name, spectra=spectra, wavelengths=x_new, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')

    # combine some classes
    classes = [('Sqw', 'S'), ('Sw', 'S'), ('Srw', 'Sr'), ('R', 'Sr'), ('Vw', 'V'), ('Sa', 'A'),
               ('Cg', 'Cgh'), ('Cb', 'C'), ('Xc', 'X')]
    for old, new in classes:
        inds = np.array([old == ast_type for ast_type in labels.ravel()])
        labels[inds] = new

    # Sq for predictions (should be between S and Q)
    inds = np.where(np.array(['Sq' == ast_type for ast_type in labels.ravel()]))[0]
    filename = save_data(final_name + '_Sq', spectra=spectra[inds], wavelengths=x_new, labels=labels[inds],
                         metadata=metadata[inds], labels_key=labels_key, metadata_key=metadata_key,
                         subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')

    # delete some classes
    classes = ['O', 'Qw', 'Sq', 'Sq:', 'Sv', 'Svw', 'Xn', 'U']
    for old in classes:
        inds = np.where(np.array([old == ast_type for ast_type in labels.ravel()]))[0]
        labels = np.delete(labels, inds, axis=0)
        spectra = np.delete(spectra, inds, axis=0)
        metadata = np.delete(metadata, inds, axis=0)

    filename = save_data(final_name + '-reduced', spectra=spectra, wavelengths=x_new, labels=labels, metadata=metadata,
                         labels_key=labels_key, metadata_key=metadata_key, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')

    # add OC from RELAB to Q type
    filename = project_dir + '/Datasets/combined-denoised-norm.npz'
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = data["metadata"][:, 6].astype(np.str)
    types = np.reshape(types, (len(types), 1))
    inds = np.where(np.array(['Ordinary Chondrite' == ast_type for ast_type in types]))[0]
    types = types[inds]

    types[:] = 'Q'
    OC = data["spectra"][inds]
    x_oc = data["wavelengths"]
    fun = interp1d(x_oc, OC, kind='cubic')
    OC = fun(x_new)

    OC_meta = np.array([np.shape(metadata)[1] * [np.nan]]).astype(np.object)
    OC_meta[0, 1] = 'Q'
    OC_meta = np.repeat(OC_meta, len(OC), axis=0)

    labels = np.concatenate((labels, types))
    spectra = np.concatenate((spectra, OC))
    metadata = np.concatenate((metadata, OC_meta))

    filename = save_data(final_name + '-reduced_Q', spectra=spectra, wavelengths=x_new, labels=labels,
                         metadata=metadata, labels_key=labels_key, metadata_key=metadata_key, subfolder='taxonomy')
    my_mv(filename, filename.replace('taxonomy', ''), 'cp')


def resave_Tuomas_Itokawa_Eros() -> None:
    print('Re-saving Itokawa and Eros data...')

    # load the data
    filename = 'polysum.h5'  # Itokawa
    with h5py.File("".join((project_dir, 'Datasets/taxonomy/', filename)), 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Removing the coordinate info from the beginning of each sample
        data = np.array(f['d'][:, 2:])

    metadata = np.array([['Itokawa', 'Hayabusa Near Infrared Spectrometer']])

    wavelengths = np.flip(np.arange(2247.9, 763.52, -23.56)[7:60])  # nm; Tuomas filtered out some wvl
    wavelengths = np.interp(np.arange(0, 53, 0.424), np.arange(53), wavelengths)  # he did this weird interpolation

    filename = save_data('Itokawa', spectra=data, wavelengths=wavelengths, metadata=metadata)
    my_mv(filename, filename.replace('-denoised', '').replace('-norm', ''), 'mv')

    filename = 'polysumeros1000.h5'  # Eros
    with h5py.File("".join((project_dir, 'Datasets/taxonomy/', filename)), 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Removing the coordinate info from the beginning of each sample
        data = np.array(f['d'][:, 2:])

    metadata = np.array([['Eros', 'NEAR Near-Infrared Spectrometer']])

    # Tuomas used this grid for Eros
    wavelengths = np.concatenate((794.6 + 21.61 * np.arange(1, 23),
                                  43.11 * np.setdiff1d(np.arange(37, 59), [47, 57]) - 50.8))

    filename = save_data('Eros', spectra=data, wavelengths=wavelengths, metadata=metadata)
    my_mv(filename, filename.replace('-denoised', '').replace('-norm', ''), 'mv')


def resave_kachr_ol_opx() -> List[str]:
    print("Re-saving Katka's data...")

    dirin = '/home/dakorda/Python/NN/Datasets/kachr_ol_opx/'

    x_new_part = np.arange(545, 2416, 5)
    x_new = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)

    # file names and suffixes
    names = ['ol', 'py']
    suffixes = ['Ar', 'H', 'He', 'laser']
    first = 1  # allocate some space during first iteration

    for name, suffix in product(names, suffixes):
        file = name + '-' + suffix + '.csv'
        tmp = np.loadtxt(dirin + file)
        x, y = tmp[:, 0], tmp[:, 1:]

        fun = interp1d(x, np.transpose(y), kind='cubic')  # do all at once
        tmp_spec = fun(x_new_part)  # one spectrum per row

        fun = interp1d(x_new_part, tmp_spec, kind='linear', fill_value='extrapolate')  # do all at once
        tmp_spec = fun(x_new)  # one spectrum per row

        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x_new, np.transpose(tmp_spec))
        """

        if first:
            spectra = denoise_and_norm(data=tmp_spec, wavelength=x_new, denoising=denoise, normalising=normalise)

            first = 0
        else:
            spectra = np.concatenate((spectra, denoise_and_norm(data=tmp_spec, wavelength=x_new,
                                                                denoising=denoise, normalising=normalise)))

    metadata = np.array([['Olivine-Ar+', 'fresh'], ['Olivine-Ar+', '1e15 Ar+ / cm2'],
                         ['Olivine-Ar+', '3e15 Ar+ / cm2'], ['Olivine-Ar+', '6e15 Ar+ / cm2'],
                         ['Olivine-Ar+', '1e16 Ar+ / cm2'], ['Olivine-Ar+', '2e16 Ar+ / cm2'],
                         ['Olivine-Ar+', '6e16 Ar+ / cm2'], ['Olivine-Ar+', '1e17 Ar+ / cm2'],
                         #
                         ['Olivine-H+', 'fresh'], ['Olivine-H+', '1e14 H+ / cm2'], ['Olivine-H+', '1e15 H+ / cm2'],
                         ['Olivine-H+', '1e16 H+ / cm2'], ['Olivine-H+', '1e17 H+ / cm2'],
                         ['Olivine-H+', '2e17 H+ / cm2'], ['Olivine-H+', '5e17 H+ / cm2'],
                         ['Olivine-H+', '1e18 H+ / cm2'],
                         #
                         ['Olivine-He+', 'fresh'], ['Olivine-He+', '1e16 He+ / cm2'], ['Olivine-He+', '3e16 He+ / cm2'],
                         ['Olivine-He+', '6e16 He+ / cm2'], ['Olivine-He+', '1e17 He+ / cm2'],
                         #
                         ['Olivine-laser', 'fresh'], ['Olivine-laser', '1.7 J / cm2'], ['Olivine-laser', '2.4 J / cm2'],
                         ['Olivine-laser', '3.8 J / cm2'], ['Olivine-laser', '4.6 J / cm2'],
                         ['Olivine-laser', '6.7 J / cm2'], ['Olivine-laser', '10.4 J / cm2'],
                         ['Olivine-laser', '15.0 J / cm2'], ['Olivine-laser', '23.4 J / cm2'],
                         ['Olivine-laser', '30.6 J / cm2'], ['Olivine-laser', '60.0 J / cm2'],
                         ['Olivine-laser', '93.8 J / cm2'], ['Olivine-laser', '375.0 J / cm2'],
                         #
                         #
                         ['Pyroxene-Ar+', 'fresh'], ['Pyroxene-Ar+', '1e15 Ar+ / cm2'],
                         ['Pyroxene-Ar+', '3e15 Ar+ / cm2'], ['Pyroxene-Ar+', '6e15 Ar+ / cm2'],
                         ['Pyroxene-Ar+', '1e16 Ar+ / cm2'], ['Pyroxene-Ar+', '2e16 Ar+ / cm2'],
                         ['Pyroxene-Ar+', '6e16 Ar+ / cm2'], ['Pyroxene-Ar+', '1e17 Ar+ / cm2'],
                         #
                         ['Pyroxene-H+', 'fresh'], ['Pyroxene-H+', '1e16 H+ / cm2'], ['Pyroxene-H+', '1e17 H+ / cm2'],
                         ['Pyroxene-H+', '2e17 H+ / cm2'], ['Pyroxene-H+', '5e17 H+ / cm2'],
                         ['Pyroxene-H+', '1e18 H+ / cm2'],
                         #
                         ['Pyroxene-He+', 'fresh'], ['Pyroxene-He+', '1e16 He+ / cm2'],
                         ['Pyroxene-He+', '3e16 He+ / cm2'], ['Pyroxene-He+', '6e16 He+ / cm2'],
                         ['Pyroxene-He+', '1e17 He+ / cm2'],
                         #
                         ['Pyroxene-laser', 'fresh'], ['Pyroxene-laser', '4.5 J / cm2'],
                         ['Pyroxene-laser', '5.6 J / cm2'], ['Pyroxene-laser', '12.5 J / cm2'],
                         ['Pyroxene-laser', '18.0 J / cm2'], ['Pyroxene-laser', '28.1 J / cm2'],
                         ['Pyroxene-laser', '36.7 J / cm2'], ['Pyroxene-laser', '50.0 J / cm2'],
                         ['Pyroxene-laser', '72.0 J / cm2'], ['Pyroxene-laser', '112.5 J / cm2'],
                         ['Pyroxene-laser', '200.0 J / cm2'], ['Pyroxene-laser', '450.0 J / cm2'],
                         ['Pyroxene-laser', '1800.0 J / cm2']
                         ])

    labels_ol = np.array([[1, 0, 0, 0, 9.9, 90.1, 0, 0, 0.0, 0, 0, 0, 0, 0, 0]])
    labels_ol = np.repeat(labels_ol, 34, axis=0)
    labels_px = np.array([[0, 84/(84+5) * 100, 5/(84+5) * 100, 0, 0, 0, 32.9, 67.1, 0, 0, 0, 0, 0, 0, 0]])
    labels_px = np.repeat(labels_px, 32, axis=0)

    labels = np.concatenate((labels_ol, labels_px))

    sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None, na_values=['NA'],
                                     usecols="A:AF", header=1, sheet_name='TK', engine='openpyxl').to_numpy()

    labels_key = sample_catalogue[0, -num_labels_CD:]
    labels_key[1] = labels_key[1].replace("vol", "wt")
    labels_key[2] = labels_key[2].replace("vol", "wt")

    filename = save_data('Kachr_ol_opx', spectra=spectra, wavelengths=x_new, metadata=metadata, labels=labels,
                         labels_key=labels_key, subfolder='kachr_ol_opx')
    my_mv(filename, filename.replace('kachr_ol_opx', ''), 'cp')

    return [filename]


def resave_didymos() -> List[str]:

    print("Re-saving Didymos' data...")

    dirin = '/home/dakorda/Python/NN/Datasets/didymos/'

    x_new = np.arange(490, 2451, 10)

    file = 'Didymos_vnir_albedo.dat'
    tmp = pd.read_csv(dirin + file, sep='\t', index_col=None, header=[0], engine='python').to_numpy()
    x, y = tmp[:, 0], tmp[:, 1:]

    fun = interp1d(x, np.transpose(y), kind='cubic')  # do all at once
    spectrum = fun(x_new)  # one spectrum per row
    spectrum = denoise_and_norm(data=spectrum, wavelength=x_new, denoising=denoise, normalising=normalise, sigma_nm=20)

    metadata = np.array([['Didymos spectrum']])

    filename = save_data('Didymos', spectra=spectrum, wavelengths=x_new, metadata=metadata, subfolder='didymos')
    my_mv(filename, filename.replace('didymos', ''), 'cp')

    return [filename]


if __name__ == '__main__':
    start_line_number, end_line_number, final_names = (2,), (591,), ('relab',)

    names_relab = collect_data_RELAB(start_line_number, end_line_number, final_names)
    names_ctape = collect_data_CTAPE()
    names_Tomas = resave_Tomas_OL_OPX_mixtures()

    # resave data
    save_names = names_relab + names_ctape + names_Tomas
    final_name = 'combined-denoised-norm.npz'
    combine_files(tuple(save_names), final_name)

    # clean_and_resave(final_name)

    resample_asteroid_taxonomy_data()
    resave_Tomas_Chelyabinsk()
    resave_Tuomas_Itokawa_Eros()
    resave_kachr_ol_opx()
