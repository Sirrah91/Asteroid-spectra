import pandas as pd
import numpy as np
import random
from scipy.interpolate import interp1d
from urllib.request import urlopen
from os import path
from typing import Tuple, List, Union

from modules.CD_parameters_RELAB import *


def combine_files(filenames: Tuple[str, ...], final_name: str) -> str:
    outfile_name = "".join((project_dir, '/Datasets/RELAB/', final_name))
    with open(outfile_name, 'w') as outfile:
        for fname in filenames:
            with open(fname, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    return outfile_name


def flatten_list(list_of_lists: List[List]) -> np.ndarray:
    return np.array([item for sub_list in list_of_lists for item in sub_list])


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
    first_index, last_index = offset, offset + num_minerals
    modal_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes[0]
    OL_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes[1]
    OPX_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes[2]
    CPX_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    first_index, last_index = last_index, last_index + subtypes[3]
    PLG_numbers = sample_catalogue[lines_in_sample_catalogue, first_index:last_index] / 100

    # Metadata
    metadata = sample_catalogue[lines_in_sample_catalogue, :-(num_labels + 1)]  # +1 because the last one is weathering

    return filenames, modal_numbers, OL_numbers, OPX_numbers, CPX_numbers, PLG_numbers, metadata


def select_numbers(numbers: np.ndarray) -> np.ndarray:
    # modals
    modal = numbers[0]
    modal = modal[:, np.where(use_minerals)[0]]

    # renormalise to unity
    norm = np.sum(modal, axis=1).astype(np.float64)
    # if an of the values is missing, norm is nan
    # modal = np.transpose(np.divide(np.transpose(modal), norm))
    # no_nan = np.where(1 - np.isnan(norm))

    for i in range(np.shape(modal)[0]):
        if not np.isnan(norm[i]):
            modal[i] = modal[i] / norm[i]

    # chemical
    indices = np.where(use_minerals)[0] + 1  # +1 for modals
    chemical = np.hstack([numbers[index] for index in indices])

    numbers = np.hstack((modal, chemical))

    return numbers


def collect_spectra(xq: np.ndarray, fnames: np.ndarray) -> np.ndarray:
    # This function collects spectra from the database, denoises and normalises them

    N = len(fnames)
    vq = np.zeros((N, len(xq)))

    # List of missing spectra (rows)
    irow = []

    if denoise:
        width = 9
        cent = int(np.round(width / 2))
        kernel = np.zeros(width)

        for ii in range(int(np.floor(width / 2))):
            kernel[ii] = 1 / (np.abs(ii - cent) + 1)
            kernel[-ii - 1] = 1 / (np.abs(ii - cent) + 1)
        kernel[cent] = 1

        kernel = kernel / np.sum(kernel)
        correction = np.convolve(np.ones(len(xq)), kernel, 'same')

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

        v_interp = fun(xq)

        if denoise:
            v_final = np.convolve(np.squeeze(v_interp), kernel, 'same') / correction
        else:
            v_final = v_interp

        # Normalised reflectance
        if normalise:
            v_norm = fun(normalised_at)
        else:
            v_norm = 1

        vq[i, :] = v_final / v_norm

    # Remove missing data
    vq = np.delete(vq, irow, axis=0)

    return vq


def save_data(final_name: str, spectra: np.ndarray, numbers: np.ndarray = None, meta: np.ndarray = None) -> str:
    if mixtures:
        final_name += '_for_mixing'
    else:
        if normalise:
            final_name += '-norm'
        if denoise:
            final_name += '-denoised'

        # Save data without labels (not needed for mixtues)
        filename = "".join((path_relab, final_name, '_nolabel.dat'))
        np.savetxt(filename, spectra, fmt='%.5f', delimiter='\t')

    # save metadata if these exist
    # data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    if meta is not None:
        filename = "".join((path_relab, final_name, '_meta.dat'))
        with open(filename, 'w') as f:
            for element in meta:
                final_string = '\t'.join([str(e) for e in element]) + '\n'

                for e in final_string:
                    f.write(e)

    # Save data with labels
    if numbers is not None:
        spectra = np.hstack((spectra, numbers))
        if mixtures:
            filename = "".join((path_MGM, final_name, '.dat'))
        else:
            filename = "".join((path_relab, final_name, '.dat'))
        np.savetxt(filename, spectra, fmt='%.5f', delimiter='\t')

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
        filenames[i_modal] = save_data(final_names[i_modal], spectra, numbers, metadata)

    return filenames


def interp_mixing(data: np.ndarray, coefs: np.ndarray) -> Tuple[np.ndarray, ...]:
    spectra, numbers = data[:, :-num_labels], data[:, -num_labels:]

    N, width_s = np.shape(spectra)
    _, width_n = np.shape(numbers)
    spectra_synthetic = np.zeros((int((N * N - N) / 2), width_s))
    numbers_synthetic = np.zeros((int((N * N - N) / 2), width_n))

    coefs /= np.sum(coefs)

    ind = 0
    for j in range(N - 1):
        for k in range(j + 1, N):
            # spectrum in ln(R)
            spectra_synthetic[ind, :] = coefs[0] * np.log(spectra[j, :]) + coefs[1] * np.log(spectra[k, :])
            numbers_synthetic[ind, :] = coefs[0] * numbers[j, :] + coefs[1] * numbers[k, :]
            ind += 1
            # ind = int((N * (N - 1) / 2) - (N - j) * ((N - j) - 1) / 2 + k - j - 1)

    spectra_synthetic = np.exp(spectra_synthetic)

    return spectra_synthetic, numbers_synthetic


def mixing_within_minerals(filenames: Tuple[str, ...] = ('OL_for_mixing', 'OPX_for_mixing', 'CPX_for_mixing')) -> None:
    final_filenames = [filenames[i] + '_mixed' for i in range(len(filenames))]

    coefs = np.array([1, 1])

    for i_modal in range(len(filenames)):
        # load the data
        filename = "".join((path_relab, filenames[i_modal], '.dat'))
        data = pd.read_csv(filename, sep='\t', header=None).to_numpy()

        # mixing - some kind of interpolation between functions
        interp_spectra, interp_numbers = interp_mixing(data, coefs)

        # saving
        save_data('RELAB/' + final_filenames[i_modal], interp_spectra, interp_numbers)

    return


def mixing_among_minerals(what_minerals: Tuple[str, ...], nmax: int = 100000) -> None:
    # linear of logarithmic mixing?
    linear = False

    path_to_data = path_relab
    # path_to_data = '/home/local/dakorda/MGM/david_mgm/results/'

    used_minerals = 4 * [False]

    # load the data
    data_file = "".join((path_to_data, '/OL_mixtures.dat'))
    OL = pd.read_csv(data_file, sep='\t', header=None).to_numpy()
    if 'OL' in what_minerals:
        used_minerals[0] = True

    data_file = "".join((path_to_data, '/OPX_mixtures.dat'))
    OPX = pd.read_csv(data_file, sep='\t', header=None).to_numpy()
    if 'OPX' in what_minerals:
        used_minerals[1] = True

    data_file = "".join((path_to_data, '/CPX_mixtures.dat'))
    CPX = pd.read_csv(data_file, sep='\t', header=None).to_numpy()
    if 'CPX' in what_minerals:
        used_minerals[2] = True

    data_file = "".join((path_to_data, '/PLG_mixtures.dat'))
    PLG = pd.read_csv(data_file, sep='\t', header=None).to_numpy()
    if 'PLG' in what_minerals:
        used_minerals[3] = True

    n_labels = np.sum(subtypes)

    N_values = np.shape(OL)[1]

    N = np.array([len(OL), len(OPX), len(CPX), len(PLG)])[used_minerals]
    n = np.prod(N)

    # write into a file
    filename = 'synthetic_' + "_".join(what_minerals)

    # randomly selected unique indices
    nmax = np.min((n, nmax))
    indices = random.sample(range(n), nmax)

    inds = list(np.unravel_index(indices, N))
    if use_minerals[0]:
        iol = inds.pop(0)
    else:
        iol = np.zeros(nmax).astype(np.int8)

    if used_minerals[1]:
        iopx = inds.pop(0)
    else:
        iopx = np.zeros(nmax).astype(np.int8)

    if used_minerals[2]:
        icpx = inds.pop(0)
    else:
        icpx = np.zeros(nmax).astype(np.int8)

    if used_minerals[3]:
        iplg = inds.pop(0)
    else:
        iplg = np.zeros(nmax).astype(np.int8)

    final_synthetic = np.zeros((len(indices), N_values))

    # This can be done even without the for loop. For loop is probably slower but needs less memory
    for i in range(len(indices)):
        # each combination has unique coefficients in order to cover modal space
        coefs = mixing_coefficients('OC')  # ORDINARY CHONDRITES
        # only used minerals
        coefs *= used_minerals  # this delete only spectra and modals, chemicals are done below
        # normalise it to 1
        coefs /= np.sum(coefs)

        if linear:
            final_synthetic[i, :-num_labels] = (coefs[0] * OL[iol[i], :-num_labels] +
                                                coefs[1] * OPX[iopx[i], :-num_labels] +
                                                coefs[2] * CPX[icpx[i], :-num_labels] +
                                                coefs[3] * PLG[iplg[i], :-num_labels])
        else:
            # mixing spectra (ln(R))
            final_synthetic[i, :-num_labels] = (coefs[0] * np.log(OL[iol[i], :-num_labels]) +
                                                coefs[1] * np.log(OPX[iopx[i], :-num_labels]) +
                                                coefs[2] * np.log(CPX[icpx[i], :-num_labels]) +
                                                coefs[3] * np.log(PLG[iplg[i], :-num_labels]))

        # mixing modals
        final_synthetic[i, -num_labels:-n_labels] = (coefs[0] * OL[iol[i], -num_labels:-n_labels] +
                                                     coefs[1] * OPX[iopx[i], -num_labels:-n_labels] +
                                                     coefs[2] * CPX[icpx[i], -num_labels:-n_labels] +
                                                     coefs[3] * PLG[iplg[i], -num_labels:-n_labels])

        # mixing chemicals
        final_synthetic[i, -n_labels:] = (OL[iol[i], -n_labels:] * used_minerals[0] +
                                          OPX[iopx[i], -n_labels:] * used_minerals[1] +
                                          CPX[icpx[i], -n_labels:] * used_minerals[2] +
                                          PLG[iplg[i], -n_labels:] * used_minerals[3])
    if not linear:
        # spectrum back in reflectance
        final_synthetic[:, :-num_labels] = np.exp(final_synthetic[:, :-num_labels])

    # normalisation spectra
    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    spectra = final_synthetic[:, :-num_labels]

    if normalise:
        for i in range(len(indices)):
            fun = interp1d(xq, spectra[i], kind='cubic')

            # Normalised reflectance
            v_norm = fun(normalised_at)
            final_synthetic[i, :-num_labels] = fun(xq) / v_norm

    save_data(filename, final_synthetic[:, :-n_labels], final_synthetic[:, -n_labels:])

    return


def mixing_coefficients(type1: str = None) -> np.ndarray:
    coefs = np.zeros(num_minerals)

    if type1 == 'OC':  # ORDINARY CHONDRITES
        fact = 2.5  # maximum deviation from sigma

        mu, sigma = np.array([0.508, 0.305, 0.082, 0.105]), np.array([0.058, 0.063, 0.019, 0.025])
        coefs = sigma * np.random.randn() + mu
        coefs /= np.sum(coefs)
        while np.any(np.abs(coefs - mu) > fact * sigma):
            coefs = sigma * np.random.randn() + mu
            coefs /= np.sum(coefs)

    return coefs / np.sum(coefs)


def resample_Tuomas_data():
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
    spectra = data[:, 1:]

    # interpolate the spectra
    fun = interp1d(x_old, spectra, kind='cubic')
    spectra = fun(x_new)

    save_data('Tuomas/' + final_name, spectra)


def find_minimum(x: np.ndarray, y: np.ndarray, x0: float, n: int = 5) -> float:
    step = x[1] - x[0]

    if step >= 1:  # x is most likely in nm
        half_width = 200
    else:  # x is most likely in um
        half_width = 0.2

    # find minimum in this interval
    ind = np.where(np.logical_and(x0 - half_width <= x, x <= x0 + half_width))
    x_int = x[ind]
    y_int = y[ind]

    # minimum value on this interval
    ix0 = np.argmin(y_int)

    x_min = x_int[ix0 - n:ix0 + n + 1]
    y_min = y_int[ix0 - n:ix0 + n + 1]

    params = np.polyfit(x_min, y_min, 2)

    return -params[1] / (2 * params[0])


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


        start_line_number, end_line_number, final_names = (2,), (73,), ('OC',)


    final_names = collect_data_RELAB(start_line_number, end_line_number, final_names)

    # full_final_data_file = combine_files(tuple(final_names), 'min_and_mix.dat')

    # mixing_among_minerals(('OL', 'OPX'))
    # resample_Tuomas_data()
