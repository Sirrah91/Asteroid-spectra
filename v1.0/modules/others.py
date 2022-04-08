"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.NN_data import load_data
from modules.NN_evaluate import evaluate_test_data, evaluate
from modules.control_plots import plot_Fa_vs_Fs_v2, plot_Fa_vs_Fs_Tuomas
from modules.NN_losses_metrics_activations import my_rmse
from copy import deepcopy

from modules.NN_config import *
from modules.CD_parameters import path_relab

# Name of the new data in ./Datasets/RELAB/ or ./Datasets/Tuomas/
# filename_data = 'Tuomas_HB_spectra-denoised-norm-nolabel.dat'
filename_data = 'AP_spectra-denoised-norm-nolabel.dat'
filename_train_data = 'OC-denoised-norm.dat'
filename_train_data = 'OC-denoised-nocont.dat'
# filename_train_data = 'achondrites-denoised-norm.dat'

x_test, y_test = load_data(filename_train_data)

model_names1 = ['20211018154939_FC.h5',
                '20211018154949_FC.h5',
                '20211018155002_FC.h5',
                '20211018155013_FC.h5',
                '20211018155023_FC.h5']  # this one is good

model_names2 = ['20211020134525_FC.h5',
                '20211020140040_FC.h5',
                '20211020141700_FC.h5',
                '20211020143237_FC.h5',  # this one is good
                '20211020144856_FC.h5',
                '20211020150621_FC.h5',
                '20211020152738_FC.h5',
                '20211020154836_FC.h5',
                '20211020160605_FC.h5',
                '20211020161912_FC.h5']

model_names3 = ['20211020205324_FC.h5']

# tenhle set je uz pro optimalizovanou sit -- 'min_and_mix-denoised-norm.dat'
# RMSE celku je 18.06
model_names01 = ['20211122125145_FC.h5',
                 '20211122125337_FC.h5',
                 '20211122125527_FC.h5',
                 '20211122125747_FC.h5',
                 '20211122130011_FC.h5',
                 '20211122130245_FC.h5',
                 '20211122130649_FC.h5',
                 '20211122131045_FC.h5',
                 '20211122131301_FC.h5',
                 '20211122131438_FC.h5']

# 'min_and_mix-denoised-nocont.dat'
# RMSE celku je 22.85
model_names02 = ['20211122125145_FC.h5',
                 '20211122125337_FC.h5',
                 '20211122125527_FC.h5',
                 '20211122125747_FC.h5',
                 '20211122130011_FC.h5',
                 '20211122130245_FC.h5',
                 '20211122130649_FC.h5',
                 '20211122131045_FC.h5',
                 '20211122131301_FC.h5',
                 '20211122131438_FC.h5']

# 'min_and_mix-denoised-nocont_CH.dat'
# RMSE celku je 22.71
model_names03 = ['20211122125145_FC.h5',
                 '20211122125337_FC.h5',
                 '20211122125527_FC.h5',
                 '20211122125747_FC.h5',
                 '20211122130011_FC.h5',
                 '20211122130245_FC.h5',
                 '20211122130649_FC.h5',
                 '20211122131045_FC.h5',
                 '20211122131301_FC.h5',
                 '20211122131438_FC.h5']

# 'synthetic_GEN_OL_OPX_CPX_PLG-denoised.dat'
# RMSE celku je
model_names04 = ['20211122125145_FC.h5',
                 '20211122125337_FC.h5',
                 '20211122125527_FC.h5',
                 '20211122125747_FC.h5',
                 '20211122130011_FC.h5',
                 '20211122130245_FC.h5',
                 '20211122130649_FC.h5',
                 '20211122131045_FC.h5',
                 '20211122131301_FC.h5',
                 '20211122131438_FC.h5']

# 'synthetic_OC_OL_OPX_CPX_PLG-denoised.dat'
# RMSE celku je
model_names05 = ['20211122125145_FC.h5',
                 '20211122125337_FC.h5',
                 '20211122125527_FC.h5',
                 '20211122125747_FC.h5',
                 '20211122130011_FC.h5',
                 '20211122130245_FC.h5',
                 '20211122130649_FC.h5',
                 '20211122131045_FC.h5',
                 '20211122131301_FC.h5',
                 '20211122131438_FC.h5']

res = [evaluate_test_data([name], x_test, y_test) for name in model_names02]
# predictions, accuracy = evaluate_test_data([model_names2[3]], x_test, y_test)
# predictions, accuracy = evaluate_test_data(model_names3, x_test, y_test)
# predictions = evaluate(model_names3, filename_data)

#  5.941257 ,  5.670361 ,  2.4510405,  3.0278194
#   5.7653446, 6.256558 , 1.9498428, 2.5815694
# constant prediction
y_pred = deepcopy(y_test)
y_pred[:, 0] = 0.508
y_pred[:, 1] = 0.305
y_pred[:, 2] = 0.082
y_pred[:, 3] = 0.105
acc = my_rmse(y_test, y_pred).numpy()

plot_Fa_vs_Fs_v2(y_test, predictions)
plot_Fa_vs_Fs_Tuomas(predictions)

ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]
Fa, Fs = predictions[:, ind_Fa] * 100, predictions[:, ind_Fs] * 100

filename = path_relab + 'OC-denoised-norm_meta.dat'
data = np.loadtxt(filename, delimiter='\t')
types = data[:, 7]
inds_H = np.array(['H' in OC_type for OC_type in types])
inds_L = np.array([('L' in OC_type) and ('LL' not in OC_type) for OC_type in types])
inds_LL = np.array(['LL' in OC_type for OC_type in types])

HFa, HFs = (16.2, 16.2 + 3.8), (14.5, 14.5 + 3.5)
LFa, LFs = (22.0, 22.0 + 4.0), (19.0, 19.0 + 3.0)
LLFa, LLFs = (26.0, 26.0 + 6.0), (22.0, 22.0 + 4.2)

FaH, FsH = Fa[inds_H], Fs[inds_H]
FaL, FsL = Fa[inds_L], Fs[inds_L]
FaLL, FsLL = Fa[inds_LL], Fs[inds_LL]

NN = 100
sigma = 1

#####

tmp = np.linspace(FaH - sigma * accuracy[ind_Fa], FaH + sigma * accuracy[ind_Fa], NN)
c1 = np.array(
    [np.array([HFa[1] >= tmp[i, j] >= HFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsH - sigma * accuracy[ind_Fs], FsH + sigma * accuracy[ind_Fs], NN)
c2 = np.array(
    [np.array([HFs[1] >= tmp[i, j] >= HFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

cH = sum(c1 * c2) / sum(inds_H) * 100

#####

tmp = np.linspace(FaL - sigma * accuracy[ind_Fa], FaL + sigma * accuracy[ind_Fa], NN)
c1 = np.array(
    [np.array([LFa[1] >= tmp[i, j] >= LFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsL - sigma * accuracy[ind_Fs], FsL + sigma * accuracy[ind_Fs], NN)
c2 = np.array(
    [np.array([LFs[1] >= tmp[i, j] >= LFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

cL = sum(c1 * c2) / sum(inds_L) * 100

#####

tmp = np.linspace(FaLL - sigma * accuracy[ind_Fa], FaLL + sigma * accuracy[ind_Fa], NN)
c1 = np.array(
    [np.array([LLFa[1] >= tmp[i, j] >= LLFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsLL - sigma * accuracy[ind_Fs], FsLL + sigma * accuracy[ind_Fs], NN)
c2 = np.array(
    [np.array([LLFs[1] >= tmp[i, j] >= LLFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

cLL = sum(c1 * c2) / sum(inds_LL) * 100

np.round(np.mean((FaH, FsH), axis=1), 1)
np.round(np.mean((FaL, FsL), axis=1), 1)
np.round(np.mean((FaLL, FsLL), axis=1), 1)

# error
np.round(accuracy[ind_Fa] / np.sqrt(np.sum(inds_H)), 1)
np.round(accuracy[ind_Fa] / np.sqrt(np.sum(inds_L)), 1)
np.round(accuracy[ind_Fa] / np.sqrt(np.sum(inds_LL)), 1)

np.round(accuracy[ind_Fs] / np.sqrt(np.sum(inds_H)), 1)
np.round(accuracy[ind_Fs] / np.sqrt(np.sum(inds_L)), 1)
np.round(accuracy[ind_Fs] / np.sqrt(np.sum(inds_LL)), 1)
"""

import numpy as np
import pandas as pd
from modules.utilities import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from modules.CD_parameters import *
from modules.mixing_models import *
import math


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

        data_denoised = np.convolve(np.squeeze(data), kernel, 'same') / correction
    else:
        data_denoised = data

    # Normalised reflectance
    if normalising:
        fun = interp1d(xq, data_denoised, kind='cubic')  # v_final differs from v
        v_norm = fun(normalised_at)
    else:
        v_norm = 1

    return data_denoised / v_norm


def load_data_relab(which='OL'):
    Sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None,
                                     na_values=['NA'], usecols="A:AG", engine='openpyxl').to_numpy()

    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)

    if which == 'OL':
        sampleIDs = ['JB-JLB-945-D', 'JB-JLB-A17', 'JB-JLB-A15', 'JB-JLB-A16', 'JB-JLB-A14-A']
        spectrumIDs = ['C1JB945D', 'C1JBA17', 'C1JBA15', 'C1JBA16', 'C1JBA14A', ]
        coefs = [0, 0.25, 0.50, 0.75, 1]  # OPX number
    else:
        sampleIDs = ['PP-CMP-021', 'XP-CMP-016', 'XP-CMP-014', 'XP-CMP-012', 'XP-CMP-010', 'XP-CMP-011',
                     'XP-CMP-013', 'XP-CMP-015', 'PE-CMP-030']
        spectrumIDs = ['C1PP21', 'C1XP16', 'C1XP14', 'C1XP12', 'C1XP10', 'C1XP11', 'C1XP13', 'C1XP15', 'C1PE30']
        coefs = [0, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 1]  # OPX number

    lines_in_sample_catalogue = flatten_list([np.where(Sample_catalogue[:, 0] == sampleID)[0]
                                              for sampleID in sampleIDs])

    PIs = Sample_catalogue[lines_in_sample_catalogue, 2]

    filenames = np.array(["".join(
        (path_relab, '/data/', PIs[ii].lower(), '/', sampleIDs[ii][:2].lower(), '/', spectrumIDs[ii].lower(),
         '.asc')) for ii in range(len(spectrumIDs))])

    vq = np.zeros((len(xq), len(coefs)))

    for ii, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            n_lines = int(f.readline())  # First line contains info about the length of the data
            data = np.array([np.array(f.readline().split()[:2], dtype=np.float64) for _ in range(n_lines)])

        to_nm = 1

        x = data[:, 0] * to_nm  # to nm
        v = data[:, 1]

        # This have to be done due to some spectra
        x, idx = np.unique(x, return_index=True)
        v = v[idx]

        fun = interp1d(x, v, kind='cubic')
        v_interp = fun(xq)

        vq[:, ii] = denoise_and_norm(v_interp, denoising=True, normalising=False)

    return xq, vq, coefs


def smooth_minmax(spectrum1, spectrum2, alpha: float = -50):
    return np.log(np.exp(alpha * spectrum1) + np.exp(alpha * spectrum2)) / alpha


def get_absorption(spectrum):
    return np.square(-np.log(spectrum))


def get_spectrum(absorption):
    return np.exp(-np.sqrt(absorption))


def mix_function_v1(spectrum1, spectrum2, coef):
    x = np.array([0, 0.10, 0.25, 0.50, 0.75, 0.9, 1])
    y = x ** (1 / 1.5)

    new_coef1 = np.polyval(np.polyfit(x, y, 4), coef)
    new_coef2 = np.polyval(np.polyfit(x, y, 4), 1 - coef)

    new_coef = np.zeros(np.shape(spectrum1))
    new_coef[np.where(spectrum1 < spectrum2)] = new_coef1
    new_coef[np.where(spectrum1 >= spectrum2)] = new_coef2

    # convolution removes the jumps in new_coef
    width = 41
    cent = int(np.round(width / 2))
    kernel = np.zeros(width)

    for ii in range(int(np.floor(width / 2))):
        kernel[ii] = 1 / (np.abs(ii - cent) + 1)
        kernel[-ii - 1] = 1 / (np.abs(ii - cent) + 1)
    kernel[cent] = 1

    kernel = kernel / np.sum(kernel)
    correction = np.convolve(np.ones(len(spectrum1)), kernel, 'same')

    new_coef = np.convolve(new_coef, kernel, 'same') / correction  # this one is for OL + OPX
    # new_coef = coef ** (1 / 1.25)  # this is for Chelyabinsk dark + bright

    # does the spectra intersect?
    if np.size(np.argwhere(np.diff(np.sign(spectrum1 - spectrum2))).flatten()):
        alpha = 10
    else:
        alpha = 500

    mixed_spectrum = np.exp(
        new_coef * np.log(smooth_minmax(spectrum1, spectrum2, alpha=-alpha)) + (1 - new_coef) * np.log(
            smooth_minmax(spectrum2, spectrum1, alpha=alpha)))

    return mixed_spectrum


def mix_function_v2(spectrum1, spectrum2, coef):
    # 1/1.5 pro NIR; 1/1.05 pro VIS.. nejaky spojity prechod?
    tmp = np.zeros(np.shape(spectrum1))
    tmp[:140] = 1 / 1.05
    tmp[140:] = 1 / 1.5
    coef = coef ** tmp

    mixed_spectrum = np.exp(coef * np.log(spectrum1) + (1 - coef) * np.log(spectrum2))

    return mixed_spectrum


def mix_function_linear(spectrum1, spectrum2, coef):
    spec1, spec2 = get_absorption(spectrum1), get_absorption(spectrum2)
    mix = spec1 * coef + spec2 * (1 - coef)
    mixed_spectrum = get_spectrum(mix)

    return mixed_spectrum


def mix_function_FAN(spectrum1, spectrum2, coef):
    spec1, spec2 = get_absorption(spectrum1), get_absorption(spectrum2)

    part_one = spec1 * coef + spec2 * (1 - coef)
    part_two = coef * (1 - coef) * spec1 * spec2

    mix = part_one + part_two
    mixed_spectrum = get_spectrum(mix)

    return mixed_spectrum


def mix_function_GBM(spectrum1, spectrum2, coef):
    spec1, spec2 = get_absorption(spectrum1), get_absorption(spectrum2)

    gamma = -0.14  # interakcni stupen

    part_one = spec1 * coef + spec2 * (1 - coef)
    part_two = coef * (1 - coef) * spec1 * spec2

    mix = part_one + gamma * part_two
    mixed_spectrum = get_spectrum(mix)

    return mixed_spectrum


def mix_function_PPNM(spectrum1, spectrum2, coef):
    spec1, spec2 = get_absorption(spectrum1), get_absorption(spectrum2)

    beta = -0.367  # interakcni stupen

    part_one = spec1 * coef + spec2 * (1 - coef)
    part_two = np.square(coef * spec1) + 2 * coef * (1 - coef) * spec1 * spec2 + np.square((1 - coef) * spec1)

    mix = part_one + beta * part_two
    mixed_spectrum = get_spectrum(mix)

    return mixed_spectrum


def mix_function_MLM(spectrum1, spectrum2, coef):
    spec1, spec2 = get_absorption(spectrum1), get_absorption(spectrum2)

    P = 0.5

    w1 = spec1 / (P * spec1 + 1 - P)
    w2 = spec2 / (P * spec2 + 1 - P)

    y = w1 * coef + w2 * (1 - coef)

    mix = (1 - P) * y / (1 - P * y)
    mixed_spectrum = get_spectrum(mix)

    return mixed_spectrum


def MaxwellGarnettFormula(spectrum1, spectrum2, vol_spec1):
    small_number_cutoff = 1e-6

    # minus signs here are formal
    eps_base, eps_incl = get_absorption(spectrum1), get_absorption(spectrum2)

    if vol_spec1 < 0 or vol_spec1 > 1:
        print('WARNING: volume portion of spectrum1 is out of range!')

    factor_up = 2 * vol_spec1 * eps_base + (3 - 2 * vol_spec1) * eps_incl
    factor_down = (3 - vol_spec1) * eps_base + vol_spec1 * eps_incl

    if np.any(np.abs(factor_down) < small_number_cutoff):
        print('WARNING: the effective medium is singular!')
        eps_mean = np.zeros(np.shape(spectrum1))
    else:
        eps_mean = eps_base * factor_up / factor_down

    spectrum = get_spectrum(eps_mean)

    return spectrum


def remove_continuum(x: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    len_data, = np.shape(spectrum)

    # 2D data for convex hull
    ch_data = np.zeros((len_data, 2))
    ch_data[:, 0] = x
    ch_data[:, 1] = spectrum

    hull = ConvexHull(ch_data).vertices

    # remove lower branch from vertices (delete all vertices between 0 and len0data - 1
    hull = np.roll(hull, -np.where(hull == 0)[0][0] - 1)  # move 0 to the end of the list
    hull = np.sort(hull[np.where(hull == len_data - 1)[0][0]:])

    # keep the UV bands
    # x0 = my_argmin(x, spectrum, x0=650, minimum=False)
    # hull = hull[np.argmin(np.abs(x[hull] - x0)):]
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

    rectified_spectra = spectrum / continuum

    if 0:
        fig, ax = plt.subplots()
        ax.plot(x, spectrum / continuum)
        ax.plot(x, spectrum)
        ax.plot(x, continuum)

    return rectified_spectra


def plot_it(x, v, v_syn):
    colours = ['k', 'r', 'b', 'g', 'c', 'm', 'y', 'k', 'r', 'b', 'g', 'c', 'm', 'y']
    fig, ax = plt.subplots()
    for i in range(np.shape(v)[1]):
        ax.plot(x, v[:, i], colours[i])
        ax.plot(x, v_syn[:, i], colours[i] + '--')

    ax.set_ylabel('Reflectance')
    ax.set_xlabel('$\lambda$ [$\mu$m]')
    ax.set_ylim(bottom=0, top=1)
    plt.show()

    # outdir = "".join((project_dir, '/figures/'))
    # fig_name = 'test.png'
    # fig.savefig("".join((outdir, '/', fig_name)), format='png', bbox_inches='tight', pad_inches=0)


def load_data_Ch(which='SD'):
    if which == 'SD':
        numbers = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        pref, suf = 'SD', ''
        ndata = 421
    else:
        numbers = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        pref, suf = '', 'IM'
        ndata = 2151

    data = np.zeros((ndata, len(numbers)))
    indir = '/home/dakorda/Python/NN/Datasets/met_test/'

    for index, num in enumerate(numbers):
        a = np.loadtxt(indir + pref + str(num) + suf + '.txt')
        data[:, index] = a[:, 1]

    x_axis = a[:, 0]

    return x_axis, data, numbers / 100


def SAM(s1, s2):
    """
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math don't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle


mixing_function = mixing_function_hapke_based

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
x_new = np.arange(400, 2501, 5)
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
x_new = np.arange(400, 2501, 5)
OPX = fun(x_new)
# plt.plot(x_new, OPX)

C = np.arange(0, 1.1, 0.1)  # OPC number
# C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
syn_spectrum = np.zeros((len(x_new), len(C)))

for i in range(len(C)):
    syn_spectrum[:, i] = mixing_function(OPX, OL, C[i])

C *= 100
C = C.astype(int).astype(str)

px10 = np.loadtxt(dirin + '10px.dat')
px25 = np.loadtxt(dirin + '25px.dat')
px50 = np.loadtxt(dirin + '50px.dat')
px75 = np.loadtxt(dirin + '75px.dat')
px90 = np.loadtxt(dirin + '90px.dat')

raw_spectra = np.zeros((len(x_new), 7))
raw_spectra[:, 0] = OL
raw_spectra[:, 1] = interp1d(px10[:, 0], px10[:, 1], kind='cubic')(x_new)
raw_spectra[:, 2] = interp1d(px25[:, 0], px25[:, 1], kind='cubic')(x_new)
raw_spectra[:, 3] = interp1d(px50[:, 0], px50[:, 1], kind='cubic')(x_new)
raw_spectra[:, 4] = interp1d(px75[:, 0], px75[:, 1], kind='cubic')(x_new)
raw_spectra[:, 5] = interp1d(px90[:, 0], px90[:, 1], kind='cubic')(x_new)
raw_spectra[:, 6] = OPX

correlation, RMSE, sam = np.zeros(np.shape(C)), np.zeros(np.shape(C)), np.zeros(np.shape(C))

try:
    for i in range(7):
        correlation[i] = np.corrcoef(raw_spectra[:, i], syn_spectrum[:, i])[0, 1]
        RMSE[i] = np.sqrt(np.mean((raw_spectra[:, i] - syn_spectrum[:, i]) ** 2))
        sam[i] = SAM(raw_spectra[:, i], syn_spectrum[:, i])

    print(', '.join('{:.5f}'.format(k) for k in correlation))
    print(', '.join('{:.5f}'.format(k) for k in RMSE))
    print(', '.join('{:.5f}'.format(k) for k in sam))
except:
    pass

fig, ax = plt.subplots()
ax.plot(x_new, syn_spectrum)

C = np.array(['Px ' + s for s in C])
ax.legend(C, loc='best')

ax.plot(x_new, raw_spectra, '--k')

ax.set_xlabel('wavelength [nm]')
ax.set_ylabel('reflectance')

"""
what = 'OL'
X, V, C = load_data_relab(what)
V_syn = np.zeros((len(X), len(C)))

for i in range(len(C)):
    V_syn[:, i] = mixing_function(V[:, -1], V[:, 0], C[i])

plot_it(X, V, V_syn)

what = 'OPX'
X, V, C = load_data_relab(what)
V_syn = np.zeros((len(X), len(C)))

for i in range(len(C)):
    V_syn[:, i] = mixing_function(V[:, -1], V[:, 0], C[i])

plot_it(X, V, V_syn)

what = 'SD'
X, V, C = load_data_Ch(what)
V_syn = np.zeros((len(X), len(C)))

for i in range(len(C)):
    V_syn[:, i] = mixing_function(V[:, -1], V[:, 0], C[i])

plot_it(X, V, V_syn)
"""

"""
x = np.arange(0, 5, 0.1)
x = np.reshape(x, (np.size(x), 1))
y = np.exp(-5 * x)

xt = np.transpose(x)
yt = np.transpose(y)

lr, num_eps = 0.1, 1e-15
t0, grad = 0, lr

while np.abs(lr * grad) > num_eps:
    y0 = np.exp(-t0 * x)
    y0t = np.transpose(y0)
    dy0 = -x*y0
    dy0t = -np.transpose(y0) * np.transpose(x)

    grad = -2 * np.matmul(yt, dy0) + 2 * np.matmul(y0t, dy0)

    t0 -= lr * grad
print(t0[0, 0])
"""
"""
import numpy as np
from scipy.interpolate import interp1d


def get_absorption_2(spectrum):
    return spectrum  # np.square(-np.log(spectrum))


def get_spectrum_2(absorption):
    return absorption  # np.exp(-np.sqrt(absorption))


abs, spec = get_absorption_2, get_spectrum_2

C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
i = 3
spectrum1, spectrum2, coef = np.reshape(OPX, (421, 1)), np.reshape(OL, (421, 1)), C[i]
spectrum_true = raw_spectra[:, i-1]

fun = interp1d(x_raw, spectrum_true)
spectrum_true = np.reshape(fun(x_new), (421, 1))

spec1, spec2 = abs(spectrum1), abs(spectrum2)
spectrum_true = abs(spectrum_true)

y0 = spec1 * coef + spec2 * (1 - coef)
y1 = coef * (1 - coef) * spec1 * spec2

lr, num_eps = 0.1, 1e-15
gamma, grad = 0.2, lr

while np.abs(lr * grad) > num_eps:
    mix = y0 + gamma * y1
    y_prime = spec(mix)

    grad = -2 * np.matmul(np.transpose(spectrum_true), y1) + 2 * np.matmul(np.transpose(y_prime), y1)

    gamma -= lr * grad

print(gamma)
"""
"""
import numpy as np
from scipy.interpolate import interp1d


def get_absorption_2(spectrum):
    return spectrum  # np.square(-np.log(spectrum))


def get_spectrum_2(absorption):
    return absorption  # np.exp(-np.sqrt(absorption))


abs, spec = get_absorption_2, get_spectrum_2

C = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
i = 3
spectrum1, spectrum2, coef = np.reshape(OPX, (421, 1)), np.reshape(OL, (421, 1)), C[i]
spectrum_true = raw_spectra[:, i-1]

fun = interp1d(x_raw, spectrum_true)
spectrum_true = np.reshape(fun(x_new), (421, 1))

spec1, spec2 = abs(spectrum1), abs(spectrum2)
spectrum_true = abs(spectrum_true)

y0 = spec1 * coef + spec2 * (1 - coef)
y1 = np.square(coef * spec1) + 2 * coef * (1-coef) * spec1 * spec2 + np.square((1-coef) * spec1)

lr, num_eps = 0.001, 1e-15
beta, grad = 0.2, lr

while np.abs(lr * grad) > num_eps:
    mix = y0 + beta * y1
    y_prime = spec(mix)

    grad = -2 * np.matmul(np.transpose(spectrum_true), y1) + 2 * np.matmul(np.transpose(y_prime), y1)

    beta -= lr * grad

print(beta)

"""

"""
def mixing_function_test():
    def smooth_minmax(spectrum1, spectrum2, alpha: float = -50):
        # alpha > 0 for approx of maximum
        # alpha < 0 for approx of minimum

        return np.log(np.exp(alpha * spectrum1) + np.exp(alpha * spectrum2)) / alpha

    def mix_function(spectrum1, spectrum2, coef, what):
        OL_OPX, SD_Ch, OPX_CPX = 0, 0, 0

        if what == 'OL':
            OL_OPX = 1
        elif what == 'OPX':
            OPX_CPX = 1
        else:
            SD_Ch = 1

        if OL_OPX:
            # x are abundances of OPX in OL + OPX mixture
            x = np.array([0, 0.25, 0.5, 0.75, 1])
            # y[1:-1] were estimated by eye
            y = x ** (1 / (1e-5 + 8 * x))

            new_coef1 = np.polyval(np.polyfit(x, y, 4), coef)
            new_coef2 = np.polyval(np.polyfit(x, y, 4), 1 - coef)

            new_coef = np.zeros(np.shape(spectrum1))
            new_coef[np.where(spectrum1 < spectrum2)] = new_coef1
            new_coef[np.where(spectrum1 >= spectrum2)] = new_coef2

            # convolution removes the jumps in new_coef
            width = 41
            cent = int(np.round(width / 2))
            kernel = np.zeros(width)

            for ii in range(int(np.floor(width / 2))):
                kernel[ii] = 1 / (np.abs(ii - cent) + 1)
                kernel[-ii - 1] = 1 / (np.abs(ii - cent) + 1)
            kernel[cent] = 1

            kernel = kernel / np.sum(kernel)
            correction = np.convolve(np.ones(len(spectrum1)), kernel, 'same')

            new_coef, alpha = np.convolve(new_coef, kernel, 'same') / correction, 10  # this one is for OL + OPX
            '''
            # does the spectra intersect?
            if np.size(np.argwhere(np.diff(np.sign(spectrum1 - spectrum2))).flatten()):
                alpha = 10
            else:
                alpha = 50
            '''
            mixed_spectrum = np.exp(
                new_coef * np.log(smooth_minmax(spectrum1, spectrum2, alpha=-alpha)) + (1 - new_coef) * np.log(
                    smooth_minmax(spectrum1, spectrum2, alpha=alpha)))
        elif SD_Ch:
            new_coef, alpha = coef ** (1 / 1.25), 40  # this one is for SD set
            mixed_spectrum = np.exp(
                new_coef * np.log(smooth_minmax(spectrum1, spectrum2, alpha=-alpha)) + (1 - new_coef) * np.log(
                    smooth_minmax(spectrum1, spectrum2, alpha=alpha)))
        elif OPX_CPX:
            alpha = 10
            new_coef = coef ** (1 / 2)
            mixed_spectrum = np.exp(
                new_coef * np.log(smooth_minmax(spectrum1, spectrum2, alpha=-alpha)) + (1 - new_coef) * np.log(
                    smooth_minmax(spectrum1, spectrum2, alpha=alpha)))

        # mixed_spectrum = 1/np.exp(coef * np.log(1/spectrum1) + (1-coef) * np.log(spectrum2))

        return mixed_spectrum

    def MaxwellGarnettFormula(spectrum1, spectrum2, vol_spec1):
        small_number_cutoff = 1e-6

        # minus signs here are formal
        eps_base, eps_incl = np.square(-np.log(spectrum1)), np.square(-np.log(spectrum2))

        if vol_spec1 < 0 or vol_spec1 > 1:
            print('WARNING: volume portion of spectrum1 is out of range!')

        factor_up = 2 * vol_spec1 * eps_base + (3 - 2 * vol_spec1) * eps_incl
        factor_down = (3 - vol_spec1) * eps_base + vol_spec1 * eps_incl

        if np.any(np.abs(factor_down) < small_number_cutoff):
            print('WARNING: the effective medium is singular!')
            eps_mean = np.zeros(np.shape(spectrum1))
        else:
            eps_mean = eps_base * factor_up / factor_down

        spectrum = np.exp(-np.sqrt(eps_mean))

        return spectrum

    def load_data_Ch(which='SD'):
        if which == 'SD':
            numbers = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
            pref, suf = 'SD', ''
            ndata = 421
        else:
            numbers = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
            pref, suf = '', 'IM'
            ndata = 2151

        data = np.zeros((ndata, len(numbers)))
        indir = '/home/dakorda/Python/NN/Datasets/met_test/'

        for index, num in enumerate(numbers):
            a = np.loadtxt(indir + pref + str(num) + suf + '.txt')
            data[:, index] = a[:, 1]

        x_axis = a[:, 0]

        return x_axis, data, numbers / 100

    def load_data_relab(which='OL'):
        Sample_catalogue = pd.read_excel("".join((path_relab, 'Sample_Catalogue.xlsx')), index_col=None,
                                         na_values=['NA'], usecols="A:AG", engine='openpyxl').to_numpy()

        xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)

        if which == 'OL':
            sampleIDs = ['JB-JLB-945-D', 'JB-JLB-A17', 'JB-JLB-A15', 'JB-JLB-A16', 'JB-JLB-A14-A']
            spectrumIDs = ['C1JB945D', 'C1JBA17', 'C1JBA15', 'C1JBA16', 'C1JBA14A', ]
            coefs = [0, 0.25, 0.50, 0.75, 1]  # OPX number
        else:
            sampleIDs = ['PP-CMP-021', 'XP-CMP-016', 'XP-CMP-014', 'XP-CMP-012', 'XP-CMP-010', 'XP-CMP-011',
                         'XP-CMP-013', 'XP-CMP-015', 'PE-CMP-030']
            spectrumIDs = ['C1PP21', 'C1XP16', 'C1XP14', 'C1XP12', 'C1XP10', 'C1XP11', 'C1XP13', 'C1XP15', 'C1PE30']
            coefs = [0, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 1]  # OPX number

        lines_in_sample_catalogue = flatten_list([np.where(Sample_catalogue[:, 0] == sampleID)[0]
                                                  for sampleID in sampleIDs])

        PIs = Sample_catalogue[lines_in_sample_catalogue, 2]

        filenames = np.array(["".join(
            (path_relab, '/data/', PIs[ii].lower(), '/', sampleIDs[ii][:2].lower(), '/', spectrumIDs[ii].lower(),
             '.asc')) for ii in range(len(spectrumIDs))])

        vq = np.zeros((len(xq), len(coefs)))

        for ii, filename in enumerate(filenames):
            with open(filename, 'r') as f:
                n_lines = int(f.readline())  # First line contains info about the length of the data
                data = np.array([np.array(f.readline().split()[:2], dtype=np.float64) for _ in range(n_lines)])

            to_nm = 1

            x = data[:, 0] * to_nm  # to nm
            v = data[:, 1]

            # This have to be done due to some spectra
            x, idx = np.unique(x, return_index=True)
            v = v[idx]

            fun = interp1d(x, v, kind='cubic')
            v_interp = fun(xq)

            vq[:, ii] = denoise_and_norm(v_interp, denoising=True, normalising=False)

        return xq, vq, coefs

    def plot_it(x, v, v_syn):
        colours = ['k', 'r', 'b', 'g', 'c', 'm', 'y', 'k', 'r', 'b', 'g', 'c', 'm', 'y']
        fig, ax = plt.subplots()
        for i in range(np.shape(v)[1]):
            ax.plot(x, v[:, i], colours[i])
            ax.plot(x, v_syn[:, i], colours[i] + '--')

        ax.set_ylabel('Reflectance')
        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylim(bottom=0, top=1)
        plt.show()

        # outdir = "".join((project_dir, '/figures/'))
        # fig_name = 'test.png'
        # fig.savefig("".join((outdir, '/', fig_name)), format='png', bbox_inches='tight', pad_inches=0)

    what = 'OL'
    X, V, C = load_data_relab(what)
    V_syn = np.zeros((len(X), len(C)))

    for i in range(len(C)):
        V_syn[:, i] = mix_function(V[:, -1], V[:, 0], C[i], what)
        # V_syn[:, i] = MaxwellGarnettFormula(V[:, -1], V[:, 0], C[i])

    plot_it(X, V, V_syn)
"""
