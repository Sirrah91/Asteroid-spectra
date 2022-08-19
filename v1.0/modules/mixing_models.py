# based on https://www.researchgate.net/publication/326921440_Hapke-based_computational_method_to_enable_unmixing_of_hyperspectral_data_of_common_salts
import random
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from modules.CD_parameters import *
from modules.utilities_spectra import denoise_and_norm, save_data
from modules.utilities import timing

incoming_angle = 0.0
emmerging_angle = 30.0
phase_angle = 30.0
Dl_Ds = 250 / 100  # pomer nejvetsi / nejmensi castice
phi = 0.0  # filling factor
b, c = -0.4, 0.25

mu0 = np.cos(incoming_angle * np.pi / 180)
mu = np.cos(emmerging_angle * np.pi / 180)
g = phase_angle * np.pi / 180


def ch(w, x):
    """
    Calculates Chandrasekhar H function
    w - single scatter albedo
    x = mu or mu_0 / K
    """
    y = np.sqrt(1 - w)
    r0 = (1 - y) / (1 + y)
    return 1. / (1 - w * x * (r0 + (1 - 2 * r0 * x) / 2. * np.log((1 + x) / x)))


def reflect(w, mu_0, mu):
    """
    Calculates reflectance of the mixture
    w - single scatter albedo
    mu - cosine of incidence angle
    mu_0 cosine of the emergence angle
    """

    num_eps = 1e-5

    if phi > num_eps:
        K = - np.log(1 - 1.209 * phi ** (2 / 3)) / (1.209 * phi ** (2 / 3))
    else:
        K = 1

    h_mu0 = ch(w, mu_0 / K)
    h_mu = ch(w, mu / K)

    P = 1 + b * np.cos(g) + c * (1.5 * np.cos(g) ** 2 - 0.5)

    hs = 3 * np.sqrt(3) / 8 * K * phi / np.log(Dl_Ds)
    if hs > num_eps:
        B = 1. / (1 + np.tan(g / 2) / hs)
    else:
        B = 0

    return K * w / 4. / (mu_0 + mu) * (P * (1 + B) + h_mu0 * h_mu - 1)


def residual_reflect(w, r):
    """
    Minimization fitting function for extracting w values
    params : w, inc_angle, emr_angle
    r : reflectance data to subtract
    """
    return reflect(w, mu0, mu) - r


def extract_ssa(r_data):
    w = np.zeros_like(r_data)
    for i in range(len(r_data)):
        try:  # only if r = 1 or is very very close to it -> w = 1
            w[i] = brentq(residual_reflect, 0., 1., args=(r_data[i]))
        except:
            w[i] = 1
    return w


def mixing_function_hapke_based(spectrum1, spectrum2, coef):
    w1, w2 = extract_ssa(spectrum1), extract_ssa(spectrum2)
    w_mix = coef * w1 + (1 - coef) * w2  # same grain size; else sum(Vi * wi / Di) / sum(Vi/Di)

    return reflect(w_mix, mu0, mu)


def mixing_function_hapke_general(spectra: np.ndarray, coefs: np.ndarray, grain_sizes: np.ndarray = None):
    if grain_sizes is None:
        grain_sizes = np.ones(np.shape(coefs))

    if not (len(spectra) == len(coefs) == len(grain_sizes)):
        raise ValueError('Dimensions are not consistent. Check inputs.')

    n_spectra = len(coefs)
    w_mix = 0

    normalisation = np.sum(coefs / grain_sizes)

    for i in range(n_spectra):
        w = extract_ssa(spectra[i])
        w_mix += coefs[i] * w / grain_sizes[i]
    w_mix /= normalisation

    return reflect(w_mix, mu0, mu)


def mixing_function_lin(spectra: np.ndarray, coefs: np.ndarray, grain_size: np.ndarray = None) -> np.ndarray:
    def get_absorption(spectrum):
        return np.square(-np.log(spectrum))

    def get_spectrum(absorption):
        return np.exp(-np.sqrt(absorption))

    spectra = np.transpose(spectra)

    part_one = np.matmul(get_absorption(spectra), coefs)
    '''
    if gamma != 0:
        tmp = spectra * coefs
        part_two = np.zeros(np.shape(part_one))
        for ii in range(len(coefs) - 1):
            part_two += tmp[:, ii] * np.sum(tmp[:, ii + 1:], axis=1)

        return get_spectrum(part_one + gamma * part_two)
    '''
    return get_spectrum(part_one)


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
