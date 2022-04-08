# based on https://www.researchgate.net/publication/326921440_Hapke-based_computational_method_to_enable_unmixing_of_hyperspectral_data_of_common_salts

import numpy as np
from scipy.optimize import brentq

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
            w[i] = brentq(residual_reflect, 0., 1, args=(r_data[i]))
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
