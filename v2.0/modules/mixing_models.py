# based on https://www.researchgate.net/publication/326921440_Hapke-based_computational_method_to_enable_unmixing_of_hyperspectral_data_of_common_salts

import numpy as np
from scipy.optimize import brentq

from modules._constants import _num_eps

incoming_angle = 0.0
emmerging_angle = 30.0
phase_angle = 30.0

Dl_Ds = 250. / 100.  # largest / smallest grain
phi = 0.0  # filling factor
b, c = -0.4, 0.25

mu0 = np.cos(incoming_angle * np.pi / 180.)
mu = np.cos(emmerging_angle * np.pi / 180.)
g = phase_angle * np.pi / 180.


def ch(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculates Chandrasekhar H function
    w - single scatter albedo
    x = mu or mu_0 / K
    """
    y = np.sqrt(1. - w)
    r0 = (1. - y) / (1. + y)
    return 1. / (1. - w * x * (r0 + (1. - 2. * r0 * x) / 2. * np.log((1. + x) / x)))


def reflect(w: np.ndarray, mu_0: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Calculates reflectance of the mixture
    w - single scatter albedo
    mu - cosine of incidence angle
    mu_0 cosine of the emergence angle
    """

    if phi > _num_eps:
        K = - np.log(1. - 1.209 * phi ** (2. / 3.)) / (1.209 * phi ** (2. / 3.))
    else:
        K = 1.

    h_mu0 = ch(w, mu_0 / K)
    h_mu = ch(w, mu / K)

    P = 1. + b * np.cos(g) + c * (1.5 * np.cos(g) ** 2 - 0.5)

    hs = 3. * np.sqrt(3.) / 8. * K * phi / np.log(Dl_Ds)
    if hs > _num_eps:
        B = 1. / (1. + np.tan(g / 2.) / hs)
    else:
        B = 0.

    return K * w / 4. / (mu_0 + mu) * (P * (1. + B) + h_mu0 * h_mu - 1.)


def residual_reflect(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Minimization fitting function for extracting w values
    params : w, inc_angle, emr_angle
    r : reflectance data to subtract
    """
    return reflect(w, mu0, mu) - r


def extract_ssa(r_data: np.ndarray):
    w = np.zeros_like(r_data)
    for i, data in enumerate(r_data):
        try:  # only if r = 1 or is very close to it -> w = 1
            w[i] = brentq(residual_reflect, 0., 1., args=(data))
        except:
            w[i] = 1.
    return w


def hapke(spectra: np.ndarray, coefs: np.ndarray, grain_sizes: np.ndarray | None = None) -> np.ndarray:
    if grain_sizes is None:
        grain_sizes = np.ones(np.shape(coefs))

    if not (len(spectra) == len(coefs) == len(grain_sizes)):
        raise ValueError("Dimensions are not consistent. Check inputs.")

    w_mix = np.zeros(len(spectra[0]))

    normalisation = np.sum(coefs / grain_sizes)

    inds = np.where(coefs > 0)[0]

    if len(inds) > 1:
        for i in inds:
            w = extract_ssa(spectra[i])
            w_mix += coefs[i] * w / grain_sizes[i]
        w_mix /= normalisation

        return reflect(w_mix, mu0, mu)
    else:
        return spectra[inds].ravel()


def linear_in_absorption_coef(spectra: np.ndarray, coefs: np.ndarray,
                              grain_size: np.ndarray | None = None, gamma: float = 0.0) -> np.ndarray:
    # based on Halimi et al. 2011 model

    def get_absorption(spectrum: np.ndarray) -> np.ndarray:
        return np.square(-np.log(spectrum))

    def get_spectrum(absorption: np.ndarray) -> np.ndarray:
        return np.exp(-np.sqrt(absorption))

    inds = np.where(coefs > 0)[0]

    if len(inds) > 1:
        absorption_coefs = get_absorption(np.transpose(spectra))

        part_one = absorption_coefs @ coefs

        if gamma != 0:
            tmp = absorption_coefs * coefs
            part_two = np.sum([tmp[:, i] * np.sum(tmp[:, i + 1:], axis=1) for i in range(len(coefs) - 1)], axis=0)

            return get_spectrum(part_one + gamma * part_two)

        return get_spectrum(part_one)
    else:
        return spectra[inds].ravel()
