from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

from modules.utilities import my_argmin, my_argmax, argnearest


def calc_BAR_BC(wavelength: np.ndarray, reflectance: np.ndarray) -> tuple[np.ndarray, ...]:
    # find extremes around these wavelengths
    pos_max_1 = 680.
    pos_max_2 = 1500.
    pos_max_3 = 2440.

    pos_min_1 = 950.
    pos_min_2 = 1900.

    if np.max(wavelength) < 100:  # most likely in um
        print("Converting wavelengths to nm.")
        wavelength = wavelength * 1000.

    reflectance = np.reshape(reflectance, (-1, len(wavelength)))

    BIC = np.zeros(len(reflectance))
    BIIC = np.zeros(len(reflectance))
    BAR = np.zeros(len(reflectance))

    for i, spectrum in enumerate(reflectance):
        try:
            BIC[i] = my_argmin(wavelength, spectrum, x0=pos_min_1, dx=100., n_points=3)
        except Exception:
            BIC[i] = np.nan
        try:
            BIIC[i] = my_argmin(wavelength, spectrum, x0=pos_min_2, dx=200., n_points=3)
        except Exception:
            BIIC[i] = np.nan

        try:
            wvl_max_1 = my_argmax(wavelength, spectrum, x0=pos_max_1, dx=200., n_points=3)
        except Exception:
            wvl_max_1 = np.nan

        try:
            wvl_max_2 = my_argmax(wavelength, spectrum, x0=pos_max_2, dx=300., n_points=3)
        except Exception:
            wvl_max_2 = np.nan

        try:
            wvl_max_3 = my_argmax(wavelength, spectrum, x0=pos_max_3, dx=200., n_points=3)
        except Exception:
            wvl_max_3 = np.nan

        if np.isnan(wvl_max_1 + wvl_max_2 + wvl_max_3):
            BAR[i] = np.nan
            continue

        # If the founded maxima are out of wavelength range, shift them at the edge
        wvl_max_1 = np.max((wvl_max_1, np.min(wavelength)))
        wvl_max_3 = np.min((wvl_max_3, np.max(wavelength)))

        fun = interp1d(wavelength, spectrum, kind="cubic")

        # area of the first band
        # y = slope * x + const
        x1, x2 = wvl_max_1, wvl_max_2
        y1, y2 = fun(wvl_max_1), fun(wvl_max_2)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)

        line = slope * wavelength + const

        arg_wvl_start = argnearest(wavelength, wvl_max_1)[0]
        arg_wvl_stop = argnearest(wavelength, wvl_max_2)[0]

        fc = line - spectrum
        band_area_1 = simps(y=fc[arg_wvl_start:arg_wvl_stop + 1], x=wavelength[arg_wvl_start:arg_wvl_stop + 1])

        # area of the second band
        if wvl_max_3 > np.max(wavelength) or wvl_max_3 < wvl_max_2 or np.isnan(wvl_max_3):
            wvl_max_3 = np.max(wavelength)

        x1, x2 = wvl_max_2, wvl_max_3
        y1, y2 = fun(wvl_max_2), fun(wvl_max_3)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)

        line = slope * wavelength + const

        arg_wvl_start = argnearest(wavelength, wvl_max_2)[0]
        arg_wvl_stop = argnearest(wavelength, wvl_max_3)[0]

        fc = line - spectrum
        band_area_2 = simps(y=fc[arg_wvl_start:arg_wvl_stop + 1], x=wavelength[arg_wvl_start:arg_wvl_stop + 1])

        BAR[i] = band_area_2 / band_area_1

    return BAR, BIC, BIIC


def calc_composition(BAR: np.ndarray, BIC: np.ndarray, BIIC: np.ndarray, asteroid_types: np.ndarray,
                     method: str = "biic") -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # method "biic" for Gaffey, Cloutis
    # method "bic" for Reddy, Dunn

    def calc_Fs_Wo(bic: np.ndarray, biic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # https://www.researchgate.net/publication/260055905_Mineralogy_of_Asteroids
        def update_wo(fs: float, bic: float) -> float:
            if fs < 10.:
                return 347.9 * bic / 1000. - 313.6
            if 10. <= fs < 25.:
                return 456.2 * bic / 1000. - 416.9
            if 25. <= fs < 50.:
                return 418.9 * bic / 1000. - 380.9
            return 0.

        def update_fs(wo: float, biic: float) -> float:
            if wo < 11.:
                return 268.2 * biic / 1000. - 483.7
            if 11. <= wo < 30.:
                return 57.5 * biic / 1000. - 72.7
            if 30. <= wo < 45.:
                return -12.9 * biic / 1000. + 45.9
            return -118.0 * biic / 1000. + 278.5

        Fs = np.zeros(len(bic))
        Wo = np.zeros(len(bic))

        for i, (b1c, b2c) in enumerate(zip(bic, biic)):
            # itinialisation
            Fs_old, Wo_old = 25., 0.
            counter = 0

            while 1:  # do-while-like cycle
                counter += 1
                Wo_new = update_wo(Fs_old, b1c)
                Fs_new = update_fs(Wo_new, b2c)

                if np.abs(Fs_new - Fs_old) <= 1e-1 or counter > 10:
                    break
                Fs_old = Fs_new

            if Wo_old < Wo_new and counter > 10:
                Fs[i] = Fs_old
                Wo[i] = Wo_old
            else:
                Fs[i] = Fs_new
                Wo[i] = Wo_new

        return Fs, Wo

    def calc_Fs(bic: np.ndarray, biic: np.ndarray, ast_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # https://arxiv.org/pdf/1502.05008.pdf Table 2

        Fs = np.zeros(np.shape(bic))  # zero is default for A type
        Wo = np.zeros(np.shape(bic))  # zero is default for A type

        for k, b1c in enumerate(bic):
            if "S" in ast_types[k] or "Q" in ast_types[k]:
                Fs[k] = -879.1 * (b1c / 1000.) ** 2 + 1824.9 * (b1c / 1000.) - 921.7
            elif "V" in ast_types[k]:
                Fs[k] = 1023.4 * (b1c / 1000.) - 913.82
                # Fs[k] = 205.9 * (biic[k] / 1000.) - 364.3

        return Fs, Wo

    def calc_Ol(bar: np.ndarray, ast_types: np.ndarray) -> np.ndarray:

        Ol = - 0.242 * bar + 0.782  # Cloutis is default

        for j, (area, ast_type) in enumerate(zip(bar, ast_types)):
            if "S" in ast_type or "Q" in ast_type:
                # Cloutis' equation (https://arxiv.org/pdf/1502.05008.pdf Table 2)
                # based on px / (ol + px) -> ol / (ol + px) = 1 - px / (ol + px) | *100 to vol%
                # OL_fraction = 0.417 * BAR + 0.052
                # OL_fraction = 100 - 100 * OL_fraction
                # based on ol / (ol + px) | *100 to vol%
                Ol[j] = -0.242 * area + 0.782
            elif "A" in ast_type:
                Ol[j] = -11.27 * area ** 2 + 0.3012 * area + 0.956

        return Ol * 100.

    Ol = calc_Ol(BAR, asteroid_types)
    if method == "biic":
        Fs, Wo = calc_Fs_Wo(BIC, BIIC)
    else:
        Fs, Wo = calc_Fs(BIC, BIIC, asteroid_types)

    return Ol, Fs, Wo


def filter_data_mask(Ol: np.ndarray, Fs: np.ndarray, Wo: np.ndarray) -> np.ndarray:
    # filter out data outside allowed ranges
    conditions = [Ol >= 0., Ol <= 100.,
                  Fs >= 0., Fs <= 100.,
                  Wo >= 0., Wo <= 50.,
                  Fs + Wo <= 100.]

    mask = np.logical_and.reduce(conditions)

    return mask
