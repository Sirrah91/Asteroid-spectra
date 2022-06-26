from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from scipy.interpolate import interp1d
from scipy.integrate import simps

from modules.utilities import *


def calc_BAR_BC(wavelength, reflectance):
    # find extremes around these wavelengths
    pos_max_1 = 680
    pos_max_2 = 1500
    pos_max_3 = 2440

    pos_min_1 = 950
    pos_min_2 = 1900

    BIC = np.zeros(len(reflectance))
    BIIC = np.zeros(len(reflectance))
    BAR = np.zeros(len(reflectance))

    dx = wavelength[1] - wavelength[0]

    for i in range(len(reflectance)):
        try:
            BIC[i] = my_argmin(wavelength, reflectance[i], pos_min_1, dx=100, n=3)
        except:
            BIC[i] = np.nan
        try:
            BIIC[i] = my_argmin(wavelength, reflectance[i], pos_min_2, dx=200, n=3)
        except:
            BIIC[i] = np.nan

        try:
            wvl_max_1 = my_argmax(wavelength, reflectance[i], pos_max_1, dx=200, n=3)
        except:
            wvl_max_1 = np.nan

        try:
            wvl_max_2 = my_argmax(wavelength, reflectance[i], pos_max_2, dx=300, n=3)
        except:
            wvl_max_2 = np.nan

        try:
            wvl_max_3 = my_argmax(wavelength, reflectance[i], pos_max_3, dx=200, n=3)
        except:
            wvl_max_3 = np.nan
        '''
        if np.isnan(wvl_max_1 + wvl_max_2 + wvl_max_3):
            BAR[i] = np.nan
            continue
        '''

        fun = interp1d(wavelength, reflectance[i], kind='cubic')

        # area of the first band
        # y = slope * x + const
        x1, x2 = wvl_max_1, wvl_max_2
        y1, y2 = fun(wvl_max_1), fun(wvl_max_2)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)

        line = slope * wavelength + const

        wvl_start = np.where(wavelength == np.round(wvl_max_1 / dx) * dx.astype(int))[0][0]
        wvl_stop = np.where(wavelength == np.round(wvl_max_2 / dx) * dx.astype(int))[0][0]

        fc = line - reflectance[i]
        band_area_1 = simps(fc[wvl_start:wvl_stop + 1], dx=dx)

        # area of the second band
        if wvl_max_3 > np.max(wavelength) or wvl_max_3 < wvl_max_2 or np.isnan(wvl_max_3):
            wvl_max_3 = np.max(wavelength)

        x1, x2 = wvl_max_2, wvl_max_3
        y1, y2 = fun(wvl_max_2), fun(wvl_max_3)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)

        line = slope * wavelength + const

        wvl_start = np.where(wavelength == np.round(wvl_max_2 / dx) * dx.astype(int))[0][0]
        wvl_stop = np.where(wavelength == np.round(wvl_max_3 / dx) * dx.astype(int))[0][0]

        fc = line - reflectance[i]
        band_area_2 = simps(fc[wvl_start:wvl_stop + 1], dx=dx)

        BAR[i] = band_area_2 / band_area_1

    return BAR, BIC, BIIC


def calc_composition(BAR, BIC, BIIC, asteroid_types, method=0):
    # method 0 for Gaffey, Cloutis
    # method 1 for Reddy, Dunn

    def calc_Fs_Wo(bic, biic):
        # https://www.researchgate.net/publication/260055905_Mineralogy_of_Asteroids
        def update_wo(fs, bic):
            if fs < 10:
                return 347.9 * bic / 1000 - 313.6
            if 10 <= fs < 25:
                return 456.2 * bic / 1000 - 416.9
            if 25 <= fs < 50:
                return 418.9 * bic / 1000 - 380.9
            return 0

        def update_fs(wo, biic):
            if wo < 11:
                return 268.2 * biic / 1000 - 483.7
            if 11 <= wo < 30:
                return 57.5 * biic / 1000 - 72.7
            if 30 <= wo < 45:
                return -12.9 * biic / 1000 + 45.9
            return -118.0 * biic / 1000 + 278.5

        Fs = np.zeros(len(bic))
        Wo = np.zeros(len(bic))

        for i in range(len(bic)):
            # itinialisation
            Fs_new, Wo_old = 25, 0
            Fs_old = 0
            counter = 0

            while np.abs(Fs_new - Fs_old) > 1e-1:
                counter += 1
                Fs_old = Fs_new
                Wo_new = update_wo(Fs_old, bic[i])
                Fs_new = update_fs(Wo_new, biic[i])
                if counter > 10:
                    break
            if Wo_old < Wo_new and counter > 10:
                Fs[i] = Fs_old
                Wo[i] = Wo_old
            else:
                Fs[i] = Fs_new
                Wo[i] = Wo_new

        return Fs, Wo

    def calc_Fs(bic, biic, ast_types):
        # https://arxiv.org/pdf/1502.05008.pdf Table 2

        Fs = np.zeros(np.shape(bic))  # zero is default for A type
        Wo = np.zeros(np.shape(bic))  # zero is default for A type

        for k in range(len(bic)):
            if 'S' in ast_types[k] or 'Q' in ast_types[k]:
                Fs[k] = -879.1 * (bic[k] / 1000) ** 2 + 1824.9 * (bic[k] / 1000) - 921.7
            elif 'V' in ast_types[k]:
                Fs[k] = 1023.4 * (bic[k] / 1000) - 913.82
                # Fs[k] = 205.9 * (biic[k] / 1000) - 364.3

        return Fs, Wo

    def calc_Ol(bar, ast_types):

        Ol = - 0.242 * bar + 0.782  # Cloutis is defauls

        for j in range(len(bar)):
            if 'S' in ast_types[j] or 'Q' in ast_types[j]:
                # Cloutis' equation (https://arxiv.org/pdf/1502.05008.pdf Table 2)
                # based on px / (ol + px) -> ol / (ol + px) = 1 - px / (ol + px) | *100 to vol%
                # OL_fraction = 0.417 * BAR + 0.052
                # OL_fraction = 100 - 100 * OL_fraction
                # based on ol / (ol + px) | *100 to vol%
                Ol[j] = -0.242 * bar[j] + 0.782
            elif 'A' in ast_types[j]:
                Ol[j] = -11.27 * bar[j] ** 2 + 0.3012 * bar[j] + 0.956

        return Ol * 100

    Ol = calc_Ol(BAR, asteroid_types)
    if method == 0:
        Fs, Wo = calc_Fs_Wo(BIC, BIIC)
    else:
        Fs, Wo = calc_Fs(BIC, BIIC, asteroid_types)

    return Ol, Fs, Wo


def filter_data_mask(Ol, Fs, Wo):
    # filter out data outside of allowed ranges

    mask1 = np.logical_and(Ol >= 0, Ol <= 100)  # OL fraction between 0 and 100 percent
    mask2 = np.logical_and(Fs >= 0, Fs <= 100)  # Fs between 0 and 100
    mask3 = np.logical_and(Wo >= 0, Wo <= 50)  # Wo between 0 and 50
    mask4 = Fs + Wo <= 100  # Fs + Wo <= 100

    mask = mask1 * mask2 * mask3 * mask4

    return mask
