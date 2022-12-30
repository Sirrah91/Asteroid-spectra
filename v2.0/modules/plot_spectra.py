import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

from modules.utilities import check_dir, my_argmin, flatten_list, safe_arange
from modules.utilities_spectra import denoise_and_norm
from modules.CD_parameters import lambda_min, lambda_max, resolution_max, denoise, normalise
from modules.CD_parameters import resolution_final, num_labels_CD, normalised_at

from modules._constants import _path_figures, _path_catalogues, _path_relab_spectra, _path_data


# check if the output directory exists
check_dir(_path_figures)

start_line_number = 93  # The first line
end_line_number = 93  # The last line
following_the_spectra_catlogue = True  # Read spectrumIDs or read SampleIDs first?

rows = np.array(range(start_line_number, end_line_number + 1)) - 2

# Read the files
if following_the_spectra_catlogue:
    Sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A, C, Ag", engine="openpyxl").to_numpy()
    Spectra_catalogue = pd.read_excel("".join((_path_catalogues, "Spectra_Catalogue.xlsx")), index_col=None,
                                      na_values=["NA"], usecols="A, B, F:H", engine="openpyxl").to_numpy()[rows]

    SpectrumIDs = Spectra_catalogue[:, 0]
    SampleIDs = Spectra_catalogue[:, 1]

    # Range for interpolation
    Start = np.array(Spectra_catalogue[:, 2])
    Stop = np.array(Spectra_catalogue[:, 3])
    Step = np.array(Spectra_catalogue[:, 4])

    Weathering = flatten_list([Sample_catalogue[np.where(Sample_catalogue[:, 0] == SampleID)[0], 2]
                               for SampleID in SampleIDs])

else:
    Sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A, C, Ag", engine="openpyxl").to_numpy()[rows]
    Spectra_catalogue = pd.read_excel("".join((_path_catalogues, "Spectra_Catalogue.xlsx")), index_col=None,
                                      na_values=["NA"], usecols="A, B, F:H", engine="openpyxl").to_numpy()

    SampleIDs = np.array(Sample_catalogue[:, 0])

    # Find Spectrum ID
    SpectrumIDs = flatten_list([Spectra_catalogue[np.where(Spectra_catalogue[:, 1] == SampleID)[0], 0]
                                for SampleID in SampleIDs])

    # Range for interpolation
    Start = flatten_list([Spectra_catalogue[np.where(Spectra_catalogue[:, 1] == SampleID)[0], 2]
                          for SampleID in SampleIDs])
    Stop = flatten_list([Spectra_catalogue[np.where(Spectra_catalogue[:, 1] == SampleID)[0], 3]
                         for SampleID in SampleIDs])
    Step = flatten_list([Spectra_catalogue[np.where(Spectra_catalogue[:, 1] == SampleID)[0], 4]
                         for SampleID in SampleIDs])

    # Find samples for the spectra (this is necessary because a spectrum can have same SampleID)
    SampleIDs = flatten_list([Spectra_catalogue[np.where(Spectra_catalogue[:, 0] == SpectrumID)[0], 1]
                              for SpectrumID in SpectrumIDs])

    Weathering = flatten_list([Sample_catalogue[np.where(Sample_catalogue[:, 0] == SampleID)[0], 2]
                               for SampleID in SampleIDs])

# Take only these spectra
mask = np.array(np.where((Start <= lambda_min) & (Stop >= lambda_max) & (Step <= resolution_max))).ravel()

SpectrumIDs = SpectrumIDs[mask]
SampleIDs = SampleIDs[mask]
X = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)

# Find corresponding PIs
PIs = flatten_list([Sample_catalogue[np.where(Sample_catalogue[:, 0] == SampleID)[0], 1] for SampleID in SampleIDs])
"""
# Sorting
idx = np.argsort(SampleIDs)
SampleIDs = SampleIDs[idx]
SpectrumIDs = SpectrumIDs[idx]
PIs = PIs[idx]
"""

if denoise:
    width = 9.
    cent = int(np.round(width / 2.))
    kernel = np.zeros(width)

    for ii in range(int(np.floor(width / 2.))):
        kernel[ii] = 1. / (np.abs(ii - cent) + 1)
        kernel[-ii - 1] = 1. / (np.abs(ii - cent) + 1)
    kernel[cent] = 1.

    kernel = kernel / np.sum(kernel)
    correction = np.convolve(np.ones(len(X)), kernel, "same")

fs = 15

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
# fig.suptitle(name, fontsize=fs * 1.3)

for i in range(len(SpectrumIDs)):
    PI = PIs[i].lower()
    Sample_prefix = SampleIDs[i][:2].lower()
    SpectrumID = SpectrumIDs[i].lower()

    filename = "".join((_path_relab_spectra, PI, "/", Sample_prefix, "/", SpectrumID, ".asc"))

    if filename.endswith(".txt"):
        # skip the first line and the second line as header
        data = pd.read_csv(filename, sep="\t", index_col=None, header=[0], skiprows=[0],
                           engine="python").to_numpy()
        to_nm = 1000
    elif filename.endswith(".asc"):
        with open(filename, "r") as f:
            nlines = int(f.readline())  # First line contains info about the length of the data
            data = np.array([np.array(f.readline().split(), dtype=np.float64)[:2] for _ in range(nlines)])
            to_nm = 1

    x = data[:, 0] * to_nm  # to nm
    v = data[:, 1]

    x, idx = np.unique(x, return_index=True)
    v = v[idx]

    fun = interp1d(x, v, kind="cubic")
    v_interp = fun(X)

    v = denoise_and_norm(data=v_interp, wavelength=X, denoising=True, normalising=False, sigma_nm=7.)
    v_norm = denoise_and_norm(data=v, wavelength=X, denoising=False, normalising=True, normalised_at_wvl=normalised_at)

    if denoise:
        v = np.convolve(np.squeeze(v_interp), kernel, "same") / correction
    else:
        v = v_interp

    # Normalised reflectance
    try:
        v_norm = fun(normalised_at)
    except:
        v_norm = 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("".join(("SampleID = ", SampleIDs[i], ", SpectrumID = ", SpectrumIDs[i], ", index = ", str(i))),
                 fontsize=fs * 1.3)

    ax1.plot(X, v)
    ax2.plot(X, v_norm)

    ax1.set_xlabel("Wavelength [nm]", fontsize=fs)
    ax1.set_ylabel("Reflectance", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=fs)

    ax2.set_xlabel("Wavelength [nm]", fontsize=fs)
    ax2.set_ylabel("Reflectance [normalised at 550 nm]", fontsize=fs)
    ax2.tick_params(axis="both", labelsize=fs)
plt.show()


def compare_mixtures_1() -> None:
    # This function plots synthetic spectrum and a meteorite spectrum which is closest to it in L2-norm sense
    data_file = "".join((_path_data, "/synthetic_mixtures.dat"))
    SM = pd.read_csv(data_file, sep="\t", header=None).to_numpy()

    data_file = "".join((_path_data, "/meteorites_OC.dat"))
    M = pd.read_csv(data_file, sep="\t", header=None).to_numpy()

    data_file = "".join((_path_data, "/meteorites_OC_meta.dat"))
    meta = pd.read_csv(data_file, sep="\t", header=None).to_numpy()

    # if all labels are present...
    spectra1, numbers1 = SM[:, :-num_labels_CD], SM[:, -num_labels_CD:]
    spectra2, numbers2 = M[:, :-num_labels_CD], M[:, -num_labels_CD:]

    SampleNames = meta[:, 1]

    xq = safe_arange(lambda_min, lambda_max, resolution_final, endpoint=True)
    fs = 16

    # Copy and past from this part

    lim = 0

    plt.close("all")
    for i in range(lim, lim + 20):

        nearest = 0
        norm = 100

        for j, num2 in enumerate(numbers2):
            tmp = np.linalg.norm(numbers1[i] - num2)

            if tmp < norm:
                nearest = j
                norm = tmp

        M, m = np.max((spectra1[i], spectra2[nearest])), np.min((spectra1[i], spectra2[nearest]))

        fig, ax1 = plt.subplots(1, 1, figsize=(20, 8))

        lns1 = ax1.plot(xq, spectra1[i], label="Syntetic mixture")
        lns2 = ax1.plot(xq, spectra2[nearest], label="".join(("Meteorite - ", SampleNames[nearest])))

        ax1.set_xlabel("Wavelength [nm]", fontsize=fs)
        ax1.set_ylabel("Reflectance", fontsize=fs)
        ax1.tick_params(axis="both", labelsize=fs)
        ax1.set_ylim(bottom=0.9 * m, top=1.1 * M)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        plt.legend(fontsize=fs)

        plt.show()

    lim += 20


def compare_mixtures_2(inds: np.ndarray, coefs: np.ndarray) -> None:
    # linear combination of spectra with the given coefs

    spectra = np.zeros((len(inds), len(X)))
    numbers = np.zeros((len(inds)))

    for c, j in enumerate(inds):
        i = int(j - 2)
        PI = PIs[i].lower()
        Sample_prefix = SampleIDs[i][:2].lower()
        SpectrumID = SpectrumIDs[i].lower()

        numbers[c] = Sample_catalogue[i, 2]

        filename = "".join((_path_relab_spectra, PI, "/", Sample_prefix, "/", SpectrumID, ".asc"))

        if filename.endswith(".txt"):
            # skip the first line and the second line as header
            data = pd.read_csv(filename, sep="\t", index_col=None, header=[0], skiprows=[0],
                               engine="python").to_numpy()
            to_nm = 1000.
        elif filename.endswith(".asc"):
            with open(filename, "r") as f:
                nlines = int(f.readline())  # First line contains info about the length of the data
                data = np.array([np.array(f.readline().split(), dtype=np.float64)[:2] for _ in range(nlines)])
                to_nm = 1.

        x = data[:, 0] * to_nm  # to nm
        v = data[:, 1]

        x, idx = np.unique(x, return_index=True)
        v = v[idx]

        fun = interp1d(x, v, kind="cubic")

        spectra[c, :] = fun(X)

    final = (coefs[0] * spectra[0] + coefs[1] * spectra[1]) / np.sum(coefs)
    compare_with = spectra[2]

    title_individual = ["".join(("Fa", str(int(numbers[i])))) for i in range(len(inds))]

    title_final = "".join(("(", str(int(coefs[0])), " * ", title_individual[0], " + ", str(int(coefs[1])), " * ",
                           title_individual[1], ") / ", str(int(np.sum(coefs))), " = Fa",
                           str(int(np.sum(coefs * numbers[:2]) / np.sum(coefs)))))

    title_save = "".join((str(int(coefs[0])), title_individual[0], "_", str(int(coefs[1])), title_individual[1]))

    ###########
    # SHIFT THE TWO SPECTRA
    a1 = abs(X - my_argmin(X, final, 1100., 200.))
    a2 = abs(X - my_argmin(X, compare_with, 1100., 200.))

    x_min_1 = np.where(a1 == np.min(a1))
    x_min_2 = np.where(a2 == np.min(a2))
    y_min_1 = final[x_min_1]
    y_min_2 = compare_with[x_min_2]

    shift = y_min_2 - y_min_1
    compare_with -= shift

    title_individual[2] += "".join((" - (", str(np.round(shift[0], 4)), ")"))
    ###########

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 8))
    fs = 16

    lns1 = ax1.plot(X, final, "r-", label=title_final)
    lns2 = ax1.plot(X, compare_with, "b-", label=title_individual[2])
    lns3 = ax1.plot(X, spectra[0], "g--", label=title_individual[0])
    lns4 = ax1.plot(X, spectra[1], "g:", label=title_individual[1])

    ax1.set_xlabel("Wavelength [nm]", fontsize=fs)
    ax1.set_ylabel("Reflectance", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=fs)

    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    plt.legend(fontsize=fs)
    plt.show()
    fig.savefig("".join((_path_figures, title_save, ".png")))
