# - DOPLNIT ZISKAVANI PARAMETRU O UHLECH
# - TO DOPLNIT I DO UKLADANYCH METADAT

from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
from typing import Literal
from functools import partial

from modules.mixing_models import hapke, linear_in_absorption_coef
from modules.utilities_spectra import denoise_and_norm, save_data
from modules.utilities import my_mv, stack
from modules.decorators import timing

from modules.NN_data import load_compositional_data as load_data
from modules.CD_parameters import normalise, normalised_at, minerals_CD, endmembers_CD
from modules.NN_config import minerals_all, model_grid

from modules._constants import _path_data, _path_catalogues

@timing
def mixing_among_minerals(what_minerals: tuple[str, ...], mix_func: str = "Hapke",
                          mix_type: str = "GEN", nmax: int = 10000) -> None:
    subfolder = "mathematical_mixtures/"

    # mixing functions are in modules.mixing_models.py
    if mix_func == "Hapke":
        mixing_function = hapke
    else:
        gamma = 0.0  # a float between 0 and 1 (see Halimi et al. 2011)
        mixing_function = partial(linear_in_absorption_coef, gamma=gamma)

    num_endmembers = int(np.sum(endmembers_CD))
    num_labels = num_endmembers + int(np.sum(minerals_CD))

    # which data
    data_suffix = "-denoised"

    used_minerals = 4 * [False]

    # load the data
    OL_spec, OL_labels, OL_gs, wvl = load_spectra_v2("OL", True)
    if "OL" in what_minerals:
        used_minerals[0] = True

    OPX_spec, OPX_labels, OPX_gs = load_spectra_v2("OPX", False)
    if "OPX" in what_minerals:
        used_minerals[1] = True

    CPX_spec, CPX_labels, CPX_gs = load_spectra_v2("CPX", False)
    if "CPX" in what_minerals:
        used_minerals[2] = True

    PLG_spec, PLG_labels, PLG_gs = load_spectra_v2("PLG", False)
    if "PLG" in what_minerals:
        used_minerals[3] = True

    if not np.any(used_minerals):
        raise ValueError("No minerals to mix.")

    N_values = np.shape(OL_spec)[1]

    """
    ###############
    # if you don"t want to mix the same grain sizes, comment the part below
    grain_sizes = np.array([x for x in set(tuple(x) for x in OL_gs) & set(tuple(x) for x in OPX_gs) &
                            set(tuple(x) for x in CPX_gs) & set(tuple(x) for x in PLG_gs)])

    # delete [0, 0] grain size ([from, to] i.e. not defined)
    grain_sizes = grain_sizes[np.sum(grain_sizes, 1) > 0]

    first = True  # to allocate space for results
    
    for grain_size in grain_sizes:
        ol = OL_spec[np.sum(OL_gs == grain_size, 1) == 2]
        opx = OPX_spec[np.sum(OPX_gs == grain_size, 1) == 2]
        cpx = CPX_spec[np.sum(CPX_gs == grain_size, 1) == 2]
        plg = PLG_spec[np.sum(PLG_gs == grain_size, 1) == 2]
        
        ol_lab = OL_labels[np.sum(OL_gs == grain_size, 1) == 2]
        opx_lab = OPX_labels[np.sum(OPX_gs == grain_size, 1) == 2]
        cpx_lab = CPX_labels[np.sum(CPX_gs == grain_size, 1) == 2]
        plg_lab = PLG_labels[np.sum(PLG_gs == grain_size, 1) == 2]
        
        # indices of the spectra
        N = np.array([len(ol), len(opx), len(cpx), len(plg)])
        
        if not np.sum(N[used_minerals]) == 0:
            warnings.warn("No spectra to mix.")
            continue
        ###############
    """

    ol, opx, cpx, plg = OL_spec, OPX_spec, CPX_spec, PLG_spec
    ol_lab, opx_lab, cpx_lab, plg_lab = OL_labels, OPX_labels, CPX_labels, PLG_labels

    # indices of the spectra
    N = np.array([len(ol), len(opx), len(cpx), len(plg)])

    # iol, iopx, icpx, iplg = give_me_indices_unique_spectra(N, nmax, used_minerals)
    iol, iopx, icpx, iplg = give_me_indices_nonunique_spectra(N, nmax, used_minerals)

    synthetic_spectra = np.zeros((len(iol), N_values))
    synthetic_labels = np.zeros((len(iol), num_labels))

    # This can be done even without the for loop. For loop is probably slower but needs less memory
    for i, (i_ol, i_opx, i_cpx, i_plg) in tqdm(enumerate(zip(iol, iopx, icpx, iplg))):
        # each combination has unique coefficients in order to cover modal space
        coefs = mixing_coefficients(mix_type, used_minerals)
        # only used minerals
        coefs *= used_minerals  # this deletes only spectra and modals, chemicals are done below
        # normalise it to 1
        coefs /= np.sum(coefs)

        gs = np.array([np.mean(OL_gs[i_ol]), np.mean(OPX_gs[i_opx]), np.mean(CPX_gs[i_cpx]), np.mean(PLG_gs[i_plg])])
        gs[gs < 5.] = 5.

        spectra = stack((ol[i_ol], opx[i_opx], cpx[i_cpx], plg[i_plg]), axis=0)

        # mixing spectra
        synthetic_spectra[i] = mixing_function(spectra, coefs, gs)

        # mixing modals
        synthetic_labels[i, :-num_endmembers] = coefs

        # mixing chemicals
        synthetic_labels[i, -num_endmembers:] = (ol_lab[i_ol][-num_endmembers:] * used_minerals[0] +
                                                 opx_lab[i_opx][-num_endmembers:] * used_minerals[1] +
                                                 cpx_lab[i_cpx][-num_endmembers:] * used_minerals[2] +
                                                 plg_lab[i_plg][-num_endmembers:] * used_minerals[3]
                                                 )
    """
    ###############
    if first:
        final_synthetic = synthetic_spectra
        first = False
    else:
        final_synthetic = stack((final_synthetic, synthetic_spectra), axis=0)
    
    final_synthetic = np.unique(final_synthetic, axis=0)
    ###############
    """

    final_synthetic, inds = np.unique(synthetic_spectra, axis=0, return_index=True)
    synthetic_labels = synthetic_labels[inds]

    # spectra normalisation
    final_synthetic = denoise_and_norm(data=final_synthetic, denoising=False, wavelength=wvl, normalising=normalise,
                                       normalised_at_wvl=normalised_at)

    metadata = np.array([["synthetic data"]])
    metadata = np.repeat(metadata, len(final_synthetic), axis=0)

    sample_catalogue = pd.read_excel("".join((_path_catalogues, "Sample_Catalogue.xlsx")), index_col=None,
                                     na_values=["NA"], usecols="A:AF", header=1,
                                     sheet_name="TK", engine="openpyxl").to_numpy()

    labels_key = sample_catalogue[0, -num_labels:]

    # write into a file
    filename = "".join(("synthetic_", model_grid, "_", mix_type, "_", "_".join(what_minerals), data_suffix))
    filename = save_data(filename, spectra=final_synthetic, wavelengths=wvl, metadata=metadata, labels=synthetic_labels,
                         labels_key=labels_key, subfolder=subfolder)
    print(filename)
    my_mv(filename, filename.replace(subfolder, ""), "cp")

    return


def load_spectra(what: str, return_wavelengths: bool = False) -> tuple[np.ndarray, ...]:
    path_to_data = _path_catalogues
    # path_to_data = "/home/local/dakorda/MGM/david_mgm/results/"
    data_suffix = "-denoised"

    data_file = "".join((path_to_data, "/", what, data_suffix, ".npz"))
    data = np.load(data_file, allow_pickle=True)

    meta = data["metadata"]
    grain_size = np.array(meta[:, 9:11], dtype=np.float32)

    if return_wavelengths:
        wvl = data["wavelengths"]

        return data["spectra"], data["labels"], grain_size, wvl
    else:
        return data["spectra"], data["labels"], grain_size


def load_spectra_v2(what: Literal["OL", "OPX", "CPX", "PLG"],
                    return_wavelengths: bool = False) -> tuple[np.ndarray, ...]:
    if not np.all(minerals_all):
        raise ValueError('Unlock all minerals in "minerals_all" in NN_config.py to prevent issues with loading.')

    filename = "combined-denoised.npz"  # NO NORMALISATION
    spectra, labels, meta = load_data(filename, clean_dataset=True, return_meta=True, keep_all_labels=True)

    if what == "OL":
        inds = np.where(labels[:, 0] == 1)
    elif what == "OPX":
        inds = np.where(labels[:, 1] == 1)
    elif what == "CPX":
        inds = np.where(labels[:, 2] == 1)
    elif what == "PLG":
        inds = np.where(labels[:, 3] == 1)
    else:
        raise ValueError("Unknown mineral.")

    if return_wavelengths:
        data_file = "".join((_path_data, filename))
        data = np.load(data_file, allow_pickle=True)

        return spectra[inds], labels[inds], np.array(meta[inds][:, 9:11], dtype=np.float32), data["wavelengths"]
    else:
        return spectra[inds], labels[inds], np.array(meta[inds][:, 9:11], dtype=np.float32)


def mixing_coefficients(type_of_mixture: str, used_minerals: list[bool]) -> np.ndarray:
    rng = np.random.default_rng()

    if type_of_mixture == "OC":  # ORDINARY CHONDRITES
        fact = 2.5  # maximum deviation from sigma

        # based on selected RELAB OC
        mu, sigma = np.array([0.508, 0.305, 0.082, 0.105]), np.array([0.058, 0.063, 0.019, 0.025])
        coefs = sigma * rng.standard_normal() + mu
        coefs /= np.sum(coefs)
        while np.any(np.abs(coefs - mu) > fact * sigma):
            coefs = sigma * rng.standard_normal() + mu
            coefs /= np.sum(coefs)

    else:  # "GEN" (general) is default
        rng = np.random.default_rng()
        n_digits = len(used_minerals)

        while True:
            used_coefs = np.array([bool(int(i)) for i in "{0:0{n_digits}b}".
                                  format(rng.integers(2 ** n_digits), n_digits=n_digits)]) * np.array(used_minerals)
            if np.sum(used_coefs) > 0:
                break

        if np.sum(used_coefs) == 1:
            coefs = np.array(used_coefs, dtype=float)
        else:
            coefs = np.zeros(np.sum(used_coefs))
            for i in range(np.sum(used_coefs) - 1):
                coefs[i] = rng.uniform(0, 1 - np.sum(coefs))
            coefs[-1] = 1 - np.sum(coefs)

            coefs = np.array([coefs[int(np.sum(used_coefs[:i]))] if coef else 0 for i, coef in enumerate(used_coefs)])

            coefs[used_coefs] = rng.permutation(coefs[used_coefs])

            if np.sum(used_coefs) > 2:
                # heuristic (more uniform distribution of modals but fewer samples with intermediate coefs)
                for _ in range(1):
                    coefs *= rng.random(len(minerals_CD))

        # normalise
        coefs /= np.sum(coefs)

    return coefs


def give_me_indices_unique_spectra(N: np.ndarray, nmax: int, used_minerals: list[bool]) -> list[np.ndarray]:
    # randomly selected unique indices (no two sets of mixed spectra are the same)
    rng = np.random.default_rng()

    n = np.prod(N[used_minerals])
    nmax_gs = np.min((n, nmax))
    indices = rng.choice(n, nmax_gs, replace=False)

    inds = np.array(np.unravel_index(indices, N[used_minerals]))

    return [inds[int(np.sum(used_minerals[:i]))] if mineral else np.zeros(nmax_gs, dtype=int)
            for i, mineral in enumerate(used_minerals)]


def give_me_indices_nonunique_spectra(N: np.ndarray, nmax: int, used_minerals: list[bool]) -> list[np.ndarray]:
    # randomly selected non-unique indices (two same sets spectra can be mixed -- different modal composition)
    rng = np.random.default_rng()

    return [rng.integers(n, size=nmax) if mineral else np.zeros(nmax, dtype=int)
            for n, mineral in zip(N, used_minerals)]


if __name__ == "__main__":
    what_minerals, mix_func, mix_type, nmax = ("OL", "OPX", "CPX", "PLG"), "Hapke", "GEN", 10000

    mixing_among_minerals(what_minerals=what_minerals, mix_func=mix_func, mix_type=mix_type, nmax=nmax)
