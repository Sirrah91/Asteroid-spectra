from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from tqdm import tqdm
import numpy as np
from collections.abc import Iterator
from functools import partial

# import pandas as pd
# import warnings

from modules.mixing_models import hapke, linear_in_absorption_coef
from modules.utilities_spectra import denoise_and_norm, save_data, load_xlsx
from modules.utilities import my_mv, stack, normalise_array, safe_arange
from modules.decorators import timing

from modules.NN_data import load_composition_data as load_data
from modules._constants import _sep_in, _sep_out, _wp

# defaults only
from modules.CD_parameters import normalise, wvl_norm


@timing
def mixing_among_minerals(what_minerals: tuple[str, ...], mix_func: str = "Hapke",
                          mix_type: str = "GEN", nmax: int = 10000,
                          normalise_spectra: bool | None = None,
                          norm_at: float | None = None,
                          rnd_seed: int | None = None) -> None:
    if normalise_spectra is None: normalise_spectra = normalise
    if norm_at is None: norm_at = wvl_norm

    subfolder = "mathematical_mixtures"

    # mixing functions are in modules.mixing_models.py
    if mix_func == "Hapke":
        mixing_function = hapke
    else:
        gamma = 0.0  # a float between 0 and 1 (see Halimi et al. 2011)
        mixing_function = partial(linear_in_absorption_coef, gamma=gamma)

    count_endmembers = 2 + 3 + 3 + 3
    num_minerals = 4

    num_labels = count_endmembers + num_minerals
    used_minerals = num_minerals * [False]

    # load the data
    spectra_generator = load_spectra()
    OL_spec, OL_labels, OL_gs, wvl = next(spectra_generator)
    if "OL" in what_minerals:
        used_minerals[0] = True

    OPX_spec, OPX_labels, OPX_gs, wvl = next(spectra_generator)
    if "OPX" in what_minerals:
        used_minerals[1] = True

    CPX_spec, CPX_labels, CPX_gs, wvl = next(spectra_generator)
    if "CPX" in what_minerals:
        used_minerals[2] = True

    PLG_spec, PLG_labels, PLG_gs, wvl = next(spectra_generator)
    if "PLG" in what_minerals:
        used_minerals[3] = True

    if not np.any(used_minerals):
        raise ValueError("No minerals to mix.")

    N_values = np.shape(OL_spec)[1]

    """
    ###############
    # If you don't want to mix the same grain sizes, comment on the part below
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
        
        # Indices of the spectra
        N = np.array([len(ol), len(opx), len(cpx), len(plg)])
        
        if not np.sum(N[used_minerals]) == 0:
            warnings.warn("No spectra to mix.")
            continue
        ###############
    """

    ol, opx, cpx, plg = OL_spec, OPX_spec, CPX_spec, PLG_spec
    ol_lab, opx_lab, cpx_lab, plg_lab = OL_labels, OPX_labels, CPX_labels, PLG_labels

    # Indices of the spectra
    N = np.array([len(ol), len(opx), len(cpx), len(plg)])

    # iol, iopx, icpx, iplg = give_me_indices_unique_spectra(N, nmax, used_minerals, rnd_seed=rnd_seed)
    iol, iopx, icpx, iplg = give_me_indices_nonunique_spectra(N, nmax, used_minerals, rnd_seed=rnd_seed)

    synthetic_spectra = np.zeros((len(iol), N_values))
    synthetic_labels = np.zeros((len(iol), num_labels))

    # This can be done even without the for loop. For loop is probably slower but needs less memory
    for i, (i_ol, i_opx, i_cpx, i_plg) in tqdm(enumerate(zip(iol, iopx, icpx, iplg))):
        # Each combination has unique coefficients to cover modal space
        coefs = mixing_coefficients(mix_type, used_minerals, rnd_seed=rnd_seed)
        # Only used minerals
        coefs *= used_minerals  # This deletes only spectra and modals, chemicals are done below
        # Normalise it to 1
        coefs = normalise_array(coefs)

        gs = np.array([np.mean(OL_gs[i_ol]), np.mean(OPX_gs[i_opx]), np.mean(CPX_gs[i_cpx]), np.mean(PLG_gs[i_plg])])
        gs[gs < 5.] = 5.

        spectra = stack((ol[i_ol], opx[i_opx], cpx[i_cpx], plg[i_plg]), axis=0)

        # mixing spectra
        synthetic_spectra[i] = mixing_function(spectra, coefs, gs)

        # mixing modals
        synthetic_labels[i, :-count_endmembers] = coefs

        # mixing chemicals
        synthetic_labels[i, -count_endmembers:] = (ol_lab[i_ol][-count_endmembers:] * used_minerals[0] +
                                                 opx_lab[i_opx][-count_endmembers:] * used_minerals[1] +
                                                 cpx_lab[i_cpx][-count_endmembers:] * used_minerals[2] +
                                                 plg_lab[i_plg][-count_endmembers:] * used_minerals[3]
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
    final_synthetic = denoise_and_norm(data=final_synthetic, wavelength=wvl, denoising=False,
                                       normalising=normalise_spectra, wvl_norm_nm=norm_at)

    metadata = np.array([["synthetic data"]])
    metadata = np.repeat(metadata, len(final_synthetic), axis=0)

    sample_catalogue = load_xlsx("Sample_Catalogue.xlsx", usecols="A:AF", header=1, sheet_name="TK").to_numpy()

    labels_key = sample_catalogue[0, -num_labels:]

    # write into a file
    filename = f"synthetic{_sep_out}{mix_type}{_sep_out}{f'{_sep_in}'.join(what_minerals)}{_sep_out}denoised"
    filename = save_data(filename, spectra=final_synthetic, wavelengths=wvl, metadata=metadata, labels=synthetic_labels,
                         labels_key=labels_key, subfolder=subfolder, denoised=True, normalised=normalise_spectra)

    my_mv(filename, filename.replace(subfolder, "", 1), "cp")

    return


def load_spectra() -> Iterator[np.ndarray, ...]:
    used_minerals = np.array([True, True, True, True])  # OL, OPX, CPX, PLG
    used_endmembers = [[True, True],  # Fa, Fo; OL
                       [True, True, True],  # Fs, En, Wo; OPX
                       [True, True, True],  # Fs, En, Wo; CPX
                       [True, True, True]]  # An, Ab, Or; PLG
    grid_setup = {"model_grid": "450-2450-5-550",
                  "instrument": None,
                  "wvl_grid": safe_arange(450, 2450, 5, endpoint=True, dtype=_wp),
                  "wvl_norm": 550.}
    filtering_setup = {"use_pure_only": False,
                       "use_mix_of_the_pure_ones": False,
                       "lim_vol_part":  0.65,
                       "chem_limits": {"OL": {"Fa": 3.},
                                       "OPX": {"Fs (OPX)": 5.},
                                       "CPX": {"Fs (CPX)": 5.}},
                       "remove_high_iron_unwanted": True,
                       "keep_if_not_used": False,
                       "red_thresh": 5.
                       }

    filename = f"mineral{_sep_in}spectra{_sep_out}denoised.npz"  # NO NORMALISATION
    spectra, labels, meta, wavelengths = load_data(filename, clean_dataset=True, reinterpolation=False,
                                                   return_meta=True, return_wavelengths=True,
                                                   used_minerals=used_minerals, used_endmembers=used_endmembers,
                                                   grid_setup=grid_setup, filtering_setup=filtering_setup)


    for what in ["OL", "OPX", "CPX", "PLG"]:
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

        yield spectra[inds], labels[inds], np.array(meta[["MinSize", "MaxSize"]].iloc[inds], dtype=np.float32), wavelengths


def mixing_coefficients(type_of_mixture: str, used_minerals: list[bool], rnd_seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed=rnd_seed)

    if type_of_mixture == "OC":  # ORDINARY CHONDRITES
        fact = 2.5  # maximum deviation from sigma

        # based on selected RELAB OC
        mu, sigma = np.array([0.508, 0.305, 0.082, 0.105]), np.array([0.058, 0.063, 0.019, 0.025])
        while True:
            coefs = rng.normal(loc=mu, scale=sigma)
            coefs = normalise_array(coefs)
            if np.all(np.abs(coefs - mu) <= fact * sigma):
                break

    else:  # "GEN" (general) is the default
        n_digits = len(used_minerals)

        while True:
            used_coefs = np.array([bool(int(i))
                                   for i in f"{rng.integers(2 ** n_digits):0{n_digits}b}"]) * np.array(used_minerals)
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
                    coefs *= rng.random(len(coefs))

        # normalise
        coefs = normalise_array(coefs)

    return coefs


def give_me_indices_unique_spectra(N: np.ndarray, nmax: int, used_minerals: list[bool],
                                   rnd_seed: int | None = None) -> list[np.ndarray]:
    # randomly selected unique indices (no two sets of mixed spectra are the same)
    rng = np.random.default_rng(seed=rnd_seed)

    n = np.prod(N[used_minerals])
    nmax_gs = np.min((n, nmax))
    indices = rng.choice(n, nmax_gs, replace=False)

    inds = np.array(np.unravel_index(indices, N[used_minerals]))

    return [inds[int(np.sum(used_minerals[:i]))] if mineral else np.zeros(nmax_gs, dtype=int)
            for i, mineral in enumerate(used_minerals)]


def give_me_indices_nonunique_spectra(N: np.ndarray, nmax: int, used_minerals: list[bool],
                                      rnd_seed: int | None = None) -> list[np.ndarray]:
    # randomly selected non-unique indices (two same sets spectra can be mixed -- different modal composition)
    rng = np.random.default_rng(seed=rnd_seed)

    return [rng.integers(n, size=nmax) if mineral else np.zeros(nmax, dtype=int)
            for n, mineral in zip(N, used_minerals)]


if __name__ == "__main__":
    what_minerals, mix_func, mix_type, nmax = ("OL", "OPX", "CPX", "PLG"), "Hapke", "GEN", 10000

    mixing_among_minerals(what_minerals=what_minerals, mix_func=mix_func, mix_type=mix_type, nmax=nmax,
                          normalise_spectra=normalise, norm_at=wvl_norm, rnd_seed=None)
