import numpy as np
from os import path
from typing import Literal
from functools import reduce

from modules.utilities_spectra import normalise_spectrum_at_wvl
from modules.utilities import stack, find_nearest, safe_arange
from modules._constants import _sep_in, _sep_out

from modules.NN_config_composition import p, bin_code, minerals_used, endmembers_used, comp_model_setup

minerals_used_range, endmembers_used_range = minerals_used, endmembers_used
model_subdir_range = comp_model_setup["model_subdir"]
monitoring = comp_model_setup["monitoring"]

model_basedir = "range_test"

constant_range_or_spacing: Literal["spacing", "range", "window", "normalisation"] = "range"
instrument = "ASPECT_vis-nir1-nir2-swir"

if constant_range_or_spacing != "range":
    instrument = None

if constant_range_or_spacing == "spacing":
    spacing = 10.  # spacing of individual model in nm
    start, stop, step = 450., 2450., 100.  # lowest and highest wavelength and shift in wavelengths in nm

    # define ranges
    from_wvl_all = safe_arange(start, stop, step)
    to_wvl_all = from_wvl_all + step

    ranges = [(from_wvl_all[j], to_wvl_all[j + i]) for j in range(len(from_wvl_all))
              for i in range(len(to_wvl_all[j:]))]

    num_ranges = len(ranges)

    # vector spacing
    spacing = spacing * np.ones(num_ranges)

    # all wavelength grids and normalisations
    wvl_all = [safe_arange(wvl_range[0], wvl_range[1], spac, endpoint=True) for wvl_range, spac in zip(ranges, spacing)]
    norm_at = [normalise_spectrum_at_wvl(wvl_range) for wvl_range in wvl_all]

    # model names
    model_grids = [_sep_in.join(str(int(x)) for x in np.round([wvl_range[0], wvl_range[1], spac, norm]))
                   for wvl_range, spac, norm in zip(ranges, spacing, norm_at)]
    instruments = [None for _ in range(num_ranges)]

elif constant_range_or_spacing == "range":
    # wavelength ranges in nm
    approx_resolution = np.array([10, 20, 30, 40, 50], dtype=int)  # nm
    norm_at = 1550.

    start = 650.

    if instrument is None:
        stop = start + 2. * np.lcm.reduce(approx_resolution)

    elif "swir" in instrument:  # ASPECT
        stop = 2450.
    else:  # ASPECT
        stop = 1650.

    # define ranges
    num_ranges = len(approx_resolution)
    ranges = [(start, stop) for _ in range(num_ranges)]

    # all wavelength grids and normalisations
    wvl_all = [safe_arange(start, stop, res, endpoint=True) for res in approx_resolution]
    common_wvl = reduce(np.intersect1d, [*wvl_all])
    norm_at = [find_nearest(common_wvl, norm_at) for _ in range(num_ranges)]

    # vector spacing
    spacing = np.round([wvl[1] - wvl[0] for wvl in wvl_all], 5)

    if instrument is None:
        model_grids = [_sep_in.join(str(int(x)) for x in np.round([wvl_range[0], wvl_range[1], spac, norm]))
                       for wvl_range, spac, norm in zip(ranges, spacing, norm_at)]
        instruments = [None for _ in range(num_ranges)]
    else:
        model_grids = [_sep_out.join((instrument, str(int(res)))) for res in approx_resolution]
        instruments = model_grids

elif constant_range_or_spacing == "window":
    # wavelength ranges and spacing outside the window in nm
    start, stop, _spacing = 600., 1600., 40.  # nm

    window_size = 200.  # nm
    window_spacing = 20.  # nm

    window_starts = safe_arange(start, stop, _spacing)
    window_ends = window_starts + window_size

    inds = window_ends <= stop
    window_starts, window_ends = window_starts[inds], window_ends[inds]

    # all wavelength grids
    wvl_all = np.array([stack((safe_arange(start, window_start, _spacing),  # preceding window
                               safe_arange(window_start, window_end, window_spacing),  # inside window
                               safe_arange(window_end, stop + _spacing, _spacing)))  # following window
                        for window_start, window_end in zip(window_starts, window_ends)])

    # define ranges
    num_ranges = len(wvl_all)
    ranges = [(start, stop) for _ in range(num_ranges)]

    # normalisations
    common_wvl = reduce(np.intersect1d, [*wvl_all])
    norm_at = [find_nearest(common_wvl, 1550.) for _ in range(num_ranges)]

    # vector spacing
    spacing = _spacing * np.ones(num_ranges)

    # model names
    model_grids = [f"{int(ranges[i][0])}({int(window_starts[i])}){_sep_in}{int(ranges[i][1])}({int(window_ends[i])})" \
                   f"{_sep_in}{int(spacing[i])}({int(window_spacing)})" \
                   f"{_sep_in}{int(norm_at[i])}"
                   for i in range(num_ranges)]
    instruments = [None for _ in range(num_ranges)]

elif constant_range_or_spacing == "normalisation":
    # test of different normalisation
    # wavelength ranges and spacing outside the window in nm
    start, stop, step = 450, 2450., 10.  # nm
    norm_step = 100.  # nm

    # all wavelength grids
    wvl_all = safe_arange(start, stop, step, endpoint=True)

    # normalisations
    norm_at = [find_nearest(wvl_all, norm) for norm in safe_arange(start, stop, norm_step, endpoint=True)]
    wvl_all = np.repeat(wvl_all[None, :], len(norm_at), axis=0)

    # define ranges
    num_ranges = len(wvl_all)
    ranges = [(start, stop) for _ in range(num_ranges)]

    # vector spacing
    spacing = step * np.ones(num_ranges)

    # model names
    model_grids = [_sep_in.join(str(int(x)) for x in np.round([wvl_range[0], wvl_range[1], spac, norm]))
                   for wvl_range, spac, norm in zip(ranges, spacing, norm_at)]
    instruments = [None for _ in range(num_ranges)]

else:
    raise ValueError('Unknown "constant_range_or_spacing" settings')

if instrument is not None and "ASPECT" not in instrument:
    raise ValueError('Unknown instrument. Must be "None" or contain "ASPECT".')

# model subdirs
model_subdirs = [path.join(model_basedir, constant_range_or_spacing, model_grid) for model_grid in model_grids]

# model names
model_names = [f"{p['model_type']}{_sep_out}{model_grid}{_sep_out}{bin_code}" for model_grid in model_grids]

# data ranges
range_grids = [{"model_grid": model_grids[index],
                "instrument": instruments[index],
                "wvl_grid": wvl_all[index],
                "wvl_norm": norm_at[index]} for index in range(len(model_grids))]
