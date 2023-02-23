import numpy as np
from typing import Literal

from modules.utilities import stack, safe_arange
from modules.NN_config import *

show_result_plot = False
show_control_plot = False
verb = 0

model_basedir = "range_test/"

instrument: Literal["ASPECT", None] = "ASPECT"
constant_range_or_spacing: Literal["spacing", "range", "window"] = "window"

if constant_range_or_spacing == "spacing":
    spacing = 10  # nm
    instrument = None

    model_subdir = "".join((model_basedir, "/spacing_", str(int(spacing)), "_nm/"))

    # wavelength ranges in nm
    start, stop, step = 450, 2450, 100

    from_wvl_all = safe_arange(start, stop, step)
    to_wvl_all = from_wvl_all + step

    ranges = [(from_wvl_all[j], to_wvl_all[j + i]) for j in range(len(from_wvl_all))
              for i in range(len(to_wvl_all[j:]))]

    num_ranges = len(ranges)

    spacing = spacing * np.ones(len(ranges))
elif constant_range_or_spacing == "range":
    approx_resolution = np.array([30, 35, 40, 45, 50])  # nm

    # wavelength ranges in nm
    start = 650
    stop = 2500

    model_subdir = "".join((model_basedir, "/range_", str(int(start)), "_", str(int(stop)), "_nm/"))

    # no. points of the input data
    n_points = 1 + (stop - start) / approx_resolution

    # floor or ceil?
    for i, n in enumerate(n_points):
        f, c = int(np.floor(n)), int(np.ceil(n))
        tmp_f, tmp_c = np.linspace(start, stop, f), np.linspace(start, stop, c)
        step_f, step_c = tmp_f[1] - tmp_f[0], tmp_c[1] - tmp_c[0]

        if np.abs(step_f - approx_resolution[i]) < np.abs(step_c - approx_resolution[i]):
            n_points[i] = np.floor(n)
        else:
            n_points[i] = np.ceil(n)

    n_points = np.array(n_points, dtype=int)

    num_ranges = len(n_points)

    # real resolution
    spacing = np.zeros(np.shape(n_points))
    for i, n in enumerate(n_points):
        tmp = np.linspace(start, stop, n)
        spacing[i] = np.round(tmp[1] - tmp[0], 5)

    ranges = [(start, stop) for _ in range(num_ranges)]
    spacing = spacing * np.ones(len(ranges))

    wvl_all = [np.linspace(start, stop, n) for n in n_points]
else:
    # wavelength ranges in nm
    start, stop, _spacing = 600, 1600, 40  # nm

    window_size = 200  # nm
    window_spacing = 20  # nm

    model_subdir = "".join((model_basedir, "/range_", str(int(start)), "_", str(int(stop)), "_window_",
                         str(int(window_size)), "_nm/"))

    wvl_all = np.array([stack((safe_arange(start, window_start, _spacing),
                               safe_arange(window_start, window_start + window_size, window_spacing),
                               safe_arange(window_start + window_size, stop + _spacing, _spacing)))
                        for window_start in safe_arange(start, stop, _spacing) if window_start + window_size <= stop])

    num_ranges = len(wvl_all)

    ranges = [(start, stop) for _ in range(num_ranges)]

    spacing = _spacing * np.ones(len(ranges))
