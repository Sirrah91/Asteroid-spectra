import numpy as np
from math import isclose
import warnings

from modules.utilities import find_nearest

# defaults only
from modules._constants import _num_eps


def check_grid(grid: np.ndarray, num_eps: float = _num_eps) -> None:
    if grid[1] < grid[0]:
        raise ValueError("Invalid grid. Ending wavelength is not larger than starting one.")

    if grid[3] < grid[0] or grid[3] > grid[1]:
        raise ValueError("Invalid grid. Normalisation is not within the grid range.")

    grid_resolution_fraction = np.mod((grid[1] - grid[0]) / grid[2], 1.0)
    if  not (isclose(grid_resolution_fraction, 0.0, abs_tol=num_eps) or
             isclose(grid_resolution_fraction, 1.0, abs_tol=num_eps)):
        warnings.warn("Warning. The grid may have incorrect resolution or limits.")


def normalise_spectrum_at_wvl(wavelength: list[float] | np.ndarray) -> float:
    # based on minimum RMSE + maximum within 10 tests (with outliers)
    where_norm = np.array([2350., 550., 1550., 1450., 650., 750., 2450., 2150., 450., 1650., 1850.,
                           2050., 2250., 1950., 1750., 850., 950., 1150., 1350., 1250., 1050.])

    wavelength = np.array(wavelength)

    for wvl in where_norm:
        if np.min(wavelength) <= wvl <= np.max(wavelength):
            return find_nearest(wavelength, wvl)

    return wavelength[0]


# pre-defined wavelengths grids (format from, to, resolution, normalised_at; nm)
data_grids = {
    "full": np.array([450., 2450., 5., 550.]),
    "Eros": np.array([820., 2360., 20., 1300.]),
    "Itokawa": np.array([820., 2080., 20., 1500.]),
}
