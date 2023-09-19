import numpy as np
from math import isclose
import warnings

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


# pre-defined wavelengths grids (format from, to, resolution, normalised_at; nm)
data_grids = {
    "full": np.array([450., 2450., 5., 550.]),
    "Eros": np.array([820., 2360., 20., 1300.]),
    "Itokawa": np.array([820., 2080., 20., 1500.]),
}
