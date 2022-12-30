from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
from pandas.core.common import flatten
from typing import Literal, Iterable

from keras.models import Functional
from sklearn.decomposition import PCA
from scipy.stats import kde
import shutil
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules._constants import _num_eps


def check_dir(dir_or_file_path: str) -> None:
    # This function check if the directory exists. If not, the function creates it.

    dir_or_file_path = Path(dir_or_file_path)

    if dir_or_file_path.suffix:
        directory = dir_or_file_path.parent  # is file
    else:
        directory = dir_or_file_path  # is folder

    if not directory.is_dir():
        print("Directory", directory.as_posix(), "doesn't exist, creating it now.")
        directory.mkdir(parents=True, exist_ok=True)


def flatten_list(nested_list: Iterable, general: bool = False) -> np.ndarray:
    # This function flattens a list of lists
    if not general:  # works for list of lists
        return np.array([item for sub_list in nested_list for item in sub_list])
    else:  # deeply nested irregular lists, dictionaries, numpy arrays, tuples, strings, ...
        return np.array(list(flatten(nested_list)))


def stack(arrays: tuple | list, axis: int | None = None, reduce: bool = False) -> np.ndarray:
    """
    concatenate arrays along the specific axis

    if reduce=True, the "arrays" tuple is processed in this way
    arrays = (A, B, C, D)
    stack((stack((stack((A, B), axis=axis), C), axis=axis), D, axis=axis)
    This is potentially slower but allows to concatenate e.g.
    A.shape = (2, 4, 4)
    B.shape = (3, 4)
    C.shape = (4,)
    res = stack((C, B, A), axis=0)
    res.shape = (3, 4, 4)
    res[0] == stack((C, B), axis=0)
    res[1:] == A
    """

    def _stack(arrays: tuple | list, ndim: np.ndarray, axis: int | None = None) -> np.ndarray:
        if np.all(ndim == 1):  # vector + vector + ...
            if axis is None:  # -> vector
                return np.concatenate(arrays, axis=axis)
            else:  # -> 2-D array
                return np.stack(arrays, axis=axis)

        elif np.var(ndim) != 0:  # N-D array + (N-1)-D array + ... -> N-D array
            max_dim = np.max(ndim)

            # longest array
            shape = list(np.shape(arrays[np.argmax(ndim)]))
            shape[axis] = -1

            arrays = [np.reshape(a, shape) if np.ndim(a) < max_dim else a for a in arrays]
            return np.concatenate(arrays, axis=axis)

        elif is_constant(ndim):  # N-D array + N-D array + -> N-D array or (N+1)-D array
            ndim = ndim[0]
            if axis < ndim:  # along existing dimensions
                return np.concatenate(arrays, axis=axis)
            else:  # along a new dimension
                return np.stack(arrays, axis=axis)

    def _check_dims(ndim: np.ndarray, reduce: bool = False) -> None:
        error_msg = "Maximum allowed difference in dimension of concatenated arrays is one."

        if np.max(ndim) - np.min(ndim) > 1:
            if reduce:
                raise ValueError(error_msg)
            else:
                raise ValueError("".join((error_msg, "\n",
                                          'Use "reduce=True" to unlock more general (but slower) stacking.')))

    # 0-D arrays to 1-D arrays (to e.g. add a number to a vector)
    arrays = tuple([np.reshape(array, (1,)) if np.ndim(array) == 0 else array for array in arrays])

    if reduce:
        result = arrays[0]

        for array in arrays[1:-1]:
            arrays_part = (result, array)

            ndim = np.array([np.ndim(array) for array in arrays_part])
            _check_dims(ndim, reduce)
            result = _stack(arrays_part, ndim, axis)
        else:
            arrays_part = (result, arrays[-1])

            ndim = np.array([np.ndim(array) for array in arrays_part])
            _check_dims(ndim, reduce)
            return  _stack(arrays_part, ndim, axis)

    else:
        ndim = np.array([np.ndim(array) for array in arrays])
        _check_dims(ndim, reduce)
        return _stack(arrays, ndim, axis)


def is_constant(array: np.ndarray, axis: int | bool = None, constant: float = None) -> bool | np.ndarray:
    if constant is None:  # return True if the array is constant along the axis
        return np.var(array, axis=axis) < _num_eps
    else:  # return True if the array is equal to "constant" along the axis
        return np.all(np.abs(array - constant) < _num_eps, axis=axis)


def safe_arange(start: float, stop: float | None = None, step: float = 1.0, dtype: str | None = None,
                endpoint: bool = False, linspace_like: bool = True) -> np.ndarray:
    if stop is None:
        start, stop = 0.0, start

    if linspace_like:
        n = int(np.round((stop - start) / step)) + int(endpoint==True)
        return np.linspace(start, stop, n, endpoint=endpoint, dtype=dtype)

    return np.array(step * np.arange(start / step, stop / step), dtype=dtype)


def plot_me(x: np.ndarray, y: np.ndarray | None = None) -> None:
    matplotlib.use("TkAgg")  # Potentially dangerous (can change backend of the following parts)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.squeeze(x)
    
    if y is None:
        if np.ndim(np.squeeze(x)) == 1:  # line plot - x axis info is missing        
            x, y = np.arange(len(x)), x
            ax.plot(x, y)

            """
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            """

        else:  # x is matrix to plot
            y_max, x_max = np.shape(x)
            im = ax.imshow(x, origin="lower", extent=[0, x_max, 0, y_max], aspect="auto")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

    else:  # line plot
        y = np.squeeze(y)
        ax.plot(x, y)

        """
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        """

    plt.tight_layout()
    plt.show()


def my_argextreme(min_or_max: Literal["min", "max"], x: np.ndarray, y: np.ndarray, x0: float, dx: float,
                  n: int = 3) -> float:
    if min_or_max not in ["min", "max"]:
        raise ValueError('"min_or_max" must be "min" or "max".')

    # This function returns a position of local extreme of y(x) around a point x0 +- dx
    step = x[1] - x[0]

    # select in this interval
    ind = np.where(np.logical_and(x0 - dx <= x, x <= x0 + dx))
    x_int = x[ind]
    y_int = y[ind]

    # extreme value on this interval
    if min_or_max == "min":
        ix0 = np.argmin(y_int)
    else:  # must be "max"
        ix0 = np.argmax(y_int)

    x_ext = x_int[ix0 - n:ix0 + n + 1]
    y_ext = y_int[ix0 - n:ix0 + n + 1]

    params = np.polyfit(x_ext, y_ext, 2)

    # position of the extreme value
    extreme = -params[1] / (2. * params[0])

    # return extreme of parabola if it is not far (3 pixels) from the local extreme
    res = extreme if np.abs(extreme - x_int[ix0]) < 3 * step else x_int[ix0]

    return res


def my_argmin(x: np.ndarray, y: np.ndarray, x0: float, dx: float, n: int = 3) -> float:
    # This function returns a position of local minimum of y(x) around a point x0 +- dx
    return my_argextreme("min", x, y, x0, dx, n)


def my_argmax(x: np.ndarray, y: np.ndarray, x0: float, dx: float, n: int = 3) -> float:
    # This function returns a position of local maximum of y(x) around a point x0 +- dx
    return my_argextreme("max", x, y, x0, dx, n)


def best_blk(num: int) -> tuple[int, int]:
    # Function to find the best rectangle with area lower or equal to num
    # Useful for subplot layouts
    col1 = np.ceil(np.sqrt(num))
    row1 = np.ceil(num / col1)

    col2 = np.ceil(np.sqrt(num)) + 1
    row2 = np.ceil(num / col2)

    if col1 * row1 <= col2 * row2:
        row, col = int(row1), int(col1)
    else:
        row, col = int(row2), int(col2)

    return row, col


def distance(rectangle: np.ndarray, point: np.ndarray) -> np.ndarray:
    # distance(rectangle, point) computes a distance between the rectangle and the point p
    # rectangle[0, 0] = x.min, rectangle[0, 1] = x.max
    # rectangle[1, 0] = y.min, rectangle[1, 1] = y.max
    # point[0], point[1] = point.x, point.y
    dx = np.max((rectangle[0, 0] - point[0], np.zeros(np.shape(point[0])), point[0] - rectangle[0, 1]), axis=0)
    dy = np.max((rectangle[1, 0] - point[1], np.zeros(np.shape(point[1])), point[1] - rectangle[1, 1]), axis=0)

    return np.sqrt(dx * dx + dy * dy)


def my_pca(x_data: np.ndarray, n_components: int = 2, standardise: bool = False,
           return_info: bool = False) -> tuple[np.ndarray, dict[str, bool | np.ndarray | PCA]] | np.ndarray:
    # Function computes first n_components principal components

    mu = np.mean(x_data, axis=0),
    if standardise:
        std = np.std(x_data, ddof=1, axis=0)
    else:
        std = 1.

    x = (x_data - mu) / std

    pca = PCA(n_components=n_components)
    x_data_pca = pca.fit_transform(x)

    if return_info:
        info = {"standardised": standardise,
                "mean": mu,
                "std": std,
                "principal components": pca.components_,
                "eigenvalues": pca.explained_variance_,
                "model": pca}

        return x_data_pca, info
    return x_data_pca


def my_mv(source: str, destination: str, mv_or_cp: str = "mv") -> None:
    check_dir(destination)

    if mv_or_cp == "mv":
        shutil.move(source, destination)
    elif mv_or_cp == "cp":
        shutil.copy(source, destination)
    else:
        print('mv_or_cp must be either "mv" or "cp"')


def normalise_in_rows(array: np.ndarray, norm_vector: np.ndarray) -> np.ndarray:
    return np.transpose(np.divide(np.transpose(array), norm_vector))


def get_weights_from_model(model: Functional) -> dict[str, np.ndarray]:
    layer_names = np.array([layer["class_name"] for layer in model.get_config()["layers"]])
    weights = {"".join((name, "_", str(i))): model.layers[i].get_weights() for i, name in enumerate(layer_names)}

    return weights


def kernel_density_estimation_2d(y_true_part: np.ndarray, y_pred_part: np.ndarray,
                                 nbins: int = 20) -> tuple[np.ndarray, ...]:
    error = 100. * (y_pred_part - y_true_part)
    quantity = 100. * y_true_part

    k = kde.gaussian_kde((quantity, error))
    # limits: 0:100 for quantity; max error range for error
    xi, yi = np.mgrid[0:100:nbins * 1j, -50:50:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    return xi, yi, zi


def kernel_density_estimation_1d(y_true_part: np.ndarray, y_pred_part: np.ndarray,
                                 nbins: int = 20) -> tuple[np.ndarray, ...]:
    error = 100. * (y_pred_part - y_true_part)

    k = kde.gaussian_kde(error)
    # limits: 0:100 for quantity; max error range for error
    xi = np.linspace(-50, 50, nbins)
    zi = k(xi)

    return xi, zi


def round_data_with_errors(data: np.ndarray, errors: np.ndarray, n_valid: int = 2) -> tuple[np.ndarray, ...]:
    n = n_valid - np.ceil(np.log10(errors))  # rounding to n_valid numbers
    n[~np.isfinite(n)] = n_valid
    n = np.array(n, dtype=int)

    data_rounded = np.array([np.round(d, prec) for d, prec in zip(data, n)])
    errors_rounded = np.array([np.round(e, prec) for e, prec in zip(errors, n)])

    return data_rounded, errors_rounded


def replace_spaces_with_phantom(str_array: np.ndarray) -> np.ndarray:
    # replace spaces with phantom numbers
    return np.array([string.replace(" ", "\\phantom{0}") for string in str_array])
