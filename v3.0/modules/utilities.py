from os import environ, path
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import re
from pandas.core.common import flatten
from typing import Literal, Iterable, Callable
from time import time
from keras.models import Functional
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import gaussian_kde, kendalltau, pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
import shutil
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tracemalloc
from linecache import getline

from modules.decorators import reduce_like
from modules._constants import _sep_out, _sep_in

# defaults only
from modules._constants import _num_eps, _rnd_seed


def check_dir(dir_or_file_path: str) -> None:
    # This function checks if the directory exists. If not, the function creates it.

    dir_or_file_path = Path(dir_or_file_path)

    if dir_or_file_path.suffix:
        directory = dir_or_file_path.parent  # is file
    else:
        directory = dir_or_file_path  # is folder

    if not directory.is_dir():
        print(f'Directory "{directory.as_posix()}" does not exist, creating it now.')
        directory.mkdir(parents=True, exist_ok=True)


def check_file(filename: str, base_folder: str, subfolder: str) -> str:
    if path.exists(filename):
        pass
    elif path.exists(path.join(base_folder, subfolder, filename)):
        filename = path.join(base_folder, subfolder, filename)
    else:
        raise FileNotFoundError(f"The file {filename} was not found.")

    return path.abspath(filename)


def flatten_list(nested_list: Iterable, general: bool = False) -> np.ndarray:
    # This function flattens a list of lists
    if not general:  # works for a list of lists
        return np.array([item for sub_list in nested_list for item in sub_list])
    else:  # deeply nested irregular lists, dictionaries, numpy arrays, tuples, strings, ...
        return np.array(list(flatten(nested_list)))


def stack(arrays: tuple | list, axis: int | None = None, reduce: bool = False) -> np.ndarray:
    """
    concatenate arrays along the specific axis

    if reduce=True, the "arrays" tuple is processed in this way
    arrays = (A, B, C, D)
    stack((stack((stack((A, B), axis=axis), C), axis=axis), D), axis=axis)
    This is potentially slower but allows for concatenating e.g.
    A.shape = (2, 4, 4)
    B.shape = (3, 4)
    C.shape = (4,)
    res = stack((C, B, A), axis=0, reduce=True)
    res.shape = (3, 4, 4)
    res[0] == stack((C, B), axis=0)
    res[1:] == A
    """

    @reduce_like
    def _stack(arrays: tuple | list, axis: int | None = None) -> np.ndarray:
        ndim = np.array([np.ndim(array) for array in arrays])
        _check_dims(ndim, reduce)

        if np.all(ndim == 1):  # vector + vector + ...
            if axis is None:  # -> vector
                return np.concatenate(arrays, axis=axis)
            else:  # -> 2-D array
                return np.stack(arrays, axis=axis)

        elif np.var(ndim) != 0:  # N-D array + (N-1)-D array + ... -> N-D array
            max_dim = np.max(ndim)

            # longest array
            shape = np.array(np.shape(arrays[np.argmax(ndim)]))
            shape[axis] = -1

            # reshape is dangerous; you can potentially stack e.g. 10x1 with 2x5x2 along axis=0 that is confusing
            # possible dimension difference is one; omit the -1 shape. The rest should be equal.
            shapes = [np.array(np.shape(array)) for array in arrays if np.ndim(array) < max_dim]
            if not np.all([sh in shape[shape > 0] for sh in shapes]):
                raise ValueError("Arrays of these dimensions cannot be stacked.")

            arrays = [np.reshape(array, shape) if np.ndim(array) < max_dim else array for array in arrays]

            return np.concatenate(arrays, axis=axis)

        elif is_constant(ndim):  # N-D array + N-D array + ... -> N-D array or (N+1)-D array
            ndim = ndim[0]
            if axis < ndim:  # along existing dimensions
                return np.concatenate(arrays, axis=axis)
            else:  # along a new dimension
                return np.stack(arrays, axis=axis)

    def _check_dims(ndim: np.ndarray, reduce: bool = False) -> None:
        error_msg = ("Maximum allowed difference in dimension of concatenated arrays is one. "
                     "If you want to stack along higher dimensions, use a combination of stack and np.reshape.")

        if np.max(ndim) - np.min(ndim) > 1:
            if reduce:
                raise ValueError(error_msg)
            else:
                raise ValueError(f'{error_msg}\nUse "reduce=True" to unlock more general (but slower) stacking.')

    # 0-D arrays to 1-D arrays (e.g. add a number to a vector)
    arrays = [np.reshape(array, (1,)) if np.ndim(array) == 0 else np.array(array) for array in arrays]
    arrays = tuple([array for array in arrays if np.size(array) > 0])
    if len(arrays) == 0: arrays = (np.array([], dtype=int),)  # enable to stack(np.array([]))

    if reduce:
        return _stack.reduce(arrays, axis)
    else:
        return _stack(arrays, axis)


def return_mean_std(array: np.ndarray, axis: int | None = None,
                    ignore_nan: bool = True) -> tuple[np.ndarray, ...] | tuple[float, ...]:
    if ignore_nan:
        mean_value = np.nanmean(array, axis=axis)
    else:
        mean_value = np.mean(array, axis=axis)

    if axis is None:
        ddof = np.min((np.size(array) - 1, 1))
    else:
        ddof = np.min((np.shape(array)[axis] - 1, 1))

    if ignore_nan:
        std_value = np.nanstd(array, axis=axis, ddof=ddof)
    else:
        std_value = np.std(array, axis=axis, ddof=ddof)

    return mean_value, std_value


def my_polyfit(x: np.ndarray | list[float], y: np.ndarray | list[float], deg: int,
               method: Literal["huber", "ransac", "theilsen", "numpy"] = "ransac", rnd_seed: int | None = _rnd_seed,
               return_fit: bool = False, return_fit_only: bool = False,
               x_fit: np.ndarray | None = None) -> np.ndarray | tuple[np.ndarray, ...]:

    if len(y) < deg + 1:
        warnings.warn("Polyfit may be poorly conditioned")

    if len(y) <= deg + 1:
        p = np.polyfit(x, y, deg)
    else:
        if method == "huber":
            p = huber_fit(x, y, deg)

        elif method == "theilsen":
            p = theilsen_fit(x, y, deg, rnd_seed=rnd_seed)

        elif method == "ransac":
            p = ransac_fit(x, y, deg, rnd_seed=rnd_seed)

        elif method == "numpy":
            p = np.polyfit(x, y, deg)

        else:
            warnings.warn('Unknown method. Implemented possibilities are "huber", "ransac", "theilsen", and "numpy". '
                          'Using ransac.')
            p = ransac_fit(x, y, deg, rnd_seed=rnd_seed)

    if return_fit or return_fit_only:
        if x_fit is None:
            x_fit = x
        y_fit = np.polyval(p, x_fit)

    if return_fit:
        return p, y_fit
    if return_fit_only:
        return y_fit
    return p


def ransac_fit(x: np.ndarray | list[float], y: np.ndarray | list[float], deg: int,
               rnd_seed: int | None = _rnd_seed) -> np.ndarray:
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(np.reshape(x, (-1, 1)))

    ransac = RANSACRegressor(random_state=rnd_seed)
    ransac.fit(x_poly, y)

    p = ransac.estimator_.coef_
    p[0] += ransac.estimator_.intercept_
    return p[::-1]


def huber_fit(x: np.ndarray | list[float], y: np.ndarray | list[float], deg: int) -> np.ndarray:
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(np.reshape(x, (-1, 1)))

    huber = HuberRegressor()
    huber.fit(x_poly, y)

    p = huber.coef_
    p[0] += huber.intercept_
    return p[::-1]


def theilsen_fit(x: np.ndarray | list[float], y: np.ndarray | list[float], deg: int,
                 rnd_seed: int | None = _rnd_seed) -> np.ndarray:
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(np.reshape(x, (-1, 1)))

    theilsen = TheilSenRegressor(random_state=rnd_seed)
    theilsen.fit(x_poly, y)

    p = theilsen.coef_
    p[0] += theilsen.intercept_
    return p[::-1]


def is_constant(array: np.ndarray | list| float, constant: float | None = None, axis: int | None = None,
                atol: float = _num_eps) -> bool | np.ndarray:
    if atol < 0.:
        raise ValueError('"atol" must be a non-negative number')

    array = np.array(array, dtype=float)

    if np.ndim(array) == 0:
        array = array[np.newaxis]

    if constant is None:  # return True if the array is constant along the axis
        # ddof = 1 for single value gives std = NaN, but I want std = 0
        if axis is None:
            ddof = np.min((np.size(array) - 1, 1))
        else:
            ddof = np.min((np.shape(array)[axis] - 1, 1))

        return np.std(array, axis=axis, ddof=ddof) < atol

    else:  # return True if the array is equal to "constant" along the axis
        return np.all(np.abs(array - constant) < atol, axis=axis)


def safe_arange(start: float, stop: float | None = None, step: float = 1.0, dtype: type = float,
                endpoint: bool = False, linspace_like: bool = True, num_eps: float = _num_eps) -> np.ndarray:
    if stop is None:
        start, stop = 0.0, start

    start, stop, step = float(start), float(stop), float(step)

    one_over_step = step ** (-1)  # step**(-1) is better in rounding

    if linspace_like:
        n = (stop - start) * one_over_step + float(endpoint == True)

        # to select the step which is closest to the targeted
        step_floor = (stop - start) / (np.floor(n) - 1.)
        step_ceil = (stop - start) / (np.ceil(n) - 1.)
        if np.abs(step_floor - step) < np.abs(step_ceil - step):
            n = int(np.floor(n))
        else:
            n = int(np.ceil(n))

        return np.linspace(start, stop, n, endpoint=endpoint, dtype=dtype)

    return np.array(step * np.arange(start * one_over_step,
                                     (stop + num_eps * np.sign(2 * int(endpoint == True) - 1)) * one_over_step),
                    dtype=dtype)


def argnearest(array: np.ndarray, value: float) -> tuple[int, ...]:
    return np.unravel_index(np.nanargmin(np.abs(array - value)), np.shape(array))


def find_nearest(array: np.ndarray, value: float) -> float:
    return array[argnearest(array=array, value=value)]


def find_outliers(y: np.ndarray, x: np.ndarray | None = None,
                  z_thresh: float = 1., num_eps: float = _num_eps) -> np.ndarray:
    if x is None: x = np.arange(len(y))

    if len(np.unique(x)) != len(x):
        raise ValueError('"x" input must be unique.')

    inds = np.argsort(x)
    x_iterate, y_iterate = x[inds], y[inds]

    while True:
        deriv = np.diff(y_iterate) / np.diff(x_iterate)
        mu, sigma = return_mean_std(deriv)
        z_score = (deriv - mu) / sigma

        positive = np.where(z_score > z_thresh + num_eps)[0]
        negative = np.where(-z_score > z_thresh + num_eps)[0]

        # noise -> the points are next to each other (overlap if compensated for "diff" shift)
        outliers = stack((np.intersect1d(positive, negative + 1), np.intersect1d(negative, positive + 1)))

        if np.size(outliers) == 0:
            break

        x_iterate, y_iterate = np.delete(x_iterate, outliers), np.delete(y_iterate, outliers)

    # outliers are shifted due to deleting the two iterables
    return np.where([x_test not in x_iterate for x_test in x])[0]


def remove_outliers(y: np.ndarray, x: np.ndarray | None = None,
                    z_thresh: float = 1., num_eps: float = _num_eps) -> np.ndarray | tuple[np.ndarray, ...]:
    inds_to_remove = find_outliers(y=y, x=x, z_thresh=z_thresh, num_eps=num_eps)

    if x is None:
        return np.delete(y, inds_to_remove)

    return np.delete(y, inds_to_remove), np.delete(x, inds_to_remove)


def plot_me(x: np.ndarray | list, *args, **kwargs) -> tuple:
    matplotlib.use("TkAgg")  # Potentially dangerous (changes backend in the following code)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    x = np.squeeze(x)
    if np.ndim(x) == 0: x = np.reshape(x, (1,))

    if np.ndim(x) == 1:  # line plot
        # Is the first arg y axis?
        if len(args) and isinstance(args[0], np.ndarray | list) and np.size(x) in np.shape(args[0]):
            y = np.squeeze(args[0])
            try:
                ax.plot(x, y, *args[1:], **kwargs)
            except ValueError:
                ax.plot(x, np.transpose(y), *args[1:], **kwargs)
        else:
            ax.plot(x, *args, **kwargs)

        """
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        """

    else:  # x is the matrix to plot
        y_max, x_max = np.shape(x)
        im = ax.imshow(x, *args, origin="lower", extent=[0, x_max, 0, y_max], aspect="auto", **kwargs)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position="right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.tight_layout()
    plt.show()

    return fig, ax


def split_path(filename: str, is_dir_check: bool = False) -> tuple[str, ...]:

    if is_dir_check and path.isdir(filename):
        return filename, "", ""

    dirname, basename = path.split(filename)

    if "." in basename:
        basename, extension = basename.split(".", 1)
    else:
        extension = ""

    return dirname, basename, extension


def timestamp(t: float, prec: int = 3) -> str:
    if t < 0.:
        raise ValueError('Elapsed time "t" must be a non-negative number.')
    n = 3.

    if t * n < 1.:  # less than 1/n seconds -> show milliseconds
        return f"{np.round(t * 1000., prec):.{prec:d}f} milliseconds"
    if t / 60. < n:  # less than n minutes -> show seconds
        return f"{np.round(t, prec):.{prec:d}f} seconds"
    if t / 60. < n * 60.:  # between n minutes and n hours -> show minutes
        return f"{np.round(t / 60., prec):.{prec:d}f} minutes"
    if t / 3600. < n * 24.:  # between n hours and n days -> show hours
        return f"{np.round(t / 3600., prec):.{prec:d}f} hours"
    return f"{np.round(t / 86400., prec):.{prec:d}f} days"  # show days


def timeit(_func: Callable, *args, num_repeats: int = 100, return_output: bool = False, **kw):
    ts = time()
    for _ in range(num_repeats - 1):
        _func(*args, **kw)
    else:
        result = _func(*args, **kw)
    te = time()

    elapsed_time = timestamp(te - ts, prec=3)
    if num_repeats == 1:
        print(f'Function "{_func.__name__}" took {elapsed_time}.')
    else:
        elapsed_time_per_repetition = timestamp((te - ts) / num_repeats, prec=5)
        print(f'Function "{_func.__name__}" took {elapsed_time} after {num_repeats} repetitions '
              f'({elapsed_time_per_repetition} per repetition).')

    if return_output:
        return result


def to_list(param) -> list:
    return param if isinstance(param, list) else [param]


def my_argextreme(min_or_max: Literal["min", "max"], x: np.ndarray, y: np.ndarray,
                  x0: float | None = None, dx: float = 50.,
                  n_points: int = 3,
                  fit_method: str = "ransac",
                  rtol: float = 2.) -> float:
    # This function returns a position of local extreme of y(x) around a point x0 +- dx
    if min_or_max not in ["min", "max"]:
        raise ValueError('"min_or_max" must be "min" or "max".')

    x, y = np.array(x), np.array(y)

    if x0 is None:
        if min_or_max == "min":
            x0 = x[np.nanargmin(y)]
        else:
            x0 = x[np.nanargmax(y)]

    # select in this interval
    ind = np.where(np.logical_and(x0 - dx <= x, x <= x0 + dx))[0]
    y_int = y[ind]

    # extreme value using all indices
    if min_or_max == "min":
        ix0 = int(ind[np.nanargmin(y_int)])
    else:  # must be "max"
        ix0 = int(ind[np.nanargmax(y_int)])

    start = ix0 - n_points if ix0 - n_points >= 0 else 0
    stop = ix0 + n_points + 1

    start, stop = int(start), int(stop)

    x_ext = x[start:stop]
    y_ext = y[start:stop]

    mask = np.logical_and(np.isfinite(x_ext), np.isfinite(y_ext))

    x_ext, y_ext = x_ext[mask], y_ext[mask]

    if len(x_ext) < 3:
        return x[ix0]

    # It is good to standardise the data
    x_mean, x_std = return_mean_std(x_ext)
    y_mean, y_std = return_mean_std(y_ext)

    x_fit = (x_ext - x_mean) / x_std
    y_fit = (y_ext - y_mean) / y_std

    if len(x_fit) == 3:  # not enough points -> use numpy
        params = my_polyfit(x_fit, y_fit, 2, method="numpy")
        try_numpy = False

    elif fit_method == "numpy":  # numpy is used when other methods fail
        # keep it here as extra possibility to not repeat it in "except"
        params = my_polyfit(x_fit, y_fit, 2, method=fit_method)
        try_numpy = False
    else:

        try:
            params = my_polyfit(x_fit, y_fit, 2, method=fit_method)
            try_numpy = True
        except ValueError:
            warnings.warn(f'Fitting failed. Switching to fit_method="numpy".')
            params = my_polyfit(x_fit, y_fit, 2, method="numpy")
            try_numpy = False

    # position of the extreme value
    extreme = -params[1] / (2. * params[0]) * x_std + x_mean

    # return extreme of a parabola if it is not far (rtol pixel) from the local extreme
    step = rtol * np.mean(np.diff(x_ext))

    if np.abs(extreme - x[ix0]) <= step:
        return extreme

    if try_numpy:
        # if it is too far from the local extreme, use standard numpy fit and try it again
        params = my_polyfit(x_fit, y_fit, 2, method="numpy")
        extreme = -params[1] / (2. * params[0]) * x_std + x_mean

        if np.abs(extreme - x[ix0]) <= step:
            return extreme

    return x[ix0]


def my_argmin(x: np.ndarray, y: np.ndarray, x0: float | None = None, dx: float = 10.,
              n_points: int = 3, fit_method: str = "ransac", rtol: float = 2.) -> float:
    # This function returns a position of local minimum of y(x) around a point x0 +- dx
    return my_argextreme("min", x, y, x0, dx, n_points, fit_method, rtol)


def my_argmax(x: np.ndarray, y: np.ndarray, x0: float | None = None, dx: float = 10.,
              n_points: int = 3, fit_method: str = "ransac", rtol: float = 2.) -> float:
    # This function returns a position of local maximum of y(x) around a point x0 +- dx
    return my_argextreme("max", x, y, x0, dx, n_points, fit_method, rtol)


def to_shape(a: np.ndarray, shape: tuple[int, int], val: float = np.nan) -> np.ndarray:
    y_, x_ = shape
    y, x = np.shape(a)
    y_pad = (y_ - y)
    x_pad = (x_ - x)
    return np.pad(a, ((y_pad // 2, y_pad // 2 + y_pad % 2),
                      (x_pad // 2, x_pad // 2 + x_pad % 2)),
                  mode='constant', constant_values=(val,))


def cropND(img: np.ndarray, bounding: tuple[int, int]) -> np.ndarray:
    start = tuple(map(lambda a, da: (a - da) // 2, np.shape(img), bounding))
    end = tuple(map(np.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def sliding_window(image: np.ndarray, kernel: np.ndarray, func: str | Callable, mode: str = "same") -> np.ndarray:
    if not (callable(func) or func in ["conv", "conv2", "min", "max", "median"]):
        raise ValueError('"func" must be a 2D function or be in ["conv", "conv2", "min", "max", "median"]')
    if mode not in ["full", "same", "valid"]:
        raise ValueError('"mode" must be in ["full", "same", "valid"]')

    # needed due to nans
    image, kernel = np.array(image, dtype=float), np.array(kernel, dtype=float)

    # We start by defining some constants, which are required for this code
    kern_h, kern_w = np.shape(kernel)
    im_h, im_w = np.shape(image)

    # full-shape output
    full_h, full_w = im_h + kern_h - 1, im_w + kern_w - 1

    # padding to the full shape of a full shape (to cover edges -> valid window in this is full in input image)
    image_padded = to_shape(image, (full_h + kern_h - 1, full_w + kern_w - 1))
    pad_im_h, pad_im_w = np.shape(image_padded)

    #
    # Preparing the indices that split the padded image into sub-images
    #

    # indices in columns (in height); only valid indices in padded image
    # Shape of filter2 is full_h * kern_h
    filter1 = np.arange(kern_h) + np.arange(pad_im_h - kern_h + 1)[:, np.newaxis]

    # indices in rows (in width); only valid indices in padded image
    # Shape of filter1 is full_w x kern_w
    filter2 = np.arange(kern_w) + np.arange(pad_im_w - kern_w + 1)[:, np.newaxis]

    # splitting the padded image into sub-images

    # intermediate is the stepped data, which has the shape full_h x kern_h x pad_im_w
    intermediate = image_padded[filter1]

    # transpose the inner dimensions of the intermediate to enact filter2
    # shape is now full_h x pad_im_w x kern_h
    intermediate = np.transpose(intermediate, (0, 2, 1))

    # Apply filter2 on the inner data piecewise, resultant shape is full_h x full_w x kern_w x kern_h
    intermediate = intermediate[:, filter2]

    # transpose inwards again to get a resultant shape of full_h x full_w x kern_h x kern_w
    intermediate = np.transpose(intermediate, (0, 1, 3, 2))

    if func in ["conv", "conv2"]:
        result = np.nansum(intermediate * np.rot90(kernel, 2), axis=(2, 3))

        if mode == "same":
            out_h, out_w = im_h, im_w

        # Matlab-like valid (different from scipy if kernel size > image size; scipy switches them)
        elif mode == "valid":
            if kern_h != 0:
                out_h = np.max(im_h - kern_h + 1, 0)
            else:
                out_h = im_h
            if kern_w != 0:
                out_w = np.max(im_w - kern_w + 1, 0)
            else:
                out_w = im_w

        else:  # full
            out_h, out_w = full_h, full_w

    else:
        # mode == "same" always if not conv or conv2
        out_h, out_w = im_h, im_w

        if callable(func):  # apply the function and crop
            # general function lambda i, k: func(i, k) -> R
            result = np.zeros((full_h, full_w))
            for ix in range(full_h):
                for iy in range(full_w):
                    result[ix, iy] = func(intermediate[ix, iy], kernel)

        # Do not apply the function on "filtered-out" data (where kernel <= 0)
        intermediate = intermediate * np.where(kernel > 0., 1., np.nan)

        if func == "median":
            result = np.nanmedian(intermediate, axis=(2, 3))

        elif func == "max":
            result = np.nanmax(intermediate, axis=(2, 3))

        elif func == "min":
            result = np.nanmin(intermediate, axis=(2, 3))

    return cropND(result, (out_h, out_w))


def best_blk(num: int) -> tuple[int, int]:
    # Function finds the best rectangle with an area lower or equal to num
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
    # distance(rectangle, point) computes the distance between the rectangle and the point p
    # rectangle[0, 0] = x.min, rectangle[0, 1] = x.max
    # rectangle[1, 0] = y.min, rectangle[1, 1] = y.max
    # point[0], point[1] = point.x, point.y
    dx = np.max((rectangle[0, 0] - point[0], np.zeros(np.shape(point[0])), point[0] - rectangle[0, 1]), axis=0)
    dy = np.max((rectangle[1, 0] - point[1], np.zeros(np.shape(point[1])), point[1] - rectangle[1, 1]), axis=0)

    return np.sqrt(dx * dx + dy * dy)


def my_pca(x_data: np.ndarray,
           n_components: int | float | None = None,
           standardise: bool = False,
           return_info: bool = False,
           svd_solver: str = "full",
           num_eps: float = _num_eps,
           **kwargs) -> tuple[np.ndarray, dict[str, bool | np.ndarray | PCA]] | np.ndarray:
    # Function computes first n_components principal components

    if standardise:
        ddof = np.min((len(x_data) - 1, 1))
        std = np.std(x_data, ddof=ddof, axis=0)
        if np.any(std <= num_eps):  # "<=" is necessary for num_eps = 0.
            raise ValueError("Not all features are determinative. Remove these features, or don't use standardisation.")
    else:
        std = np.full(np.shape(x_data)[1], fill_value=1.)

    x = x_data / std  # mean is removed and stored within PCA

    pca = PCA(n_components=n_components, svd_solver=svd_solver, **kwargs)
    x_data_pca = pca.fit_transform(x)

    """
    mu = np.mean(x, axis=0)
    x = x - mu
    cov = np.cov(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    inds = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[inds]
    PC = np.transpose(eigenvectors[:, inds])  # may have different orientation and length

    explained_variance = eigenvalues / np.sum(eigenvalues)

    x_data_pca = x @ np.transpose(PC[:n_components])  # may have different orientation and length

    back_projection = (x_data_pca @ PC[:n_components] + mu) * std
    """

    if return_info:
        pca.std_correction_ = std

        info = {"standardised": standardise,
                "PC": pca.components_,
                "eigenvalues": pca.explained_variance_,
                "explained_variance": pca.explained_variance_ratio_,
                "back_projection": pca.inverse_transform(x_data_pca) * std,
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


def is_float(element: any) -> bool:
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def substring_counts(string: str, sub_string: str) -> int:
    return len(re.findall(f"(?={re.escape(sub_string)})", string))


def find_all(string: str, sub_string: str) -> list[int]:
    return [m.start() for m in re.finditer(f"(?={re.escape(sub_string)})", string)]


def sort_df_with_keys(df: pd.DataFrame, sorted_keys: list[str]) -> pd.DataFrame:
    used_keys = [key for key in pd.unique(sorted_keys) if key in df.keys()]  # pd.unique does not sort
    other_keys = [key for key in df.keys() if key not in used_keys]

    used_keys += other_keys

    return df[used_keys]


def normalise_array(array: np.ndarray,
                    axis: int | None = None,
                    norm_vector: np.ndarray | None = None,
                    norm_constant: float = 1.,
                    num_eps: float = _num_eps) -> np.ndarray:
    if norm_vector is None:
        norm_vector = np.nansum(array, axis=axis, keepdims=True)

    # to force correct dimensions (e.g. when passing the output of interp1d)
    if np.ndim(norm_vector) != np.ndim(array) and np.ndim(norm_vector) > 0:
        norm_vector = np.expand_dims(norm_vector, axis=axis)

    if np.any(np.abs(norm_vector) < num_eps):
        warnings.warn("You normalise with (almost) zero values. Check the normalisation vector.")

    return array / norm_vector * norm_constant


def normalise_in_columns(array: np.ndarray,
                         norm_vector: np.ndarray | None = None,
                         norm_constant: float = 1.) -> np.ndarray:
    return normalise_array(array, axis=0, norm_vector=norm_vector, norm_constant=norm_constant)


def normalise_in_rows(array: np.ndarray,
                      norm_vector: np.ndarray | None = None,
                      norm_constant: float = 1.) -> np.ndarray:
    return normalise_array(array, axis=1, norm_vector=norm_vector, norm_constant=norm_constant)


def denoise_array(array: np.ndarray, sigma_px: float, remove_mean: bool = True) -> np.ndarray:

    *_, cols = np.shape(array)

    correction = gaussian_filter1d(np.ones(cols), sigma=sigma_px, mode="constant")
    array_denoised = gaussian_filter1d(array, sigma=sigma_px, mode="constant")

    array_denoised = normalise_in_columns(array_denoised, norm_vector=correction)

    if remove_mean:  # here I assume that the noise has a zero mean
        mn = np.mean(array_denoised - array, axis=-1, keepdims=True)
    else:
        mn = 0.

    return array_denoised - mn


def maxdiff(arr1: np.ndarray, arr2: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.nanmax(np.abs(arr1 - arr2), axis=axis)


def nanequal(x, y, axis: int | None = None):
    return np.all(np.logical_or(x == y, np.logical_and(pd.isna(x), pd.isna(y))), axis=axis)


def npz_to_dat(filename: str) -> None:
    data = np.load(filename, allow_pickle=True)

    for file in data.files:
        filename_dat = f"{filename.replace('.npz', '')}{_sep_out}{file.replace(' ', _sep_in)}.dat"
        fmt = "%.5f" if np.issubdtype(np.result_type(data[file]), np.number) else "%s"
        np.savetxt(filename_dat, data[file], fmt=fmt, delimiter='\t')


def change_files_in_npz(filename: str, old_files: list[str], new_files: list[str]) -> None:
    if len(old_files) != len(new_files):
        raise ValueError('Different lengths of "old_files" and "new_files" parameters.')

    data = np.load(filename, allow_pickle=True)
    data = dict(data)

    for old_file, new_file in zip(old_files, new_files):
        if old_file in data.keys():
            data[new_file] = data.pop(old_file)

    with open(filename, "wb") as f:
        np.savez_compressed(f, **data)


def is_empty(data: np.ndarray) -> bool:
    return data is None or np.size(data) == 0


def display_top(snapshot, traced_memory, key_type: str = "lineno", limit: int = 3, memory_format: str = "MiB"):
    if memory_format == "GiB":
        coef = 1. / 1024. / 1024. / 1024.
    elif memory_format == "MiB":
        coef = 1. / 1024. / 1024.
    elif memory_format == "KiB":
        coef = 1. / 1024.

    elif memory_format == "GB":
        coef = 1. / 1000. / 1000. / 1000.
    elif memory_format == "MB":
        coef = 1. / 1000. / 1000.
    elif memory_format == "kB":
        coef = 1. / 1000.

    elif memory_format == "B":
        coef = 1.

    else:
        raise ValueError('Unknown format. Possible formats are "B", "kB", "KiB", "MB", "MiB, "GB", and "GiB".')

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = path.join(frame.filename.split(path.sep)[-2:])
        print(f"#{index}: {filename}:{frame.lineno}: {np.round(stat.size * coef, 1):.1f} {memory_format}")
        line = getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"{line:>5}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {np.round(size * coef, 1):.1f} {memory_format}")
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {np.round(total * coef, 1):.1f} {memory_format}")
    print(f"Peak allocated size: {np.round(traced_memory[1] * coef, 1):.1f} {memory_format}")


def get_weights_from_model(model: Functional) -> dict[str, np.ndarray]:
    layer_names = np.array([layer["class_name"] for layer in model.get_config()["layers"]])
    weights = {f"{name}_{i}": model.layers[i].get_weights() for i, name in enumerate(layer_names)}

    return weights


def kernel_density_estimation_2d(y_true_part: np.ndarray, y_pred_part: np.ndarray,
                                 nbins: int = 20) -> tuple[np.ndarray, ...]:
    error = 100. * (y_pred_part - y_true_part)
    quantity = 100. * y_true_part

    k = gaussian_kde((quantity, error))
    # limits: 0:100 for quantity; max error range for error
    # 1j means there is nbins * 1steps
    xi, yi = np.mgrid[0:100:nbins * 1j, -50:50:nbins * 1j]
    zi = k(stack([xi.flatten(), yi.flatten()], axis=0))

    zi = np.reshape(zi, np.shape(xi))

    return xi, yi, zi


def kernel_density_estimation_1d(y_true_part: np.ndarray, y_pred_part: np.ndarray,
                                 nbins: int = 20) -> tuple[np.ndarray, ...]:
    error = 100. * (y_pred_part - y_true_part)

    k = gaussian_kde(error)
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


def kendall_pval(x, y):
    # use these with DataFrame.corr(method=kendall_pval) to get p-value for the variables
    return kendalltau(x, y)[1]


def pearsonr_pval(x, y):
    # use these with DataFrame.corr(method=pearsonr_pval) to get p-value for the variables
    return pearsonr(x, y)[1]


def spearmanr_pval(x, y):
    # use these with DataFrame.corr(method=spearmanr_pval) to get p-value for the variables
    return spearmanr(x, y)[1]
