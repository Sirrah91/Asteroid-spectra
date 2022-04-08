from pathlib import Path
import numpy as np
from typing import List, Tuple
from functools import wraps
from time import time
import scipy.optimize as opt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def check_dir(dir_or_file_path: str) -> None:
    dir_or_file_path = Path(dir_or_file_path)

    if dir_or_file_path.suffix:
        directory = dir_or_file_path.parent  # is file
    else:
        directory = dir_or_file_path  # is folder

    if not directory.exists():
        print('Directory ' + directory.as_posix() + " doesn't exist, creating it now.")
        directory.mkdir(parents=True, exist_ok=True)


def flatten_list(list_of_lists: List[List]) -> np.ndarray:
    return np.array([item for sub_list in list_of_lists for item in sub_list])


def my_argmin(x: np.ndarray, y: np.ndarray, x0: float, n: int = 3, minimum: bool = True) -> float:
    step = x[1] - x[0]

    if step >= 1:  # x is most likely in nm
        half_width = 200
    else:  # x is most likely in um
        half_width = 0.2

    # select in this interval
    ind = np.where(np.logical_and(x0 - half_width <= x, x <= x0 + half_width))
    x_int = x[ind]
    y_int = y[ind]

    # extreme value on this interval
    if minimum:
        ix0 = np.argmin(y_int)
    else:
        ix0 = np.argmax(y_int)

    x_ext = x_int[ix0 - n:ix0 + n + 1]
    y_ext = y_int[ix0 - n:ix0 + n + 1]

    params = np.polyfit(x_ext, y_ext, 2)

    # position of the extreme value
    return -params[1] / (2 * params[0])


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func: %r args: [%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def gimme_indices() -> np.ndarray:
    from modules.NN_config import num_minerals, subtypes

    indices = np.zeros((len(subtypes) + 1, 2), dtype=int)
    indices[0, :] = 0, num_minerals

    for k in range(len(subtypes)):
        indices[k + 1, :] = indices[k, 1], indices[k, 1] + subtypes[k]

    if num_minerals == 0:  # delete the first pair
        indices = indices[1:, :]

    return indices


def print_accuracy(accuracy: np.ndarray, what: str) -> None:
    indices = gimme_indices()
    tmp = np.array([np.mean(accuracy[range(indices[j, 0], indices[j, 1])]) for j in range(len(indices))])
    print('Mean', what, 'RMSE:', '[' + ', '.join('{:.1f}'.format(k) for k in tmp) + ']')

    """
    print('Mean', what, 'RMSE:', str("{:.1f};").format(np.mean(accuracy)), '[' + '; '.join(
        ', '.join('{:.1f}'.format(k) for k in accuracy[range(indices[j, 0], indices[j, 1])]) for j in
        range(len(indices))) + ']')
    """

    return


def best_blk(N: int) -> Tuple[int, int]:
    col1 = np.ceil(np.sqrt(N))
    row1 = np.ceil(N / col1)

    col2 = np.ceil(np.sqrt(N)) + 1
    row2 = np.ceil(N / col2)

    if col1 * row1 <= col2 * row2:
        row, col = int(row1), int(col1)
    else:
        row, col = int(row2), int(col2)

    return row, col


def distance(rect: np.ndarray, p: np.ndarray) -> np.ndarray:
    # rect[0, 0] = x.min, rect[0, 1] = x.max
    # rect[1, 0] = y.min, rect[1, 1] = y.max
    # p[0], p[1] = p.x, p.y
    dx = np.max((rect[0, 0] - p[0], np.zeros(np.shape(p[0])), p[0] - rect[0, 1]), axis=0)
    dy = np.max((rect[1, 0] - p[1], np.zeros(np.shape(p[1])), p[1] - rect[1, 1]), axis=0)

    return np.sqrt(dx * dx + dy * dy)


def my_pca(x_data: np.ndarray, n_components: int = 2, standardise: bool = False) -> np.ndarray:
    """
    filename_data = 'AP_spectra-denoised-norm-nolabel.dat'

    project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.
    data_file = "".join((project_dir, '/Datasets/', filename_data))
    x_data = np.loadtxt(data_file, delimiter='\t')
    """

    if standardise:
        x = StandardScaler().fit_transform(x_data)
    else:
        x = x_data

    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)



'''
def gauss_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def error_estimation(y_true, y_pred):
    data = y_true - y_pred
    error = np.zeros(np.shape(data)[1])
    # indexy pres subtypes...
    ind = np.array([3, 5, 7])
    for i in range(np.shape(data)[1]):
        if i < 3:
            a = data[:, i].ravel() * 100
        else:
            a = data[np.where(data[:, i] > 0), ind[i]].ravel() * 100
        b = np.histogram(a, 7)
        y = b[0]
        x = b[1]
        dx = x[1] - x[0]
        x += dx / 2
        x = x[:-1]
        x = np.linspace(x.min(), x.max(), len(x))

        mean = np.sum(x * y) / np.sum(y)
        sigma = np.sqrt(np.sum(y * (x - mean)**2) / np.sum(y))

        p0 = [np.max(y), mean, sigma]
        popt, pcov = opt.curve_fit(gaus_function, x, y, p0=p0)

        error[i] = popt[2]
        g = gaus_function(x, *p0)
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, g)
'''
