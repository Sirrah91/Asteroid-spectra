from pathlib import Path
import numpy as np
from typing import List, Tuple
from functools import wraps
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shutil


def check_dir(dir_or_file_path: str) -> None:
    # This function check if the directory exists. If not, the function creates it.

    dir_or_file_path = Path(dir_or_file_path)

    if dir_or_file_path.suffix:
        directory = dir_or_file_path.parent  # is file
    else:
        directory = dir_or_file_path  # is folder

    if not directory.exists():
        print('Directory ' + directory.as_posix() + " doesn't exist, creating it now.")
        directory.mkdir(parents=True, exist_ok=True)


def flatten_list(list_of_lists: List[List]) -> np.ndarray:
    # This function flattens a list of lists
    return np.array([item for sub_list in list_of_lists for item in sub_list])


def my_argmin(x: np.ndarray, y: np.ndarray, x0: float, dx: float = 200, n: int = 3) -> float:
    # This function returns a position of local minimum of y(x) around a point x0 +- dx
    step = x[1] - x[0]

    if step < 1 and dx > 1:  # x is most likely in um and dx in nm
        dx /= 1000

    # select in this interval
    ind = np.where(np.logical_and(x0 - dx <= x, x <= x0 + dx))
    x_int = x[ind]
    y_int = y[ind]

    # extreme value on this interval
    ix0 = np.argmin(y_int)

    x_ext = x_int[ix0 - n:ix0 + n + 1]
    y_ext = y_int[ix0 - n:ix0 + n + 1]

    params = np.polyfit(x_ext, y_ext, 2)

    # position of the extreme value
    extreme = -params[1] / (2 * params[0])

    # return extreme of parabola if it is not far (3 pixels) from the local extreme
    res = extreme if np.abs(extreme - x_int[ix0]) < 3 * step else x_int[ix0]

    return res


def my_argmax(x: np.ndarray, y: np.ndarray, x0: float, dx: float = 200, n: int = 3) -> float:
    # This function returns a position of local maximum of y(x) around a point x0 +- dx
    step = x[1] - x[0]

    if step < 1 and dx > 1:  # x is most likely in um and dx in nm
        dx /= 1000

    # select in this interval
    ind = np.where(np.logical_and(x0 - dx <= x, x <= x0 + dx))
    x_int = x[ind]
    y_int = y[ind]

    # extreme value on this interval
    ix0 = np.argmax(y_int)

    x_ext = x_int[ix0 - n:ix0 + n + 1]
    y_ext = y_int[ix0 - n:ix0 + n + 1]

    params = np.polyfit(x_ext, y_ext, 2)

    # position of the extreme value
    extreme = -params[1] / (2 * params[0])

    # return extreme of parabola if it is not far (3 pixels) from the local extreme
    res = extreme if np.abs(extreme - x_int[ix0]) < 3 * step else x_int[ix0]

    return res


def timing(f):
    # @timing
    # def some_function(arguments)
    # timing(f) measures and print execution time of some_function(arguments)
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func: %r args: [%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def gimme_indices() -> np.ndarray:
    # This function returns first and last indices of modal/mineral groups
    from modules.NN_config import num_minerals, subtypes

    indices = np.zeros((len(subtypes) + 1, 2), dtype=int)
    indices[0, :] = 0, num_minerals

    for k in range(len(subtypes)):
        indices[k + 1, :] = indices[k, 1], indices[k, 1] + subtypes[k]

    if num_minerals == 0:  # delete the first pair
        indices = indices[1:, :]

    return indices


def print_accuracy(accuracy: np.ndarray, what: str) -> None:
    # Function to print vectoc accuracy
    indices = gimme_indices()
    tmp = np.array([np.mean(accuracy[range(indices[j, 0], indices[j, 1])]) for j in range(len(indices))])
    print('Mean', what, 'RMSE:', '[' + ', '.join('{:.1f}'.format(k) for k in tmp) + ']')

    """
    print('Mean', what, 'RMSE:', str("{:.1f};").format(np.mean(accuracy)), '[' + '; '.join(
        ', '.join('{:.1f}'.format(k) for k in accuracy[range(indices[j, 0], indices[j, 1])]) for j in
        range(len(indices))) + ']')
    """

    return


def best_blk(num: int) -> Tuple[int, int]:
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


def my_pca(x_data: np.ndarray, n_components: int = 2, standardise: bool = False) -> np.ndarray:
    # Function computes first n_components principal components
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


def my_mv(source: str, destination: str, mv_or_cp: str = 'mv') -> None:
    from modules.NN_config import project_dir
    check_dir(destination)

    if mv_or_cp == 'mv':
        shutil.move(source, destination)
    elif mv_or_cp == 'cp':
        shutil.copy(source, destination)
    else:
        print('mv_or_cp must be either "mv" or "cp"')
