from pathlib import Path
import numpy as np
from typing import List, Tuple
from functools import wraps
from time import time

from keras.models import Functional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kde
import shutil
from glob import glob
import os


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


def print_accuracy(accuracy: np.ndarray, what: str, all_types_to_one: bool = False) -> None:
    # Function to print vector accuracy

    pref = 'Mean ' + what + ' RMSE:'

    if all_types_to_one:
        indices = gimme_indices()
        tmp = np.array([np.mean(accuracy[range(indices[j, 0], indices[j, 1])]) for j in range(len(indices))])
        print('Mean', what, 'RMSE:', '[' + ', '.join('{:.1f}'.format(k) for k in tmp) + ']')
    else:
        indices = unique_indices()
        print('{:21s}'.format(pref), '[' + ', '.join('{:4.1f}'.format(k) for k in accuracy[indices]) + ']')

    return


def print_accuracy_header(all_types_to_one: bool = False) -> None:
    # Function to print header of the vector accuracy

    header = np.array(['OL', 'OPX', 'CPX', 'PLG',
                       'Fa', 'Fo',
                       'Fs', 'En', 'Wo',
                       'Fs', 'En', 'Wo',
                       'An', 'Ab', 'Or'])

    if all_types_to_one:
        pass
    else:
        indices = unique_indices(all_minerals=True)
        print('{:23s}'.format(''), ''.join('{:6s}'.format(k) for k in header[indices]))

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
    check_dir(destination)

    if mv_or_cp == 'mv':
        shutil.move(source, destination)
    elif mv_or_cp == 'cp':
        shutil.copy(source, destination)
    else:
        print('mv_or_cp must be either "mv" or "cp"')


def normalise_in_rows(array: np.ndarray, norm_vector: np.ndarray) -> np.ndarray:
    return np.transpose(np.divide(np.transpose(array), norm_vector))


def get_weights_from_model(model: Functional) -> dict:
    # indices of conv layers if not known
    indices = np.array([index for index, layer in enumerate(model.get_config()['layers'])
                        if 'Conv' in layer['class_name']])

    weights = {}

    for i, layer_index in enumerate(indices):
        weights[f'Conv_{i}'] = model.layers[layer_index].get_weights()[0]

    return weights


def kernel_density_estimation_2d(y_true_part: np.ndarray, y_pred_part: np.ndarray,
                                 nbins: int = 20) -> Tuple[np.ndarray, ...]:
    error = 100 * (y_pred_part - y_true_part)
    quantity = 100 * y_true_part

    k = kde.gaussian_kde((quantity, error))
    # limits: 0:100 for quantity; max error range for error
    xi, yi = np.mgrid[0:100:nbins * 1j, -50:50:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    return xi, yi, zi


def kernel_density_estimation_1d(y_true_part: np.ndarray, y_pred_part: np.ndarray,
                                 nbins: int = 20) -> Tuple[np.ndarray, ...]:
    error = 100 * (y_pred_part - y_true_part)

    k = kde.gaussian_kde(error)
    # limits: 0:100 for quantity; max error range for error
    xi = np.linspace(-50, 50, nbins)
    zi = k(xi)

    return xi, zi


def unique_indices(all_minerals: bool = False) -> np.ndarray:
    from copy import deepcopy
    from modules.NN_config import use_minerals_all, subtypes_all, used_indices

    # modification of used indices (if there are two labels, their absolute errors are the same; one is enough)
    use_inds = deepcopy([list(use_minerals_all)] + [*subtypes_all])
    for i in range(len(use_inds)):
        if np.sum(use_inds[i]) == 2:
            use_inds[i][np.where(use_inds[i])[0][-1]] = False
    use_inds = flatten_list(use_inds) * used_indices

    if all_minerals:
        return np.where(use_inds)[0]
    else:
        # delete indices, where used_indices are not use_indices
        return np.array([i for i in range(len(np.where(used_indices)[0]))
                         if np.where(used_indices)[0][i] in np.where(use_inds)[0]])


def collect_all_models(suffix: str, subfolder_model: str, full_path: bool = True) -> List:
    from modules.NN_config import project_dir

    model_str = "".join((project_dir, '/Models/', subfolder_model, '/', '*', suffix, '.h5'))

    if full_path:
        return glob(model_str)
    else:
        return [os.path.basename(x) for x in glob(model_str)]


def find_outliers(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 40.0, px_only: bool = True,
                  meta: np.ndarray = None) -> np.ndarray:

    from modules.NN_losses_metrics_activations import my_ae
    from modules.NN_config import num_minerals

    absolute_error = my_ae(num_minerals=num_minerals, all_to_one=False)(y_true, y_pred).numpy()

    inds_samples, inds_quantities = np.where(absolute_error > threshold)

    unique_samples = np.unique(inds_samples)

    if px_only:
        samples = np.array([i for i in unique_samples
                            if np.logical_and(1 in inds_quantities[np.where(inds_samples == i)],
                                              2 in inds_quantities[np.where(inds_samples == i)])])
    else:
        samples = unique_samples

    if meta is None:
        return samples
    else:
        return np.array(list(zip(meta[samples], samples)), dtype=np.object)


def round_data_with_errors(data: np.ndarray, errors: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = 2 - np.ceil(np.log10(errors))  # rounding to 2 valid number
    n[~np.isfinite(n)] = 1

    data_rounded, errors_rounded = np.zeros(np.shape(data)), np.zeros(np.shape(errors))

    for i in range(len(data)):
        data_rounded[i] = np.round(data[i], n[i])
        errors_rounded[i] = np.round(errors[i], n[i])

    return np.round(data_rounded, 1), np.round(errors_rounded, 1)


def replace_spaces_with_phantom(str_array: np.ndarray) -> np.ndarray:
    # replace spaces with phantom numbers
    for i in range(len(str_array)):
        str_array[i] = str_array[i].replace(" ", "\\phantom{0}")

    return str_array

