import numpy as np
from warnings import warn
from modules.utilities import safe_arange, flatten_list
from modules.NN_data_grids import data_grids, check_grid, normalise_spectrum_at_wvl
from modules.NN_classes import gimme_list_of_classes

from modules._constants import _wp, _sep_in


def config_check(output_setup: dict, grid_setup: dict, data_split_setup: dict, model_options: dict) -> None:
    if "minerals" in output_setup:  # composition config
        if not np.any(output_setup["minerals"]):
            raise ValueError('There is no mineral in "minerals".')

        num_endmembers = int(np.sum(gimme_endmember_counts(used_endmembers=output_setup["used_endmembers"])))

        # If there is no end-member, return a warning
        if num_endmembers == 0:
            warn('Warning. There is no valid end-member in "endmembers".')

    else:  # taxonomy config
        list_of_classes = gimme_list_of_classes(grid_option=grid_setup["model_grid"])

        if len(list_of_classes) != len(set(list_of_classes)):
            raise ValueError('"list_of_classes" in NN_classes.py is not unique.')

    if data_split_setup["val_portion"] + data_split_setup["test_portion"] >= 1.:
        raise ValueError('Too high sum of "val_portion" and "test_portion.')

    if not (0. <= model_options["trim_mean_cut"] < 0.5):
        raise ValueError(f"Trimming parameter must be a non-negative number lower than 0.5 "
                         f"but is {model_options['trim_mean_cut']}.")

    if output_setup["num_labels"] == 0:
        raise ValueError("There is no valid label.")


def check_used_quantities(used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                          raise_error: bool = True) -> bool:
    # check the input
    endmember_counts = gimme_endmember_counts(used_endmembers=used_endmembers)

    # no "single" labels
    if np.sum(used_minerals) == 1 or np.any(endmember_counts == 1):
        error_msg = "No singleton labels."
        if raise_error:
            raise ValueError(error_msg)
        else:
            warn(error_msg)
            return False

    # no mineral for end-member (one allowed if no minerals)
    if np.sum(used_minerals) == 0 and np.sum(endmember_counts[~used_minerals] > 0) > 1:
        error_msg = ("Missing mineral label for one of endmember group "
                     "(only one group is allowed if no minerals are present).")
        if raise_error:
            raise ValueError(error_msg)
        else:
            warn(error_msg)
            return False

    # no mineral for endmember
    if np.sum(used_minerals) > 1 and np.sum(endmember_counts[~used_minerals] > 0) > 0:
        error_msg = "Missing mineral label for one of endmember group."
        if raise_error:
            raise ValueError(error_msg)
        else:
            warn(error_msg)
            return False

    return True


def gimme_minerals_all(used_minerals: np.ndarray, used_endmembers: list[list[bool]]) -> np.ndarray:
    return np.where(gimme_endmember_counts(used_endmembers) > 0, True, used_minerals)


def gimme_num_minerals(used_minerals: np.ndarray) -> int:
    num_minerals = int(np.sum(used_minerals))
    return num_minerals if num_minerals > 1 else 0


def gimme_endmember_counts(used_endmembers: list[list[bool]]) -> np.ndarray:
    return np.array([np.sum(endmember) for endmember in used_endmembers])


def gimme_model_grid(instrument: str | None, interpolate_to: str | None,
                     wvl_grid: np.ndarray, wvl_norm: float | None) -> dict:

    if instrument is None:
        if interpolate_to in data_grids.keys():
            grid = data_grids[interpolate_to]
            check_grid(grid)

            *new_wvl_grid, new_wvl_grid_normalisation = grid
            new_wvl_grid = safe_arange(*new_wvl_grid, endpoint=True, dtype=_wp)
        else:
            new_wvl_grid, new_wvl_grid_normalisation = np.array(wvl_grid, dtype=_wp), wvl_norm

        m, M = int(np.round(np.min(new_wvl_grid))), int(np.round(np.max(new_wvl_grid)))
        res = int(np.round(np.mean(np.diff(new_wvl_grid))))

        if new_wvl_grid_normalisation == "adaptive":
            new_wvl_grid_normalisation = normalise_spectrum_at_wvl(new_wvl_grid)

        if new_wvl_grid_normalisation is None:
            norm_for_grid = None
        else:
            norm_for_grid = int(np.round(new_wvl_grid_normalisation))

        model_grid = _sep_in.join(str(x) for x in [m, M, res, norm_for_grid])

    else:
        model_grid = instrument
        new_wvl_grid = None
        new_wvl_grid_normalisation = wvl_norm

    return {"model_grid": model_grid,
            "instrument": instrument,
            "wvl_grid": new_wvl_grid,
            "wvl_norm": new_wvl_grid_normalisation}


def gimme_used_quantities(minerals: np.ndarray,
                          endmembers: list[list[bool]]) -> tuple[np.ndarray, list[list[bool]]]:
    # If there is only one end-member for a given mineral, the information is redundant and worsens the optimisation
    endmembers_used = [endmember if (mineral and np.sum(endmember) > 1) else len(endmember) * [False]
                       for mineral, endmember in zip(minerals, endmembers)]

    # If there is only one mineral, the modal information is redundant and worsens the optimisation
    minerals_used = minerals if np.sum(minerals) > 1 else np.array([False] * len(minerals))

    return minerals_used, endmembers_used


def gimme_num_labels(used_minerals: np.ndarray, used_endmembers: list[list[bool]]) -> int:
    num_endmembers = int(np.sum(gimme_endmember_counts(used_endmembers=used_endmembers)))
    num_minerals = gimme_num_minerals(used_minerals=used_minerals)

    return num_minerals + num_endmembers


def used_to_bin(used_minerals: np.ndarray, used_endmembers: list[list[bool]]) -> str:
    bin_min = "".join(str(int(x)) for x in used_minerals)
    bin_comp = ["".join(str(int(x)) for x in endmember) for endmember in used_endmembers]

    return _sep_in.join([bin_min] + bin_comp)


def gimme_separator(bin_code: str) -> str:
    non_digit = np.array([not e.isdigit() for e in list(bin_code)])
    separator = np.unique(np.array(list(bin_code))[non_digit])

    if len(separator) > 1:
        raise ValueError(f"Non-unique separator in {bin_code}.")

    return separator[0]


def bin_to_used(bin_code: str, separator: str | None = None,
                return_all: bool = False) -> tuple[np.ndarray, list] | tuple[np.ndarray, list, np.ndarray]:
    # This function converts info from binary code in the name of a model into mineral abundances and composition
    # and gives you minerals_used and end_member used
    # e.g. bin_to_composition("1110-11-110-000-000") = np.array(["OL", "OPX", "CPX", "Fa" "Fo", "Fs (OPX)", "En (OPX)"])

    error_msg = f'Invalid bin code input "{bin_code}".'

    if separator is None: separator = gimme_separator(bin_code=bin_code)

    used_quantities = [list(quantity) for quantity in bin_code.split(separator)]

    used_quantities_flat = flatten_list(used_quantities)
    if not np.all(np.logical_or(used_quantities_flat == "1", used_quantities_flat == "0")):
        raise ValueError(f'{error_msg} Bin code must be made of "1" and "0" only.')

    used_quantities = [[quantity == "1" for quantity in quantities] for quantities in used_quantities]

    used_minerals = np.array(used_quantities[0], dtype=bool)
    used_endmembers = used_quantities[1:]

    if len(used_minerals) != len(used_endmembers):
        raise ValueError(f'The length of the used minerals does not equal the length of the used end-members.\n'
                         f'Probably an incorrect separator. Bin code "{bin_code}", separator "{separator}".')

    # check the input
    check_used_quantities(used_minerals=used_minerals, used_endmembers=used_endmembers, raise_error=True)

    if return_all:
        used_minerals = gimme_minerals_all(used_minerals, used_endmembers)

    return used_minerals, used_endmembers


def cls_to_bin(used_classes: dict[str, int]) -> str:
    use_unknown_class = "Other" in used_classes
    if use_unknown_class:
        return f"{len(used_classes) - 1}{_sep_in}{int(use_unknown_class)}"
    return f"{len(used_classes)}{_sep_in}{int(use_unknown_class)}"


def bin_to_cls(bin_code: str, separator: str | None = None) -> dict[str, int]:
    error_msg = f'Invalid bin code input "{bin_code}".'

    if separator is None: separator = gimme_separator(bin_code=bin_code)

    model_grid, use_unknown_class = bin_code.split(separator)

    if not model_grid.isdigit():
        raise ValueError(f'{error_msg} Bin code must contain only digits and a separator.')

    if use_unknown_class not in ["0", "1"]:
        raise ValueError(f'{error_msg} Use unknown classes can be "0" or "1".')

    return gimme_classes(model_grid=model_grid, use_unknown_class=bool(int(use_unknown_class)))


def gimme_classes(model_grid: str | None = None, use_unknown_class: bool | None = None) -> dict[str, int]:
    # Definition of classes
    list_of_classes = gimme_list_of_classes(grid_option=model_grid)

    if int(use_unknown_class):
        list_of_classes.append("Other")  # Name should be changed in NN_data.labels_to_categories too.

    return {cls: i for i, cls in enumerate(list_of_classes)}
