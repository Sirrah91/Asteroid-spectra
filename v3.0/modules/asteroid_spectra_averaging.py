import h5py
from os import path
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import affinity
from scipy.interpolate import interp1d
from typing import Literal
import json
from tqdm import tqdm

from modules.decorators import timing
from modules.utilities_spectra import load_h5
from modules.utilities import stack, safe_arange, normalise_array
from modules._constants import _path_data, _spectra_name, _wavelengths_name, _coordinates_name, _sep_out

# ------------------------------------------------------------------------------------------------------------------------
what_to_run = "averaging"  # can be either "indices", "averaging", or "both"

what_asteroid = "Itokawa"
spectra_dir = path.join(_path_data, "asteroids", what_asteroid)

# longitude and latitude resolution
dlon, dlat = 1., 1.
dlon_half, dlat_half = dlon / 2., dlat / 2.

lon_grid, lat_grid = safe_arange(0., 360., dlon, endpoint=False), safe_arange(-90., 90., dlat, endpoint=True)

# interpolation method
# linear is much faster; cubic interpolation may cause troubles if there are wider gaps in spectra
interp_method = "linear"

# data polygon vs point or vs patch polygon
# can be either "point" or "polygon"
point_or_polygon = "polygon"  # "point" is much faster

if "Itokawa" == what_asteroid:
    # limits for data filtering
    phase_angle_limit = 30.  # deg
    incidence_angle_limit = 50.  # deg
    emission_angle_limit = 50.  # deg
    distance_limit = 5.  # km
    mean_value_limit = 0.01

    # wavelength range
    wavelengths_to_keep = np.arange(7, 62)

    # maximum based on wavelengths_to_keep; cutting in collect_data
    wavelengths_new = safe_arange(820., 2080., 20., endpoint=True)  # nm
    normalised_at_wvl = 1500.  # nm

else:
    # limits for data filtering
    phase_angle_limit = 40.  # deg
    incidence_angle_limit = 60.  # deg
    emission_angle_limit = 60.  # deg
    area_limit = 750.  # deg2
    mean_value_limit = 0.01

    # wavelength range
    wavelengths_to_keep = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 36,
                                    37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57])

    # maximum based on wavelengths_to_keep; cutting in collect_data
    wavelengths_new = safe_arange(820., 2440., 20., endpoint=True)  # nm
    normalised_at_wvl = 1300.  # nm


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
#                                       PREPARING INDICES OF THE SPECTRA
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
def read_data(asteroid_name: str, file_dir: str) -> np.ndarray:
    # import and save the datasets
    data_name = "data" if asteroid_name == "Itokawa" else "d"
    filename = f"{asteroid_name}{_sep_out}data.h5"

    obs_data = load_h5(path.join(file_dir, filename), list_keys=[data_name])

    return obs_data[data_name]


def read_indices(asteroid_name: str, file_dir: str) -> list[str]:
    # import indices file to array
    filename = f"{asteroid_name}{_sep_out}indices.txt"

    with open(path.join(file_dir, filename), "r") as text_file:
        lines = text_file.read().splitlines()

    return [line for line in lines if line]  # remove empty lines at the end of the file (if there are some)


def save_indices(asteroid_name: str, file_dir: str, indices_to_save: np.ndarray) -> None:
    filename = f"{asteroid_name}{_sep_out}indices.txt"

    with open(path.join(file_dir, filename), "w") as f:
        for i in indices_to_save:
            f.write(f"{list(i)}\n")


def saving_spectra(asteroid_name: str, file_dir: str, filtered_pt_coords: np.ndarray, weighted_spectra: np.ndarray,
                   wavelengths: np.ndarray) -> None:
    # Save averaged spectra into an .h5 file

    # remove NaNs (if polygons only touch but do not overlap)
    inds_to_delete = np.unique(np.where(~np.isfinite(weighted_spectra))[0])
    weighted_spectra = np.delete(weighted_spectra, inds_to_delete, axis=0)
    filtered_pt_coords = np.delete(filtered_pt_coords, inds_to_delete, axis=0)

    # Creates the h5-file and saves the data as a new dataset
    filename = f"{asteroid_name}{_sep_out}averaged.h5"
    with h5py.File(path.join(file_dir, filename), "w") as f:
        f.create_dataset(_spectra_name, data=weighted_spectra)
        f.create_dataset(_coordinates_name, data=filtered_pt_coords)
        f.create_dataset(_wavelengths_name, data=wavelengths)


# ------------------------------------------------------------------------------------------------------------------------
def process_lat(asteroid_name: str, coord_vector: np.ndarray) -> tuple[float, ...]:
    if asteroid_name == "Itokawa":
        # 1st corner y-coordinate etc.
        y1, y2, y3, y4 = np.transpose(coord_vector[..., [0, 6, 4, 2]])

        return y1, y2, y3, y4
    else:
        # lat min, max
        y1, y2 = np.transpose(coord_vector[..., [0, 1]])

        return y1, y2, y2, y1


def preprocess_lon(asteroid_name: str, coord_vector: np.ndarray) -> tuple[float, ...]:
    # It is not possible for measurement to be over 180 degrees
    # This function only shift points in longitude to make them close together

    if asteroid_name == "Itokawa":
        for k in [3, 5, 7]:
            coord_vector[..., k] = np.where(coord_vector[..., k] - coord_vector[..., 1] >= 180.,
                                            coord_vector[..., k] - 360., coord_vector[..., k])
            coord_vector[..., k] = np.where(coord_vector[..., k] - coord_vector[..., 1] <= -180.,
                                            coord_vector[..., k] + 360., coord_vector[..., k])

        return coord_vector[..., 1], coord_vector[..., 7], coord_vector[..., 5], coord_vector[..., 3]

    else:
        coord_vector[..., 3] = np.where(coord_vector[..., 3] - coord_vector[..., 2] >= 180.,
                                        coord_vector[..., 3] - 360., coord_vector[..., 3])
        coord_vector[..., 3] = np.where(coord_vector[..., 3] - coord_vector[..., 2] <= -180.,
                                        coord_vector[..., 3] + 360., coord_vector[..., 3])

        return coord_vector[..., 2], coord_vector[..., 2], coord_vector[..., 3], coord_vector[..., 3]


def process_coords(asteroid_name: str, coord_vector: np.ndarray) -> np.ndarray:
    y1, y2, y3, y4 = process_lat(asteroid_name, coord_vector)
    x1, x2, x3, x4 = preprocess_lon(asteroid_name, coord_vector)

    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


def coordinates_to_polygons(asteroid_name: str, coord_vector: np.ndarray) -> np.ndarray:
    coordinates = process_coords(asteroid_name, coord_vector)
    coordinates = np.rollaxis(coordinates, -1)

    return np.array([Polygon(coordinate).convex_hull for coordinate in coordinates])


# ------------------------------------------------------------------------------------------------------------------------
def filter_incomplete_observations(obs_data: np.ndarray) -> np.ndarray:
    # filter out missing or wrong data

    if np.ndim(obs_data) == 1:
        obs_data = np.reshape(obs_data, (1, len(obs_data)))

    angles = np.all(obs_data[:, 68:] > -99., axis=1)
    distance = obs_data[:, 67] > 0.
    reflectance = np.all(obs_data[:, wavelengths_to_keep] > 0., axis=1)

    return np.logical_and.reduce((angles, distance, reflectance))


def filter_observation_geometry(asteroid_name: str, obs_data: np.ndarray) -> np.ndarray:
    # filter out specific observations geometries

    print("Filtering geometries before index computation can speed up the process but if you need different filtering, "
          "you must recompute the indices.")

    if np.ndim(obs_data) == 1:
        obs_data = np.reshape(obs_data, (1, len(obs_data)))

    phase_angle = obs_data[:, 64] <= phase_angle_limit
    incidence_angle = obs_data[:, 65] <= incidence_angle_limit
    emission_angle = obs_data[:, 66] <= emission_angle_limit

    if asteroid_name == "Itokawa":
        distance = obs_data[:, 67] <= distance_limit
    else:
        distance = np.abs((obs_data[:, 69] - obs_data[:, 68]) * (obs_data[:, 71] - obs_data[:, 70])) <= area_limit

    return np.logical_and.reduce((phase_angle, incidence_angle, emission_angle, distance))


def filter_spectra(obs_data: np.ndarray) -> np.ndarray:
    # filter our specific spectra

    print("Filtering spectra before index computation can speed up the process but if you need different filtering, "
          "you must recompute the indices.")

    if np.ndim(obs_data) == 1:
        obs_data = np.reshape(obs_data, (1, len(obs_data)))

    mean_value = np.mean(obs_data[:, wavelengths_to_keep], axis=1) >= mean_value_limit

    return mean_value


def filter_specific_spectra(asteroid_name: str, obs_data: np.ndarray) -> np.ndarray:
    # this function filter out damaged spectra

    mask = np.ones(len(obs_data), dtype=bool)

    if asteroid_name == "Itokawa":
        mask[29747] = False
        mask[30164] = False
    else:
        mask[3703] = False

    return mask


# ------------------------------------------------------------------------------------------------------------------------
def shift_figure(poly, xoff: int = 0, yoff: int = 0):
    return affinity.translate(poly, xoff=xoff, yoff=yoff)


def polygon_within_intersect_polygon(target_poly, data_poly, brief_description: bool = False) -> int:
    # conditions were set to minimise processing time

    if target_poly.distance(data_poly) > 0.:  # no overlap
        return 0

    if brief_description:  # overlaps somehow
        return 4

    # how does it overlap?
    if target_poly.intersects(data_poly):  # intersects or touches
        # touches are efficiently removed by multiplying with intersect area in
        # the calc_weights; it can raise warnings "division by zero" when applying weights
        return 3

    elif data_poly.within(target_poly):  # data are inside target
        return 1

    else:  # target is inside the data; target_poly.within(data_poly)
        return 2


def point_within_polygon(point, polygon) -> int:
    if point.distance(polygon) > 0.:
        return 0
    else:  # target point is inside the data
        return 2


def figure_layout(pt_or_polygon, polygon, brief_description: bool = False) -> int:
    for shift_lon in [0, 360, -360]:
        if pt_or_polygon.geom_type == "Polygon":
            layout = polygon_within_intersect_polygon(shift_figure(pt_or_polygon, xoff=shift_lon), polygon,
                                                      brief_description=brief_description)
        else:
            layout = point_within_polygon(shift_figure(pt_or_polygon, xoff=shift_lon), polygon)

        if layout > 0:
            return layout

    return 0


# ------------------------------------------------------------------------------------------------------------------------

@timing
def get_indices_v1(asteroid_name: str, obs_data: np.ndarray, used_indices: np.ndarray,
                   polygon_or_point: Literal["polygon", "point"] = "polygon") -> np.ndarray:
    # This is faster if number of spectra > number of grid points

    # Checking if point is inside polygon
    print(f"\n[INFO] Testing how many polygons overlap given {polygon_or_point}")

    if polygon_or_point not in ["polygon", "point"]:
        raise ValueError('"polygon_or_point" must be either "polygon" or "point".')

    indices_to_save = len(lat_grid) * len(lon_grid) * [0]

    c = 0  # index in indices

    for lat in tqdm(lat_grid):
        for lon in lon_grid:

            row = [lon, lat]

            if polygon_or_point == "polygon":
                coordinates = np.array([[lon - dlon_half, lat - dlat_half], [lon - dlon_half, lat + dlat_half],
                                        [lon + dlon_half, lat + dlat_half], [lon + dlon_half, lat - dlat_half]])
                patch = Polygon(coordinates).convex_hull
            else:
                patch = Point(lon, lat)

            for index in used_indices:
                # Converting lat/lon to 360x180 pixel grid
                if asteroid_name == "Itokawa":
                    vector_lon_lat = obs_data[index, 70:]
                else:
                    vector_lon_lat = obs_data[index, 68:]

                coordinates = process_coords(asteroid_name, vector_lon_lat)
                polygon_data = Polygon(coordinates).convex_hull

                # this is slow but precise
                if figure_layout(patch, polygon_data, brief_description=True) > 0:
                    row.append(index)

            indices_to_save[c] = row
            c += 1

    return np.array(indices_to_save, dtype=object)


@timing
def get_indices_v2(asteroid_name: str, obs_data: np.ndarray, used_indices: np.ndarray,
                   polygon_or_point: Literal["polygon", "point"] = "polygon") -> np.ndarray:
    # This is faster if number of spectra < number of grid points

    # Checking if point is inside polygon
    print(f"\n[INFO] Testing how many polygons overlap given {polygon_or_point}")

    if polygon_or_point not in ["polygon", "point"]:
        raise ValueError('"polygon_or_point" must be either "polygon" or "point".')

    indices_to_save = len(obs_data) * [0]

    c = 0  # index in indices

    for index in tqdm(used_indices):

        row = [index]

        if asteroid_name == "Itokawa":
            vector_lon_lat = obs_data[index, 70:]
        else:
            vector_lon_lat = obs_data[index, 68:]

        coordinates = process_coords(asteroid_name, vector_lon_lat)
        polygon_data = Polygon(coordinates).convex_hull

        for lat in lat_grid:
            for lon in lon_grid:

                if polygon_or_point == "polygon":
                    coordinates = np.array([[lon - dlon_half, lat - dlat_half], [lon - dlon_half, lat + dlat_half],
                                            [lon + dlon_half, lat + dlat_half], [lon + dlon_half, lat - dlat_half]])
                    patch = Polygon(coordinates).convex_hull
                else:
                    patch = Point(lon, lat)

                # this is slow but precise
                if figure_layout(patch, polygon_data, brief_description=True) > 0:
                    row.append(index)

        indices_to_save[c] = row
        c += 1

    indices_to_save = reverse_indices(np.array(indices_to_save, dtype=object))

    return np.array(indices_to_save, dtype=object)


def reverse_indices(computed_indices: np.ndarray) -> np.ndarray:
    # clean the unused spectra
    inds_to_delete = [ind for ind, comp_inds in enumerate(computed_indices) if len(comp_inds) == 1]
    computed_indices = np.delete(computed_indices, inds_to_delete)

    # re-saving indices
    reversed_indices = len(lat_grid) * len(lon_grid) * [0]

    c = 0  # index in indices

    for lat in lat_grid:
        for lon in lon_grid:
            reversed_indices[c] = [lon, lat, *[line[0] for line in computed_indices if [lon, lat] in line]]
            c += 1

    return np.array(reversed_indices, dtype=object)


@timing
def get_indices(polygons_data: np.ndarray, used_indices: np.ndarray,
                polygon_or_point: Literal["polygon", "point"] = "polygon") -> np.ndarray:
    # This is faster if number of spectra > number of grid points

    # Checking if point is inside polygon
    print(f"\n[INFO] Testing how many polygons overlap given {polygon_or_point}")

    if polygon_or_point not in ["polygon", "point"]:
        raise ValueError('"polygon_or_point" must be either "polygon" or "point".')

    indices_to_save = len(lat_grid) * len(lon_grid) * [0]

    c = 0  # index in indices

    for lat in tqdm(lat_grid):
        for lon in lon_grid:

            row = [lon, lat]

            if polygon_or_point == "polygon":
                coordinates_patch = np.array([[lon - dlon_half, lat - dlat_half],
                                              [lon - dlon_half, lat + dlat_half],
                                              [lon + dlon_half, lat + dlat_half],
                                              [lon + dlon_half, lat - dlat_half]])
                patch = Polygon(coordinates_patch)
            else:
                patch = Point(lon, lat)

            for poly_data, index in zip(polygons_data, used_indices):

                if figure_layout(patch, poly_data, brief_description=True) > 0:
                    row.append(index)

            indices_to_save[c] = row
            c += 1

    return np.array(indices_to_save, dtype=object)


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
#                                            AVERAGING OF THE SPECTRA
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
def import_data(asteroid_name: str, file_dir: str) -> tuple[np.ndarray, ...]:
    obs_data = read_data(asteroid_name, file_dir)

    lines = read_indices(asteroid_name, file_dir)

    # lines which will be combined
    df = np.array([json.loads(line) for line in lines], dtype=object)

    inds_to_delete = [ind for ind, d in enumerate(df) if len(d) == 2]
    df = np.delete(df, inds_to_delete)

    if asteroid_name == "Itokawa":
        wavelengths = np.flip(obs_data[0, wavelengths_to_keep])
    else:
        wavelengths = stack((794.6 + 21.61 * np.arange(1., 33.),
                             43.11 * np.arange(33., 64.) - 50.8))[wavelengths_to_keep]

    return obs_data, df, wavelengths


# ------------------------------------------------------------------------------------------------------------------------
def combine_indices(df: np.ndarray, obs_data: np.ndarray) -> np.ndarray:
    # Combines the indices from ray_casing_file with the original dataset.
    # over spectra; skipping the 2 first values including coordinates
    return np.array([[obs_data[inds_spectra] for inds_spectra in line[2:]] for line in df], dtype=object)


# ------------------------------------------------------------------------------------------------------------------------
def filtering_data(asteroid_name: str, df: np.ndarray, combined: np.ndarray) -> tuple[np.ndarray, ...]:
    # Filtering away unwanted samples
    filtered_spectra = len(df) * [0]  # Used to calculate the mean spectra
    filtered_poly_coords = len(df) * [0]  # Used for calculating weight
    filtered_pt_coords = len(df) * [0]  # Finding the pixel-coordinates for filtered spectra
    for i, comb in enumerate(combined):
        spectra = len(comb) * [0]
        poly_coord = len(comb) * [0]
        # coordinates = len(combined[i]) * [0]

        for j, c in enumerate(comb):
            # additional filtering if needed
            # if something:
            if asteroid_name == "Itokawa":
                spectra[j] = np.flip(c[wavelengths_to_keep])
                poly_coord[j] = c[70:]
            else:
                spectra[j] = c[wavelengths_to_keep]
                poly_coord[j] = c[68:]

        # if "if" is added, this must be within the if
        # coordinates = [df[i][0], df[i][1]]  # coordinates (need to be done only once)

        filtered_spectra[i] = spectra
        filtered_poly_coords[i] = poly_coord
        filtered_pt_coords[i] = np.array([df[i][0], df[i][1]])

    filtered_spectra = np.array(filtered_spectra, dtype=object)
    filtered_poly_coords = np.array(filtered_poly_coords, dtype=object)
    filtered_pt_coords = np.array(filtered_pt_coords, dtype=np.float32)

    inds_to_delete = [ind for ind, filtered_spectrum in enumerate(filtered_spectra) if len(filtered_spectrum) == 0]
    filtered_spectra = np.delete(filtered_spectra, inds_to_delete)
    filtered_poly_coords = np.delete(filtered_poly_coords, inds_to_delete)
    filtered_pt_coords = np.delete(filtered_pt_coords, inds_to_delete, axis=0)

    return filtered_spectra, filtered_poly_coords, filtered_pt_coords


# ------------------------------------------------------------------------------------------------------------------------
def interp_and_norm(filtered_spectra: np.ndarray, wvl_old: np.ndarray,
                    wvl_new: np.ndarray, norm_at: float) -> np.ndarray:
    # Interpolate and normalise each filtered spectra
    interpolated = len(filtered_spectra) * [0]

    for i, filtered_spectrum in enumerate(filtered_spectra):
        spectra = len(filtered_spectrum) * [0]
        for j, single_spectrum in enumerate(filtered_spectrum):
            if interp_method == "linear":
                spectrum = np.interp(wvl_new, wvl_old, single_spectrum)
                norm_reflectance = np.interp(norm_at, wvl_new, spectrum)
            else:
                spectrum = interp1d(wvl_old, single_spectrum, kind=interp_method)(wvl_new)
                norm_reflectance = interp1d(wvl_new, spectrum, kind=interp_method)(norm_at)

            spectra[j] = spectrum / norm_reflectance
        interpolated[i] = spectra
    return np.array(interpolated, dtype=object)


# ------------------------------------------------------------------------------------------------------------------------
def calc_areas(asteroid_name: str, polygon_data_coords: np.ndarray, polygon_patch_coords: np.ndarray) -> np.ndarray:
    # Weighting each mean spectra with an inverse polygon area
    weighted_areas = len(polygon_data_coords) * [0]
    # Finding polygon areas and indexes
    for i, (data_coords, patch_coords) in enumerate(zip(polygon_data_coords, polygon_patch_coords)):
        area = len(data_coords) * [0]

        coordinates = np.array([[patch_coords[0] - dlon_half, patch_coords[1] - dlat_half],
                                [patch_coords[0] - dlon_half, patch_coords[1] + dlat_half],
                                [patch_coords[0] + dlon_half, patch_coords[1] + dlat_half],
                                [patch_coords[0] + dlon_half, patch_coords[1] - dlat_half]])
        patch = Polygon(coordinates).convex_hull
        patch_area = patch.area

        for j, coords in enumerate(data_coords):
            # It is not possible for measurement to be over 180 degrees (or across the 360 boundary)
            coordinates = process_coords(asteroid_name, coords)
            polygon_data = Polygon(coordinates).convex_hull
            data_area = polygon_data.area

            # this is valid for both point and polygon targets
            patch_polygon_layout = figure_layout(patch, polygon_data, brief_description=False)

            if patch_polygon_layout == 1:  # data inside patch (intersection.area = data_area)
                area[j] = 1. / patch_area
            elif patch_polygon_layout == 2:  # patch inside data (intersect.area = patch_area); applied for point target
                area[j] = 1. / data_area
            elif patch_polygon_layout == 3:  # general case
                area[j] = polygon_data.intersection(patch).area / data_area / patch_area

        weighted_areas[i] = np.array(area)
    return np.array(weighted_areas, dtype=object)


def weighting(filtered_spectra: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # do not add something like if normalise_array(weight) > 0. because you lose correct information about coordinates
    return np.array([np.sum(np.transpose(np.transpose(filtered_spectrum) * normalise_array(weight)), axis=0)
                     for filtered_spectrum, weight in zip(filtered_spectra, weights)])


# ------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # for what_asteroid in asteroid_list:  # run it in a loop but keep in mind some setting is asteroid-dependent
    # ------------------------------------------------------------------------------------------------------------------------
    if what_to_run in ["both", "indices"]:
        data = read_data(what_asteroid, spectra_dir)
        filtered_indices = np.where(np.logical_and.reduce((filter_incomplete_observations(data),
                                                           filter_observation_geometry(what_asteroid, data),
                                                           filter_specific_spectra(what_asteroid, data),
                                                           filter_spectra(data))))[0]

        # Converting lat/lon to 360x180 pixel grid
        if what_asteroid == "Itokawa":
            polygons = coordinates_to_polygons(what_asteroid, data[filtered_indices, 70:])
        else:
            polygons = coordinates_to_polygons(what_asteroid, data[filtered_indices, 68:])

        data = get_indices(polygons, filtered_indices, point_or_polygon)
        save_indices(what_asteroid, spectra_dir, data)
    # ------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------
    if what_to_run in ["both", "averaging"]:
        data, indices, wavelengths_old = import_data(what_asteroid, spectra_dir)

        data = combine_indices(indices, data)
        data, poly_coords, indices = filtering_data(what_asteroid, indices, data)

        data = interp_and_norm(data, wavelengths_old, wavelengths_new, normalised_at_wvl)

        areas = calc_areas(what_asteroid, poly_coords, indices)
        data = weighting(data, areas)

        saving_spectra(what_asteroid, spectra_dir, indices, data, wavelengths_new)
    # ------------------------------------------------------------------------------------------------------------------------
