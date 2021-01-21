import itertools as it

import numpy as np
import xarray as xr


def init_static_file():
    """ create an empty xarray dataset"""
    static = xr.Dataset()
    return static


def subsample_supergrid(
    ds,
    variable,
    pointtype,
    inputgrid="symetric",
    outputgrid="nonsymetric",
    dimvar_map=None,
):
    """ subsample on a given pointtype variable from dataset ds """

    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)
    # check grid for consistency
    check_grid(tmp, inputgrid)
    # getting start point for array
    isc, jsc, IscB, JscB = define_start_point(inputgrid, outputgrid)

    if pointtype == "T":
        data = tmp[variable].values[jsc::2, isc::2]
        dims = ("yh", "xh")
    elif pointtype == "U":
        data = tmp[variable].values[jsc::2, IscB::2]
        dims = ("yh", "xq")
    elif pointtype == "V":
        data = tmp[variable].values[JscB::2, isc::2]
        dims = ("yq", "xh")
    elif pointtype == "Q":
        data = tmp[variable].values[JscB::2, IscB::2]
        dims = ("yq", "xq")

    if data.dtype == np.dtype("f8"):
        data = data.astype("f4")

    out = xr.DataArray(data=data, dims=dims)
    return out


def sum_on_supergrid(
    ds,
    variable,
    pointtype,
    inputgrid="symetric",
    outputgrid="nonsymetric",
    dimvar_map=None,
    dimsum=("x", "y"),
):
    """ sum variables on super grid (e.g. dx, dy) """
    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)
    # check grid for consistency
    check_grid(tmp, inputgrid)
    # getting start point for array
    isc, jsc, IscB, JscB = define_start_point(inputgrid, outputgrid)

    if pointtype == "T":
        dims = ("yh", "xh")
        workarray = tmp[variable].values[0:, 0:]
    elif pointtype == "U":
        dims = ("yh", "xq")
        workarray = np.roll(tmp[variable].values[0:, 0:], -1, axis=1)
    elif pointtype == "V":
        dims = ("yq", "xh")
        workarray = np.roll(tmp[variable].values[0:, 0:], -1, axis=0)
        workarray[-1, :] = workarray[-2, :]
    elif pointtype == "Q":
        dims = ("yq", "xq")
        workarray = np.roll(tmp[variable].values[0:, 0:], -1, axis=0)
        workarray[-1, :] = workarray[-2, :]
        workarray = np.roll(workarray, -1, axis=1)
    else:
        raise NotImplementedError("Unknown point type")

    if "x" in dimsum:
        out_x = sum_by_2_elements(workarray, axis=1)
    else:
        out_x = workarray[:, 1::2]
    if "y" in dimsum:
        out_xy = sum_by_2_elements(out_x, axis=0)
    else:
        out_xy = out_x[1::2, :]

    out = xr.DataArray(data=out_xy.astype("f4"), dims=dims)
    return out


def apply_name_mapping(ds, dimvar_map=None):
    """handle non-standard variable/dimension names by applying a
    reverse mapping"""
    if dimvar_map is not None:
        map_reversed = dict([[v, k] for k, v in dimvar_map.items()])
        out = ds.rename(map_reversed)
    else:
        out = ds
    return out


def check_grid(ds, inputgrid):
    # sanity check on grid
    if inputgrid == "symetric":
        if (len(ds["nxp"]) != len(ds["nx"]) + 1) or (
            len(ds["nyp"]) != len(ds["ny"]) + 1
        ):
            raise ValueError("input supergrid is not symetric")
    return None


def define_start_point(inputgrid, outputgrid):
    """ define start index for array according to grid symmetry"""
    if inputgrid == "symetric":
        isc = 1
        jsc = 1
        if outputgrid == "nonsymetric":
            IscB = 2
            JscB = 2
        elif outputgrid == "symetric":
            IscB = 0
            JscB = 0
    elif inputgrid == "nonsymetric":
        raise NotImplementedError("Supergrid should never be nonsymetric")
    else:
        raise ValueError("input grid can only be symetric or nonsymetric")
    return isc, jsc, IscB, JscB


def correct_longitude_northfold(da, point="V"):
    """ correct longitude on north fold (only for geolon_v, geolon_c) """
    # get shape of array
    ny, nx = da.values.shape
    # find second node of tripolar grid
    node2 = 3 * nx / 4
    inode2 = int(node2)
    if inode2 != node2:
        raise ValueError("cannot divide nx by 4")

    data = da.values
    data[-1, inode2:] -= 360.0
    if point == "Q":
        data[-1, inode2 - 1] -= 180.0
    out = xr.DataArray(data=data, dims=da.dims)
    return out


def sum_by_2_elements(array, axis=0):
    """ sum pair of elements of array along axis"""
    if axis == 0:
        out = np.array(
            [sum(r) for r in it.zip_longest(array[::2, :], array[1::2, :], fillvalue=0)]
        )
    elif axis == 1:
        out = np.array(
            [sum(r) for r in it.zip_longest(array[:, ::2], array[:, 1::2], fillvalue=0)]
        )
    return out
