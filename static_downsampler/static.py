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
    outputgrid="nonsymetric",
    dimvar_map=None,
    dimsum=("x", "y"),
    is_regional=False,
):
    """ sum variables on super grid (e.g. dx, dy) """

    if is_regional:
        raise NotImplementedError("this is not suited for regional grids yet")

    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)
    # check grid for consistency, supergrid must be symetric
    check_grid(tmp, "symetric")
    # getting start point for array
    isc, jsc, IscB, JscB = define_start_point("symetric", outputgrid)

    if not is_regional:
        # use periodicity and north pole fold to extend supergrid
        ext_array = extend_supergrid_array(tmp[variable])
        # we need to shift indices to account for the extra
        # row/column on the south/east
        isc += 1
        jsc += 1
        IscB += 1
        JscB += 1
    else:
        ext_array = tmp[variable]

    ny, nx = ext_array.shape
    last = 0 if is_regional else 1

    if pointtype == "T":
        dims = ("yh", "xh")
        workarray = ext_array.isel(
            x=slice(isc - 1, nx - last), y=slice(jsc - 1, ny - last)
        )
    elif pointtype == "U":
        dims = ("yh", "xq")
        workarray = ext_array.isel(x=slice(IscB - 1, nx), y=slice(jsc - 1, ny - last))
    elif pointtype == "V":
        dims = ("yq", "xh")
        workarray = ext_array.isel(x=slice(isc - 1, nx - last), y=slice(JscB - 1, ny))
    elif pointtype == "Q":
        dims = ("yq", "xq")
        workarray = ext_array.isel(x=slice(IscB - 1, nx), y=slice(JscB - 1, ny))

    ny, nx = workarray.shape
    if "x" in dimsum:
        out_x = workarray.coarsen(x=2).sum()
    else:
        out_x = workarray.isel(x=slice(1, nx, 2))
    if "y" in dimsum:
        out_xy = out_x.coarsen(y=2).sum()
    else:
        out_xy = out_x.isel(y=slice(1, ny, 2))

    out = xr.DataArray(data=out_xy.values.astype("f4"), dims=dims)
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
        raise ValueError("Supergrid should never be nonsymetric")
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


def extend_supergrid_array(array):
    """ extend supergrid array, assuming periodic/tripolar grid"""
    tmp = array.values
    inner = xr.DataArray(tmp, dims=("y", "x"))
    # northfold: take areas of north row and mirror it
    northfold = xr.DataArray(tmp[-1, ::-1], dims=("x"))
    # southern boundary
    south = xr.DataArray(tmp[0, :], dims=("x"))
    inner_w_poles = xr.concat([south, inner, northfold], dim="y")
    # apply E-W boundary condition
    west = xr.DataArray(inner_w_poles[-1, :], dims=("y"))
    east = xr.DataArray(inner_w_poles[0, :], dims=("y"))
    out = xr.concat([west, inner_w_poles, east], dim="x")
    return out
