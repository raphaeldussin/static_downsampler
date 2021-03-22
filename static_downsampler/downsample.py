#!/usr/bin/env python
import argparse

import numpy as np
import xarray as xr
from numba import njit

from static_downsampler.make_basin_mask_d2 import create_basin_code
from static_downsampler.static import apply_name_mapping, sum_by_2_elements


def downsample2_dataset(ds, sym=False):
    """ downsample the given dataset ds by 2 """

    ds_d2 = downsample2_nominal_coord(ds, sym=sym)
    ds_d2["geolon"] = downsample2_coord(ds, coord="lon", point="T", sym=sym)
    ds_d2["geolon_u"] = downsample2_coord(ds, coord="lon", point="U", sym=sym)
    ds_d2["geolon_v"] = downsample2_coord(ds, coord="lon", point="V", sym=sym)
    ds_d2["geolon_c"] = downsample2_coord(ds, coord="lon", point="Q", sym=sym)

    ds_d2["geolat"] = downsample2_coord(ds, coord="lat", point="T", sym=sym)
    ds_d2["geolat_u"] = downsample2_coord(ds, coord="lat", point="U", sym=sym)
    ds_d2["geolat_v"] = downsample2_coord(ds, coord="lat", point="V", sym=sym)
    ds_d2["geolat_c"] = downsample2_coord(ds, coord="lat", point="Q", sym=sym)

    ds_d2["Coriolis"] = downsample2_coord(ds, coord="Coriolis", point="Q", sym=sym)

    ds_d2["dxt"] = downsample2_gridmetric(ds, metric="dx", point="T", sym=sym)
    ds_d2["dxCu"] = downsample2_gridmetric(ds, metric="dx", point="U", sym=sym)
    ds_d2["dxCv"] = downsample2_gridmetric(ds, metric="dx", point="V", sym=sym)

    ds_d2["dyt"] = downsample2_gridmetric(ds, metric="dy", point="T", sym=sym)
    ds_d2["dyCu"] = downsample2_gridmetric(ds, metric="dy", point="U", sym=sym)
    ds_d2["dyCv"] = downsample2_gridmetric(ds, metric="dy", point="V", sym=sym)

    ds_d2["areacello"] = downsample2_2dvar(ds, "areacello", "T", op="sum", sym=sym)
    ds_d2["areacello_bu"] = downsample2_2dvar(
        ds, "areacello_bu", "Q", op="sum", sym=sym
    )
    ds_d2["areacello_cu"] = downsample2_2dvar(
        ds, "areacello_cu", "U", op="sum", sym=sym
    )
    ds_d2["areacello_cv"] = downsample2_2dvar(
        ds, "areacello_cv", "V", op="sum", sym=sym
    )

    ds_d2["deptho"] = downsample2_2dvar(ds, "deptho", "T", op="masked_avg", sym=sym)
    ds_d2["hfgeou"] = downsample2_2dvar(ds, "hfgeou", "T", op="masked_avg", sym=sym)
    ds_d2["sftof"] = downsample2_2dvar(ds, "sftof", "T", op="masked_avg", sym=sym)

    ds_d2["wet"] = downsample2_2dvar(ds, "wet", "T", op="mask", sym=sym)
    ds_d2["wet_c"] = downsample2_2dvar(ds, "wet_c", "Q", op="mask", sym=sym)
    ds_d2["wet_u"] = downsample2_2dvar(ds, "wet_u", "U", op="mask", sym=sym)
    ds_d2["wet_v"] = downsample2_2dvar(ds, "wet_v", "V", op="mask", sym=sym)

    ds_d2["basin"] = create_basin_code(ds_d2)

    override = downsample2_nominal_coord(ds, sym=sym)
    ds_d2["xh"] = override["xh"]
    ds_d2["xq"] = override["xq"]
    ds_d2["yh"] = override["yh"]
    ds_d2["yq"] = override["yq"]

    return ds_d2


def downsample2_nominal_coord(ds, sym=False):
    """downsample nominal coordinates by a factor of 2
    ds: xarray.Dataset
    """

    offset = 1 if sym else 0

    Is_h_d2 = 0 + offset
    Js_h_d2 = 0 + offset

    Is_q_d2 = np.mod(Is_h_d2 + 1, 2)
    Js_q_d2 = np.mod(Js_h_d2 + 1, 2)

    xh_d2 = ds["xq"].values[Is_h_d2::2]
    yh_d2 = ds["yq"].values[Js_h_d2::2]

    xq_d2 = ds["xq"].values[Is_q_d2::2]
    yq_d2 = ds["yq"].values[Js_q_d2::2]

    downsampled_ds = xr.Dataset(
        coords={
            "xh": (("xh"), xh_d2, {"units": "degrees_east", "cartesian_axis": "X"}),
            "yh": (("yh"), yh_d2, {"units": "degrees_north", "cartesian_axis": "Y"}),
            "xq": (("xq"), xq_d2, {"units": "degrees_east", "cartesian_axis": "X"}),
            "yq": (("yq"), yq_d2, {"units": "degrees_north", "cartesian_axis": "Y"}),
        }
    )
    return downsampled_ds


def define_staggering(point, sym=False):
    """ define starting I, J point and resulting dims """

    offset = 1 if sym else 0

    if point == "T":
        dims = ("yh", "xh")
        Is = 0 + offset
        Js = 0 + offset
    elif point == "U":
        dims = ("yh", "xq")
        Is = 1 - offset
        Js = 0 + offset
    elif point == "V":
        dims = ("yq", "xh")
        Is = 0 + offset
        Js = 1 - offset
    elif point == "Q":
        dims = ("yq", "xq")
        Is = 1 - offset
        Js = 1 - offset
    else:
        raise ValueError("unknow point type")
    return Is, Js, dims


def downsample2_coord(ds, coord="lon", point="T", sym=False):
    """downsample coord onto target point, all resulting points are coming from
    Q-points on the original grid with particular offset"""
    N = 2
    Is, Js, dims = define_staggering(point, sym=sym)

    if coord == "lon":
        varin = "geolon_c"
    elif coord == "lat":
        varin = "geolat_c"
    elif coord == "Coriolis":
        varin = "Coriolis"
    else:
        raise ValueError("unknown coordinate")

    data = ds[varin].values[Js::N, Is::N]
    out = xr.DataArray(data=data, dims=dims)
    return out


def downsample2_gridmetric(ds, metric="dx", point="T", sym=False):
    """get downsampled dx, dy from dxCv and dyCu: the axis of the metric
    is summed for each 2 elements, whereas the perpendicular axis is subsampled
    """

    # metrics on the subsampled grid will be derived from dxCv and dyCu
    # since the downsampled grid is build from Q-point of the native grid
    if metric == "dx":
        varin = "dxCv"
        dimsin = ("yq", "xh")
    elif metric == "dy":
        varin = "dyCu"
        dimsin = ("yh", "xq")

    # get the dimensions of the output array
    _, _, dims = define_staggering(point, sym=sym)
    # define the starting point following the N-E C-grid convention
    if point == "T":
        Is, Js = 0, 0
    elif point == "U":
        Is, Js = 1, 0
    elif point == "V":
        Is, Js = 0, 1
    # symetric grid add a row/column on the left/bottom
    # which will shift the starting point
    offset = 1 if sym else 0
    # on the subsample axis, add offset and limit to (0, 1)
    if metric == "dx":
        Js = np.mod(Js + offset, 2)
    elif metric == "dy":
        Is = np.mod(Is + offset, 2)

    # the summation axis has even number of values regardless of the grid type
    # hence it needs to be rolled to the starting index so it keeps an even
    # number of points, and subsampled starting from the index on the other axis
    if metric == "dx":
        rollaxis = "xh"  # rolling axis is X
        subaxis = "yq"  # subsample axis is Y
        sumaxis = 1  # summation axis is X (=1 for numpy)
        ny, nx = ds[varin].shape

        # subsample along Y
        tmp = ds[varin].isel({subaxis: slice(Js, ny + 1, 2)})
        # roll along X (correct if E-W periodic)
        tmp = tmp.roll({rollaxis: -Is}, roll_coords=False)
        # prepare for numpy
        workarray = tmp.transpose(*dimsin).values

    elif metric == "dy":
        rollaxis = "yh"  # rolling axis is Y
        subaxis = "xq"  # subsample axis is Y
        sumaxis = 0  # summation axis is Y (=0 for numpy)
        ny, nx = ds[varin].shape

        # subsample along X
        tmp = ds[varin].isel({subaxis: slice(Is, nx + 1, 2)})
        # roll along Y (towards south)
        tmp = tmp.roll({rollaxis: -Js}, roll_coords=False)
        # prepare for numpy
        workarray = tmp.transpose(*dimsin).values
        # correct north pole fold (assuming Q-pivot)
        if Js != 0:
            workarray[-1, :] = workarray[-2, ::-1]

    else:
        raise ValueError("unknown metric")

    data = sum_by_2_elements(workarray, axis=sumaxis)
    out = xr.DataArray(data=data, dims=dims)

    # in symetric grids, we need to add a row/column
    # if the point type of the metric is a corner type
    # on the metric axis (e.g. downsampled dxU has coords yh, xq)
    if sym:
        if (metric == "dx") and ("xq" in dims):
            # use E-W periodicity for dxU
            out = xr.concat([out.isel(xq=-1), out], dim="xq")
        if (metric == "dy") and ("yq" in dims):
            # duplicate southernmost row for dyV
            out = xr.concat([out.isel(yq=0), out], dim="yq")

    return out.transpose(*dims)


def downsample2_2dvar(ds, variable, point, op="sum", dimvar_map=None, sym=False):
    """ sum variables on super grid (e.g. dx, dy) """
    # if present, apply name mapping
    ds = apply_name_mapping(ds, dimvar_map=dimvar_map)

    # the areas all derive from areacello of the native grid
    areaname = "areacello"
    maskname = "wet"

    # get the dimensions of the output array
    _, _, dims = define_staggering(point, sym=sym)
    # define the starting point following the N-E C-grid convention
    if point == "T":
        Is, Js = 0, 0
        dims = ("yh", "xh")
    elif point == "U":
        Is, Js = 1, 0
        dims = ("yh", "xq")
    elif point == "V":
        Is, Js = 0, 1
        dims = ("yq", "xh")
    elif point == "Q":
        Is, Js = 1, 1
        dims = ("yq", "xq")
    else:
        raise ValueError("Unknown point type")

    dimsin = ("yh", "xh")
    xdim = find_dimname(dimsin, "x")
    ydim = find_dimname(dimsin, "y")

    # symetric grid do not matter since all native variables
    # are on the T-point

    area = ds[areaname]
    mask = ds[maskname]

    # roll along X (correct if E-W periodic)
    area = area.roll({xdim: -Is}, roll_coords=False)
    mask = mask.roll({xdim: -Is}, roll_coords=False)
    # roll along Y (towards south)
    area = area.roll({ydim: -Js}, roll_coords=False)
    mask = mask.roll({ydim: -Js}, roll_coords=False)
    # prepare for numpy
    workarea = area.transpose(*dimsin).values
    workmask = mask.transpose(*dimsin).values
    # correct north pole fold (assuming Q-pivot)
    if Js != 0:
        workarea[-1, :] = workarea[-2, ::-1]
        workmask[-1, :] = workmask[-2, ::-1]

    if "area" in variable:
        workarray = workarea
    elif "wet" in variable:
        workarray = workmask
    else:
        tmp = ds[variable]
        assert tmp.dims == dimsin
        # roll along X (correct if E-W periodic)
        tmp = tmp.roll({xdim: -Is}, roll_coords=False)
        # roll along Y (towards south)
        tmp = tmp.roll({ydim: -Js}, roll_coords=False)
        # prepare for numpy
        workarray = tmp.transpose(*dimsin).values
        # correct north pole fold (assuming Q-pivot)
        if Js != 0:
            workarray[-1, :] = workarray[-2, ::-1]

    ny, nx = workarea.shape
    assert int(ny / 2) == ny / 2
    assert int(nx / 2) == nx / 2

    ny, nx = workmask.shape
    assert int(ny / 2) == ny / 2
    assert int(nx / 2) == nx / 2

    ny, nx = workarray.shape
    assert int(ny / 2) == ny / 2
    assert int(nx / 2) == nx / 2

    if op == "sum":
        data = sum_kernel_d2(workarray)
    elif op == "masked_avg":
        data = masked_mean_kernel_d2(workarray, workarea * workmask)
    elif op == "mask":
        data = mask_kernel_d2(workarray)
    else:
        raise ValueError("Unknown operation")

    out = xr.DataArray(data=data, dims=dims)

    # in symetric grids, we need to add a row/column
    # if the point type of the metric is a corner type
    # on the metric axis (e.g. downsampled dxU has coords yh, xq)
    if sym:
        if "xq" in dims:
            # use E-W periodicity for dxU
            out = xr.concat([out.isel(xq=-1), out], dim="xq")
        if "yq" in dims:
            # duplicate southernmost row for dyV
            out = xr.concat([out.isel(yq=0), out], dim="yq")

    return out.transpose(*dims)


def sum_kernel_d2(array):
    """ sum of masked variables """
    out_x = sum_by_2_elements(array, axis=1)
    out_xy = sum_by_2_elements(out_x, axis=0)
    return out_xy


@njit
def masked_mean_kernel_d2(array, areacell):
    ny, nx = array.shape
    ny_d2, nx_d2 = int(ny / 2), int(nx / 2)
    out = np.zeros((ny_d2, nx_d2))
    for jj in range(0, ny_d2):
        for ji in range(0, nx_d2):
            # load values
            a00 = array[2 * jj, 2 * ji]
            a01 = array[2 * jj, 2 * ji + 1]
            a10 = array[2 * jj + 1, 2 * ji]
            a11 = array[2 * jj + 1, 2 * ji + 1]

            b00 = areacell[2 * jj, 2 * ji]
            b01 = areacell[2 * jj, 2 * ji + 1]
            b10 = areacell[2 * jj + 1, 2 * ji]
            b11 = areacell[2 * jj + 1, 2 * ji + 1]

            # count ocean values
            area_tot = 0
            if a00 != 0:
                area_tot += b00
            if a01 != 0:
                area_tot += b01
            if a10 != 0:
                area_tot += b10
            if a11 != 0:
                area_tot += b11

            if area_tot == 0:
                out[jj, ji] = 0
            else:
                out[jj, ji] = (a00 * b00 + a01 * b01 + a10 * b10 + a11 * b11) / area_tot
    return out


@njit
def mask_kernel_d2(array):
    """ downsample binary mask """
    ny, nx = array.shape
    ny_d2, nx_d2 = int(ny / 2), int(nx / 2)
    out = np.zeros((ny_d2, nx_d2))
    for jj in range(0, ny_d2):
        for ji in range(0, nx_d2):
            # load values
            a00 = array[2 * jj, 2 * ji]
            a01 = array[2 * jj, 2 * ji + 1]
            a10 = array[2 * jj + 1, 2 * ji]
            a11 = array[2 * jj + 1, 2 * ji + 1]
            # count ocean values
            npts = 0
            if a00 != 0:
                npts += 1
            if a01 != 0:
                npts += 1
            if a10 != 0:
                npts += 1
            if a11 != 0:
                npts += 1

            if npts == 0:
                out[jj, ji] = 0
            else:
                out[jj, ji] = 1
    return out


def is_symetric(ds):
    """check if grid is symetric or not"""
    if (len(ds["xq"]) == len(ds["xh"]) + 1) and (len(ds["yq"]) == len(ds["yh"]) + 1):
        return True
    elif (len(ds["xq"]) == len(ds["xh"])) and (len(ds["yq"]) == len(ds["yh"])):
        return False
    else:
        raise ValueError("inconsistent dimensions sizes")


def find_dimname(mylist, tag):
    """find the name of the dimension containing tag from mylist"""
    for dim in mylist:
        if dim.find(tag) == 0:
            return dim


def fix_attributes(ds, ds_d2):
    """ copy variable attributes from a source ds """
    listvar = list(ds_d2.variables)
    for v in listvar:
        if v in ds.variables:
            ds_d2[v].attrs = ds[v].attrs
        ds_d2[v].encoding = {"_FillValue": 1.0e20, "missing_value": 1.0e20}
    return ds_d2


def add_FREGRID_hack(ds_d2):
    """add netcdf attributes to areacello_* so that fregrid
    does not die when regridding static file"""
    ds_d2["areacello_bu"].attrs.update({"interp_method": "none"})
    ds_d2["areacello_cu"].attrs.update({"interp_method": "none"})
    ds_d2["areacello_cv"].attrs.update({"interp_method": "none"})
    return ds_d2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="create downsampled" + "static file")

    parser.add_argument("-s", "--static", required=True, help="input static file")
    parser.add_argument("-o", "--output", required=True, help="output d2 static file")

    args = parser.parse_args()

    ds = xr.open_dataset(args.static)
    sym = is_symetric(ds)
    ds_d2 = downsample2_dataset(ds, sym=sym)
    ds_d2 = fix_attributes(ds, ds_d2)
    ds_d2.attrs.update({"external_variables": "areacello areacello"})
    ds_d2 = add_FREGRID_hack(ds_d2)
    ds_d2.to_netcdf(args.output, format="NETCDF3_64BIT")
