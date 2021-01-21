#!/usr/bin/env python
import argparse

import numpy as np
import xarray as xr
from numba import njit

from static_downsampler.make_basin_mask_d2 import create_basin_code
from static_downsampler.static import apply_name_mapping, sum_by_2_elements


def downsample2_dataset(ds):
    """ downsample the given dataset ds by 2 """
    ds_d2 = downsample2_nominal_coord(ds)
    ds_d2["geolon"] = downsample2_coord(ds, coord="lon", point="T")
    ds_d2["geolon_u"] = downsample2_coord(ds, coord="lon", point="U")
    ds_d2["geolon_v"] = downsample2_coord(ds, coord="lon", point="V")
    ds_d2["geolon_c"] = downsample2_coord(ds, coord="lon", point="Q")

    ds_d2["geolat"] = downsample2_coord(ds, coord="lat", point="T")
    ds_d2["geolat_u"] = downsample2_coord(ds, coord="lat", point="U")
    ds_d2["geolat_v"] = downsample2_coord(ds, coord="lat", point="V")
    ds_d2["geolat_c"] = downsample2_coord(ds, coord="lat", point="Q")

    ds_d2["Coriolis"] = downsample2_coord(ds, coord="Coriolis", point="Q")

    ds_d2["dxt"] = downsample2_gridmetric(ds, metric="dx", point="T")
    ds_d2["dxCu"] = downsample2_gridmetric(ds, metric="dx", point="U")
    ds_d2["dxCv"] = downsample2_gridmetric(ds, metric="dx", point="V")

    ds_d2["dyt"] = downsample2_gridmetric(ds, metric="dy", point="T")
    ds_d2["dyCu"] = downsample2_gridmetric(ds, metric="dy", point="U")
    ds_d2["dyCv"] = downsample2_gridmetric(ds, metric="dy", point="V")

    ds_d2["areacello"] = downsample2_2dvar(ds, "areacello", "T", op="sum")
    ds_d2["areacello_bu"] = downsample2_2dvar(ds, "areacello_bu", "Q", op="sum")
    ds_d2["areacello_cu"] = downsample2_2dvar(ds, "areacello_cu", "U", op="sum")
    ds_d2["areacello_cv"] = downsample2_2dvar(ds, "areacello_cv", "V", op="sum")

    ds_d2["deptho"] = downsample2_2dvar(ds, "deptho", "T", op="masked_avg")
    ds_d2["hfgeou"] = downsample2_2dvar(ds, "hfgeou", "T", op="masked_avg")
    ds_d2["sftof"] = downsample2_2dvar(ds, "sftof", "T", op="masked_avg")

    ds_d2["wet"] = downsample2_2dvar(ds, "wet", "T", op="mask")
    ds_d2["wet_c"] = downsample2_2dvar(ds, "wet_c", "Q", op="mask")
    ds_d2["wet_u"] = downsample2_2dvar(ds, "wet_u", "U", op="mask")
    ds_d2["wet_v"] = downsample2_2dvar(ds, "wet_v", "V", op="mask")

    ds_d2["basin"] = create_basin_code(ds_d2)

    override = downsample2_nominal_coord(ds)
    ds_d2["xh"] = override["xh"]
    ds_d2["xq"] = override["xq"]
    ds_d2["yh"] = override["yh"]
    ds_d2["yq"] = override["yq"]

    return ds_d2


def downsample2_nominal_coord(ds):
    """downsample nominal coordinates by a factor of 2
    ds: xarray.Dataset
    """

    Is = 0  # MOM6 notation: grid edge
    Js = 0
    xh_d2 = ds["xq"].values[Is::2]
    yh_d2 = ds["yq"].values[Js::2]

    xq_d2 = ds["xq"].values[(Is + 1)::2]
    yq_d2 = ds["yq"].values[(Js + 1)::2]

    downsampled_ds = xr.Dataset(
        coords={
            "xh": (("xh"), xh_d2, {"units": "degrees_east", "cartesian_axis": "X"}),
            "yh": (("yh"), yh_d2, {"units": "degrees_north", "cartesian_axis": "Y"}),
            "xq": (("xq"), xq_d2, {"units": "degrees_east", "cartesian_axis": "X"}),
            "yq": (("yq"), yq_d2, {"units": "degrees_north", "cartesian_axis": "Y"}),
        }
    )
    return downsampled_ds


def define_staggering(point):
    """ define starting I, J point and resulting dims """
    if point == "T":
        dims = ("yh", "xh")
        Is = 0
        Js = 0
    elif point == "U":
        dims = ("yh", "xq")
        Is = 1
        Js = 0
    elif point == "V":
        dims = ("yq", "xh")
        Is = 0
        Js = 1
    elif point == "Q":
        dims = ("yq", "xq")
        Is = 1
        Js = 1
    else:
        raise ValueError("unknow point type")
    return Is, Js, dims


def downsample2_coord(ds, coord="lon", point="T"):
    """downsample coord onto target point, all resulting points are coming from
    Q-points on the original grid with particular offset"""
    N = 2
    Is, Js, dims = define_staggering(point)

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


def downsample2_gridmetric(ds, metric="dx", point="T"):
    """get dx, dy as sum of dxCv and dyCu with appropriate
    staggering"""

    Is, Js, dims = define_staggering(point)

    if metric == "dx":
        varin = "dxCv"
        axis = 1
        workarray = np.roll(ds[varin].values, -1, axis=1)[::2, :]
    elif metric == "dy":
        varin = "dyCu"
        axis = 0
        workarray = np.roll(ds[varin].values, -1, axis=0)[:, ::2]
        workarray[-1, :] = workarray[-2, :]
    else:
        raise ValueError("unknown coordinate")

    data = sum_by_2_elements(workarray, axis=axis)
    out = xr.DataArray(data=data, dims=dims)
    return out


def downsample2_2dvar(ds, variable, pointtype, op="sum", dimvar_map=None):
    """ sum variables on super grid (e.g. dx, dy) """
    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)

    if pointtype == "T":
        dims = ("yh", "xh")
        workarray = tmp[variable].values
        areacell = tmp["areacello"].values
    elif pointtype == "U":
        dims = ("yh", "xq")
        workarray = np.roll(tmp[variable].values, -1, axis=1)
        areacell = np.roll(tmp["areacello_cu"].values, -1, axis=1)
    elif pointtype == "V":
        dims = ("yq", "xh")
        workarray = np.roll(tmp[variable].values, -1, axis=0)
        workarray[-1, :] = workarray[-2, :]
        areacell = np.roll(tmp["areacello_cv"].values, -1, axis=0)
        areacell[-1, :] = areacell[-2, :]
    elif pointtype == "Q":
        dims = ("yq", "xq")
        workarray = np.roll(tmp[variable].values, -1, axis=0)
        workarray[-1, :] = workarray[-2, :]
        workarray = np.roll(workarray, -1, axis=1)
        areacell = np.roll(tmp["areacello_bu"].values, -1, axis=0)
        areacell[-1, :] = areacell[-2, :]
        areacell = np.roll(areacell, -1, axis=1)
    else:
        raise NotImplementedError("Unknown point type")

    ny, nx = workarray.shape
    assert int(ny / 2) == ny / 2
    assert int(nx / 2) == nx / 2

    if op == "sum":
        data = sum_kernel_d2(workarray)
    elif op == "masked_avg":
        data = masked_mean_kernel_d2(workarray, areacell)
    elif op == "mask":
        data = mask_kernel_d2(workarray)
    else:
        raise ValueError("Unknown operation")

    out = xr.DataArray(data=data, dims=dims)
    return out


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
    ds_d2 = downsample2_dataset(ds)
    ds_d2 = fix_attributes(ds, ds_d2)
    ds_d2.attrs.update({"external_variables": "areacello areacello"})
    ds_d2 = add_FREGRID_hack(ds_d2)
    ds_d2.to_netcdf(args.output, format="NETCDF3_64BIT")
