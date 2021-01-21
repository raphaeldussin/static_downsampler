#!/usr/bin/env python
import argparse

import netCDF4
import numpy as np
import xarray as xr

from static_downsampler.static import sum_by_2_elements

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="create downsampled" + "ocean_hgrid file"
    )

    parser.add_argument("-g", "--grid", required=True, help="input ocean_hgrid.nc file")
    parser.add_argument(
        "-o", "--output", required=True, help="output ocean_hgrid.nc.d2 file"
    )

    args = parser.parse_args()

    hgrid = xr.open_dataset(args.grid)

    # coordinates
    x_d2 = hgrid["x"].values[0::2, 0::2]
    y_d2 = hgrid["y"].values[0::2, 0::2]

    # grid metrics
    dx_tmp = hgrid["dx"].values[::2, :]
    dy_tmp = hgrid["dy"].values[:, ::2]

    dx_d2 = sum_by_2_elements(dx_tmp, axis=1)
    dy_d2 = sum_by_2_elements(dy_tmp, axis=0)

    # cell areas
    area_x = sum_by_2_elements(hgrid["area"].values, axis=1)
    area_xy_d2 = sum_by_2_elements(area_x, axis=0)

    out = xr.Dataset()

    out["x"] = xr.DataArray(data=x_d2, dims=("nyp", "nxp"))
    out["y"] = xr.DataArray(data=y_d2, dims=("nyp", "nxp"))
    out["dx"] = xr.DataArray(data=dx_d2, dims=("nyp", "nx"))
    out["dy"] = xr.DataArray(data=dy_d2, dims=("ny", "nxp"))
    out["area"] = xr.DataArray(data=area_xy_d2, dims=("ny", "nx"))

    out.to_netcdf(args.output, format="NETCDF3_64BIT")

    # re-open to append tile information
    fout = netCDF4.Dataset(args.output, "a")
    string = fout.createDimension("string", 255)
    tile = fout.createVariable("tile", "S1", ("string"))
    stringvals = np.empty(1, "S" + repr(len(tile)))
    stringvals[0] = "tile1"
    tile[:] = netCDF4.stringtochar(stringvals)
    fout.close()
