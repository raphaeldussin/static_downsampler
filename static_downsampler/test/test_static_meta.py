# requires pytest-datafiles
import os

import numpy as np
import pytest
import xarray as xr

from static_downsampler.static import (
    correct_longitude_northfold,
    init_static_file,
    subsample_supergrid,
    sum_on_supergrid,
)

GRID_OM4_125 = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "files_OM4_125/",
)

GRID_OM4_025 = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "files_OM4_025/",
)

GRID_OM4_05 = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "files_OM4_05/",
)


@pytest.mark.parametrize("FIXTURE_DIR", [GRID_OM4_125, GRID_OM4_025, GRID_OM4_05])
def test_lonlat(datafiles, FIXTURE_DIR):
    ds = xr.open_dataset(FIXTURE_DIR + "ocean_hgrid.nc")
    ds_check = xr.open_dataset(FIXTURE_DIR + "ocean_static.nc")

    static = init_static_file()
    static["geolon"] = subsample_supergrid(ds, "x", "T")
    static["geolat"] = subsample_supergrid(ds, "y", "T")
    assert np.array_equal(static["geolon"].values, ds_check["geolon"].values)
    assert np.array_equal(static["geolat"].values, ds_check["geolat"].values)

    static["geolon_u"] = subsample_supergrid(ds, "x", "U")
    static["geolat_u"] = subsample_supergrid(ds, "y", "U")
    assert np.array_equal(static["geolon_u"].values, ds_check["geolon_u"].values)
    assert np.array_equal(static["geolat_u"].values, ds_check["geolat_u"].values)

    static["geolon_v"] = subsample_supergrid(ds, "x", "V")
    static["geolon_v"] = correct_longitude_northfold(static["geolon_v"])
    static["geolat_v"] = subsample_supergrid(ds, "y", "V")
    assert np.array_equal(static["geolon_v"].values, ds_check["geolon_v"].values)
    assert np.array_equal(static["geolat_v"].values, ds_check["geolat_v"].values)

    static["geolon_c"] = subsample_supergrid(ds, "x", "Q")
    static["geolon_c"] = correct_longitude_northfold(static["geolon_c"], point="Q")
    static["geolat_c"] = subsample_supergrid(ds, "y", "Q")
    assert np.array_equal(static["geolon_c"].values, ds_check["geolon_c"].values)
    assert np.array_equal(static["geolat_c"].values, ds_check["geolat_c"].values)


@pytest.mark.parametrize("FIXTURE_DIR", [GRID_OM4_125, GRID_OM4_025, GRID_OM4_05])
def test_dxdy(datafiles, FIXTURE_DIR):
    ds = xr.open_dataset(FIXTURE_DIR + "ocean_hgrid.nc")
    ds_check = xr.open_dataset(FIXTURE_DIR + "ocean_static.nc")

    static = init_static_file()
    static["dxt"] = sum_on_supergrid(ds, "dx", "T", dimsum=("x"))
    assert np.array_equal(static["dxt"].values, ds_check["dxt"].values)

    static["dyt"] = sum_on_supergrid(ds, "dy", "T", dimsum=("y"))
    assert np.array_equal(static["dyt"].values, ds_check["dyt"].values)

    static["dxCu"] = sum_on_supergrid(ds, "dx", "U", dimsum=("x"))
    static["dyCu"] = sum_on_supergrid(ds, "dy", "U", dimsum=("y"))
    assert np.array_equal(static["dxCu"].values, ds_check["dxCu"].values)
    # model cannot have zero dy but supergrid can, absolute tolerance is 2 meters
    assert np.allclose(static["dyCu"].values, ds_check["dyCu"].values, atol=2.0)

    static["dxCv"] = sum_on_supergrid(ds, "dx", "V", dimsum=("x"))
    static["dyCv"] = sum_on_supergrid(ds, "dy", "V", dimsum=("y"))
    assert np.array_equal(static["dxCv"].values, ds_check["dxCv"].values)
    assert np.array_equal(static["dyCv"].values, ds_check["dyCv"].values)


@pytest.mark.parametrize("FIXTURE_DIR", [GRID_OM4_125, GRID_OM4_025, GRID_OM4_05])
def test_area(datafiles, FIXTURE_DIR):
    ds = xr.open_dataset(FIXTURE_DIR + "ocean_hgrid.nc")
    ds_check = xr.open_dataset(FIXTURE_DIR + "ocean_static.nc")

    static = init_static_file()
    static["areacello"] = sum_on_supergrid(ds, "area", "T", dimsum=("x", "y"))
    assert np.allclose(
        static["areacello"].values, ds_check["areacello"].values, atol=50000
    )
