# requires pytest-datafiles
import xarray as xr
import pytest
import numpy as np
import os
from static_downsampler.static import init_static_file
from static_downsampler.static import subsample_supergrid
from static_downsampler.static import correct_longitude_northfold

GRID_OM4_125 = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'files_OM4_125/',
    )

GRID_OM4_025 = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'files_OM4_025/',
    )

GRID_OM4_05 = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'files_OM4_05/',
    )


@pytest.mark.parametrize("FIXTURE_DIR", [GRID_OM4_125, GRID_OM4_025, GRID_OM4_05])
def test_lonlat(datafiles, FIXTURE_DIR):
    ds = xr.open_dataset(FIXTURE_DIR + 'ocean_hgrid.nc')
    ds_check = xr.open_dataset(FIXTURE_DIR + 'ocean_static.nc')

    static = init_static_file()
    static['geolon'] = subsample_supergrid(ds, 'x', 'T')
    static['geolat'] = subsample_supergrid(ds, 'y', 'T')
    assert np.array_equal(static['geolon'].values, ds_check['geolon'].values)
    assert np.array_equal(static['geolat'].values, ds_check['geolat'].values)

    static['geolon_u'] = subsample_supergrid(ds, 'x', 'U')
    static['geolat_u'] = subsample_supergrid(ds, 'y', 'U')
    assert np.array_equal(static['geolon_u'].values,
                          ds_check['geolon_u'].values)
    assert np.array_equal(static['geolat_u'].values,
                          ds_check['geolat_u'].values)

    static['geolon_v'] = subsample_supergrid(ds, 'x', 'V')
    static['geolon_v'] = correct_longitude_northfold(static['geolon_v'])
    static['geolat_v'] = subsample_supergrid(ds, 'y', 'V')
    assert np.array_equal(static['geolon_v'].values, ds_check['geolon_v'].values)
    assert np.array_equal(static['geolat_v'].values, ds_check['geolat_v'].values)

    static['geolon_c'] = subsample_supergrid(ds, 'x', 'Q')
    static['geolon_c'] = correct_longitude_northfold(static['geolon_c'], point='Q')
    static['geolat_c'] = subsample_supergrid(ds, 'y', 'Q')
    assert np.array_equal(static['geolon_c'].values, ds_check['geolon_c'].values)
    assert np.array_equal(static['geolat_c'].values, ds_check['geolat_c'].values)
