# requires pytest-datafiles
import numpy as np
import pytest
import xarray as xr

from static_downsampler.static import (check_grid, define_start_point,
                                       sum_on_supergrid)

# define global dims size
ny = 180
nx = 360
lonE = -300
lonW = 60
latS = -90
latN = 90
hres = 1

ds_sym = xr.Dataset(
    data_vars=dict(
        dx=(["ny", "nx"], np.random.rand(ny, nx)),
        dy=(["ny", "nx"], np.random.rand(ny, nx)),
        area=(["ny", "nx"], np.random.rand(ny, nx)),
        angle=(["ny", "nx"], np.random.rand(ny, nx)),
    ),
    coords=dict(
        x=xr.DataArray(
            np.arange(lonE, lonW + hres),
            dims=["nxp"],
            attrs={
                "units": "degrees",
            },
        ),
        y=xr.DataArray(
            np.arange(latS, latN + hres),
            dims=["nyp"],
            attrs={
                "units": "degrees",
            },
        ),
    ),
)

ds_nonsym = xr.Dataset(
    data_vars=dict(
        dx=(["ny", "nx"], np.random.rand(ny, nx)),
        dy=(["ny", "nx"], np.random.rand(ny, nx)),
        area=(["ny", "nx"], np.random.rand(ny, nx)),
        angle=(["ny", "nx"], np.random.rand(ny, nx)),
    ),
    coords=dict(
        x=xr.DataArray(
            np.arange(lonE, lonW),
            dims=["nxp"],
            attrs={
                "units": "degrees",
            },
        ),
        y=xr.DataArray(
            np.arange(latS, latN),
            dims=["nyp"],
            attrs={
                "units": "degrees",
            },
        ),
    ),
)


@pytest.mark.parametrize("INPUTGRID", ["symetric"])
def test_check_grid(INPUTGRID):
    check_grid(ds_sym, INPUTGRID)
    # non-symetric supergrid is not allowed
    with pytest.raises(ValueError):
        check_grid(ds_nonsym, INPUTGRID)


@pytest.mark.parametrize("OUTPUTGRID", ["symetric", "nonsymetric"])
def test_define_start_point(OUTPUTGRID):
    isc, jsc, IscB, JscB = define_start_point("symetric", OUTPUTGRID)
    assert isc == 1
    assert jsc == 1
    assert int(IscB) == IscB
    assert int(JscB) == JscB
    # non-symetric supergrid is not allowed
    with pytest.raises(ValueError):
        _, _, _, _ = define_start_point("nonsymetric", OUTPUTGRID)


@pytest.mark.parametrize("POINTTYPE", ["T", "U", "V", "Q"])
@pytest.mark.parametrize("DIMSUM", [("x", "y"), ("x"), ("y")])
def test_sum_on_supergrid(POINTTYPE, DIMSUM):

    out = sum_on_supergrid(
        ds_sym, "dx", POINTTYPE, outputgrid="nonsymetric", dimsum=DIMSUM
    )
    assert out.shape == (ny / 2, nx / 2)

    out = sum_on_supergrid(
        ds_sym, "dx", POINTTYPE, outputgrid="symetric", dimsum=DIMSUM
    )
    if POINTTYPE == "Q":
        assert out.shape == (ny / 2 + 1, nx / 2 + 1)
    elif POINTTYPE == "U":
        assert out.shape == (ny / 2, nx / 2 + 1)
    elif POINTTYPE == "V":
        assert out.shape == (ny / 2 + 1, nx / 2)
    elif POINTTYPE == "T":
        assert out.shape == (ny / 2, nx / 2)

    # non-symetric supergrid is not allowed
    with pytest.raises(ValueError):
        out = sum_on_supergrid(ds_nonsym, "dx", POINTTYPE, outputgrid="symetric")
    with pytest.raises(ValueError):
        out = sum_on_supergrid(ds_nonsym, "dx", POINTTYPE, outputgrid="nonsymetric")

    # regional not yet implemented
    with pytest.raises(NotImplementedError):
        out = sum_on_supergrid(
            ds_sym, "dx", POINTTYPE, outputgrid="symetric", is_regional=True
        )
