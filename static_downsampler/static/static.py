import xarray as xr
import itertools as it
import numpy as np


def init_static_file():
    """ create an empty xarray dataset"""
    static = xr.Dataset()
    return static


def subsample_supergrid(ds, variable, pointtype, inputgrid='symetric',
                        outputgrid='nonsymetric', dimvar_map=None):
    """ subsample on a given pointtype variable from dataset ds """

    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)
    # check grid for consistency
    check_grid(tmp, inputgrid)
    # getting start point for array
    isc, jsc, IscB, JscB = define_start_point(inputgrid, outputgrid)

    if pointtype == 'T':
        data = tmp[variable].values[jsc::2, isc::2]
        dims = ('yh', 'xh')
    elif pointtype == 'U':
        data = tmp[variable].values[jsc::2, IscB::2]
        dims = ('yh', 'xq')
    elif pointtype == 'V':
        data = tmp[variable].values[JscB::2, isc::2]
        dims = ('yq', 'xh')
    elif pointtype == 'Q':
        data = tmp[variable].values[JscB::2, IscB::2]
        dims = ('yq', 'xq')

    if data.dtype == np.dtype('f8'):
        data = data.astype('f4')

    out = xr.DataArray(data=data, dims=dims)
    return out


def sum_on_supergrid(ds, variable, pointtype, inputgrid='symetric',
                     outputgrid='nonsymetric', dimvar_map=None,
                     dimsum=('x', 'y')):
    """ sum variables on super grid (e.g. dx, dy) """
    #print(pointtype)
    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)
    # check grid for consistency
    check_grid(tmp, inputgrid)
    # getting start point for array
    isc, jsc, IscB, JscB = define_start_point(inputgrid, outputgrid)

    if pointtype == 'T':
        dims = ('yh', 'xh')
        workarray = tmp[variable].values[0:, 0:]
    elif pointtype == 'U':
        dims = ('yh', 'xq')
        workarray = np.roll(tmp[variable].values[0:, 0:], -1, axis=1)
    elif pointtype == 'V':
        dims = ('yq', 'xh')
        workarray = np.roll(tmp[variable].values[0:, 0:], -1, axis=0)
        workarray[-1, :] = workarray[-2, :]
    elif pointtype == 'Q':
        dims = ('yq', 'xq')
        workarray = np.roll(tmp[variable].values[0:, 0:], -1, axis=0)
        workarray[-1, :] = workarray[-2, :]
        workarray = np.roll(workarray, -1, axis=1)
    else:
        raise NotImplementedError('Unknown point type')

    if 'x' in dimsum:
        out_x = sum_by_2_elements(workarray, axis=1)
    else:
        out_x = workarray[:, 1::2]
    if 'y' in dimsum:
        out_xy = sum_by_2_elements(out_x, axis=0)
    else:
        out_xy = out_x[1::2, :]

    out = xr.DataArray(data=out_xy.astype('f4'), dims=dims)
    return out


def apply_name_mapping(ds, dimvar_map=None):
    """ handle non-standard variable/dimension names by applying a
    reverse mapping"""
    if dimvar_map is not None:
        map_reversed = dict([[v, k] for k, v in dimvar_map.items()])
        out = ds.rename(map_reversed)
    else:
        out = ds
    return out


def check_grid(ds, inputgrid):
    # sanity check on grid
    if inputgrid == 'symetric':
        if ((len(ds['nxp']) != len(ds['nx']) + 1) or
           (len(ds['nyp']) != len(ds['ny']) + 1)):
            raise ValueError('input supergrid is not symetric')
    return None


def define_start_point(inputgrid, outputgrid):
    """ define start index for array according to grid symmetry"""
    if inputgrid == 'symetric':
        isc = 1
        jsc = 1
        if outputgrid == 'nonsymetric':
            IscB = 2
            JscB = 2
        elif outputgrid == 'symetric':
            IscB = 0
            JscB = 0
    elif inputgrid == 'nonsymetric':
        raise NotImplementedError('Supergrid should never be nonsymetric')
    else:
        raise ValueError('input grid can only be symetric or nonsymetric')
    return isc, jsc, IscB, JscB


def correct_longitude_northfold(da, point='V'):
    """ correct longitude on north fold (only for geolon_v, geolon_c) """
    # get shape of array
    ny, nx = da.values.shape
    # find second node of tripolar grid
    node2 = 3 * nx / 4
    inode2 = int(node2)
    if inode2 != node2:
        raise ValueError("cannot divide nx by 4")

    data = da.values
    data[-1, inode2:] -= 360.
    if point == 'Q':
        data[-1, inode2-1] -= 180.
    out = xr.DataArray(data=data, dims=da.dims)
    return out


def sum_by_2_elements(array, axis=0):
    """ sum pair of elements of array along axis"""
    if axis == 0:
        out = np.array([sum(r) for r in it.zip_longest(array[::2, :],
                                                       array[1::2, :],
                                                       fillvalue=0)])
    elif axis == 1:
        out = np.array([sum(r) for r in it.zip_longest(array[:, ::2],
                                                       array[:, 1::2],
                                                       fillvalue=0)])
    return out


if __name__ == "__main__":
    ds = xr.open_dataset('data/ocean_hgrid.nc')
    ds_check = xr.open_dataset('data/ocean_monthly_z.static.nc')

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

    print('lon/lat OK')

    static['dxt'] = sum_on_supergrid(ds, 'dx', 'T', dimsum=('x'))
    assert np.array_equal(static['dxt'].values, ds_check['dxt'].values)

    static['dyt'] = sum_on_supergrid(ds, 'dy', 'T', dimsum=('y'))
    assert np.array_equal(static['dyt'].values, ds_check['dyt'].values)

    static['dxCu'] = sum_on_supergrid(ds, 'dx', 'U', dimsum=('x'))
    static['dyCu'] = sum_on_supergrid(ds, 'dy', 'U', dimsum=('y'))
    assert np.array_equal(static['dxCu'].values, ds_check['dxCu'].values)
    # model cannot have zero dy but supergrid can, absolute tolerance is 0.1m
    assert np.allclose(static['dyCu'].values, ds_check['dyCu'].values, atol=0.1)

    static['dxCv'] = sum_on_supergrid(ds, 'dx', 'V', dimsum=('x'))
    static['dyCv'] = sum_on_supergrid(ds, 'dy', 'V', dimsum=('y'))
    assert np.array_equal(static['dxCv'].values, ds_check['dxCv'].values)
    assert np.array_equal(static['dyCv'].values, ds_check['dyCv'].values)

    print('dx/dy ok')

    static['areacello'] = sum_on_supergrid(ds, 'area', 'T', dimsum=('x', 'y'))
    assert np.allclose(static['areacello'].values, ds_check['areacello'].values, atol=1000)

    static['areacello_bu'] = sum_on_supergrid(ds, 'area', 'Q', dimsum=('x', 'y'))
    #assert np.allclose(static['areacello_bu'].values, ds_check['areacello_bu'].values, atol=1000)


    #assert np.array_equal(static['dyCv'].values, ds_check['dyCv'].values)
    static['areacello_cu'] = sum_on_supergrid(ds, 'area', 'U', dimsum=('x', 'y'))
    static['areacello_cv'] = sum_on_supergrid(ds, 'area', 'V', dimsum=('x', 'y'))


    #print('area ok')
    #exit()
    #print(ds_check['dyCu'].values[-1,1439])
    #print(static['dyCu'].values[-1,1439])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.pcolormesh(np.ma.masked_values((static['areacello'].values - ds_check['areacello'].values), 0))
    plt.colorbar()

    plt.figure()
    plt.pcolormesh(np.ma.masked_values((static['areacello_bu'].values - ds_check['areacello_bu'].values), 0))
    plt.colorbar()

    plt.figure()
    generated = static['areacello_cu'].values
    original = ds_check['areacello_cu'].values
    generated[np.where(original == 0)] = 0

    print(generated - original)
    plt.pcolormesh(np.ma.masked_values(generated - original, 0))
    plt.colorbar()

    plt.figure()
    
    generated = static['areacello_cv'].values
    original = ds_check['areacello_cv'].values
    generated[np.where(original == 0)] = 0

    print((generated - original).max())

    plt.pcolormesh(np.ma.masked_values(generated - original, 0))
    
    plt.pcolormesh(np.ma.masked_values((static['areacello_cv'].values - ds_check['areacello_cv'].values), 0))
    plt.colorbar()

    plt.show()


    #print(static['dxt'].values - ds_check['dxt'].values)
    #static['areacello'] = sum_on_supergrid(ds, 'area', 'T', dimsum=('x', 'y'))
    #assert np.allclose(static['areacello'].values, ds_check['areacello'].values, rtol=0.01)
    #assert np.array_equal(static['areacello'].values, ds_check['areacello'].values)
