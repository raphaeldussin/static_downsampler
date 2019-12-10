import xarray as xr
# import numpy as np


def init_static_file():
    """ create an empty xarray dataset"""
    static = xr.Dataset()
    return static


def subsample_supergrid(ds, variable, pointtype, inputgrid='symetric',
                        outputgrid='nonsymetric', dimvar_map=None):
    """ subsample on a given pointtype variable from dataset ds """

    # if present, apply name mapping
    tmp = apply_name_mapping(ds, dimvar_map=dimvar_map)

    # sanity check on grid
    if inputgrid == 'symetric':
        if ((len(tmp['nxp']) != len(tmp['nx']) + 1) or
           (len(tmp['nyp']) != len(tmp['ny']) + 1)):
            raise ValueError('input supergrid is not symetric')

    # define start/end points depending on grid properties
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
        raise NotImplementedError()
    else:
        raise ValueError('input grid can only be symetric or nonsymetric')

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

    out = xr.DataArray(data=data, dims=dims)
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



if __name__ == "__main__":
    ds = xr.open_dataset('data/ocean_hgrid.nc')
    #ds2 = ds.rename({'x': 'lon'})
    #print(ds2)
    #dimvar_map = {'x': 'lon'}
    geolon = subsample_supergrid(ds, 'x', 'T', inputgrid='symetric')
    #print(geolon)
    static = init_static_file()
    static['geolon'] = subsample_supergrid(ds, 'x', 'T', inputgrid='symetric')
    static['geolon_c'] = subsample_supergrid(ds, 'x', 'Q', inputgrid='symetric')
    print(static)