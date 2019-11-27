import numpy as np
import xarray as xr


def downsample_dataset(dataset, methods, N=2):
    """ downsample the given dataset by N using the methods
    given in the methods dictionary (key: variable,
    value: [method, opt other var])
    """

    # first create empty dataset with the downsampled coordinates
    downsampled_ds = downsample_coord(dataset, N=N)

    for var in methods.keys():
        method, optvar = methods[var]
        if optvar is None:
            if method == 'sum':
                downsampled_ds[var] = downsample_sum(dataset[var],
                                                     downsampled_ds, N=N)
        pass

    return downsampled_ds


def downsample_sum(da, ds_down, N=2):

    data = da.values
    
    return out


def downsample_coord(ds, N=2):
    """ downsample nominal coordinates by a factor of N
    ds: xarray.Dataset
    N: int
    """

    if np.mod(N, 2) == 0:
        indstart = int(N/2)-1
        xh_down = ds['xq'].values[indstart::N]
        yh_down = ds['yq'].values[indstart::N]
    else:
        indstart = int((N-1)/2)
        xh_down = ds['xh'].values[indstart::N]
        yh_down = ds['yh'].values[indstart::N]

    xq_down = ds['xq'].values[(N-1)::N]
    yq_down = ds['yq'].values[(N-1)::N]

    downsampled_ds = xr.Dataset(coords={'xh': (('xh'), xh_down),
                                        'yh': (('yh'), yh_down),
                                        'xq': (('xq'), xq_down),
                                        'yq': (('yq'), yq_down)})
    return downsampled_ds


if __name__ == "__main__":
    ds = xr.open_dataset('data/ocean_monthly_z.static.nc')
    methods = {}
    ds_d2 = downsample_dataset(ds, methods)
    print(ds_d2)
