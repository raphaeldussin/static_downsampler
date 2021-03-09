# static_downsampler

The MOM6 ocean model cannot build the downsampled grid because the downsampled
grid variables are not built from the same variables (e.g. h-point geolon downsampled is
built from q-point geolon_c). This python package is a workaround for the code limitation.

## Install

```
git clone https://github.com/raphaeldussin/static_downsampler.git
python setup.py install
```

## Downsampling of ocean_static files:

This works with symetric and non-symetric grids:

```
downsample.py -s ocean_static_no_mask_table.nc -o ocean_static_d2.nc
downsample.py -s ocean_static_no_mask_table_sym.nc -o ocean_static_d2_sym.nc
```

## Create downsampled hgrid (needed for ocean_mosaic_d2)

```
create_hgrid_d2.py -g ocean_hgrid.nc -o ocean_hgrid_d2.nc
```

## Create ocean_mosaic_d2 (using FRE-NCtools)

```
make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic_d2 --tile_file ocean_hgrid_d2.nc
```
