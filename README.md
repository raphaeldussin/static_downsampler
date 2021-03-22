# static_downsampler

The MOM6 ocean model cannot build the downsampled grid because the downsampled
grid variables are not built from the same variables (e.g. h-point geolon downsampled is
built from q-point geolon_c). This python package is a workaround for the code limitation.

## Install

```
git clone https://github.com/raphaeldussin/static_downsampler.git
python setup.py install
```

## Downsampling of ocean_static files

From the native resolution ocean_static (no_mask_table), create the ocean_static_d2
with the command line tool. This works with symetric and non-symetric grids:

```
downsample.py -s ocean_static_no_mask_table.nc -o ocean_static_d2.nc
downsample.py -s ocean_static_no_mask_table_sym.nc -o ocean_static_d2_sym.nc
```

## Creating the remapping weights from d2 to 1x1deg

### Creating the downsampled hgrid (needed for ocean_mosaic_d2)

From the native grid, you need to provide ocean_hgrid.nc. Create a downsampled ocean_hgrid_d2 
using the utility code from static_downsampler:

```
create_hgrid_d2.py -g ocean_hgrid.nc -o ocean_hgrid_d2.nc
```

### Create ocean_mosaic_d2 (using FRE-NCtools)

Create the ocean_mosaic grid for the downsampled grid. For this, you need the FRE-NCtools (https://github.com/NOAA-GFDL/FRE-NCtools)
compiled on your system. Make sure you have netcdf/hdf and mpich loaded and then run in command line:

```
module load netcdf/4.2 mpich2
make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic_d2 --tile_file ocean_hgrid_d2.nc
```

### Create the remapping weights

fregrid is going to look for ocean_hgrid.nc to infer dimensions so we need to link the downsampled file to a bogus ocean_hgrid.nc.
In this example, we interpolate the depth of the ocean for a downsampled static file (created with downsample.py).
The remapping weights will be saved in remapwgts_d2.nc and an interpolated file for ocean depth will be created (remap_d2_to_1x1.nc).
It can be used to check that the remap worked correctly then tossed.

NB: we're using fregrid_parallel because larger grids cannot be handle by the serial fregrid code.

```
ln -s ocean_hgrid_d2.nc ocean_hgrid.nc
mpirun -n 16 fregrid_parallel --input_mosaic ocean_mosaic_d2.nc --input_file ocean_annual_z_d2.static.nc --remap_file remapwgts_d2 --interp_method conserve_order1 --scalar_field deptho --nlon 360 --nlat 180 --output_file remap_d2_to_1x1
ncks -3 remapwgts_d2 -o OM4p125_grid_d2_remap_file_1140x1120_to_360x180.nc
```
