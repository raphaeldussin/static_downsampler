#!/bin/bash

# download/install code https://github.com/raphaeldussin/static_downsampler.git
# assuming you have generated the ocean_static without mask table:
# 1. add basins to ocean_static
make_basin_mask_d2.py -i ocean_static_no_mask_table.nc -o ocean_static.nc

# 2. run downsampling
downsample.py -s ocean_static.nc -o ocean_static_d2.nc

# 3. create downsampled hgrid (needed for ocean_mosaic_d2)
create_hgrid_d2.py -g ocean_hgrid.nc -o ocean_hgrid_d2.nc

# 4. Create ocean_mosaic_d2
make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic_d2 --tile_file ocean_hgrid_d2.nc
