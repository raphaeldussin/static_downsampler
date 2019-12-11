all: ocean_hgrid.nc ocean_mask.nc ocean_static.nc

ocean_hgrid.nc ocean_mask.nc ocean_static.nc:
	wget -nv ftp://ftp.gfdl.noaa.gov/perm/Alistair.Adcroft/MOM6-testing/OM4_05/$@
	md5sum -c $@.md5

