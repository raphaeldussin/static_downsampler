""" setup for static_downsampler """
import setuptools


INSTALL_REQUIRES = ["numpy", "xarray", "dask", "netCDF4", "numba"]
TESTS_REQUIRE = ['pytest >= 2.8', 'pytest_datafiles']

setuptools.setup(
    name="static_downsampler",
    version="0.0.2",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("A tool to downsample MOM static files"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/static_downsampler",
    packages=["static_downsampler"],
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    scripts=[
        "static_downsampler/downsample.py",
        "static_downsampler/create_hgrid_d2.py",
        "static_downsampler/make_basin_mask_d2.py",
    ],
)
