#!/usr/bin/env python

import argparse

import numpy
import xarray as xr


def ice9it(i, j, depth, minD=0.0):
    """
    Recursive implementation of "ice 9".
    Returns 1 where depth>minD and is connected to depth[j,i], 0 otherwise.
    """
    wetMask = 0 * depth

    (nj, ni) = wetMask.shape
    stack = set()
    stack.add((j, i))
    while stack:
        (j, i) = stack.pop()
        if wetMask[j, i] or depth[j, i] <= minD:
            continue
        wetMask[j, i] = 1

        if i > 0:
            stack.add((j, i - 1))
        else:
            stack.add((j, ni - 1))  # Periodic beyond i=0

        if i < ni - 1:
            stack.add((j, i + 1))
        else:
            stack.add((j, 0))  # Periodic beyond i=ni-1

        if j > 0:
            stack.add((j - 1, i))

        if j < nj - 1:
            stack.add((j + 1, i))
        else:
            stack.add((j, ni - 1 - i))  # Tri-polar fold beyond j=nj-1

    return wetMask


def ice9(x, y, depth, xy0):
    ji = nearestJI(x, y, xy0[0], xy0[1])
    return ice9it(ji[1], ji[0], depth)


def nearestJI(x, y, x0, y0):
    """
    Find (j,i) of cell with center nearest to (x0,y0).
    """
    return numpy.unravel_index(((x - x0) ** 2 + (y - y0) ** 2).argmin(), x.shape)


def southOf(x, y, xy0, xy1):
    """
    Returns 1 for point south/east of the line that passes through xy0-xy1, 0 otherwise.
    """
    x0 = xy0[0]
    y0 = xy0[1]
    x1 = xy1[0]
    y1 = xy1[1]
    dx = x1 - x0
    dy = y1 - y0
    Y = (x - x0) * dy - (y - y0) * dx
    Y[Y >= 0] = 1
    Y[Y <= 0] = 0
    return Y


def create_basin_code(ds):
    # read data in
    x = ds["geolon"].values
    y = ds["geolat"].values

    wet = ds["wet"].values
    wet[numpy.isnan(wet)] = 0
    code = 0 * wet

    print("Finding Cape of Good Hope ...", end=" ")
    tmp = 1 - wet
    tmp[x < -30] = 0
    tmp = ice9(x, y, tmp, (20, -30.0))
    yCGH = (tmp * y).min()
    print("done.", yCGH)

    print("Finding Melbourne ...", end=" ")
    tmp = 1 - wet
    tmp[x > -180] = 0
    tmp = ice9(x, y, tmp, (-220, -25.0))
    yMel = (tmp * y).min()
    print("done.", yMel)

    print("Processing Persian Gulf ...")
    tmp = wet * (1 - southOf(x, y, (55.0, 23.0), (56.5, 27.0)))
    tmp = ice9(x, y, tmp, (53.0, 25.0))
    code[tmp > 0] = 11
    wet = wet - tmp  # Removed named points

    print("Processing Red Sea ...")
    tmp = wet * (1 - southOf(x, y, (40.0, 11.0), (45.0, 13.0)))
    tmp = ice9(x, y, tmp, (40.0, 18.0))
    code[tmp > 0] = 10
    wet = wet - tmp  # Removed named points

    print("Processing Black Sea ...")
    tmp = wet * (1 - southOf(x, y, (26.0, 42.0), (32.0, 40.0)))
    tmp = ice9(x, y, tmp, (32.0, 43.0))
    code[tmp > 0] = 7
    wet = wet - tmp  # Removed named points

    print("Processing Mediterranean ...")
    tmp = wet * (southOf(x, y, (-5.7, 35.5), (-5.7, 36.5)))
    tmp = ice9(x, y, tmp, (4.0, 38.0))
    code[tmp > 0] = 6
    wet = wet - tmp  # Removed named points

    print("Processing Baltic ...")
    tmp = wet * (southOf(x, y, (8.6, 56.0), (8.6, 60.0)))
    tmp = ice9(x, y, tmp, (10.0, 58.0))
    code[tmp > 0] = 9
    wet = wet - tmp  # Removed named points

    print("Processing Hudson Bay ...")
    tmp = wet * (
        (
            1
            - (1 - southOf(x, y, (-95.0, 66.0), (-83.5, 67.5)))
            * (1 - southOf(x, y, (-83.5, 67.5), (-84.0, 71.0)))
        )
        * (1 - southOf(x, y, (-70.0, 58.0), (-70.0, 65.0)))
    )
    tmp = ice9(x, y, tmp, (-85.0, 60.0))
    code[tmp > 0] = 8
    wet = wet - tmp  # Removed named points

    print("Processing Arctic ...")
    tmp = wet * (
        (1 - southOf(x, y, (-171.0, 66.0), (-166.0, 65.5)))
        * (1 - southOf(x, y, (-64.0, 66.4), (-50.0, 68.5)))  # Lab Sea
        + southOf(x, y, (-50.0, 0.0), (-50.0, 90.0))
        * (1 - southOf(x, y, (0.0, 65.39), (360.0, 65.39)))  # Denmark Strait
        + southOf(x, y, (-18.0, 0.0), (-18.0, 65.0))
        * (1 - southOf(x, y, (0.0, 64.9), (360.0, 64.9)))  # Iceland-Sweden
        + southOf(x, y, (20.0, 0.0), (20.0, 90.0))  # Barents Sea
        + (1 - southOf(x, y, (-280.0, 55.0), (-200.0, 65.0)))
    )
    tmp = ice9(x, y, tmp, (0.0, 85.0))
    code[tmp > 0] = 4
    wet = wet - tmp  # Removed named points

    print("Processing Pacific ...")
    tmp = wet * (
        (1 - southOf(x, y, (0.0, yMel), (360.0, yMel)))
        - southOf(x, y, (-257, 1), (-257, 0)) * southOf(x, y, (0, 3), (1, 3))
        - southOf(x, y, (-254.25, 1), (-254.25, 0)) * southOf(x, y, (0, -5), (1, -5))
        - southOf(x, y, (-243.7, 1), (-243.7, 0)) * southOf(x, y, (0, -8.4), (1, -8.4))
        - southOf(x, y, (-234.5, 1), (-234.5, 0)) * southOf(x, y, (0, -8.9), (1, -8.9))
    )
    tmp = ice9(x, y, tmp, (-150.0, 0.0))
    code[tmp > 0] = 3
    wet = wet - tmp  # Removed named points

    print("Processing Atlantic ...")
    tmp = wet * (1 - southOf(x, y, (0.0, yCGH), (360.0, yCGH)))
    tmp = ice9(x, y, tmp, (-20.0, 0.0))
    code[tmp > 0] = 2
    wet = wet - tmp  # Removed named points

    print("Processing Indian ...")
    tmp = wet * (1 - southOf(x, y, (0.0, yCGH), (360.0, yCGH)))
    tmp = ice9(x, y, tmp, (55.0, 0.0))
    code[tmp > 0] = 5
    wet = wet - tmp  # Removed named points

    print("Processing Southern Ocean ...")
    tmp = ice9(x, y, wet, (0.0, -55.0))
    code[tmp > 0] = 1
    wet = wet - tmp  # Removed named points

    code[wet > 0] = -9
    (j, i) = numpy.unravel_index(wet.argmax(), x.shape)
    if j:
        print("There are leftover points unassigned to a basin code")
        while j:
            print(x[j, i], y[j, i], [j, i])
            wet[j, i] = 0
            (j, i) = numpy.unravel_index(wet.argmax(), x.shape)
    else:
        print("All points assigned a basin code")

    out = xr.DataArray(
        data=code, coords={"yh": ds["yh"], "xh": ds["xh"]}, dims=("yh", "xh")
    )

    out.attrs[
        "flag_meanings"
    ] = "1:Southern Ocean, 2:Atlantic Ocean, 3:Pacific Ocean, 4:Arctic Ocean, 5:Indian Ocean, 6:Mediterranean Sea, 7:Black Sea, 8:Hudson Bay, 9:Baltic Sea, 10:Red Sea, 11:Persian Gulf"
    out.attrs["flag_values"] = "1,2,3,4,5,6,7,8,9,10,11"
    return out


if __name__ == "__main__":
    """ cmd line version """
    parser = argparse.ArgumentParser(description="Create basin codes for OM")
    parser.add_argument("-i", "--infile", type=str, required=True, help="input file")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="output file")

    args = parser.parse_args()

    ds = xr.open_dataset(args.infile)
    ds["basin"] = create_basin_code(ds)
    ds.to_netcdf(args.outfile, format="NETCDF3_64BIT")
