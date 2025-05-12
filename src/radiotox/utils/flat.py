import cv2 as cv
import numpy as np

import glob
import os
import sys

from radiotox.utils.get_views import get_views


def get_flat(directory):
    _, flats_stack = get_views(directory)
    flats_mean = np.mean(flats_stack, axis=0).astype(flats_stack.dtype)
    return flats_mean


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/get_flat.py <directory>")

    directory = sys.argv[1]
    flat = get_flat(directory)
    cv.imwrite("flat_mean.tif", flat)


if __name__ == "__main__":
    main()

