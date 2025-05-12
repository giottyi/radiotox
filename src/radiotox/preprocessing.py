import cv2 as cv
import numpy as np

import glob
import os
import sys


def get_views(directory):
    views_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
    if not views_paths:
        raise ValueError("No TIFF files found in directory.")

    sample_view = cv.imread(views_paths[0], \
            cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
    if sample_view is None:
        raise ValueError(f"Failed to read sample image: {views_paths[0]}")

    height, width = sample_view.shape
    n_views = len(views_paths)
    views_stack = np.zeros((n_views, height, width), dtype=sample_view.dtype)
    views_angles = np.zeros(n_views, dtype=int)

    views_stack[0] = sample_view
    views_angles[0] = int(views_paths[0].split('.')[1].split('_')[2])
    for i, view_path in enumerate(views_paths[1:], start=1):
        view = cv.imread(view_path, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
        if view is not None:
            views_stack[i] = view
            views_angles[i] = int(views_paths[i].split('.')[1].split('_')[2])
        else:
            print(f"Warning: Failed to read {view_path}")

    return views_angles, views_stack


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

