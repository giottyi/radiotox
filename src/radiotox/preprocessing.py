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
        views_stack[i] = cv.imread(view_path, \
                cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
        views_angles[i] = int(view_path.split('.')[1].split('_')[2])
    return views_angles, views_stack


def get_flat(directory):
    """
    averages flats without accounting for afterglow maintaining precision,
    used in minus log correction
    """
    _, flats_stack = get_views(directory)
    return np.mean(flats_stack, axis=0).astype(flats_stack.dtype)


def main():
    """
    currently crops dataset to reconstruct with astra
    """
    flat_directory, views_directory = sys.argv[1], sys.argv[2]
    flat = get_flat(flat_directory)
    _, views = get_views(views_directory)
    np.save("cropped_data.npy", \
            (-np.log(views/flat)[:,700:,700:2000]).astype(np.float32))


if __name__ == "__main__":
    main()

