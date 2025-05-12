import cv2 as cv
import numpy as np

import glob
import os
import sys
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


def read_image(view_path, views_stack, views_angles, index, lock):
    view = cv.imread(view_path, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
    if view is None:
        print(f"Warning: Failed to read {view_path}")
        return
    angle = int(view_path.split('.')[1].split('_')[2])
    with lock:
        views_stack[index] = view
        views_angles[index] = angle


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

    lock = Lock()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(read_image, path, views_stack, views_angles, i, lock)
            for i, path in enumerate(views_paths)]
        for future in futures:
            future.result()
    return views_angles, views_stack


def get_flat(directory):
    _, flats_stack = get_views(directory)
    return np.mean(flats_stack, axis=0).astype(flats_stack.dtype)


def main():
    flat_directory, views_directory = sys.argv[1], sys.argv[2]
    flat = get_flat(flat_directory)
    _, views = get_views(views_directory)
    a = -np.log(views/flat)
    #cv.imwrite("cropped_normalized.tif", (views[0]/flat)[700:,700:2000])


if __name__ == "__main__":
    main()

