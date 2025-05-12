import cv2 as cv
import numpy as np

import glob
import os


def get_views(directory):
    views_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
    views = []

    for view_path in views_paths:
        view = cv.imread(view_path, cv.IMREAD_GRAYSCALE)
        if view is not None:
            views.append(view)
        else:
            print(f"Warning: Failed to read {view_path}")

        if not views:
            raise ValueError("No valid view images found. Check directory")

    views_stack = np.stack(views, axis=0)
    return views_stack

