import cv2 as cv
import jax.numpy as jnp
import matplotlib.pyplot as plt

import glob
import os
import sys


def get_flats(directory):
    flats_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
    flats = []

    for flat_path in flats_paths:
        flat = cv.imread(flat_path, cv.IMREAD_GRAYSCALE)
        if flat is not None:
            flats.append(flat)
        else:
            print(f"Warning: Failed to read {flat_path}")

        if not flats:
            raise ValueError("No valid flat images found.")

    flats_stack = jnp.stack(flats, axis=0)
    flat_mean = jnp.mean(flats_stack, axis=0)
    return flat_mean


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/get_flat.py <directory>")

    directory = sys.argv[1]
    flat = get_flats(directory)
    plt.imshow(flat, cmap="gray")
    plt.axis("off")
    plt.savefig("flat_mean.tif")


if __name__ == "__main__":
    main()

