import jax.numpy as jnp
import matplotlib.pyplot as plt

import glob
import os
import sys

from utils.get_views import get_views


def crop(directory):
    views_stack = get_views(directory)
    return view_stack[:,]


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/get_view.py <directory>")

    directory = sys.argv[1]
    cropped_view = crop(directory)
    for cropped_view in cropped_views:
    plt.imshow(view, cmap="gray")
    plt.axis("off")
    plt.savefig("view_mean.tif")


if __name__ == "__main__":
    main()

