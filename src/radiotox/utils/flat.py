import jax.numpy as jnp
import matplotlib.pyplot as plt

import glob
import os
import sys

from radiotox.utils.get_views import get_views


def get_flat(directory):
    flats_stack = get_views(directory)
    flats_mean = jnp.mean(flats_stack, axis=0)
    return flats_mean


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/get_flat.py <directory>")

    directory = sys.argv[1]
    flat = get_flat(directory)
    plt.imshow(flat, cmap="gray")
    plt.axis("off")
    plt.savefig("flat_mean.tif")


if __name__ == "__main__":
    main()

