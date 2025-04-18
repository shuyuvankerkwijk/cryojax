"""
Routines for loading voxel‑size–dependent mean and variance values
for the Parkhurst 2024 experimental‑ice model.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import os

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def get_voxel_mean_variance(
    voxel_sizes: Float[Array, " n_query"] | Float[np.ndarray, " n_query"],
) -> Float[Array, " 2 n_query"]:
    """
    Function to get the mean potential and variance corresponding to each
    voxel size in `voxel_sizes`.

    **Arguments**

    - `voxel_sizes`
      1D array of voxel sizes.

    **Returns**

    The mean potential and variance for each voxel size in
    `voxel_sizes`. The returned array has shape (2, n_query), where
    row 0 contains the mean potential and row 1 contains the
    variance. The returned array is in the same order as the input.

    **Raises**

    `ValueError` if any requested voxel size is below the minimum or above
    the maximum voxel size in the data.
    """
    table = _read_voxel_mean_variance_table()

    sizes = table[0]
    means = table[1]
    variances = table[2]

    voxel_sizes = jnp.asarray(voxel_sizes)

    # check bounds
    min_size = sizes.min()
    max_size = sizes.max()
    if jnp.any(voxel_sizes < min_size) or jnp.any(voxel_sizes > max_size):
        out_of_bounds = voxel_sizes[(voxel_sizes < min_size) | (voxel_sizes > max_size)]
        raise ValueError(
            f"Voxel sizes {out_of_bounds.tolist()}  are outside the "
            f"tabulated range [{float(min_size):.2f}, {float(max_size):.2f}] ."
        )

    # for every queried value, compute (sizes - v) and take argmin
    idx = jnp.argmin(jnp.abs(sizes[:, None] - voxel_sizes[None, :]), axis=0)

    return jnp.vstack([means[idx], variances[idx]])


def _read_voxel_mean_variance_table() -> Float[np.ndarray, " 3 n_entries"]:
    """
    Load the experimentally determined voxel–mean/variance table
    stored in relaxed_small_box_tip3p_voxel_mean_variance.npy
    """
    with pkg_resources.as_file(
        pkg_resources.files("cryojax").joinpath("constants")
    ) as path:
        arr = jnp.load(
            os.path.join(path, "relaxed_small_box_tip3p_voxel_mean_variance.npy")
        )

    return np.asarray(arr)
