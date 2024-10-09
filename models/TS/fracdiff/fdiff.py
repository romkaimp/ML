from functools import partial
from typing import Optional

import numpy as np

# found module but no type hints or library stubs
from scipy.special import binom  # type: ignore


def fdiff_coef(d: float, window: int) -> np.ndarray:
    return (-1) ** np.arange(window) * binom(d, np.arange(window))


def fdiff(
    a: np.ndarray,
    n: float = 1.0,
    axis: int = -1,
    prepend: Optional[np.ndarray] = None,
    append: Optional[np.ndarray] = None,
    window: int = 10,
    mode: str = "same",
) -> np.ndarray:
    if mode == "full":
        mode = "same"
        raise DeprecationWarning("mode 'full' was renamed to 'same'.")

    if isinstance(n, int) or n.is_integer():
        prepend = np._NoValue if prepend is None else prepend  # type: ignore
        append = np._NoValue if append is None else append  # type: ignore
        return np.diff(a, n=int(n), axis=axis, prepend=prepend, append=append)

    if a.ndim == 0:
        raise ValueError("diff requires input that is at least one dimensional")

    a = np.asanyarray(a)
    # Mypy complains:
    # fracdiff/fdiff.py:135: error: Module has no attribute "normalize_axis_index"
    axis = np.core.multiarray.normalize_axis_index(axis, a.ndim)  # type: ignore
    dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64

    combined = []
    if prepend is not None:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    if mode == "valid":
        D = partial(np.convolve, fdiff_coef(n, window).astype(dtype), mode="valid")
        a = np.apply_along_axis(D, axis, a)
    elif mode == "same":
        # Convolve with the mode 'full' and cut last
        D = partial(np.convolve, fdiff_coef(n, window).astype(dtype), mode="full")
        s = tuple(
            slice(a.shape[axis]) if i == axis else slice(None) for i in range(a.ndim)
        )
        a = np.apply_along_axis(D, axis, a)
        a = a[s]
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    return a
