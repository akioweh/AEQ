"""
Functions for processing data
related to filters and frequency responses.

Useful for filter design and analysis.
"""

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from scipy.interpolate import Akima1DInterpolator, make_smoothing_spline
from scipy.signal import get_window


def load_resp(file: str, skip_header: int = 1) -> tuple[NDArray[float64], NDArray[float64]]:
    """Load frequency response data from a CSV file.

    Optionally skip n header rows.
    """
    data = np.loadtxt(file, delimiter=',', skiprows=skip_header)
    return data[:, 0], data[:, 1]


def resample_log(f_s, y, f_s_new):
    """Re/upsample a frequency response using
    modified-akima interpolation.

    The interpolation is done with f_s under log scaling.
    """

    assert f_s[0] <= f_s_new[0] and f_s[-1] >= f_s_new[-1], 'f_s_new must be within f_s range'

    f_s_log = np.log(f_s)
    f_s_new_log = np.log(f_s_new)
    interp = Akima1DInterpolator(f_s_log, y, method='makima')

    y_new = interp(f_s_new_log)
    return y_new


def smooth_resp_section_sparse(f_s, y, transition_start: float = 12_000, transition_end: float = 13_500, lam: float = None):
    func = make_smoothing_spline(f_s, y, lam=lam)
    y_smooth = func(f_s)

    mask_low = f_s <= transition_start
    mask_transition = (f_s > transition_start) & (f_s <= transition_end)
    mask_high = f_s > transition_end

    len_ = np.count_nonzero(mask_transition)
    transition_weights = np.linspace(0, 1, len_)

    res = np.concatenate((
        y[mask_low],
        y[mask_transition] * (1 - transition_weights) + y_smooth[mask_transition] * transition_weights,
        y_smooth[mask_high]
    ))

    return res


def smooth_resp_section(
    f_s,
    y,
    start: float = None,
    end: float = None,
    transition: float = None,
    window_len: int = None
):
    # window-based smoothing
    if start is None:
        start = f_s[0]
        if transition is not None:
            start -= transition
    if end is None:
        end = f_s[-1]
        if transition is not None:
            end += transition
    assert start < end, 'start must be less than end'
    if transition is None:
        transition = (end - start) / 10
    else:
        assert transition > 0, 'transition must be positive'
        assert transition < (end - start) / 2, 'transition must be less than half of the range'

    mask = (f_s >= start) & (f_s <= end)

    if window_len is None:
        window_len = len(mask) // 35
    window = get_window('hamming', window_len)

    y_smooth = np.convolve(y, window, mode='same') / np.sum(window)
    mask_start_transition = (f_s >= start) & (f_s <= start + transition)
    mask_end_transition = (f_s >= end - transition) & (f_s <= end)
    mask_mid = mask & ~mask_start_transition & ~mask_end_transition
    transition_weights = np.zeros_like(y)
    transition_weights[mask_start_transition] = np.linspace(0, 1, np.count_nonzero(mask_start_transition))
    transition_weights[mask_end_transition] = np.linspace(1, 0, np.count_nonzero(mask_end_transition))
    transition_weights[mask_mid] = 1

    y_res = y * (1 - transition_weights) + y_smooth * transition_weights
    return y_res
