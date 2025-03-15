"""
The most generic model of filter -- its frequency response.

Main functionality includes constructing from
various other representations and custom arithmetic
operators to Pythonically combine multiple filters.
"""

from typing import Literal

import numpy as np
from numpy import float64, ndarray
from numpy.typing import NDArray
from scipy.interpolate import Akima1DInterpolator

from .base import Filter

type FloatArray = NDArray[float64]


class DiscreteResponse:
    def __init__(self, f_s: FloatArray, y: FloatArray):
        assert len(f_s.shape) == 1
        assert len(y.shape) == 1
        assert f_s.shape == y.shape
        assert np.all(np.isreal(f_s))
        assert np.all(np.isreal(y))

        self._f_s = f_s
        self._y = y

    @property
    def f_s(self) -> FloatArray:
        # noinspection PyTypeChecker
        return 10. ** self._f_s

    @property
    def y(self) -> FloatArray:
        return self._y

    @y.setter
    def y(self, y: FloatArray):
        assert isinstance(y, ndarray)
        assert y.shape == self._y.shape
        self._y = y

    def __iter__(self):
        yield self.f_s
        yield self.y

    @classmethod
    def from_arrays(
            cls,
            f_s: FloatArray | list[float],
            y: FloatArray | list[float],
            *,
            f_s_type: Literal['linear', 'log10', 'ln'] = 'linear',
            y_type: Literal['magnitude', 'decibel'] = 'decibel'
    ):
        assert f_s_type in ('linear', 'log10', 'ln')
        assert y_type in ('magnitude', 'decibel')
        if isinstance(f_s, list):
            f_s = np.array(f_s, dtype=float64)
        if isinstance(y, list):
            y = np.array(y, dtype=float64)

        if f_s_type == 'linear':
            f_s = np.log10(f_s)
        elif f_s_type == 'ln':
            f_s = np.log10(np.exp(f_s))
        if y_type == 'magnitude':
            y = 20 * np.log10(y)

        return cls(f_s, y)

    @classmethod
    def from_table(
            cls,
            data: ndarray[tuple[int, int], float64],
            /,
            *,
            f_s_type: Literal['linear', 'log10', 'ln'] = 'linear',
            y_type: Literal['magnitude', 'decibel'] = 'decibel'
    ):
        assert isinstance(data, ndarray)
        assert len(data.shape) == 2
        assert data.shape[1] == 2
        return cls.from_arrays(data[0], data[1], f_s_type=f_s_type, y_type=y_type)

    @classmethod
    def from_filter(cls, flt: Filter, *, f_s: FloatArray = None):
        assert hasattr(flt, 'frequency_resp_at')
        if f_s is None:  # default f_s
            f_s = np.linspace(20, 20_000, 10_000, dtype=float64)
        y = flt.frequency_resp_at(f_s)
        return cls.from_arrays(f_s, y, f_s_type='linear', y_type='magnitude')

    def __copy__(self):
        return DiscreteResponse(self._f_s.copy(), self._y.copy())

    def copy(self):
        return self.__copy__()

    def normalized(self, direction: Literal['pos', 'neg'] = 'neg') -> 'DiscreteResponse':
        """Returns a copy where the FR is shifted, so
        either the maximum or minimum response is 0.0 dB.
        (So the entire FR is either +dB or -dB.)
        """
        assert direction in ('pos', 'neg')
        copy = self.copy()
        if direction == 'neg':
            copy._y -= np.max(copy._y)
        else:
            copy._y -= np.min(copy._y)
        return copy

    def resample(self, new_f_s: FloatArray) -> 'DiscreteResponse':
        """Uses interpolation to change ``f_s``."""
        interp = Akima1DInterpolator(self._f_s, self._y, method='makima', extrapolate=False)
        new_y: FloatArray = interp(new_f_s)
        new_y[np.isnan(new_y)] = 0
        return DiscreteResponse(new_f_s.copy(), new_y)

    def expand(self, min_f: float = None, max_f: float = None, *,
               behavior: Literal['zero', 'last'] = 'zero') -> 'DiscreteResponse':
        assert behavior in ('zero', 'last')
        assert min_f is not None or max_f is not None, 'At least one of min_f or max_f must be set.'
        copy = self.copy()

        if min_f is not None:
            cur_min_f = copy._f_s[0]
            assert min_f < cur_min_f
            copy._f_s = np.insert(copy._f_s, 0, [min_f, (cur_min_f + min_f) / 2])
            if behavior == 'zero':
                new_vals = [0, 0]
            else:
                new_vals = [copy._y[0], copy._y[0]]
            copy._y = np.insert(copy._y, 0, new_vals)

        if max_f is not None:
            cur_max_f = copy._f_s[-1]
            assert max_f > cur_max_f
            copy._f_s = np.append(copy._f_s, [(max_f + cur_max_f) / 2, max_f])
            if behavior == 'zero':
                new_vals = [0, 0]
            else:
                new_vals = [copy._y[-1], copy._y[-1]]
            copy._y = np.append(copy._y, new_vals)

        return copy

    def __neg__(self) -> 'DiscreteResponse':
        return DiscreteResponse(self._f_s, -self._y)

    def __add__(self, other: 'DiscreteResponse') -> 'DiscreteResponse':
        """Combines two filters."""
        if other.f_s == self._f_s:
            res_f_s = self._f_s.copy()
            res_y = self._y + other.y
        else:  # resample
            res_f_s = np.unique(np.concatenate(self._f_s, other.f_s))
            res_y = self.resample(res_f_s)._y + other.resample(res_f_s).y

        # noinspection PyTypeChecker
        return DiscreteResponse(res_f_s, res_y)

    def __sub__(self, other: 'DiscreteResponse') -> 'DiscreteResponse':
        """Returns the difference between two filters."""
        return self.__add__(other.__neg__())

    def __mul__(self, other: 'DiscreteResponse'):
        raise RuntimeError('It does not make sense to *multiply* frequency responses.'
                           'They are stored in logarithmic decibel form; addition suffices.')

    def __truediv__(self, other: 'DiscreteResponse'):
        pass
