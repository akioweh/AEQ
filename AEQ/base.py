"""
Base classes for filters.
"""
import cmath
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import SupportsComplex, SupportsFloat, SupportsIndex, overload

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray

type Float = SupportsFloat | np.floating
type Complex = SupportsComplex | np.complexfloating
type Int = SupportsIndex | np.integer


class Filter(ABC):
    """
    Abstract base class for (digital) filters.
    """

    @property
    @abstractmethod
    def fs(self) -> float:
        """The sampling frequency of the filter."""
        raise NotImplementedError

    @fs.setter
    @abstractmethod
    def fs(self, fs: float):
        """Set the sampling frequency of the filter."""
        raise NotImplementedError

    @abstractmethod
    def transfer_function[A_C: (Complex, NDArray[complex128])](self, z: A_C) -> A_C:
        """The transfer function ``H(z)`` of the filter."""
        raise NotImplementedError

    @overload
    def response_at(self, f: Float) -> Complex:
        ...

    @overload
    def response_at(self, f: NDArray[float64]) -> NDArray[complex128]:
        ...

    def response_at(self, f):
        """Returns the complex response factor ``H(z)`` at frequency ``f``.

        ``z = exp(jw)`` where ``w = 2 * pi * freq / sampling_frequency``

          - ``|H(z)|`` -> amplitude response factor
          - ``arg(H(z))`` -> phase shift (rads)
        """
        w = f * 2 * math.pi / self.fs
        if isinstance(w, (np.floating, np.ndarray)):
            z = np.exp(1j * w)
        else:
            z = cmath.rect(1, w)
        return self.transfer_function(z)

    def frequency_resp_at[A_F: (Float, NDArray[float64])](self, f: A_F) -> A_F:
        """The frequency response of the filter at the given frequency."""
        return abs(self.response_at(f))

    def phase_resp_at[A_F: (Float, NDArray[float64])](self, f: A_F) -> A_F:
        """The phase response of the filter at the given frequency."""
        resp = self.response_at(f)
        if isinstance(resp, (np.floating, np.ndarray)):
            return np.angle(resp)
        return cmath.phase(resp)

    @abstractmethod
    def apply[A: (list[float], NDArray[float64])](self, x: A) -> A:
        """Apply the filter to a list of samples."""
        raise NotImplementedError

    @abstractmethod
    def apply_on(self, stream: Iterator[Float]) -> Iterator[Float]:
        """Continuously apply the filter to a stream of samples."""
        raise NotImplementedError

    def impulse_resp(self, n: Int) -> NDArray[float64]:
        """The discrete impulse response of the filter for ``n`` samples."""
        x = np.zeros(n)
        x[0] = 1
        return self.apply(x)
