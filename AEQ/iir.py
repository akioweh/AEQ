"""
Infinite Impulse Response (IIR) filters.
Casual recursive filters.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Collection, Generator, Iterator
from typing import TypeAlias

from numpy import float64
from numpy.typing import NDArray
from scipy.signal import lfilter

from .base import Filter


class IIRFilter(Filter, ABC):
    """
    Abstract base class for Infinite Impulse Response (IIR) filters.
    """
    @abstractmethod
    def difference_eq(self, *args: float) -> float:
        """The difference equation of the filter."""
        raise NotImplementedError


t3ple: TypeAlias = tuple[float, float, float]


class BiquadFilter(IIRFilter):
    """Bi-quadratic filter object.

    Represents a generic biquad transfer function with 6 coefficients:
      - b0, b1, b2: numerator coefficients
      - a0, a1, a2: denominator coefficients
    """
    def __init__(self, b: t3ple = (0., 0., 0.), a: t3ple = (1., 0., 0.)):
        self._coefs_b = b
        self._coefs_a = a

    @property
    def coefs_a(self) -> t3ple:
        """Returns the denominator coefficients (a0, a1, a2)."""
        return self._coefs_a

    @property
    def coefs_b(self) -> t3ple:
        """Returns the numerator coefficients (b0, b1, b2)."""
        return self._coefs_b

    @property
    def coefs(self) -> tuple[t3ple, t3ple]:
        """Returns the 6 filter coefficients as the nested tuple:
        ((b0, b1, b2), (a0, a1, a2)).
        """
        return self._coefs_b, self._coefs_a

    @property
    def coefs_a_n(self) -> t3ple:
        """Returns the denominator coefficients normalized against a0."""
        a = self._coefs_a
        a0 = a[0]
        return 1., a[1] / a0, a[2] / a0

    @property
    def coefs_b_n(self) -> t3ple:
        """Returns the numerator coefficients normalized against a0."""
        b = self._coefs_b
        a0 = self._coefs_a[0]
        return b[0] / a0, b[1] / a0, b[2] / a0

    @property
    def coefs_n(self) -> tuple[t3ple, t3ple]:
        """Returns the filter coefficients but normalized against a0:
        ((b0/a0, b1/a0, b2/a0), (1., a1/a0, a2/a0)).
        """
        return self.coefs_b_n, self.coefs_a_n

    def transfer_function(self, z: complex) -> complex:
        """The bi-quadratic complex-valued transfer function ``H(z)``
        for the given coefficients.
        """
        b0, b1, b2 = self._coefs_b
        a0, a1, a2 = self._coefs_a
        return (b0 * z ** 0 + b1 * z ** -1 + b2 * z ** -2) / (a0 * z ** 0 + a1 * z ** -1 + a2 * z ** -2)

    def difference_eq(self, x: float, x1: float, x2: float, y1: float, y2: float) -> float:
        """The difference equation implementation of the filter.
        Requires the past two inputs and outputs (x1, x2, y1, and y2)
        in addition to the current input x.
        """
        b0, b1, b2 = self._coefs_b
        a0, a1, a2 = self._coefs_a
        return 1 / a0 * (b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2)

    def _apply(self, x: Iterable[float]) -> list[float]:
        """Apply using vanilla python."""
        x = list(x)
        n = len(x)
        y = [0.] * n
        y[0] = self.difference_eq(x[0], 0, 0, 0, 0)
        y[1] = self.difference_eq(x[1], x[0], 0, y[0], 0)
        for i in range(2, n):
            y[i] = self.difference_eq(x[i], x[i - 1], x[i - 2], y[i - 1], y[i - 2])
        return y

    def _apply_scipy(self, x: Collection[float]) -> NDArray[float64]:
        """Apply using scipy's lfilter function."""
        return lfilter(self._coefs_b, self._coefs_a, x)

    def apply(self, x: Collection[float]) -> list[float] | NDArray[float64]:
        if len(x) > 1000:
            return self._apply_scipy(x)
        return self._apply(x)

    def apply_on(self, stream: Iterator[float]) -> Generator[float, None, None]:
        x2, x1 = 0., 0.
        y2, y1 = 0., 0.
        for x in stream:
            y = self.difference_eq(x, x1, x2, y1, y2)
            yield y
            y2, y1 = y1, y
            x2, x1 = x1, x

    def impulse_resp(self, n: int) -> list[float] | NDArray[float64]:
        x = [0.] * n
        x[0] = 1
        return self.apply(x)
