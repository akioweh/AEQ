"""
Base classes for filters.
"""
import cmath
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Generator


class Filter(ABC):
    """
    Abstract base class for (digital) filters.
    """
    # @property
    # @abstractmethod
    # def coefs(self) -> tuple[list[float], list[float]]:
    #     """The coefficients of the filter, (b, a)."""
    #     raise NotImplementedError
    #
    # @property
    # def coefs_n(self) -> tuple[list[float], list[float]]:
    #     """The coefficients of the filter, normalized against a0."""
    #     b, a = self.coefs
    #     a0 = a[0]
    #     return ([i / a0 for i in b],
    #             [i / a0 for i in a])
    #
    # @property
    # def coefs_b(self) -> list[float]:
    #     """The numerator coefficients of the filter."""
    #     return self.coefs[0]
    #
    # @property
    # def coefs_a(self) -> list[float]:
    #     """The denominator coefficients of the filter."""
    #     return self.coefs[1]

    @abstractmethod
    def transfer_function(self, z: complex) -> complex:
        """The transfer function of the filter."""
        raise NotImplementedError

    def response_at(self, w: float) -> complex:
        """Returns the complex response factor ``H(z)`` at given NORMALIZED ANGULAR frequency ``w``.
        (``z = exp(jw)`` where ``w = 2 * pi * freq / sampling_frequency``)

          - ``|H(z)|`` -> amplitude response factor
          - ``arg(H(z))`` -> phase shift (rads)
        """
        return self.transfer_function(cmath.rect(1, w))

    def frequency_resp_at(self, f: float) -> float:
        """The frequency response of the filter at the given frequency."""
        return abs(self.response_at(f))

    def phase_resp_at(self, f: float) -> float:
        """The phase response of the filter at the given frequency."""
        return cmath.phase(self.response_at(f))

    @abstractmethod
    def apply(self, x: Collection[float]) -> list[float]:
        """Apply the filter to a list of samples."""
        raise NotImplementedError

    @abstractmethod
    def apply_on(self, stream: Iterator[float]) -> Generator[float, None, None]:
        """Continuously apply the filter to a stream of samples."""
        raise NotImplementedError

    def impulse_resp(self, n: int) -> list[float]:
        """The discrete impulse response of the filter for ``n`` samples."""
        x = [0.] * n
        x[0] = 1
        return self.apply(x)
