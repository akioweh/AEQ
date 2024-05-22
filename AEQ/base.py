"""
Base classes for filters.
"""
from abc import ABC, abstractmethod


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
    def response_at(self, f: float) -> complex:
        """The response of the filter at the given frequency (as a complex output).

        The magnitude (modulus) of the output is the frequency response,
        and the angle (argument) of the output is the phase response.
        """
        raise NotImplementedError

    @abstractmethod
    def frequency_resp_at(self, f: float) -> float:
        """The frequency response of the filter at the given frequency."""
        raise NotImplementedError

    @abstractmethod
    def phase_resp_at(self, f: float) -> float:
        """The phase response of the filter at the given frequency."""
        raise NotImplementedError

    @abstractmethod
    def impulse_resp(self, n: int) -> list[float]:
        """The discrete impulse response of the filter for ``n`` samples."""
        raise NotImplementedError
