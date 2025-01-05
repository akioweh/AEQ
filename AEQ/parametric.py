import math
from abc import abstractmethod
from collections.abc import Collection, Generator, Iterable, Iterator
from typing import TypeAlias

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.signal import freqz_sos, sosfilt

from .base import Filter
from .iir import BiquadFilter


class PEQFilter(BiquadFilter):
    """Parametric filter implemented using biquad (second-order) sections.

    Subclasses should offer a specific filter type (low-pass, high-pass, etc.)
    and calculate the coefficients from more human-friendly parameters
    (corner frequency, Q factor, gain, etc.).

    :param f0: center frequency
    :param gain: gain (for applicable filters, else 0)
    :param Q: quality factor
    :param fs: sample rate (default is normalized; 1)
    :param allow_S: whether to allow Slope variant of Q or BW
    :param name: filter type name (default is subclass name)
    """

    def __init__(
            self,
            f0: float | None = None,
            gain: float | None = None,
            Q: float | None = None,
            fs: float = 1.,
            allow_S: bool = False,
            name: str | None = None,
            # parity parameters
            BW: float | None = None,
            S: float | None = None
    ):
        """
        Setting either f0, Q, or gain to None (default) will disable that parameter.
        (Disallow its querying and setting)

        :param f0: center frequency
        :param gain: gain (for applicable filters, else 0)
        :param Q: quality factor
        :param fs: sample rate (default is normalized; 1)
        :param allow_S: whether to allow Slope variant of Q or BW
        :param name: filter type name (default is subclass name)
        """
        super().__init__(fs=fs)

        if S is not None and not allow_S:
            raise ValueError('Slope variant of Q or BW is not allowed.')
        n_count = sum([1 for i in (Q, BW, S) if i is not None])
        if n_count > 1:
            raise ValueError('Only one of S, BW, or Q can be set.')

        # filter properties
        self._name = name if name is not None else self.__class__.__name__  # filter type name
        self._allow_S = allow_S  # allow slope variant of Q or BW
        self._has_f0 = f0 is not None  # allow the setting of a center frequency
        self._has_gain = gain is not None  # allow the setting of a gain
        self._has_Q = (Q is not None or BW is not None or S is not None) and self._has_f0
        # allow the setting of a Q or BW (or S)

        # filter parameters
        if BW is not None:
            Q = self.BW_to_Q(BW, f0, fs)
        if S is not None:
            Q = self.S_to_Q(S, gain)
        self._f0 = f0
        self._Q = Q
        self._gain = gain

        # calculate intermediate values
        self._calc_intervars()

    @staticmethod
    def BW_to_Q(BW: float, f0: float, fs: float) -> float:
        w0 = 2 * math.pi * f0 / fs
        return 1 / (2 * math.sinh((math.log10(2) * w0 * BW) / (2 * math.sin(w0))))

    @staticmethod
    def S_to_Q(S: float, gain: float) -> float:
        A = 10 ** (gain / 40)
        return 1 / math.sqrt((A + 1 / A) * (1 / S - 1) + 2)

    @staticmethod
    def Q_to_BW(Q: float, f0: float, fs: float) -> float:
        w0 = 2 * math.pi * f0 / fs
        return (2 * math.sin(w0)) / (w0 * math.log10(2)) * math.asinh(1 / (2 * Q))

    @staticmethod
    def Q_to_S(Q: float, gain: float) -> float:
        A = 10 ** (gain / 40)
        return 1 / ((1 / Q ** 2 - 2) / (A + 1 / A) + 1)

    def _calc_intervars(self):
        """Calculate intermediate variables for the filter coefficients.

        Recall when f0, Q, gain, or fs changes.
        """
        self._A = 10 ** (self._gain / 40) if self._has_gain else None
        if self._has_f0:
            self._w0 = 2 * math.pi * self._f0 / self._fs
            self._sinw0 = math.sin(self._w0)
            self._cosw0 = math.cos(self._w0)
        else:
            self._w0 = self._sinw0 = self._cosw0 = None
        self._alph = self._sinw0 / (2 * self._Q) if self._has_Q else None

    @abstractmethod
    def _calc_coefs(self):
        """Calculate the biquad coefficients for the specific filter type.

        Recall when intervars change.
        """
        raise NotImplementedError('This method must be implemented in each specific filter class,'
                                  'based on how that filter works.')

    def _set_coefs(self, a0: float, a1: float, a2: float, b0: float, b1: float, b2: float):
        """Helper method to set the biquad coefficients."""
        self._coefs_b = b0, b1, b2
        self._coefs_a = a0, a1, a2

    def _calc_all(self):
        """Calculate intermediate variables and coefficients."""
        self._calc_intervars()
        self._calc_coefs()

    @BiquadFilter.fs.setter
    def fs(self, fs: float):
        self._fs = fs
        self._calc_all()

    @property
    def f0(self) -> float:
        return self._f0

    @f0.setter
    def f0(self, f0: float):
        self._f0 = f0
        self._calc_all()

    @property
    def Q(self) -> float:
        if not self._has_Q:
            raise AttributeError(f'Quality factor does not exist for filter type {self._name}.')
        return self._Q

    @Q.setter
    def Q(self, Q: float):
        if not self._has_Q:
            raise AttributeError(f'Quality factor cannot be set for filter type {self._name}.')
        self._Q = Q
        self._calc_all()

    @property
    def BW(self) -> float:
        if not self._has_Q:
            raise AttributeError(f'Bandwidth does not exist for filter type {self._name}.')
        # convert Quality factor to Bandwidth
        return self.Q_to_BW(self._Q, self._f0, self._fs)

    @BW.setter
    def BW(self, BW: float):
        if not self._has_Q:
            raise AttributeError(f'Bandwidth cannot be set for filter type {self._name}.')
        # convert Bandwidth to Quality factor
        self._Q = self.BW_to_Q(BW, self._f0, self._fs)
        self._calc_all()

    @property
    def S(self) -> float:
        if not (self._allow_S and self._has_Q):
            raise AttributeError(f'Slope does not exist for filter type {self._name}.')
        # convert Quality factor to Slope
        return self.Q_to_S(self._Q, self._gain)

    @S.setter
    def S(self, S: float):
        if not (self._allow_S and self._has_Q):
            raise AttributeError(f'Slope cannot be set for filter type {self._name}.')
        # convert Slope to Quality factor
        self._Q = self.S_to_Q(S, self._gain)
        self._calc_all()

    @property
    def gain(self) -> float:
        if not self._has_gain:
            raise AttributeError(f'Gain does not exist for filter type {self._name}.')
        return self._gain

    @gain.setter
    def gain(self, gain: float):
        if not self._has_gain:
            raise AttributeError(f'Gain cannot be set for filter type {self._name}.')
        self._gain = gain
        self._calc_all()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def __str__(self):
        return (f'{self._name} ==> f0: {self._f0}, Q: {self._Q}, gain: {self._gain} \n'
                f'b: {self.coefs_b_n} \n'
                f'a: {self.coefs_a_n}')


class LowPassFilter(PEQFilter):
    def __init__(self, f0: float, Q: float = None, BW: float = None, fs: float = 1):
        if Q is None and BW is None:
            raise ValueError('Either Q or BW must be set.')
        super().__init__(f0, Q=Q, BW=BW, fs=fs)
        self._calc_coefs()

    def _calc_coefs(self):
        a0 = 1 + self._alph
        a1 = -2 * self._cosw0
        a2 = 1 - self._alph
        b0 = (1 - self._cosw0) / 2
        b1 = 1 - self._cosw0
        b2 = (1 - self._cosw0) / 2

        self._set_coefs(a0, a1, a2, b0, b1, b2)


class HighPassFilter(PEQFilter):
    def __init__(self, f0: float, Q: float = None, BW: float = None, fs: float = 1):
        if Q is None and BW is None:
            raise ValueError('Either Q or BW must be set.')
        super().__init__(f0, Q=Q, BW=BW, fs=fs)
        self._calc_coefs()

    def _calc_coefs(self):
        a0 = 1 + self._alph
        a1 = -2 * self._cosw0
        a2 = 1 - self._alph
        b0 = (1 + self._cosw0) / 2
        b1 = -(1 + self._cosw0)
        b2 = (1 + self._cosw0) / 2

        self._set_coefs(a0, a1, a2, b0, b1, b2)


class NotchFilter(PEQFilter):
    def __init__(self, f0: float, Q: float, BW: float = None, fs: float = 1):
        super().__init__(f0, Q=Q, BW=BW, fs=fs)
        self._calc_coefs()

    def _calc_coefs(self):
        a0 = 1 + self._alph
        a1 = -2 * self._cosw0
        a2 = 1 - self._alph
        b0 = 1
        b1 = -2 * self._cosw0
        b2 = 1

        self._set_coefs(a0, a1, a2, b0, b1, b2)


class PeakingFilter(PEQFilter):
    def __init__(self, f0: float, gain: float, Q: float = None, BW: float = None, fs: float = 1):
        if Q is None and BW is None:
            raise ValueError('Either Q or BW must be set.')
        super().__init__(f0, gain, Q=Q, BW=BW, fs=fs)
        self._calc_coefs()

    def _calc_coefs(self):
        a0 = 1 + self._alph / self._A
        a1 = -2 * self._cosw0
        a2 = 1 - self._alph / self._A
        b0 = 1 + self._alph * self._A
        b1 = -2 * self._cosw0
        b2 = 1 - self._alph * self._A

        self._set_coefs(a0, a1, a2, b0, b1, b2)


class LowShelfFilter(PEQFilter):
    def __init__(self, f0: float, gain: float, Q: float = None, BW: float = None, S: float = None, fs: float = 1):
        if Q is None and BW is None and S is None:
            raise ValueError('Either Q, BW, or S must be set.')
        super().__init__(f0, gain, Q=Q, BW=BW, S=S, fs=fs, allow_S=True)
        self._calc_coefs()

    def _calc_coefs(self):
        A = self._A
        rootA = math.sqrt(A)
        alpha = self._alph

        a0 = (A + 1) + (A - 1) * self._cosw0 + 2 * rootA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * self._cosw0)
        a2 = (A + 1) + (A - 1) * self._cosw0 - 2 * rootA * alpha
        b0 = A * ((A + 1) - (A - 1) * self._cosw0 + 2 * rootA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * self._cosw0)
        b2 = A * ((A + 1) - (A - 1) * self._cosw0 - 2 * rootA * alpha)

        self._set_coefs(a0, a1, a2, b0, b1, b2)


class HighShelfFilter(PEQFilter):
    def __init__(self, f0: float, gain: float, Q: float = None, BW: float = None, S: float = None, fs: float = 1):
        if Q is None and BW is None and S is None:
            raise ValueError('Either Q, BW, or S must be set.')
        super().__init__(f0, gain, Q=Q, BW=BW, S=S, fs=fs, allow_S=True)
        self._calc_coefs()

    def _calc_coefs(self):
        A = self._A
        rootA = math.sqrt(A)
        alpha = self._alph

        a0 = (A + 1) - (A - 1) * self._cosw0 + 2 * rootA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * self._cosw0)
        a2 = (A + 1) - (A - 1) * self._cosw0 - 2 * rootA * alpha
        b0 = A * ((A + 1) + (A - 1) * self._cosw0 + 2 * rootA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * self._cosw0)
        b2 = A * ((A + 1) + (A - 1) * self._cosw0 - 2 * rootA * alpha)

        self._set_coefs(a0, a1, a2, b0, b1, b2)


class PreAmpFilter(PEQFilter):
    def __init__(self, gain: float, fs: float = 1):
        super().__init__(gain=gain, fs=fs)
        self._calc_coefs()

    def _calc_coefs(self):
        a0 = 1
        a1 = 0
        a2 = 0
        b0 = 10 ** (self._gain / 20)
        b1 = 0
        b2 = 0

        self._set_coefs(a0, a1, a2, b0, b1, b2)


t6ple: TypeAlias = tuple[float, float, float, float, float, float]


class ParametricEqualizer(Filter):
    """
    Parametric equalizer object.
    Helps aggregate a collection of PEQ filters and
    provides methods to render the total frequency/phase response.
    """

    def __init__(self, sample_rate: float = 1, filters: list[PEQFilter] = None):
        self.sample_rate = sample_rate
        self.filters = list(filters) if filters is not None else []

    @property
    def fs(self) -> float:
        return self.sample_rate

    @fs.setter
    def fs(self, fs: float):
        self.sample_rate = fs
        for flt in self.filters:
            flt.fs = fs

    @property
    def coefs_sos(self) -> list[t6ple]:
        """Returns the second-order sections (SOS) representation of the equalizer.

        Each tuple contains the 6 NORMALIZED coefficients of a biquad filter,
        a SOS, in the form: (b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0).
        """
        return [flt.coefs_b_n + flt.coefs_a_n for flt in self.filters]

    def transfer_function(self, z):
        if isinstance(z, np.ndarray):
            temp = np.array([flt.transfer_function(z) for flt in self.filters], ndmin=2)
            return np.prod(temp, axis=0)

        temp = np.fromiter((flt.transfer_function(z) for flt in self.filters), complex128)
        return np.prod(temp)

    def add_filter(self, filter_: PEQFilter):
        """Add a pre-made ``PEQFilter`` object.
        Sample rate ``fs`` must match.
        """
        assert filter_.fs == self.sample_rate, 'Filter sample rate does not match the equalizer sample rate.'
        self.filters.append(filter_)
        return filter_

    def __getitem__(self, item):
        return self.filters[item]

    def __setitem__(self, key, value):
        self.filters[key] = value

    def __delitem__(self, key):
        del self.filters[key]

    def __len__(self):
        return len(self.filters)

    def __iter__(self):
        return iter(self.filters)

    def add_peaking(self, f0: float, gain: float, Q: float = None, BW: float = None):
        return self.add_filter(PeakingFilter(f0, gain, Q=Q, BW=BW, fs=self.sample_rate))

    def add_loshelf(self, f0: float, gain: float, Q: float = None, BW: float = None, S: float = None):
        return self.add_filter(LowShelfFilter(f0, gain, Q=Q, BW=BW, S=S, fs=self.sample_rate))

    def add_hishelf(self, f0: float, gain: float, Q: float = None, BW: float = None, S: float = None):
        return self.add_filter(HighShelfFilter(f0, gain, Q=Q, BW=BW, S=S, fs=self.sample_rate))

    def add_notch_f(self, f0: float, Q: float = None, BW: float = None):
        return self.add_filter(NotchFilter(f0, Q=Q, BW=BW, fs=self.sample_rate))

    def add_lo_pass(self, f0: float, Q: float = None, BW: float = None):
        return self.add_filter(LowPassFilter(f0, Q=Q, BW=BW, fs=self.sample_rate))

    def add_hi_pass(self, f0: float, Q: float = None, BW: float = None):
        return self.add_filter(HighPassFilter(f0, Q=Q, BW=BW, fs=self.sample_rate))

    def add_pre_amp(self, gain: float):
        return self.add_filter(PreAmpFilter(gain, fs=self.sample_rate))

    @staticmethod
    def decibelize(x: float | Collection[float]) -> float | list[float]:
        """Converts linear scale (response function output) to decibel scale.
        Can accept either a single value or an iterable of values.
        """
        if isinstance(x, Iterable):
            return [20 * math.log10(xi) for xi in x]
        return 20 * math.log10(x)

    def frequency_resp(
            self, nsamples: int = 10_000,
            min_f: float = 15,
            max_f: float | None = 22_000,
            log: bool = True
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """Renders a discretized frequency response of the entire equalizer by sampling the frequency range.

        The first list contains the sampled frequencies; the second list contains the corresponding responses.
        The response are magnitude factors (not gain; for that use ``frequency_response_db``).
        """
        if max_f is None:
            max_f = self.sample_rate / 2
        assert max_f <= self.sample_rate / 2, 'Maximum frequency must not be greater than half the sample rate.'
        assert min_f < max_f, 'Minimum frequency must be smaller than maximum frequency.'
        assert min_f >= 0, 'Minimum frequency cannot be negative.'

        if log:  # craft a logarithmic frequency range
            assert min_f > 0, 'Minimum frequency must be positive for logarithmic scale.'
            f_s = np.geomspace(min_f, max_f, nsamples)
        else:  # craft a linear frequency range
            f_s = np.linspace(min_f, max_f, nsamples)

        # calculate response
        f_s, resp = freqz_sos(self.coefs_sos, worN=f_s, fs=self.sample_rate)
        fr = np.abs(resp)  # convert to magnitude factor
        return f_s, fr

    def frequency_resp_fast(self, min_samples: int = 10_000) -> tuple[NDArray[float64], NDArray[float64]]:
        """Like ``frequency_resp``, but has a linear sampling range
        spanning the entire frequency range and with a non-specific number of samples.

        This allows the frequency response to be rendered faster using FFT rather than
        direct calculation through the transfer function.
        """
        worN = next_fast_len(min_samples)
        f_s, resp = freqz_sos(self.coefs_sos, worN=worN, fs=self.sample_rate)
        fr = np.abs(resp)
        return f_s, fr

    def apply(self, x: Collection[float]) -> list[float] | NDArray[float64]:
        return sosfilt(self.coefs_sos, x)

    def apply_on(self, stream: Iterator[float]) -> Generator[float, None, None]:
        x2, x1 = 0., 0.
        y2, y1 = 0., 0.
        for x in stream:
            y = x
            for flt in self.filters:
                y = flt.difference_eq(y, x1, x2, y1, y2)
            yield y
            y2, y1 = y1, y
            x2, x1 = x1, x
