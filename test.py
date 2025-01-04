from math import sqrt

import numpy as np
from scipy import fft, signal as sgn

from AEQ import parametric as peq, plot_fr
from AEQ.fileio import create_format

fs = 384_000

EQ = peq.ParametricEqualizer(fs)

# Harman in-ear 2019 Target.
# !!=> with -3.5 dB sqrt(2)/2 Q low shelf @ 105 Hz
# !!=> On autoeq.app, this means adjusting the 9.5 dB (default) bass boost to 6.0 dB
EQ.add_pre_amp(-6.00)
EQ.add_loshelf(  105, -0.7, 0.70)
EQ.add_peaking(  143, -3.9, 0.42)
EQ.add_peaking(  792,  3.5, 0.84)
EQ.add_peaking( 1978, -3.6, 2.22)
EQ.add_peaking( 3356,  5.3, 3.28)
EQ.add_peaking( 4517, -3.3, 5.97)
EQ.add_peaking( 5202,  1.2, 5.65)
EQ.add_peaking( 6093,  4.2, 4.52)
EQ.add_peaking( 8610,  1.1, 4.48)
EQ.add_hishelf(10000,  3.4, 0.70)

# Personal resonance adjustments
EQ.add_peaking( 5500, -5.0, 5.00)
EQ.add_peaking( 5900,-12.5, 2.30)
EQ.add_peaking( 6500, -4.0,12.00)
EQ.add_peaking( 7000, -4.0,12.00)
EQ.add_peaking( 8500,  7.5, 3.50)
EQ.add_peaking(11500, -6.7, 7.00)

# Personal bass tuning
EQ.add_loshelf(110, 3.0, sqrt(2) / 2)
EQ.add_loshelf(38, 3.0, 1.00)
# EQ.add_pre_amp(-2.40)
# EQ.add_peaking(11000, 2.5, 0.50)

# EQ.add_lo_pass(26000, sqrt(2) / 2)
# EQ.add_lo_pass(26000, sqrt(2) / 2)
# EQ.add_lo_pass(26000, sqrt(2) / 2)
# EQ.add_lo_pass(26000, sqrt(2) / 2)
# EQ.add_hi_pass(24, sqrt(2) / 2)
# EQ.add_hi_pass(24, sqrt(2) / 2)
# EQ.add_hi_pass(24, sqrt(2) / 2)
# EQ.add_hi_pass(24, sqrt(2) / 2)


f_s, fr = EQ.frequency_resp(1_000_000, min_f=0, max_f=fs / 2, log=False)
nt = fs // 32

fir = sgn.firwin2(nt + 1, f_s, fr, fs=fs, antisymmetric=False, window=None)

# fir = EQ.impulse_resp(nt // 2 + 1)
# fir = np.concatenate((fir[:0:-1], fir))

# fir = EQ.impulse_resp(fs)
# fir = np.concatenate((fir[::-1], fir)) / 2
# wind = sgn.get_window('hamming', len(fir))
# fir = fir * wind

print(len(fir))
print(sum(fir))
fmt = create_format(1, fs, 64, True)
# write_wave('eq2.wav', fir, fmt)

# Freq and phase response using transfer function
w, h = sgn.freqz(fir, fs=fs, worN=1_000_000)
mag = 20 * np.log10(np.abs(h))
phase = np.angle(h)
plot_fr(w, mag, log=True, x_min=16, x_max=22_000, y_max=1, y_min=-20)
plot_fr(w, phase, log=False, x_min=16, x_max=22_000, title='Phase Response')

# Freq response using FFT from IR
y = fft.rfft(fir)
x = fft.rfftfreq(len(fir), 1 / fs)
mag = 20 * np.log10(np.abs(y))
plot_fr(x, mag, log=True, x_min=16, x_max=22_000, y_max=1, y_min=-20)
