import numpy as np
from scipy import signal as sgn
from scipy.fft import next_fast_len

from AEQ import ParametricEqualizer, load_resp, plot_fr, smooth_resp_section
from AEQ.fileio import create_format, write_wave


def decibelize(y):
    return 20 * np.log10(y)


fs = 384_000

EQ = ParametricEqualizer(fs)
EQ.add_loshelf(38, 3.0, 1.00)

target_f_s, target_y_db = load_resp('Harman in-ear 2019.csv')
# # add custom EQ
target_y_db += decibelize(EQ.frequency_resp_at(target_f_s))

measured_f_s, measured_y_db = load_resp('Moondrop Chu 2.csv')

assert np.array_equal(target_f_s, measured_f_s)
f_s = target_f_s

# this is the EQ we want to generate
correction_y_db = measured_y_db - target_y_db  # negative error
# smooth high frequencies
correction_y_db = smooth_resp_section(f_s, correction_y_db, start=12_500, transition=1_500)

# gain normalize
correction_y_db -= np.max(correction_y_db)
# convert to linear scale
correction_y = 10 ** (correction_y_db / 20)

# add datapoints for 0 Hz and fs/2, because firwin2 requires it
f_s = np.concatenate((np.array([0]), f_s, np.array([fs / 2])))
correction_y = np.concatenate((np.array([correction_y[0]]), correction_y, np.array([correction_y[-1]])))

# plot_fr(f_s=f_s, y=correction_y, title='Correction, Full')

### adjustment cycle 1

EQ1 = ParametricEqualizer(fs)
EQ1.add_peaking(13000, -6.2, 1.98)
EQ1.add_peaking( 9200,  3.0, 0.58)
EQ1.add_peaking(12800, -4.8, 7.00)
EQ1.add_peaking( 6400,  2.0, 1.67)
EQ1.add_peaking( 9750,  3.0, 4.00)
EQ1.add_peaking( 9900,  3.5, 4.84)
EQ1.add_peaking(11900, -3.2, 10.7)
EQ1.add_peaking( 8800, -3.4, 10.0)
EQ1.add_peaking(14100, -5.6, 13.4)
EQ1.add_peaking(12000, -3.9, 12.6)
EQ1.add_peaking( 6150,  2.5, 5.80)
EQ1.add_peaking(10200,  2.5, 9.44)
EQ1.add_peaking(11400,  2.5, 9.35)
EQ1.add_peaking(15000, -3.0, 9.84)
EQ1.add_peaking( 7500, -1.8, 10.0)
EQ1.add_peaking(10800,  2.2, 10.0)
EQ1.add_peaking(13330, -4.1, 19.0)

correction_y *= EQ1.frequency_resp_at(f_s)

# smooth high frequencies
correction_y_db = decibelize(correction_y)
correction_y_db = smooth_resp_section(f_s, correction_y_db, start=11_000, transition=1_500, window_len=10)
correction_y = 10 ** (correction_y_db / 20)

# normalize
correction_y /= np.max(correction_y)

# create FIR filter
nt = fs // 32
fir = sgn.firwin2(nt + 1, f_s, correction_y, fs=fs, antisymmetric=False)
w, h = sgn.freqz(fir, worN=next_fast_len(fs), fs=fs)
h = np.abs(h)

# plot_fr(f_s=w, y=h, title='Filter Response, Full')

plot_fr(f_s=f_s, y=correction_y, x_min=5, x_max=24_000, y_min=-20, y_max=3, title='Correction')
plot_fr(f_s=w, y=h, x_min=5, x_max=24_000, y_min=-20, y_max=3, title='Filter Response')

fmt = create_format(1, fs, 64, True)
write_wave('chu2.wav', fir, fmt)
