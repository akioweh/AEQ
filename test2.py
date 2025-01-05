import numpy as np

from AEQ.parametric import ParametricEqualizer

fs = 384_000

EQ = ParametricEqualizer(fs)
EQ.add_loshelf(38, 3.0, 1.00)

# read freq, resp values from Harman in-ear 2019.csv
# using numpy
target = np.loadtxt("Harman in-ear 2019.csv", delimiter=",", skiprows=1)
# # add EQ response to target response
# target[:, 1] += EQ.frequency_resp_at(target[:, 0])
# # normalize target response
target[:, 1] -= np.max(target[:, 1])
# # insert 0, target[0, 0] in front
# # and 0, 0 at the end
# target = np.concatenate((np.array((0, target[0, 1]), ndmin=2), target, np.array((fs / 2, 0.0), ndmin=2)), axis=0)

# plot_fr(f_s=r, y=EQ.frequency_resp_at(r), x_min=1, x_max=fs / 2, log=True, title='EQ')

# plot target
# plot_fr(f_s=target[:, 0], y=target[:, 1], x_min=1, x_max=fs / 2, log=True, title='Target')

# create FIR filter
# nt = fs // 32
# fir = sgn.firwin2(nt + 1, target[:, 0], target[:, 1], fs=fs, antisymmetric=False, window=None)
# w, h = sgn.freqz(fir, worN=fs // 2)
# mag = 20 * np.log10(np.abs(h))
# plot_fr(f_s=w, y=mag, x_min=1, x_max=fs / 2, log=True, title='Frequency Response')

