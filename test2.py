from AEQ.parametric import ParametricEqualizer

fs = 384_000

# flt1 = LowPassFilter(1000, sqrt(2) / 2, fs)
# ir1 = flt1.impulse_resp(fs // 8)
#
# flt2 = HighPassFilter(500, 2, fs)
# ir2 = flt2.impulse_resp(fs // 8)
#
# ir3_1 = convolve(ir1, ir2, method='direct', mode='same')
#
# x = [0] * (fs // 8)
# x[0] = 1
# ir3_2 = flt2.apply(flt1.apply(x))
# ir3_2 = np.array(ir3_2)
#
# print(ir3_1)
# print(ir3_2)

EQ = ParametricEqualizer(fs)
EQ.add_pre_amp(-6)

print(EQ.impulse_resp(10))
