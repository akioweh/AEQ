from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# plt.ion()


def plot_fr(f_s, y, y_max=None, y_min=None, x_min=None, x_max=None, log=True, title='Frequency Response'):
    fig = plt.figure()
    fig.set_size_inches(20, 6)
    fig.set_dpi(200)
    ax: Axes = fig.subplots()
    ax.set_xlabel('Frequency / Hz')
    ax.set_ylabel('Response / dB')
    ax.set_title(title)
    ax.plot(f_s, y)
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if x_max is not None:
        ax.set_xlim(right=x_max)
    if log:
        ax.set_xscale('log')
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.grid()
    fig.tight_layout()
    plt.show()
