import numpy as np
from matplotlib import pyplot as plt


def plot_resp(
        f_s,
        y,
        y_max: float = None,
        y_min: float = None,
        x_min: float = None,
        x_max: float = None,
        log: bool = True,
        title: str = 'Frequency Response',
        clip_warning: bool = False
):
    # plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 6)
    fig.set_dpi(200)
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
    if clip_warning:
        zero = np.zeros_like(y)
        ax.fill_between(f_s, y, zero, where=(y > 0), interpolate=True, color='red')

    ax.grid()
    fig.tight_layout()
    plt.show()


def plot_fr(
        f_s,
        y,
        y_max: float = None,
        y_min: float = None,
        x_min: float = None,
        x_max: float = None,
        log: bool = True,
        title: str = 'Frequency Response',
        decibelize: bool = True
):
    if decibelize:
        y = 20 * np.log10(y)
    plot_resp(f_s, y, y_max, y_min, x_min, x_max, log, title, clip_warning=True)