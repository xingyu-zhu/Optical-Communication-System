"""常用绘图：星座图、功率谱、眼图。"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def constellation(x, ax=None, title='', bins=200):
    ax = ax or plt.gca()
    x = np.ravel(x)
    lim = 1.3 * np.max(np.abs(np.r_[x.real, x.imag]))
    ax.hist2d(x.real, x.imag, bins=bins, range=[[-lim, lim], [-lim, lim]], cmap='turbo', cmin=1)
    ax.set_aspect('equal'); ax.set_title(title)
    return ax


def spectrum(x, fs, ax=None, label=None):
    ax = ax or plt.gca()
    f, p = welch(np.atleast_2d(x)[0], fs, nperseg=2048, return_onesided=False)
    ax.plot(np.fft.fftshift(f) / 1e9, 10 * np.log10(np.fft.fftshift(p) + 1e-30), label=label)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('PSD (dB/Hz)')
    return ax


def eye(x, sps, ax=None, n=500, span=2):
    ax = ax or plt.gca()
    L = span * sps
    x = np.real(np.ravel(x))
    seg = x[: min(n, len(x) // L) * L].reshape(-1, L)
    ax.plot(np.arange(span * sps) / sps, seg.T, 'b', alpha=0.05)
    ax.set_xlabel('Time (UI)')
    return ax
