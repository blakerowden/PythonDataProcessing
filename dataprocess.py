import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from dataclasses import dataclass, field
from global_ import *

# Current Experimental Data Settings
WIND_SPEED = 3
FAULT_TYPE = FaultType.POLE3
FAULT_DURATION = 0.15
FAULT_START_TIME = 0.2
FAULT_START_ANGLE = 0
VOLTAGE_DIP = 0.6
CONTROLLER_GAIN = 2
FILE_NAME = "3Phase_60.csv"


@dataclass(order=True)
class ExperimentalData:
    wind_speed: int
    fault_type: FaultType
    fault_duration: float
    fault_start_time: float
    fault_start_angle: int
    voltage_dip: int
    controller_gain: int
    fulldata: np.array
    t_step: float = field(init=False)
    Fs: float = field(init=False)
    prefault_data: dict = field(default_factory=dict)
    fault_data: dict = field(default_factory=dict)
    postfault_data: dict = field(default_factory=dict)


def extact_data(none) -> ExperimentalData:
    dirname = os.path.dirname(__file__)
    fullpath = os.path.join(dirname, FILE_NAME)
    rawdata = np.genfromtxt(fullpath, dtype=float, delimiter=",", names=True)
    data = ExperimentalData(
        WIND_SPEED,
        FAULT_TYPE,
        FAULT_DURATION,
        FAULT_START_TIME,
        FAULT_START_ANGLE,
        VOLTAGE_DIP,
        CONTROLLER_GAIN,
        rawdata,
    )
    data.t_step = data.fulldata["t_s"][1] - data.fulldata["t_s"][0]
    data.Fs = 1 / data.t_step
    fault_start_index = int(data.fault_start_time / data.t_step) - 1
    fault_end_index = int((data.fault_start_time + data.fault_duration) / data.t_step)
    for key in data.fulldata.dtype.names:
        data.prefault_data[key] = data.fulldata[key][0:fault_start_index]
        data.fault_data[key] = data.fulldata[key][
            fault_start_index - 1 : fault_end_index
        ]
        data.postfault_data[key] = data.fulldata[key][fault_end_index - 1 :]
    return data


def compute_fft(x, Fs) -> tuple:
    # Compute the FFT of x(t)
    N = len(x)
    X = np.fft.fft(x)
    X_mag = np.abs(X) / N
    freq_bins = np.linspace(0, (N - 1) * Fs / N, N)
    # Remove the negative frequencies
    X_mag = X_mag[0 : int(N / 2)]
    freq_bins = freq_bins[0 : int(N / 2)]
    # Remove the DC component
    X_mag = X_mag[1:]
    freq_bins = freq_bins[1:]
    idx = np.argsort(X_mag)[::-1]
    peak_freq = freq_bins[idx[0]]
    seccond_peak_freq = freq_bins[idx[1]]
    return X_mag, freq_bins


def plot_fft(ExperimentalData, FaultSection, ax, key, label=None, number_of_max=0):
    if FaultSection == PREFAULT:
        x = ExperimentalData.prefault_data[key]
        position = 1
    elif FaultSection == FAULT:
        x = ExperimentalData.fault_data[key]
        position = 2
    elif FaultSection == POSTFAULT:
        x = ExperimentalData.postfault_data[key]
        position = 3

    X_mag, freqs = compute_fft(x, ExperimentalData.Fs)
    ax.plot(freqs, np.abs(X_mag), label=label)
    ax.set_xlim(0, 250)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    ax.set_title(key + " Frequency Domain")
    if number_of_max == 1:
        annot_max(freqs, np.abs(X_mag), position, ax)
    elif number_of_max == 2:
        annot_max(freqs, np.abs(X_mag), position, ax, 2)


def plot_time_data(ExperimentalData, FaultSection, ax, key, label=None) -> None:
    if FaultSection == PREFAULT:
        ax.plot(
            ExperimentalData.prefault_data[t],
            ExperimentalData.prefault_data[key],
            label=label,
        )
    elif FaultSection == FAULT:
        ax.plot(
            ExperimentalData.fault_data[t],
            ExperimentalData.fault_data[key],
            label=label,
        )
    elif FaultSection == POSTFAULT:
        ax.plot(
            ExperimentalData.postfault_data[t],
            ExperimentalData.postfault_data[key],
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(key + " (p.u)")
    ax.grid(True)
    ax.legend()
    ax.set_title(key + " Time Domain")


def annot_max(x, y, pos, ax=None, number_of_max=1):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "{:.3f}Hz".format(xmax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha="right",
        va="top",
    )
    if pos == 1:
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.7, 0.9), **kw)
    elif pos == 2:
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.7, 0.7), **kw)
    elif pos == 3:
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.7, 0.5), **kw)

    if number_of_max == 2:
        xmax = x[np.argsort(y)[-2]]
        ymax = y[np.argsort(y)[-2]]
        text = "{:.3f}Hz".format(xmax)
        if pos == 1:
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.4, 0.8), **kw)
        elif pos == 2:
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.4, 0.6), **kw)
        elif pos == 3:
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.4, 0.4), **kw)


def main() -> None:
    """
    Main function for the application.
    """
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    data = extact_data(None)

    fig, axs = plt.subplots(2, 1)
    plot_time_data(data, PREFAULT, axs[0], Ig_pos_d, label="Pre-Fault")
    plot_time_data(data, FAULT, axs[0], Ig_pos_d, label="Fault")
    plot_time_data(data, POSTFAULT, axs[0], Ig_pos_d, label="Post-Fault")
    plot_fft(data, PREFAULT, axs[1], Ig_pos_d, label="Pre-Fault", number_of_max=1)
    plot_fft(data, FAULT, axs[1], Ig_pos_d, label="Fault", number_of_max=2)
    plot_fft(data, POSTFAULT, axs[1], Ig_pos_d, label="Post-Fault")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Only run the main function if this file is run directly.
    """
    main()
