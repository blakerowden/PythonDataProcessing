import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("3Phase_60.csv", dtype=float, delimiter=",", names=True)


# make data
t = data["t_s"]  # time
tstep = t[1] - t[0]  # time step
Fs = 1 / tstep  # sampling frequency


Ig_pos_q = data["Ig_pos_q"]
Ig_pos_d = data["Ig_pos_d"]
Udc = data["Udc"]
Ilsc_pos_d = data["Ilsc_pos_d"]
Ilsc_pos_q = data["Ilsc_pos_q"]

# extract all data from t=0.2 to t=0.35
fault_start_index = int(0.2 / tstep)
fault_end_index = int(0.35 / tstep)
t = t[fault_start_index:fault_end_index]
Ig_pos_q = Ig_pos_q[fault_start_index:fault_end_index]
Ig_pos_d = Ig_pos_d[fault_start_index:fault_end_index]
Udc = Udc[fault_start_index:fault_end_index]
Ilsc_pos_d = Ilsc_pos_d[fault_start_index:fault_end_index]
Ilsc_pos_q = Ilsc_pos_q[fault_start_index:fault_end_index]


def compute_fft(x) -> tuple:
    # Compute the FFT of x(t)
    N = len(x)
    X = np.fft.fft(x)
    X_mag = np.abs(X) / N
    freqs = np.linspace(0, (N - 1) * Fs / N, N)
    return X_mag, freqs


def plot_fft(x, ax, label=None):
    X_mag, freqs = compute_fft(x)
    ax.plot(freqs, np.abs(X_mag), label=label)
    ax.set_xlim(0, 250)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


fig, axs = plt.subplots(3, 1)
axs[0].plot(t, Ig_pos_d, label="d-component")
axs[0].plot(t, Ig_pos_q, label="q-component")
axs[0].set_xlim(0, 1)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Grid Current")
axs[0].grid(True)

axs[1].plot(t, Udc, label="DC Voltage")
axs[1].plot(t, Ilsc_pos_d, label="ILSC d-component")
axs[1].set_xlim(0, 1)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("DC Bus Control")
axs[1].grid(True)
axs[0].legend()
axs[1].legend()  # Add a legend.

plot_fft(Ig_pos_d, axs[2], label="d-component")

fig.tight_layout()
plt.show()
