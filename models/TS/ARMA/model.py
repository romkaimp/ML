import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

time = np.linspace(0, 10 * np.pi, 1000)

series = np.random.normal(0, 1, 1000) * np.sin(np.linspace(0, 10 * np.pi, 1000))
noise = np.random.rand(1000) * 2 - 1
noise_disp = np.std(noise) ** 2
print(noise_disp)
print(noise_disp ** 0.5)

# plt.plot(time, noise, label="noise")
# plt.plot(time, series, label="series")
# plt.legend()
# plt.show()

def arma_process(p, q, series: np.ndarray, noise: np.ndarray, time: np.ndarray) -> np.ndarray:
    length = series.shape[0]
    g = np.empty(series.shape[0] - 1)
    g_left = np.mean(series[:np.arange(length)] * noise[length - np.arange(length):], axis=0)
    g_right = np.mean(series[length - np.arange(length):] * noise[:np.arange(length)], axis=0)
    print(g_left)
    print(g_right)

arma_process(2, 4, series, noise, time)