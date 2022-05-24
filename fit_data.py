import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import model

filename = sys.argv[1]
try:
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
except OSError:
    raise FileNotFoundError(f"File {filename} not found")

plt.plot(data[:, 0], data[:, 1], "ro")

popt, pcov = curve_fit(model, data[:, 0], data[:, 1], p0=(1, 0.2))
xmin = np.min(data[:, 0])
xmax = np.max(data[:, 0])
x = np.linspace(xmin, xmax, 50)
plt.plot(x, model(x, *popt))
plt.show()
