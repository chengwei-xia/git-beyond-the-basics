import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import model

filename = sys.argv[1]
data = np.loadtxt(filename, delimiter=",", skiprows=1)
plt.plot(data[:, 0], data[:, 1], "ro")

popt, pcov = curve_fit(model, data[:, 0], data[:, 1], p0=(1, 0.2))
x = np.linspace(-2, 2.9, 100)
plt.plot(x, model(x, *popt))
plt.show()
