import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import model

filename = "example_data.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
plt.plot(data[:, 0], data[:, 1], "ro")

popt, pcov = curve_fit(model, data[:, 0], data[:, 1], p0=(1, 0.2))
xmin = np.min(data[:, 0])
xmax = np.max(data[:, 0])
x = np.linspace(xmin, xmax, 100)
plt.plot(x, model(x, *popt))
plt.show()
