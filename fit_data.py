"""
Read data points and plot them along fitted model
Usage: python fit_data.py data_file
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import model
import logging

if len(sys.argv) < 2:
    raise RuntimeError("Missing filename argument")

if "--help" in sys.argv:
    print("usage: python {sys.argv[0]} file [--plot]")
    sys.exit()

filename = sys.argv[1]
try:
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
except OSError:
    raise FileNotFoundError(f"File {filename} not found")

popt, pcov = curve_fit(model, data[:, 0], data[:, 1], p0=(1, 0.2))
logging.info("p1 = {popt[0]}, p2 = {popt[1]}")

if "--plot" in sys.argv:
    plt.plot(data[:, 0], data[:, 1], color="r", "*", label="data points")
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
    plt.plot(x, model(x, *popt), "b", linewidth=2, label="fitted model")
    plt.show()
