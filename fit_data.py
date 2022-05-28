"""
Read data points and plot them along fitted model
Usage: python fit_data.py data_file
"""
import sys
import argparse
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import model
import logging

parser = argparse.ArgumentParser(description="Plot data and fitted model")
parser.add_argument("filename", type=str, help="Name of data file")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()


if "--help" in sys.argv:
    print("usage: python {sys.argv[0]} file [--plot]")
    sys.exit()

if args.save and not args.plot:
    warn("Ignoring --save option. Use --plot to enable plotting")

try:
    data = np.loadtxt(args.filename, delimiter=",", skiprows=1)
except OSError:
    raise FileNotFoundError(f"File {args.filename} not found")

popt, pcov = curve_fit(model, data[:, 0], data[:, 1], p0=(1, 0.2))
logging.info("p1 = {popt[0]}, p2 = {popt[1]}")

if args.plot in sys.argv:
    plt.plot(data[:, 0], data[:, 1], color="r", "*", label="data points")
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
    plt.plot(x, model(x, *popt), "b", linewidth=2, label="fitted model")
    if args.save:
        plt.savefig("figure.png")
    plt.show()
