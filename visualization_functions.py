import numpy as np
import matplotlib.pyplot as plt

from lorentzian import multi_lorentz

# TODO: Add from Colab

def plot_fixed_parameters(popt, xData, yData):
  x = np.linspace(687, 810, 1340)
  off, x0_1, a1, gam1, x0_2, a2, gam2 = popt

  y = [multi_lorentz(x_, off, x0_1, a1, gam1, x0_2, a2, gam2) for x_ in x]
  plt.plot(xData, yData)
  plt.plot(x, y)

  plt.show()