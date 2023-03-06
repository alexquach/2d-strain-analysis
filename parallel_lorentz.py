import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import time
start_time = time.time()

df_filtered_PL = pd.read_csv("multi_lossy_x-75to75_y-95to55_50by50.csv")

xData = df_filtered_PL['W']

df_iter = df_filtered_PL.copy()
df_iter.drop(["W"], axis=1, inplace=True)
# df_iter.drop("Unnamed: 0", axis=1, inplace=True)

def lorentzian(x, x0, a, gam):
    return a / (1 + (2 * (x - x0)/gam)**2 )

# Define a function that takes in a set of parameters and returns the sum of multiple Lorentzian functions evaluated at the given x values.
# This function uses the parameters in params to calculate the offset (params[0]) and the parameters for each individual Lorentzian function.
# The parameters for each individual Lorentzian function are grouped into sets of three (x0, a, and gam).
def multi_lorentz(x, off, x0_1, a1, gam1, x0_2, a2, gam2):
    # off = params[0] # extract the offset parameter
    # x0_1, a1, gam1, x0_2, a2, gam2 = params[1:] # extract the parameters for the individual Lorentzian functions

    # assert not (len(paramsRest) % 3) # check that there are a multiple of three parameters for the Lorentzian functions
    # loop through the parameters for the individual Lorentzian functions and evaluate each function at the given x values
    # then sum up the results and add the offset to get the final result
    return off + lorentzian(x, x0_1, a1, gam1) + lorentzian(x, x0_2, a2, gam2)

# Define a function that takes in a set of parameters, x values, and y values, and returns the difference between the predicted y values (based on the parameters) and the actual y values.
# This function is used by the curve_fit function to minimize the sum of squared residuals between the predicted and actual y values.
def res_multi_lorentz(params, xData, yData):
    # use the multi_lorentz function to predict the y values based on the given parameters and x values
    print(type(xData))
    y_pred = [multi_lorentz(x, params) for x in xData]
    # calculate the difference between the predicted and actual y values
    diff = [y_pred[i] - yData[i] for i in range(len(yData))]
    return diff

def fit_lorentzians(xData, yData, plot=True): 
  startValues = [0.1, 725, 1, 30, 750, 1, 30]
  bounds = [-np.inf, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

  popt, pcov, idict, mesg, ier = curve_fit( multi_lorentz, xData, yData, startValues, bounds=bounds, full_output=True )   # Fit multiple Lorentzian functions to the data

  print(idict)

  if plot:
    print(f"offset: {popt[0]}")
    print(f"x0, a, gam: {popt[1]}, {popt[2]}, {popt[3]}")
    print(f"x0, a, gam: {popt[4]}, {popt[5]}, {popt[6]}")
    plot_fixed_parameters(popt)

  return popt

def mse(popt, x, y):
  off, x0_1, a1, gam1, x0_2, a2, gam2 = popt
  y_pred = [multi_lorentz(x_, off, x0_1, a1, gam1, x0_2, a2, gam2) for x_ in x]
  return ((y-y_pred)**2).mean()

def fit_lorentzians_given_yData(yData): 
  label_name = yData.name
  yData = yData / max(yData)
  startValues = [0.1, 725, 1, 30, 750, 1, 30]
  bounds = [-np.inf, 0, 0, 5, 0, 0, 5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

  try:
    popt, pcov = curve_fit( multi_lorentz, xData, yData.squeeze(), startValues, bounds=bounds, maxfev=100000 )   # Fit multiple Lorentzian functions to the data
  except RuntimeError as e:
    print(e)
    print(label_name)
    return [0, 0, 0, 0, 0, 0, 0, 420]
  popt = np.append(popt, [mse(popt, xData.values, yData.values)])
  return popt

def plot_fixed_parameters(popt):
  x = np.linspace(687, 810, 1340)
  off, x0_1, a1, gam1, x0_2, a2, gam2 = popt

  y = [multi_lorentz(x_, off, x0_1, a1, gam1, x0_2, a2, gam2) for x_ in x]
  plt.plot(xData, yData)
  plt.plot(x, y)

  plt.show()

def run_fit(inp):
    print(f"Getting input {inp}")
    out = fit_lorentzians_given_yData(inp)

    print(f"Got output {out}")
    return list(out)

test_df = df_iter
def process_image(name):
    # print(f"Fitting {name}")
    return fit_lorentzians_given_yData(test_df[name])

if __name__ == "__main__":
    from tqdm.auto import tqdm
    # x = pd.DataFrame({k: fit_lorentzians_given_yData(test_df[k]) for k in tqdm(test_df.columns)})
    from multiprocessing import Pool
    import os

    pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
    result = pool.map(process_image, test_df.columns)  # process data_inputs iterable with pool
    result = pd.DataFrame({
        k: v for k, v in zip(test_df.columns, result)
    })
    print(f"Result:\n{result}")
    print("--- %s seconds ---" % (time.time() - start_time))
    result.to_csv('multi_results_x-75to75_y-95to55_50by50.csv')