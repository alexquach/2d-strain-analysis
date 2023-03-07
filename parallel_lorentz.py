import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lorentzian import fit_lorentzians_given_yData

import time
start_time = time.time()
is_raman = True
theoretical_raman_peak = 250
xlim = (theoretical_raman_peak-10, theoretical_raman_peak+10)

csv_path = 'multi_lossy_region15_raman_1mW_filter1_3.5K_grating2400_center250_exp.3s_x-150to150_y-150to150_100by100.csv'
df_filtered = pd.read_csv(csv_path)

# Set initial conditions and constraints
if is_raman:
    xData = df_filtered['Wavenumber']
    startValues = [0.1, 0, theoretical_raman_peak, 1, 1, theoretical_raman_peak+5, 1, 1]
    bounds = [-np.inf, -np.inf, 0, 0, 1, 0, 0, 1], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
else:
    xData = df_filtered['W']
    startValues = [0.1, 725, 1, 30, 750, 1, 30]
    bounds = [-np.inf, 0, 0, 5, 0, 0, 5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

# print(df_filtered[(xData > xlim[0]) & (xData < xlim[1])])

# Create a copy and preprocess
if xlim:
    df_iter = df_filtered[(xData > xlim[0]) & (xData < xlim[1])]
    xData = xData[(xData > xlim[0]) & (xData < xlim[1])]
else:
    df_iter = df_filtered.copy()

# Drop the wavelength / wavenumber column
if is_raman:
    df_iter = df_iter.drop(["Wavenumber"], axis=1)
    df_iter = df_iter.drop(["Unnamed: 0"], axis=1)
else:
    df_iter = df_iter.drop(["W"], axis=1, inplace=True)


def run_fit(name):
    print(f"Fitting {name}")
    return fit_lorentzians_given_yData(df_iter[name], xData, startValues, bounds)

if __name__ == "__main__":
    from multiprocessing import Pool
    import os

    print(df_iter)

    pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
    result = pool.map(run_fit, df_iter.columns)  # process data_inputs iterable with pool
    result = pd.DataFrame({
        k: v for k, v in zip(df_iter.columns, result)
    })
    print(f"Result:\n{result}")
    print("--- %s seconds ---" % (time.time() - start_time))
    result.to_csv(f'multi_results_{xlim[0]}to{xlim[1]}_{csv_path}')