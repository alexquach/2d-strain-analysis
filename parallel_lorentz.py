import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lorentzian import fit_lorentzians_given_yData

import time
start_time = time.time()

is_raman = False
with_slope = False
type_process = "centroid" # "lorentz" or "centroid

theoretical_raman_peak = 308
# xlim = (theoretical_raman_peak-10, theoretical_raman_peak+10)
# xlim = (700, 800)
xlim = None

# Raman
# csv_path = 'multi_lossy_region15_raman_1mW_filter1_3.5K_grating2400_center250_exp.3s_x-150to150_y-150to150_100by100.csv'
# PL
# csv_path = 'multi_lossy_x-75to75_y-95to55_50by50.csv'
# csv_path = 'Hyperspectral PL/region22_2mW_filter4_3.5K_grating600_center750_exp2s/x-150to150_y-150to150_100by100.csv'
csv_path = 'Filter4/23_02_03_PLmap_600grat_750nmcenter_1mW_filter4_exp3s_3point9K_3features_x-40to10_y-200to0_50by200.csv'
csv_file = csv_path.split('/')[-1]
csv_filename_without_ext = csv_file.split('.')[0]
csv_folder = "/".join(csv_path.split('/')[:-1]) + '/'

df_filtered = pd.read_csv(f"{csv_folder}{csv_file}")

# Set initial conditions and constraints
if is_raman and with_slope:
    xData = df_filtered['Wavenumber']
    startValues = [0.1, 0, theoretical_raman_peak, 1, 1, theoretical_raman_peak+5, 1, 1]
    bounds = [-np.inf, -np.inf, 0, 0, 1, 0, 0, 1], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
elif is_raman:
    xData = df_filtered['Wavenumber']
    startValues = [0.1, theoretical_raman_peak, 1, 1, theoretical_raman_peak+5, 1, 1]
    bounds = [-np.inf, 0, 0, 1, 0, 0, 1], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
else:
    xData = df_filtered["W"]
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
    df_iter = df_iter.drop(["W"], axis=1)
    # df_iter = df_iter.drop(["Unnamed: 0"], axis=1)


def run_lorentz_peak_fit(name):
    print(f"Fitting {name}")
    return fit_lorentzians_given_yData(df_iter[name], xData, startValues, bounds)

def run_centroid_fit(name):
    # print(f"Fitting {name}")
    return [centroid(df_iter[name], xData)]

# calculate centroid from 1d data:
def centroid(yData, xData):
    return np.sum(yData * xData) / np.sum(yData)

if type_process == "lorentz":
    run_func = run_lorentz_peak_fit
elif type_process == "centroid":
    run_func = run_centroid_fit

if __name__ == "__main__":
    from multiprocessing import Pool
    import os

    print(df_iter)

    pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
    result = pool.map(run_func, df_iter.columns)  # process data_inputs iterable with pool
    result = pd.DataFrame({
        k: v for k, v in zip(df_iter.columns, result)
    })
    print(f"Result:\n{result}")
    print("--- %s seconds ---" % (time.time() - start_time))
    if xlim:
        result.to_csv(f'{csv_folder}{type_process}_{xlim[0]}to{xlim[1]}_{"wslope" if with_slope else "noslope"}_{csv_file}', index=False)
    else:
        result.to_csv(f'{csv_folder}{type_process}_{"wslope" if with_slope else "noslope"}_{csv_file}', index=False)