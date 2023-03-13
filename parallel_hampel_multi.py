import pandas as pd
import numpy as np
import time

start_time = time.time()

window_size = 11
sigma_threshold = 3
# csv_path = 'Hyperspectral PL/region22_2mW_filter4_3.5K_grating600_center750_exp2s/x-150to150_y-150to150_100by100.csv'
csv_path = 'Hyperspectral PL/region15_1mW_filter3_3.5K_grating600_center750_exp.1s/x-75to75_y-95to55_50by50.csv'
csv_file = csv_path.split('/')[-1]
csv_filename_without_ext = csv_file.split('.')[0]
csv_folder = "/".join(csv_path.split('/')[:-1]) + '/'

# input_csv = 'region15_raman_1mW_filter1_3.5K_grating2400_center250_exp.3s_x-150to150_y-150to150_100by100.csv'
df = pd.read_csv(csv_folder+csv_file)

is_raman = False
if is_raman:
  EXCITATION_WAVELENGTH = 532
  df['W'] = 1e7 / EXCITATION_WAVELENGTH - 1e7 / df['W']
  df.rename({"W": "Wavenumber"}, axis=1, inplace=True)


def hampel(vals_orig, k=5, threshold=3):
    med = np.median(vals_orig)
    std = np.std(vals_orig)
    if np.abs(vals_orig.iloc[k//2] - med) > threshold * std:
        vals_orig.iloc[k//2] = med
    return vals_orig.iloc[k//2]

def parallel_lossy_hampel(df, window_size=window_size, sigma_threshold=sigma_threshold):
  return (df.rolling(window=window_size).apply(hampel, raw=False, args=(window_size, sigma_threshold))) #.iloc[window_size - 1:, :]

def perform_parallel(column_name):
  return parallel_lossy_hampel(df[column_name])

if __name__ == "__main__":
  from multiprocessing import Pool
  import os

  print(df)

  pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
  result = pool.map(perform_parallel, df.columns)     # process data_inputs iterable with pool
  result = pd.DataFrame({
      k: v for k, v in zip(df.columns, result)
  })

  # truncate the NAN rows (resulting from hampel filter)
  result = result.iloc[window_size - 1:, :].reset_index(drop=True)

  print(f"Result:\n{result}")
  result.to_csv(f'{csv_folder}lossy_{csv_file}', index=False)
  print("--- %s seconds ---" % (time.time() - start_time))