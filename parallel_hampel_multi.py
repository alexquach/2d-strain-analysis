import pandas as pd
import numpy as np
import time

start_time = time.time()

df_PL = pd.read_csv('x-75to75_y-95to55_50by50.csv')

def hampel(vals_orig, k=5, threshold=3):
    med = np.median(vals_orig)
    std = np.std(vals_orig)
    if np.abs(vals_orig.iloc[k//2] - med) > threshold * std:
        vals_orig.iloc[k//2] = med
    return vals_orig.iloc[k//2]

def parallel_lossy_hampel(df, window_size=11, sigma_threshold=3):
  return (df.rolling(window=window_size).apply(hampel, raw=False, args=(window_size, sigma_threshold))) #.iloc[window_size - 1:, :]

def perform_parallel(column_name):
  return parallel_lossy_hampel(df_PL[column_name])

if __name__ == "__main__":
  window_size = 11
  print(df_PL)

  # x = pd.DataFrame({k: fit_lorentzians_given_yData(test_df[k]) for k in tqdm(test_df.columns)})
  from multiprocessing import Pool
  import os

  pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
  result = pool.map(perform_parallel, df_PL.columns)  # process data_inputs iterable with pool
  result = pd.DataFrame({
      k: v for k, v in zip(df_PL.columns, result)
  })

  result = result.iloc[window_size - 1:, :]
  print(f"Result:\n{result}")
  print("--- %s seconds ---" % (time.time() - start_time))
  result.to_csv('multi_lossy_x-75to75_y-95to55_50by50.csv')