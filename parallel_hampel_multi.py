import pandas as pd
import numpy as np
# from pandarallel import pandarallel
import dask.dataframe as dd
import math
import time

start_time = time.time()


# df_PL = pd.read_csv('x-300to0_y-150to150_80by80.csv')
# df_PL = dd.read_csv('x-300to0_y-150to150_80by80.csv')
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
  # df_PL = df_PL.drop("W", axis=1)
  # df_PL = df_PL.iloc[:, :20]
  print(df_PL)

  from tqdm.auto import tqdm
  # x = pd.DataFrame({k: fit_lorentzians_given_yData(test_df[k]) for k in tqdm(test_df.columns)})
  from multiprocessing import Pool
  import os

  pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
  result = pool.map(perform_parallel, df_PL.columns)  # process data_inputs iterable with pool
  result = pd.DataFrame({
      k: v for k, v in zip(df_PL.columns, result)
  })

  print(f"Result:\n{result}")
  result = result.iloc[window_size - 1:, :]
  print(f"Result:\n{result}")
  print("--- %s seconds ---" % (time.time() - start_time))
  result.to_csv('multi_lossy_x-75to75_y-95to55_50by50.csv')

# print(df_filtered_PL)

# df_size = int(1e6)
# df = pd.DataFrame(dict(a=np.random.randint(1, 300, df_size),
#                        b=np.random.rand(df_size)))
# def func(x):
#     return x.iloc[0] + x.iloc[1] ** 2 + x.iloc[2] ** 3 + x.iloc[3] ** 4
# print(1)
# res = df.groupby('a').b.rolling(4).apply(func, raw=False)
# print(2)
# res_parallel = df.groupby('a').b.rolling(4).parallel_apply(func, raw=False)
# res.equals(res_parallel)