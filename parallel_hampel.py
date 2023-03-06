import pandas as pd
import numpy as np
# from pandarallel import pandarallel
import dask.dataframe as dd
import math

# pandarallel.initialize(progress_bar = True)

# df_PL = pd.read_csv('x-300to0_y-150to150_80by80.csv')
# df_PL = dd.read_csv('x-300to0_y-150to150_80by80.csv')
df_PL = dd.read_csv('x-75to75_y-95to55_50by50.csv')

def hampel(vals_orig, k=5, threshold=3):
    med = np.median(vals_orig)
    std = np.std(vals_orig)
    if np.abs(vals_orig.iloc[k//2] - med) > threshold * std:
        vals_orig.iloc[k//2] = med
    return vals_orig.iloc[k//2]

def parallel_lossy_hampel(df, window_size, sigma_threshold):
  # return (df.groupby(df.columns, axis=1).rolling(window=window_size, axis=1).apply(hampel, raw=False, args=(window_size, sigma_threshold))).iloc[window_size - 1:, :]
  return (df.rolling(window=window_size).apply(hampel, raw=False, args=(window_size, sigma_threshold))) #.iloc[window_size - 1:, :]

# print(df_PL.groupby('W').rolling(window=11).apply(lambda x: print(x)))

# df_PL.index = df_PL.iloc[:, 0].values
# df_PL = df_PL.drop("W", axis=1)
# print(df_PL)

import time
start_time = time.time()
df_filtered_PL = parallel_lossy_hampel(df_PL, 11, 3)
df_filtered_PL.mean().compute()
print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
df_filtered_PL.to_csv('dask_lossy_x-75to75_y-95to55_50by50.csv', single_file=True)
df_filtered_PL.mean().compute()
print("--- %s seconds ---" % (time.time() - start_time))

df_filtered_PL.mean().compute()

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