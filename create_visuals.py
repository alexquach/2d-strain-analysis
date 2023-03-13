import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import time
start_time = time.time()

is_raman = False
with_slope = False
type_process = "centroid" # "lorentz" or "centroid

csv_path = './Hyperspectral PL/region48_1mW_filter3_3.5K_grating600_center750_exp.1s/centroid_700to800_noslope_fullscan_80by80.csv'

csv_file = csv_path.split('/')[-1]
csv_filename_without_ext = csv_file.split('.')[0]
csv_folder = "/".join(csv_path.split('/')[:-1]) + '/'

df_results = pd.read_csv(f"{csv_folder}{csv_file}")
print(df_results.shape)

reshape_dim = math.floor(math.sqrt(df_results.shape[1]))

if type_process == "lorentz":
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
    fig.suptitle('Peaks 1 and 2')

    plot_index = [1, 4] if not is_raman else [2, 5]
    plot_names = ["Peak 1 wavelength (nm)", "Peak 2 wavelength (nm)"]
    vmins = [700, 750]
    vmaxs = [735, 800]

    for i, ax in enumerate(axs):
        im = axs[i].imshow(df_results.iloc[plot_index[i], :].values.reshape(reshape_dim, reshape_dim), vmin=vmins[i], vmax=vmaxs[i])
        plt.colorbar(im, ax = axs[i])
        axs[i].set_title(f'{plot_names[i]}')

    fig.savefig(f"{csv_folder}peaks_{csv_filename_without_ext}.png")
elif type_process == "centroid":
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
    fig.suptitle(f'Centroids  {csv_filename_without_ext}')

    axs = axs.flatten()

    im = axs[1].imshow(df_results.values.reshape(reshape_dim, reshape_dim), vmin=740, vmax=800)
    plt.colorbar(im, ax = axs[1])
    axs[1].set_title(f'Centroids')

    fig.savefig(f"{csv_folder}peaks_{csv_filename_without_ext}.png")
