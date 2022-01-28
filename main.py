#!/usr/bin/env python3.6
"""
This code is used to calculate wind gust peaks as described in:

Steinheuer, Detring, Beyrich, Löhnert, Friederichs, and Fiedler
(2021): A new scanning scheme and flexible retrieval to derive both
mean winds and gusts from Doppler lidar measurements,
Atmos. Meas. Tech
DOI:

This program is a free software distributed under the terms of the GNU
General Public License as published by the Free Software Foundation,
version 3 (GNU-GPLv3).

You can redistribute and/or modify by citing the mentioned publication,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

For a description of the methods, refer to Steinheuer et al. (2021).
To test the script with an example day use main_testday.py
"""

########################################################################
# Julian Steinheuer; November 2021                                     #
# main.py: run DWL_retrieval for paper                                 #
########################################################################

import numpy as np
from os import getcwd
import multiprocessing as mp
from DWL_retrieval import \
    files_and_folders_of_dir, \
    wrap_flag_qdv_of_l1
from DWL_retrieval import \
    uvw3_retrievals
from DWL_retrieval import \
    wind_and_gust_netcdf
from DWL_retrieval import \
    lidar_quicklook

max_cpu = 12  # how many kernels to use?

########################################################################
# STEP A: Quality check                                                #
########################################################################

dir_l1 = getcwd() + '/data/'
f_and_f = files_and_folders_of_dir(directory=dir_l1)
path_folders_in = f_and_f['folders']
nc_files_in = f_and_f['files']
for i in range(len(nc_files_in) - 1, -1, -1):
    if '_l1_' not in nc_files_in[i]:  # wrong level
        del path_folders_in[i]
        del nc_files_in[i]
    elif 'level_1_js' in path_folders_in[i]:  # already quality checked
        del path_folders_in[i]
        del nc_files_in[i]

path_folders_out = [f.replace(
    'level_1_mk', 'level_1_js').replace(
    'level_1_rl', 'level_1_js') for f in path_folders_in]
pool = mp.Pool(np.min((mp.cpu_count() - 1, max_cpu)))  # parallel on
pool.starmap_async(wrap_flag_qdv_of_l1,
                   [(nc_file_in, path_folder_in, path_folder_out,
                     -18.2, 1e-4, True, 0.6)
                    for nc_file_in, path_folder_in, path_folder_out, in
                    zip(nc_files_in, path_folders_in, path_folders_out)]).get()
pool.close()  # parallel off

########################################################################
# STEP B: Retrieval                                                    #
########################################################################

heights_fix = np.array([90.3])  # sonic anemometer height to interpolate
quality_control_snr = False  # not necessary with this retrieval (but possible)
check_exist = True
iteration_stopping_sigma = 1
dir_l1_qc = getcwd() + '/data/'
f_and_f = files_and_folders_of_dir(directory=dir_l1_qc)
path_folders_in = f_and_f['folders']
nc_files_in = f_and_f['files']
for i in range(len(nc_files_in) - 1, -1, -1):
    if '_l1_' not in nc_files_in[i]:  # wrong level
        del path_folders_in[i]
        del nc_files_in[i]
    elif '/level_1_rl/' in path_folders_in[i]:  # not quality checked
        del path_folders_in[i]
        del nc_files_in[i]
    elif '/level_1_mk/' in path_folders_in[i]:  # not quality checked
        del path_folders_in[i]
        del nc_files_in[i]

########################################################################
# STEP B (600s): For all level 1 files in all folders that are in      #
#                dir_l1_qc and its subfolders, if the files are of     #
#                level 1, do!                                          #
########################################################################

duration = 600
circ = False
lowest_frac = 0.5
highest_allowed_sigma = 3
n_ef = 12
path_folders_out = [f.replace('/level_1_js/', '/level_2_js/uvw-') +
                    str(duration) + 's/' for f in path_folders_in]
pool = mp.Pool(np.min((mp.cpu_count() - 1, max_cpu)))  # parallel on
pool.starmap_async(uvw3_retrievals,
                   [(nc_file_in, path_folder_in, path_folder_out, duration,
                     circ, heights_fix, quality_control_snr, check_exist,
                     lowest_frac, highest_allowed_sigma,
                     iteration_stopping_sigma, n_ef)
                    for nc_file_in, path_folder_in, path_folder_out, in
                    zip(nc_files_in, path_folders_in, path_folders_out)]).get()
pool.close()  # parallel off

########################################################################
# STEP B (circ): For all level 1 files in all folders that are in      #
#                dir_l1_qc and its subfolders, if the files are of     #
#                level 1, do!                                          #
########################################################################

duration = np.nan
circ = True
lowest_frac = 0.66
highest_allowed_sigma = 1
n_ef = 2
path_folders_out = [f.replace('/level_1_js/', '/level_2_js/uvw-circ/')
                    for f in path_folders_in]
pool = mp.Pool(np.min((mp.cpu_count() - 1, max_cpu)))  # parallel on
pool.starmap_async(uvw3_retrievals,
                   [(nc_file_in, path_folder_in, path_folder_out, duration,
                     circ, heights_fix, quality_control_snr, check_exist,
                     lowest_frac, highest_allowed_sigma,
                     iteration_stopping_sigma, n_ef)
                    for nc_file_in, path_folder_in, path_folder_out, in
                    zip(nc_files_in, path_folders_in, path_folders_out)]).get()
pool.close()  # parallel off

########################################################################
# STEP B (classic): For Sabine level 1 files process data without new  #
#                   retrieval (so with quality check and no sigma      #
#                   thresholds).                                       #
########################################################################

heights_fix = np.array([90.3])  # sonic anemometer height to interpolate
quality_control_snr = True  # classic
check_exist = True
iteration_stopping_sigma = 999  # dummy
dir_l1_qc = getcwd() + '/data/fesstval_2020/wl_177/CSM2/level_1_js/'
f_and_f = files_and_folders_of_dir(directory=dir_l1_qc)
path_folders_in = f_and_f['folders']
nc_files_in = f_and_f['files']
for i in range(len(nc_files_in) - 1, -1, -1):
    if '_l1_' not in nc_files_in[i]:  # wrong level
        del path_folders_in[i]
        del nc_files_in[i]
    elif '202002' not in nc_files_in[i]:  # only Sabine
        del path_folders_in[i]
        del nc_files_in[i]

duration = 600
circ = False
lowest_frac = 0.5
highest_allowed_sigma = 999
n_ef = 12
path_folders_out = [f.replace('/level_1_js/', '/level_2_js_SNR/uvw-') +
                    str(duration) + 's/' for f in path_folders_in]
pool = mp.Pool(np.min((mp.cpu_count() - 1, max_cpu)))  # parallel on
pool.starmap_async(uvw3_retrievals,
                   [(nc_file_in, path_folder_in, path_folder_out, duration,
                     circ, heights_fix, quality_control_snr, check_exist,
                     lowest_frac, highest_allowed_sigma,
                     iteration_stopping_sigma, n_ef)
                    for nc_file_in, path_folder_in, path_folder_out, in
                    zip(nc_files_in, path_folders_in, path_folders_out)]).get()
pool.close()  # parallel off

duration = np.nan
circ = True
lowest_frac = 0.66
highest_allowed_sigma = 999
n_ef = 2
path_folders_out = [f.replace('/level_1_js/', '/level_2_js_SNR/uvw-circ/')
                    for f in path_folders_in]
pool = mp.Pool(np.min((mp.cpu_count() - 1, max_cpu)))  # parallel on
pool.starmap_async(uvw3_retrievals,
                   [(nc_file_in, path_folder_in, path_folder_out, duration,
                     circ, heights_fix, quality_control_snr, check_exist,
                     lowest_frac, highest_allowed_sigma,
                     iteration_stopping_sigma, n_ef)
                    for nc_file_in, path_folder_in, path_folder_out, in
                    zip(nc_files_in, path_folders_in, path_folders_out)]).get()
pool.close()  # parallel off

########################################################################
# STEP C: Wind product                                                 #
########################################################################

dir_l2 = getcwd() + '/data/'
f_and_f = files_and_folders_of_dir(directory=dir_l2)
path_folders_in_mean = f_and_f['folders']
nc_files_in_mean = f_and_f['files']
for i in range(len(nc_files_in_mean) - 1, -1, -1):
    if '_l2_' not in nc_files_in_mean[i]:  # wrong level
        del path_folders_in_mean[i]
        del nc_files_in_mean[i]
    elif 'uvw-600s' not in nc_files_in_mean[i]:
        del path_folders_in_mean[i]
        del nc_files_in_mean[i]

nc_files_in_circ = [f.replace('600s', 'circ') for f in nc_files_in_mean]
path_folders_in_circ = [f.replace('600s', 'circ') for
                        f in path_folders_in_mean]
path_folders_out = [f.replace('uvw-600s', 'uvw-gust') for
                    f in path_folders_in_mean]
pool = mp.Pool(np.min((mp.cpu_count(), max_cpu)))  # parallel on
pool.starmap_async(wind_and_gust_netcdf,
                   [(nc_file_in_mean, path_folder_in_mean,
                     nc_file_in_circ, path_folder_in_circ,
                     path_folder_out, 0.5, True, 1)
                    for nc_file_in_mean, path_folder_in_mean, nc_file_in_circ,
                    path_folder_in_circ, path_folder_out in
                    zip(nc_files_in_mean, path_folders_in_mean,
                        nc_files_in_circ, path_folders_in_circ,
                        path_folders_out)]).get()
pool.close()  # parallel off

########################################################################
# STEP D: Quicklooks                                                   #
########################################################################

dir_l2 = getcwd() + '/data/'
f_and_f = files_and_folders_of_dir(directory=dir_l2)
path_folders_in = f_and_f['folders']
nc_files_in = f_and_f['files']
for i in range(len(nc_files_in) - 1, -1, -1):
    if '_l2_' not in nc_files_in[i]:  # wrong level
        del path_folders_in[i]
        del nc_files_in[i]
    elif '_mett_' in nc_files_in[i]:  # sonic anemometer
        del path_folders_in[i]
        del nc_files_in[i]

path_folders_out = [f.replace('level_2', 'quicklooks')
                    for f in path_folders_in]
for nc_file_in, path_folder_in, path_folder_out in zip(nc_files_in,
                                                       path_folders_in,
                                                       path_folders_out):
    if nc_file_in.split('_')[-3][0:8] == 'uvw-gust':
        lidar_quicklook(nc_file_in, path_folder_in,
                        path_folder_out.replace('uvw-gust', 'uvw-gust-peak'),
                        gust=True, name_prefix='quicklook_gust_peak_')
        # lidar_quicklook(nc_file_in, path_folder_in,
        #                 path_folder_out.replace('uvw-gust','uvw-gust-peak'),
        #                 gust=False, name_prefix='quicklook_mean-wind_')
    else:
        lidar_quicklook(nc_file_in, path_folder_in, path_folder_out)
