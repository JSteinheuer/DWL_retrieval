#!/usr/bin/env python3.6
"""
This code is used to calculate wind gust peaks as described in:

Steinheuer, Detring, Beyrich, Löhnert, Friederichs, and Fiedler (2022):
A new scanning scheme and flexible retrieval to derive both mean winds and
gusts from Doppler lidar measurements, Atmos. Meas. Tech
DOI: https://doi.org/10.5194/amt-15-3243-2022

AND:

Steinheuer, Vertical wind gust profiles, Dissertation, University of Cologne,
URL: https://kups.ub.uni-koeln.de/65655/

This program is a free software distributed under the terms of the GNU General
Public License as published by the Free Software Foundation, version 3
(GNU-GPLv3).

You can redistribute and/or modify by citing the mentioned publication, but
WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.

For a description of the methods, refer to Steinheuer et al. (2022).
To test the script with an example day use main_testday.py or
main_testday_qCSM.py

The 29th June 2021 in quck continuous scanning mode is too big for git, so
  please downlaad here: https://doi.org/10.25592/uhhfdm.11227, i.e.,
  the example is: https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/fesstval/wind_and_gust/falkenberg_dlidcsm/level1/csm02/2021/catalog.html?dataset=ftpthreddsscan/fesstval/wind_and_gust/falkenberg_dlidcsm/level1/csm02/2021/sups_rao_dlidCSM02_l1_any_v00_20210629.nc
"""


###############################################################################
# Julian Steinheuer; June 2023                                                #
# main_testday_qCSM.py: run DWL_retrieval for test day 20210629               #
# Update in 2023: - change variable names                                     #
#                 - w-correction                                              #
###############################################################################

import numpy as np
from os import getcwd
from DWL_retrieval import \
    files_and_folders_of_dir, \
    wrap_flag_qdv_of_l1
from DWL_retrieval import \
    uvw3_retrievals
from DWL_retrieval import \
    wind_and_gust_netcdf
from DWL_retrieval import \
    lidar_quicklook
from DWL_retrieval import \
    w_correction

max_cpu = 12  # how many kernels to use?

###############################################################################
# STEP A: Quality check                                                       #
###############################################################################

wrap_flag_qdv_of_l1(nc_file_in='sups_rao_dlidCSM02_l1_any_v00_20210629.nc',
                    path_folder_in=getcwd() + '/data/qCSM_testday/DWL_l1/',
                    path_folder_out=getcwd() +
                    '/data/qCSM_testday/DWL_l1_QC/',
                    snr_threshold=-18.2, beta_threshold=1e-4,
                    check_exist=True, weight_starting_azimuth=0.6,
                    azimuth_shift=True)

###############################################################################
# STEP B: Retrieval                                                           #
###############################################################################
# STEP B (600s): For all level 1 files in all folders that are in dir_l1_qc   #
#                and its subfolders, if the files are of level 1, do!         #
###############################################################################

uvw3_retrievals(nc_file_in='sups_rao_dlidCSM02_l1_any_v01_20210629.nc',
                path_folder_in=getcwd() + '/data/qCSM_testday/DWL_l1_QC/',
                path_folder_out=getcwd() +
                '/data/qCSM_testday/DWL_l2/wind-600s/',
                duration=600, circ=False, heights_fix=np.array([90.3]),
                quality_control_snr=False, check_exist=True,
                lowest_frac=0.5, highest_allowed_sigma=3,
                iteration_stopping_sigma=1, n_ef=12)

###############################################################################
# STEP B (circ): For all level 1 files in all folders that are in dir_l1_qc   #
#                and its subfolders, if the files are of  level 1, do!        #
###############################################################################

uvw3_retrievals(nc_file_in='sups_rao_dlidCSM02_l1_any_v01_20210629.nc',
                path_folder_in=getcwd() + '/data/qCSM_testday/DWL_l1_QC/',
                path_folder_out=getcwd() +
                '/data/qCSM_testday/DWL_l2/wind-circ/',
                duration=np.nan, circ=True, heights_fix=np.array([90.3]),
                quality_control_snr=False, check_exist=True,
                lowest_frac=0.66, highest_allowed_sigma=1,
                iteration_stopping_sigma=1, n_ef=2)

###############################################################################
# STEP C: Wind product                                                        #
###############################################################################

wind_and_gust_netcdf(nc_file_in_mean='sups_rao_dlidCSM02_l2_wind'
                                     '-600s_v01_20210629.nc',
                     path_folder_in_mean=getcwd() +
                     '/data/qCSM_testday/DWL_l2/wind-600s/',
                     nc_file_in_circ='sups_rao_dlidCSM02_l2_wind'
                                     '-circ_v01_20210629.nc',
                     path_folder_in_circ=getcwd() +
                     '/data/qCSM_testday/DWL_l2/wind-circ/',
                     path_folder_out=getcwd() +
                     '/data/qCSM_testday/DWL_l2/wind-gust/',
                     circulations_fraction=0.5,
                     check_exist=True, max_out=1)

###############################################################################
# STEP D: Quicklooks                                                          #
###############################################################################

dir_l2 = getcwd() + '/data/qCSM_testday/DWL_l2/'
f_and_f = files_and_folders_of_dir(directory=dir_l2)
path_folders_in = f_and_f['folders']
nc_files_in = f_and_f['files']
path_folders_out = [f.replace('DWL_l2', 'quicklooks') for f in path_folders_in]
for nc_file_in, path_folder_in, path_folder_out in zip(nc_files_in,
                                                       path_folders_in,
                                                       path_folders_out):
    if nc_file_in.split('_')[-3][0:9] == 'wind-gust':
        lidar_quicklook(nc_file_in, path_folder_in,path_folder_out, gust=True)
    else:
        lidar_quicklook(nc_file_in, path_folder_in, path_folder_out)

###############################################################################
# STEP E: W-CORRECTIN                                                         #
###############################################################################

dir_l2 = getcwd() + '/data/qCSM_testday/DWL_l2/'
f_and_f = files_and_folders_of_dir(directory=dir_l2)
path_folders_in = f_and_f['folders']
nc_files_in = f_and_f['files']
path_folders_out = [f.replace('DWL_l2', 'DWL_l2_WC') for f in path_folders_in]
for nc_file_in, path_folder_in, path_folder_out in zip(nc_files_in,
                                                       path_folders_in,
                                                       path_folders_out):
    w_correction(nc_file_in, path_folder_in, path_folder_out)
