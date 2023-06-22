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
"""


###############################################################################
# Julian Steinheuer; November 2021                                            #
# DWL_retrieval.py                                                            #
# Update in 2023: - change variable names                                     #
#                 - w-correction                                              #
###############################################################################

import numpy as np
import netCDF4 as nc
import math as m
import time
import datetime
from os.path import isfile, join, isdir
from os import system, makedirs, listdir
import warnings as wn
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt


###############################################################################
# STEP A: Quality check                                                       #
#                                                                             #
# Process lidar-netcdf's l1. The qdv value is set on 0 if SNR (given as       #
# intensity = 1 + SNR) is not above certain (e.g SNR>-18) threshold.          #
# Further a beta threshold could be set for cloud detection.                  #
# For quick scanning modi the azimuth tends to shift which can be repaired.   #
###############################################################################


def files_and_folders_of_dir(directory, not_nc=False, tmp=False, error=False):
    """ Return all files and corresponding full folder-paths in directory dir

    Args:
      directory: Directory where all files and folders are.
      not_nc: If False only .nc files are returned.
      tmp: If False only files are returned that not starts with 'tmp'.
      error: If False only files are returned that not starts with 'ERROR'.
    Returns:
      folders: Complete path to folder where same index file is located.
      files: Files in corresponding index folders.
    """

    if type(directory) == str:
        dir_to_look = [directory]
    elif not type(directory) == list:
        return
    else:
        dir_to_look = directory

    folders = []
    files = []
    for f in dir_to_look:
        f_or_f = listdir(f)
        for ff in f_or_f:
            if isdir(f + ff):
                dir_to_look.append(f + ff + '/')
            else:
                folders.append(f)
                files.append(ff)

    if not not_nc:
        for i in range(len(files) - 1, -1, -1):
            if not files[i][-3:] == '.nc':
                del folders[i]
                del files[i]

    if not tmp:
        for i in range(len(files) - 1, -1, -1):
            if files[i][:3] == 'tmp':
                del folders[i]
                del files[i]

    if not error:
        for i in range(len(files) - 1, -1, -1):
            if files[i][:5] == 'ERROR':
                del folders[i]
                del files[i]

    return {'folders': folders, 'files': files}


def nc_name_check_level(nc_file_in):
    """ Return level of nc_file_in

      Args:
        nc_file_in: file name of nc-file in SAMD 1.2 convention
                    (doi.org/10.3390/ijgi5070124).
      Returns:
        level: integer between 0 and 4.
      """

    level = None
    counter = 0
    name_control = nc_file_in.split('_')
    for char in name_control:
        if (len(char) == 2) & (char[0] == 'l'):
            level = int(char[1])
            counter = counter + 1

    if name_control[0] == 'tmp':
        return None

    if counter == 1:
        return level
    else:
        return None


def nc_name_raise_version(nc_file_in):
    """ Return name of nc_file_in with raised version number

      Args:
        nc_file_in: file name of nc-file in SAMD 1.2 convention
                    (doi.org/10.3390/ijgi5070124).
      Returns:
        nc_file_out: file name of nc-file in SAMD 1.2  with raised version nr.
      """

    name_control = nc_file_in.split('_')
    if ((name_control[len(name_control) - 2][0]) != 'v') or \
            (len(name_control[len(name_control) - 2]) != 3):
        name_control.insert(len(name_control) - 1, "v00")

    version = str(int(name_control[len(name_control) - 2][1:3]) + 1)
    if len(version) == 1:
        name_control[len(name_control) - 2] = 'v0' + version
    else:
        name_control[len(name_control) - 2] = 'v' + version

    nc_file_out = '_'.join(name_control)
    return nc_file_out


def mean_azimuth(a, b, weight_a=0.6):
    """ Return weighted mean azimuth in the middle of a and b

      Args:
        a: starting azimuth.
        b: ending azimuth.
        weight_a: weight of a; so (1-weight_a) is the weight of b. The value
                  0.6 was empirically chosen and is still questionable. See:
                  Steinheuer (2023), Vertical wind gust profiles,
                  Dissertation, University of Cologne.
      Returns:
        w_mean: weighted mean.
      """

    if abs(a - b) > 180:
        if a < b:
            return ((a + 360) * weight_a + b * (1 - weight_a)) % 360
        else:
            return ((b + 360) * (1 - weight_a) + a * weight_a) % 360
    else:
        return a * weight_a + b * (1 - weight_a)


def flag_qdv_of_l1(nc_file_in, path_folder_in='', path_folder_out=None,
                   snr_threshold=-18.2, beta_threshold=1e-4,
                   check_exist=True, azimuth_shift=False,
                   weight_starting_azimuth=0.6):
    """ Flag those lidar data (daily nc-file) where SNR is below threshold
        and/or which are in or above clouds (beta>10^-4)

      Args:
        nc_file_in: file name of nc-file in SAMD 1.2 convention
                    (doi.org/10.3390/ijgi5070124). The file needs to contain
                    the variable 'intensity' (=SNR+1) and 'qdv' (flagged (=0)
                    for SNR values below threshold).
        path_folder_in: path to folder containing nc-file.
        path_folder_out: folder path for output nc-file which has +1 version as
                         input.
        snr_threshold: SNR value where values below or equal are flagged.
        beta_threshold: beta value where values higher or equal are flagged
                       (and those above a certain height of detection).
        check_exist: If True then approach is canceled when output file is
                     already existing.
        azimuth_shift: if True the azimuth is corrected towards th weighted
                       mean to the next azimuth.
        weight_starting_azimuth: if azimuth_shift=True, this is the weight to
                                 shift the starting azimuth, and
                                 (1-weight_starting_azimuth) is the weight of
                                 the ending azimuth.
      Returns:
      """

    if path_folder_out is None:
        path_folder_out = path_folder_in

    if nc_name_check_level(nc_file_in) != 1:
        wn.warn(nc_file_in + ' is not of level 1. No quality control is done')
        return

    if not isdir(path_folder_out):
        try:
            makedirs(path_folder_out)
        except:
            wn.warn(path_folder_out + ' simultaniously created!')

    nc_file_out = nc_name_raise_version(nc_file_in)
    exist = isfile(join(path_folder_out + nc_file_out))
    if exist and check_exist:
        print('Consider: ' + nc_file_out +
              ' is existing and not again calculated. Please set '
              '>check_exist< to FALSE, if recalculation is needed')
        return

    system('cp ' + path_folder_in + nc_file_in + ' ' + path_folder_out +
           'tmp_' + nc_file_out)
    try:
        nc_lidar = nc.Dataset(path_folder_out + 'tmp_' + nc_file_out, 'r+')
        if 'qdv' not in nc_lidar.variables.keys():
            qdv_nc = nc_lidar.createVariable('qdv', np.float32,
                                             ('time', 'range'),
                                             fill_value=1)
            qdv_nc.long_name = 'quality mask of Doppler velocity'

        if 'time_bnds' not in nc_lidar.variables.keys():
            time_bnds_nc = nc_lidar.createVariable('time_bnds', np.float64,
                                                   ('time', 'nv'),
                                                   fill_value=-999.0)
            time_bnds_nc.units = 'seconds since 1970-01 00:00:00'
            time_bnds_nc[:, 0] = nc_lidar['time'][:]
            time_bnds_nc[:, 1] = nc_lidar['time'][:]

        if 'intensity' not in nc_lidar.variables.keys():
            wn.warn('No intensity in lidar.variables')
            print('Input error: The input netcdf-file ' + nc_file_in +
                  ' has no intensity in lidar variables',
                  file=open(path_folder_out + 'ERROR_' +
                            nc_file_out[0:-3] + '.txt', 'w+'))
            system('rm ' + path_folder_out + 'tmp_' + nc_file_out)
            return

        if 'beta' not in nc_lidar.variables.keys():
            wn.warn('No beta in lidar.variables')
            print('Input error: The input netcdf-file ' + nc_file_in +
                  ' has no beta in lidar variables',
                  file=open(path_folder_out + 'ERROR_' +
                            nc_file_out[0:-3] + '.txt', 'w+'))
            system('rm ' + path_folder_out + 'tmp_' + nc_file_out)
            return

        qdv = nc_lidar.variables['qdv'][:]
        i_bad = np.where(nc_lidar.variables['intensity'][:] <= 1 +
                         10 ** (snr_threshold / 10))
        qdv[i_bad] = 0
        i_bad = np.where(
            nc_lidar.variables['beta'][:] >= beta_threshold)
        for i in np.unique(i_bad[0]):
            cols = np.arange(i_bad[1][np.where(i_bad[0] == i)].min(),
                             qdv.shape[1])
            rows = np.repeat(i, cols.size)
            qdv[rows, cols] = 0

        nc_lidar.variables['qdv'][:] = qdv
        azimuth_str = ''
        if azimuth_shift:
            azimuth_str = 'all azimuth values [a] are shifted towards ' + \
                          'following azimuth [b], by a_new=a*' + \
                          str(weight_starting_azimuth) + \
                          ' + b*(1-' + str(weight_starting_azimuth) + ');'

        # global attributes to overwrite:
        nc_lidar.History = 'qdv is threshold based quality flagged: values ' \
                           'are updated on 0 if SNR is below ' + \
                           str(snr_threshold) + 'and beta higher than ' + \
                           '1e-4 m^-1 sr^-1 (or behind such value), ' + \
                           azimuth_str + 'version information: ' + nc_file_in
        nc_lidar.Processing_date = time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.gmtime()) + '(UTC)'
        nc_lidar.Author = 'Julian Steinheuer, Julian.Steinheuer@uni-koeln.de'
        t_lidar = nc_lidar['time'][:].data
        # sometimes time jumps appear for some reasons in l1 data :(
        if not np.unique(np.sort(t_lidar) == t_lidar)[0]:
            max_time_shift = max(abs(np.sort(t_lidar) - t_lidar))
            counter = 0
            while max_time_shift >= 5 and counter < 100:
                a = abs(np.sort(t_lidar) - t_lidar)
                i = np.where(a != 0)[0][0]
                if i > 1:
                    ti_0 = t_lidar[i]
                    ti_p1 = t_lidar[i + 1]
                    ti_m1 = t_lidar[i - 1]
                    ti_m2 = t_lidar[i - 2]
                    if ti_m1 > ti_p1:
                        t_lidar[i] = (ti_m2 + ti_0) / 2
                    else:
                        t_lidar[i] = (ti_m1 + ti_p1) / 2

                max_time_shift = max(abs(np.sort(t_lidar) - t_lidar))
                counter = counter + 1

            if counter == 100:
                system('rm ' + path_folder_out + 'tmp_' + nc_file_out)
                print('Too many time jumps in level1 file!',
                      file=open(path_folder_out + 'ERROR_' +
                                nc_file_out[0:-3] + '.txt', 'w+'))
                wn.warn('Error: Too many time jumps in level1 file!')
                return

            nc_lidar['time'][:] = t_lidar
            nc_lidar['time_bnds'][:, 0] = t_lidar
            nc_lidar['time_bnds'][:, 1] = t_lidar
            if max_time_shift < 5 and \
                    np.unique(np.sort(t_lidar) == t_lidar)[0] != True:
                t_lidar_unsorted_i = np.where(np.sort(t_lidar) != t_lidar)[0]
                sorting_i = np.argsort(t_lidar[t_lidar_unsorted_i])
                nc_lidar['time'][t_lidar_unsorted_i] = \
                    nc_lidar['time'][t_lidar_unsorted_i[sorting_i]]
                nc_lidar['time_bnds'][t_lidar_unsorted_i, :] = \
                    nc_lidar['time_bnds'][t_lidar_unsorted_i[sorting_i], :]

        if azimuth_shift:
            azimuth = nc_lidar['azi'][:].data
            azimuth_folow = np.concatenate((azimuth[1:], [azimuth[-1]]))
            for i in range(len(azimuth)):
                nc_lidar['azi'][i] = mean_azimuth(azimuth[i], azimuth_folow[i],
                                                  weight_starting_azimuth)

        nc_lidar.close()
        system('mv ' + path_folder_out + 'tmp_' + nc_file_out + ' ' +
               path_folder_out + nc_file_out)

    except Exception as e:
        print(e)
        system('rm ' + path_folder_out + 'tmp_' + nc_file_out)


def wrap_flag_qdv_of_l1(nc_file_in, path_folder_in, path_folder_out,
                        snr_threshold=-18.2, beta_threshold=1e-4,
                        check_exist=True, weight_starting_azimuth=0.6,
                        azimuth_shift=np.nan):
    """ Wrap the function 'flag_qdv_of_l1' in order to parallelize

      Args:
        nc_file_in: file name of nc-file in SAMD 1.2 convention
                    (doi.org/10.3390/ijgi5070124). The file needs to contain
                    the variable 'intensity' (=SNR+1) and 'qdv' (flagged (=0)
                    for SNR values below threshold).
        path_folder_in: path to folder containing nc-file.
        path_folder_out: folder path for output nc-file which has +1 version as
                         input.
        snr_threshold: SNR value where values below or equal are flagged.
        beta_threshold: Beta value where values higher or equal are flagged
                        (and those above a certain height of detection). This
                        follows Marke et al. (doi.org/10.1175/JAMC-D-17-0341.1)
        check_exist: If True then approach is canceled when output file is
                     already existing.
        weight_starting_azimuth: if azimuth_shift is True, this is the weight
                                 to shift the starting azimuth, and
                                 (1-weight_starting_azimuth) is the weight of
                                 the ending azimuth. The value 0.6 was
                                 empirically chosen and is still questionable.
        azimuth_shift: True or False, to force to do or not do the correction.
                       If default np.nan, then the folder path should include
                       somehow csm if correction is wanted.
      Returns:
      """

    print(path_folder_in + nc_file_in)
    if np.isnan(azimuth_shift):
        if '/CSM1/' in path_folder_in or \
                '/CSM2/' in path_folder_in or \
                '/CSM3/' in path_folder_in or \
                'csm' in path_folder_in or \
                'CSM' in path_folder_in or \
                '/wind_and_gust/' in path_folder_in:
            azimuth_shift = True
        else:
            azimuth_shift = False

    print(nc_file_in + ', azimuth shift: ' + str(azimuth_shift))
    flag_qdv_of_l1(nc_file_in=nc_file_in,
                   path_folder_in=path_folder_in,
                   path_folder_out=path_folder_out,
                   snr_threshold=snr_threshold,
                   beta_threshold=beta_threshold,
                   check_exist=check_exist,
                   azimuth_shift=azimuth_shift,
                   weight_starting_azimuth=weight_starting_azimuth)


###############################################################################
# STEP B: Retrieval                                                           #
#                                                                             #
# Process lidar-netcdf's l1 towards wind vectors which are derived from       #
# consecutive beams within a duration window at given heights. The maximal    #
# duration gives the time dimension, since one day is divided into pieces of  #
# the given duration. Further one retrieval per circulation is calculable to  #
# obtain later gust peaks.                                                    #
###############################################################################


def uvw0_retrieval(dv, zenith, azimuth, lowest_frac=0.5,
                   highest_allowed_sigma=3, iteration_stopping_sigma=1,
                   n_ef=12, nqv=20):
    """Retrieve wind components from lidar measurements with quality control.
    Most inner retrieval function

    Args:
      dv: Vector of Doppler velocities (m/s) relative to beam line.
      zenith: Vector of zenith angels (deg.).
      azimuth: Vector of azimuth angels (deg.).
      lowest_frac: Lowest allowed fraction of useful beams to input number of
                   beams.
      highest_allowed_sigma: Assume wind field is isotropic gaussian; highest
                             allowed sigma to retrieve than v.
      iteration_stopping_sigma: Assume wind field is isotropic gaussian;
                               ending criterion for iteration that improve v.
      n_ef: Effective degrees of freedom to scale the covariance matrix.
      nqv: Nyquist velocity.
    Raises:
      ValueError: The vector lengths of zenith, azimuth, error and dv are
                  different.
    Returns:
      uvw: The three components of retrieved wind vector.
      covariance_matrix: The error covariance matrix for uvw (3 x 3).
      nvrad: Number of used radial beams.
      r2: R^2 value as in (E. Paeschke et al. 2015)
      cn: Condition number of A*diag((A^T*A)^(-1/2)).
    """

    if not ((dv.size == zenith.size) & (dv.size == azimuth.size)):
        raise ValueError('The vector lengths of zenith(%s), azimuth(%s), '
                         'and dv(%s) are different. '
                         % (zenith.size, azimuth.size, dv.size))

    mask = np.where(np.where(dv == -999.0, False, True) *
                    ~np.isnan(dv), True, False)
    n = dv.size
    lowest_frac_threshold = max(m.ceil(n * lowest_frac), 3)
    iteration = 0
    sigma_fit = iteration_stopping_sigma + highest_allowed_sigma + 1
    while sigma_fit > iteration_stopping_sigma:
        # jump out of loop?
        a_tilted = np.sort((azimuth[mask])[np.where(zenith[mask] > 3.0)])
        if a_tilted.size < 3 or \
                dv[mask].size < lowest_frac_threshold:  # jump out directly
            if sigma_fit < highest_allowed_sigma:
                break
            else:
                return {'uvw': np.full([3], np.nan),
                        'covariance_matrix': np.full([3, 3], np.nan),
                        'nvrad': 0, 'r2': np.nan, 'cn': np.nan}

        # new iteration round:
        dv = dv[mask]
        zenith = zenith[mask]
        azimuth = azimuth[mask]
        dv_used_frac = dv.size / n
        ze = 2 * np.pi * np.array(zenith) / 360
        az = 2 * np.pi * np.array(azimuth) / 360
        A = np.transpose([np.sin(ze[:]) * np.sin(az[:]),
                          np.sin(ze[:]) * np.cos(az[:]), np.cos(ze[:])])

        iteration += 1
        A_ginv = np.linalg.pinv(A, rcond=1e-15)  # Moore-Penrose inverse
        uvw = A_ginv.dot(dv)
        dv_back = A.dot(uvw)
        if dv.size == 3:  # 3Beam one circulation
            return {'uvw': uvw,
                    'covariance_matrix': np.full([3, 3], np.nan),
                    'nvrad': 3, 'r2': np.nan, 'cn': np.nan}

        sigma_fit = np.sqrt(1 / (dv.size - 3) *
                            sum((dv - dv_back) * (dv - dv_back))
                            )  # -3 because 3 degrees of freedom for u,v,w
        # which data to neglect for next round?
        if sigma_fit > iteration_stopping_sigma:
            neglect_frac = 0.05  # max 5%
            mask = np.repeat(True, dv.size)
            frac = max(min(int(n * neglect_frac),
                           dv.size - int(lowest_frac_threshold) - 1
                           ), 1)
            wdist = (dv - A.dot(uvw)) ** 2
            mask[np.argsort(-wdist)[0:frac]] = False

    cn = np.linalg.cond(A.dot(
        np.diag(1 / np.sqrt(A.transpose().dot(A)[[0, 1, 2], [0, 1, 2]]))))
    dv_m = np.mean(dv)
    e = (dv - dv_back)
    r2 = 1 - np.sum(e ** 2) / np.sum((dv - dv_m) ** 2)
    nvrad = dv.size
    if dv_used_frac < 1:
        p = (1 - dv_used_frac) / 2
        sigma_approx = sigma_fit / np.sqrt(
            1 + (2 * norm.ppf(p / 2) * norm.pdf(norm.ppf(p / 2))) / (1 - p))
    else:
        sigma_approx = sigma_fit

    highest_detectable_vh = np.sqrt(2) * nqv / np.sin(np.nanmean(ze))
    if np.sqrt(
            uvw[0] ** 2 + uvw[1] ** 2) + sigma_approx > highest_detectable_vh:
        return {'uvw': np.full([3], np.nan),
                'covariance_matrix': np.full([3, 3], np.nan),
                'nvrad': 0, 'r2': np.nan, 'cn': np.nan}

    covariance_matrix = (dv.size - 3) / n_ef * np.linalg.inv(
        np.transpose(A).dot(A)) * sigma_approx ** 2
    return {'uvw': uvw, 'covariance_matrix': covariance_matrix,
            'nvrad': nvrad, 'r2': r2, 'cn': cn}


def lin_inter_pol_dv(DV, zenith, m_range, heights):
    """ Retrieve interpolated doppler velocities from one beam on heights of
       vertical stare

    Args:
      zenith: A vector of zenith angels (dim t).
      DV: Doppler velocities relative to beam lines in each center of range.
          (dim t x r).
      m_range: Distance from sensor to center of range gate along the line of
               sight, and therefore the representative heights above lidar
               (dim r).
      heights: Heights for retrieval (linear interpolation of tilted beams).
    Raises:
      ValueError: The vector lengths of zenith, m_range, and dv are
      different.
    Returns:
      DV_ip: Interpolated Doppler velocities and belonging errors
    """

    if not (DV.shape[1] == len(m_range)):
        raise ValueError(
            'The vector lengths of m_range (%s) and the columns of'
            ' DV(%s) are different' %
            (len(m_range), DV.shape[1]))

    if not (DV.shape[0] == len(zenith)):
        raise ValueError('The vector lengths of zenith (%s) and the rows of '
                         'DV(%s) are different' %
                         (len(zenith), DV.shape[0]))

    if not np.unique(heights == np.unique(heights))[0]:
        raise ValueError('Please use a sorted vector for heights, eg '
                         'heights = ' + str(np.unique(heights)))
    ze = 2 * np.pi * np.array([zenith]) / 360
    heights_dv = np.matmul(np.cos(ze).transpose(), np.array([m_range]))
    heights_dv = np.around(heights_dv, 1)

    def lin_in(x1, x2, y1, y2, x):
        y = (((y2 - y1) * x + y1 * x2 - y2 * x1) / (x2 - x1))
        y[np.where(abs(x1 - x) < 0.1)] = y1[np.where(abs(x1 - x) < 0.1)]
        y[np.where(abs(x2 - x) < 0.1)] = y2[np.where(abs(x2 - x) < 0.1)]
        return y

    DV[np.where(DV == -999.0)] = np.nan
    DV_ip = np.full((DV.shape[0], len(heights)), np.nan)
    index = np.repeat(0, len(zenith))
    for a in range(len(heights)):
        raise_i = [([np.nan, np.nan])]
        while len(raise_i[0]) and max(index) < len(m_range):
            raise_i = np.where(heights_dv[(np.array(range(len(zenith)))),
                                          index] < heights[a])
            index[raise_i] = index[raise_i] + 1

        index_below_top = np.where(index < len(m_range))[0]
        if len(index_below_top) > 0 and heights[a] >= np.min(heights_dv):
            x1 = heights_dv[(np.array(range(len(index_below_top)))),
                            index[index_below_top] - 1]
            x2 = heights_dv[(np.array(range(len(index_below_top)))),
                            index[index_below_top]]
            dv1 = DV[(np.array(range(len(index_below_top)))),
                     index[index_below_top] - 1]
            dv2 = DV[(np.array(range(len(index_below_top)))),
                     index[index_below_top]]
            DV_ip[index_below_top, a] = lin_in(x1, x2, dv1, dv2, heights[a])

    return DV_ip


def uvw1_retrieval_vertical(nc_lidar, i_t_start, i_t_end, heights,
                            quality_control_snr=False, lowest_frac=0.5,
                            highest_allowed_sigma=3,
                            iteration_stopping_sigma=1, n_ef=12):
    """ Retrieve wind information of lidar between starting and ending index
        in all heights by using uvw0_retrieval

    Args:
      nc_lidar: A loaded netcdf file with lidar data.
      i_t_start: Starting index belonging to the time dimension of nc_lidar.
      i_t_end: Ending index belonging to the time dimension of nc_lidar.
      heights: Heights of absolute horizontal wind (2 Point interpolation).
      quality_control_snr: If True, and if in nc_file_in qdv = 0 these dvs are
                           not used (because of bad SNR value).
      lowest_frac: Lowest allowed fraction of useful beams to input number of
                   beams.
      highest_allowed_sigma: Assume wind field is isotropic gaussian; highest
                             allowed sigma to retrieve than v.
      iteration_stopping_sigma: Assume wind field is isotropic gaussian;
                               ending criterion for iteration that improve v.
      n_ef: Effective degrees of freedom to scale the covariance matrix.
    Returns:
      u: u-components of wind vector (with height).
      v: v-component of wind vector (with height).
      w: w-component of wind vector (with height).
      covariance_matrix: The error covariance matrix for uvw (heights x 3 x 3).
      abs_vh: Absolute horizontal wind speeds.
      vh_dir: Direction of horizontal winds.
      sd_abs_vh: Standard deviation of absolute horizontal winds derived
                 from covariance_matrix.
      nvrad: Number of used radial beams.
      r2: R^2 value as in (E. Paeschke et al. 2015)
      cn: Condition number of A*diag((A^T*A)^(-1/2)).
    """

    DV = nc_lidar['dv'][:].data[i_t_start:i_t_end, :]
    DV[np.where(DV == -999.0)] = np.nan
    zenith = nc_lidar['zenith'][:].data[i_t_start:i_t_end]
    azimuth = nc_lidar['azi'][:].data[i_t_start:i_t_end]
    m_range = nc_lidar['range'][:].data
    qdv = nc_lidar['qdv'][:].data[i_t_start:i_t_end, :]
    nqv = 20
    try:
        nqv = nc_lidar['nqv'][:].data
    except:
        wn.warn('Variable nqv is not in netcdf, so nqv=20m/s is used!')

    bad_q = np.where(qdv == 0)
    if not quality_control_snr:
        qdv[bad_q] = 1
        bad_q = np.where(qdv == 0)

    DV[bad_q] = np.nan
    DV_ip = lin_inter_pol_dv(DV, zenith, m_range, heights)
    u1 = np.full(len(heights), np.nan)
    v1 = np.full(len(heights), np.nan)
    w1 = np.full(len(heights), np.nan)
    covariance_matrix1 = np.full((len(heights), 3, 3), np.nan)
    abs_vh1 = np.full(len(heights), np.nan)
    vh_dir1 = np.full(len(heights), np.nan)
    sd_abs_vh1 = np.full(len(heights), np.nan)
    nvrad1 = np.full(len(heights), np.nan)
    r2_1 = np.full(len(heights), np.nan)
    cn1 = np.full(len(heights), np.nan)
    for i_h in range(len(heights)):
        dv = DV_ip[:, i_h]
        uvw0 = uvw0_retrieval(dv=dv, zenith=zenith, azimuth=azimuth,
                              lowest_frac=lowest_frac,
                              highest_allowed_sigma=highest_allowed_sigma,
                              iteration_stopping_sigma=
                              iteration_stopping_sigma,n_ef=n_ef, nqv=nqv)
        u1[i_h] = uvw0['uvw'][0]
        v1[i_h] = uvw0['uvw'][1]
        w1[i_h] = uvw0['uvw'][2]
        covariance_matrix1[i_h, :, :] = uvw0['covariance_matrix']
        abs_vh1[i_h] = np.sqrt(uvw0['uvw'][0] ** 2 + uvw0['uvw'][1] ** 2)
        vh_dir1[i_h] = ((180 / np.pi) *
                        m.atan2(-uvw0['uvw'][0], -uvw0['uvw'][1]) +
                        360) % 360
        sd_abs_vh1[i_h] = np.sqrt(
            uvw0['covariance_matrix'][0, 0] * uvw0['uvw'][0] ** 2 +
            uvw0['covariance_matrix'][1, 1] * uvw0['uvw'][1] ** 2 +
            uvw0['covariance_matrix'][0, 1] *
            2 * uvw0['uvw'][0] * uvw0['uvw'][1]) / abs_vh1[i_h]
        nvrad1[i_h] = uvw0['nvrad']
        r2_1[i_h] = uvw0['r2']
        cn1[i_h] = uvw0['cn']

    return {'u': u1, 'v': v1, 'w': w1,
            'covariance_matrix': covariance_matrix1,
            'abs_vh': abs_vh1, 'vh_dir': vh_dir1, 'sd_abs_vh': sd_abs_vh1,
            'nvrad': nvrad1, 'r2': r2_1,
            'cn': cn1}


def most_zenith(nc_lidar):
    """ Retrieve most often used zenith angle in degree

    Args:
      nc_lidar: A loaded netcdf file with lidar data.
    Returns:
      most_z: Most often used zenith angle in degree.
    """

    z = nc_lidar['zenith'][:].data
    most_z = z[0]
    counter = 1
    for z_check in np.unique(nc_lidar['zenith'][:].data):
        if np.where(z == z_check)[0].size > counter:
            counter = np.where(z == z_check)[0].size
            most_z = z_check

    return most_z


def t_indices_circuit_borders(nc_lidar, i_t_start, i_t_end):
    """ Retrieve the bordering indices of each lidar circuit between starting
       and ending index with at least step-wide 3

    Args:
      nc_lidar: A loaded netcdf file with lidar data.
      i_t_start: Starting index belonging to the time dimension of nc_lidar.
      i_t_end: Ending index belonging to the time dimension of nc_lidar.
    Returns:
      index: Bordering indices starting at i_t_start and ending at i_tend.
    """

    azimuth_all = nc_lidar['azi'][:].data[i_t_start:i_t_end]
    azimuth_all = (azimuth_all - azimuth_all[0] + 180) % 360 - 180
    clockwise = (np.mean(np.sign(azimuth_all[1:] - azimuth_all[0: -1])) > 0)
    azimuth_diff = azimuth_all - azimuth_all[0]
    if clockwise:  # check start of next round:
        index = np.where((azimuth_diff[1:] > 0) &
                         (azimuth_diff[0: -1] <= 0))[0]
    else:
        index = np.where((azimuth_diff[1:] < 0) &
                         (azimuth_diff[0: -1] >= 0))[0]

    raise_i = np.where(index[1:] - index[: -1] < 3)[0] + 1  # Problem: for
    # 3-beam mode, the raise_i procedure will cause set of beams with included
    # vertical stare and without 3 different azimuth directions -> nan's.
    while raise_i.shape[0] > 0:  # some azimuths occur that are swinging
        index[raise_i[0]:] = index[raise_i[0]:] + 1
        raise_i = np.where(index[1:] - index[: -1] < 3)[0] + 1

    while index[-1] > i_t_end:
        index = index[: -1]

    index = index + i_t_start
    index[-1] = i_t_end
    return {'i_t_circ_start': index[0:-1], 'i_t_circ_end': index[1:]}


def nc_comments(nc_file_in, nc_lidar):
    """ Comment to describe the scanning mode

    Args:
      nc_file_in: Filename of nc-file with the lidar data located in folder
                  in SAMD 1.2 convention (doi.org/10.3390/ijgi5070124).
      nc_lidar: A loaded netcdf file with lidar data.
    Returns:
      comment: A comment describing the scanning mode.
      mode: The scanning mode.
      most_elevation: The most often used elevation angle.
      most_circulation_time: The most often used circulation time.
      most_nr_beams: The most often used number of beams.
    """

    name_control = nc_file_in.split('_')
    n = name_control.__len__()
    mode = name_control[n - 5][4:7]
    elevation = str(np.round(90 - most_zenith(nc_lidar), 1))
    t_lidar = nc_lidar['time'][:].data
    if mode[0:2] == 'ST':
        return {'comment': 'STARE', 'mode': 'STARE', 'most_elevation': np.nan,
                'most_circulation_time': np.nan, 'most_nr_beams': np.nan}

    elif mode == 'RHI':
        return {'comment': mode, 'mode': mode, 'most_elevation': np.nan,
                'most_circulation_time': np.nan, 'most_nr_beams': np.nan}

    elif mode == 'PPI':
        return {'comment': mode, 'mode': mode, 'most_elevation': np.nan,
                'most_circulation_time': np.nan, 'most_nr_beams': np.nan}

    elif mode == 'DBS':
        i_t_starts = np.arange(0, t_lidar.size - 2, 4)
        i_t_ends = np.append(i_t_starts[1:], t_lidar.size - 1)

    else:
        i_t_circ = t_indices_circuit_borders(nc_lidar, i_t_start=0,
                                             i_t_end=t_lidar.size - 1)
        i_t_starts = i_t_circ['i_t_circ_start']
        i_t_ends = i_t_circ['i_t_circ_end']

    t_diff = t_lidar[i_t_ends] - t_lidar[i_t_starts]
    circ_time = np.round(np.nanmean(t_diff), 1)
    if circ_time < 20:
        circ_duration = 'quickly'

    else:
        circ_duration = 'slowly'

    (i_t_diff, counts) = np.unique(i_t_ends - i_t_starts, return_counts=True)
    n_beams = i_t_diff[np.where(counts == np.max(counts))][0]
    comment = mode + '(~' + elevation + '° elevation, ' + circ_duration + \
              ' (~' + str(circ_time) + ' s) with ~' + str(n_beams) + \
              ' beams per circulation' + ', npls = ' + \
              str(nc_lidar['npls'][:].data) + ')'
    return {'comment': comment, 'mode': mode, 'most_elevation': elevation,
            'most_circulation_time': circ_time, 'most_nr_beams': n_beams}


def name_nc_file_out(nc_file_in, circ=True, duration=np.nan,
                     quality_control_snr=False):
    """ Create new name for netcdf output file

    Args:
      nc_file_in: Filename of nc-file with the lidar data located in folder in.
      duration: Duration window with consecutive beams inside in s.
      circ: True if a retrieval per each lidar circuit should be proceeded
            (and a given duration is ignored).
      quality_control_snr: If True, and if in nc_file_in qdv = 0 these dvs are
                           not used (because of bad SNR value).
    Returns:
      nc_file_out: Filename of nc-file for output retrieval.
    """

    if type(duration) == list:
        if len(duration) == 1:
            duration = duration[0]
        else:
            duration = np.nan

    name_control = nc_file_in.split('_')
    for i in range(len(name_control)):
        char = name_control[i]
        if (len(char) == 2) & (char[0] == 'l'):
            name_control[i] = 'l2'

        if char == 'any':
            if circ:
                name_control[i] = 'wind-circ'

            else:
                name_control[i] = 'wind-' + str(int(duration)) + 's'

            if quality_control_snr:
                name_control[i] = name_control[i] + '-SNR'

    nc_file_out = '_'.join(name_control)
    return nc_file_out


def uvw2_time_series_netcdf(nc_file_in, path_folder_in, path_folder_out,
                            duration, heights, heights_corg=np.nan, circ=True,
                            quality_control_snr=False, check_exist=True,
                            lowest_frac=0.5, highest_allowed_sigma=3,
                            iteration_stopping_sigma=1, n_ef=12):
    """ Full wind retrieval of lidar data with calling uvw1_retrieval_vertical

    Args:
      nc_file_in: Filename of nc-file with the lidar data located in folder in.
      path_folder_in: The path to folder containing the lidar data nc-file.
      path_folder_out: Path to folder where the retrieval nc file is saved.
      duration: Duration window with consecutive beams inside in s.
      heights: Heights for retrieval (linear interpolation of tilted beams).
      heights_corg: Heights for retrieval corg (1) or interpolated (0). If nan
                    the information will be omitted in netcdf.
      circ: True if a retrieval per each lidar circuit should be proceeded
            (and a given duration is ignored).´
      quality_control_snr: If True, and if in nc_file_in qdv = 0 these dvs are
                           not used (because of bad SNR value).
      check_exist: If True then approach is canceled when output file is
                   already existing (without checking weather it contains the
                   same heights).
      lowest_frac: Lowest allowed fraction of useful beams to input number of
                   beams.
      highest_allowed_sigma: Assume wind field is isotropic gaussian; highest
                             allowed sigma to retrieve than v.
      iteration_stopping_sigma: Assume wind field is isotropic gaussian;
                               ending criterion for iteration that improve v.
      n_ef: Effective degrees of freedom to scale the covariance matrix.
    Returns:
    """

    start_time = time.time()
    if not isdir(path_folder_out):
        try:
            makedirs(path_folder_out)
        except:
            wn.warn(path_folder_out + ' simultaniously created!')

    nc_file_out = name_nc_file_out(nc_file_in=nc_file_in, circ=circ,
                                   duration=duration,
                                   quality_control_snr=quality_control_snr)
    exist = isfile(join(path_folder_out + nc_file_out))
    if exist and check_exist:
        print('Consider: ' + nc_file_out +
              ' is existing and not again calculated. Please set '
              '>check_exist< to FALSE, if recalculation is needed')
        return

    nc_lidar = nc.Dataset(path_folder_in + nc_file_in)
    nc_lidar_new = nc.Dataset(path_folder_out + 'tmp_' + nc_file_out,
                              'w', format='NETCDF4_CLASSIC')
    nc_description = nc_comments(nc_file_in, nc_lidar)
    # global attributes
    # nc_lidar_new.setncatts(nc_lidar.__dict__)
    nc_lidar_new.Title = 'Wind vectors derived from Doppler wind lidars'
    if 'institution' in nc_lidar.__dict__:
        nc_lidar_new.Institution = nc_lidar.institution
    elif 'Institution' in nc_lidar.__dict__:
        nc_lidar_new.Institution = nc_lidar.Institution
    else:
        nc_lidar_new.Institution = 'University of Cologne'

    if 'instrument_contact' in nc_lidar.__dict__:
        nc_lidar_new.Contact_person = nc_lidar.instrument_contact
    elif 'Contact_person' in nc_lidar.__dict__:
        nc_lidar_new.Contact_person = nc_lidar.Contact_person
    elif 'contact_person' in nc_lidar.__dict__:
        nc_lidar_new.Contact_person = nc_lidar.contact_person
    else:
        nc_lidar_new.Contact_person = 'Dr. Julian Steinheuer, ' \
                                      'Julian.Steinheuer@uni-koeln.de'

    if 'instrument_type' in nc_lidar.__dict__ and 'instrument_mode' in nc_lidar.__dict__:
        nc_lidar_new.Source = 'Ground based remote sensing by a ' + \
                              nc_lidar.instrument_type + ' in ' + \
                              nc_lidar.instrument_mode + \
                              ' processed with JSteinheuer/DWL_retrieval ' \
                              '(doi.org/10.5281/zenodo.5780949)'
    else:
        nc_lidar_new.Source = 'Ground based remote sensing by a DWL' \
                              ' processed with JSteinheuer/DWL_retrieval ' \
                              '(doi.org/10.5281/zenodo.5780949)'

    if 'history' in nc_lidar.__dict__:
        nc_lidar_new.History = 'Retrieval from Steinheuer et al. 2022 ' \
                               '(doi.org/10.5194/amt-15-3243-2022) and from ' \
                               'level1: ' + nc_lidar.history
    elif 'History' in nc_lidar.__dict__:
        nc_lidar_new.History = 'Retrieval from Steinheuer et al. 2022 ' \
                               '(doi.org/10.5194/amt-15-3243-2022) and from ' \
                               'level1: ' + nc_lidar.History
    else:
        nc_lidar_new.History = 'Retrieval from Steinheuer et al. 2022 ' \
                               '(doi.org/10.5194/amt-15-3243-2022) and from ' \
                               'level1 file: ' + nc_file_in

    nc_lidar_new.Dependencies = nc_file_in
    nc_lidar_new.Conventions = 'CF-1.8'
    nc_lidar_new.Processing_date = time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.gmtime()) + '(UTC)'
    nc_lidar_new.Author = 'Dr. Julian Steinheuer, ' \
                          'Julian.Steinheuer@uni-koeln.de'
    nc_lidar_new.Comments = nc_description['comment']
    nc_lidar_new.License = 'For non-commercial use only. This data is ' \
                           'subject to the SAMD data policy to be found at ' \
                           'doi.org/10.25592/uhhfdm.9824 and in ' \
                           'the SAMD Observation Data Product standard v2.0'
    # dimensions
    heights_all = heights
    nc_lidar_new.createDimension('time', None)
    t_nc = nc_lidar_new.createVariable('time', np.float64, ('time',))
    t_nc.long_name = 'time'
    t_nc.units = 'seconds since 1970-01 00:00:00'
    t_nc.calender = 'gregorian'
    t_nc.bounds = 'time_bnds'
    t_nc.comments = 'Timestamp belongs to the start of the interval and is ' \
                    'in UTC'
    nc_lidar_new.createDimension('height', len(heights_all))
    height_nc = nc_lidar_new.createVariable('height', np.float32, ('height',))
    height_nc.long_name = 'vertical distance from sensor'
    height_nc.comments = 'Height does not automatically correspond to a ' \
                         'center of range gate, since interpolated values ' \
                         'could be used. Check the flag height_corg for this'
    height_nc.units = 'm'
    height_nc[:] = heights_all[:].data
    if not np.isnan(heights_corg[0]):
        height_interpolated_nc = nc_lidar_new.createVariable('height_corg',
                                                             np.byte,
                                                             ('height',))
        height_interpolated_nc.long_name = 'height mask for the center of ' \
                                           'range gates heights'
        height_interpolated_nc.units = '1'
        height_interpolated_nc.comments = 'Flag values mean: ' \
                                          '0 = interpolated height; ' \
                                          '1 = corg height.'
        height_interpolated_nc[:] = heights_corg

    nc_lidar_new.createDimension('row', 3)
    # rows_nc = nc_lidar_new.createVariable('rows', np.float32, ('rows',))
    # rows_nc.long_name = 'rows of covariance matrix'
    # rows_nc.units = '1'
    nc_lidar_new.createDimension('col', 3)
    # cols_nc = nc_lidar_new.createVariable('cols', np.float32, ('cols',))
    # cols_nc.long_name = 'columns of covariance matrix'
    # cols_nc.units = '1'
    nc_lidar_new.createDimension('nv', 2)
    # variables copy d
    var = 'lat'
    nc_lidar_new.createVariable(var, nc_lidar[var].datatype,
                                nc_lidar[var].dimensions)
    nc_lidar_new[var].long_name = 'latitude'
    nc_lidar_new[var].comments = 'Latitude of instrument location'
    nc_lidar_new[var].units = 'degrees_north'
    nc_lidar_new[var][:] = nc_lidar[var][:].data
    var = 'lon'
    nc_lidar_new.createVariable(var, nc_lidar[var].datatype,
                                nc_lidar[var].dimensions)
    nc_lidar_new[var].long_name = 'longitude'
    nc_lidar_new[var].comments = 'Longitude of instrument location'
    nc_lidar_new[var].units = 'degrees_east'
    nc_lidar_new[var][:] = nc_lidar[var][:].data
    var = 'zsl'
    nc_lidar_new.createVariable(var, nc_lidar[var].datatype,
                                nc_lidar[var].dimensions)
    nc_lidar_new[var].long_name = 'altitude'
    nc_lidar_new[var].comments = 'Altitude of instrument above mean sea level'
    nc_lidar_new[var].units = 'm'
    nc_lidar_new[var][:] = nc_lidar[var][:].data



    # for var in ['lat', 'lon', 'zsl']:
    # # for var in ['focus', 'lat', 'lon', 'lrg', 'nfft', 'npls', 'nqf', 'nqv',
    # #             'nrg', 'nsmpl', 'pd', 'prf', 'resf', 'resv', 'smplf', 'tgint',
    # #             'wl', 'zsl']:
    #     try:
    #         nc_lidar_new.createVariable(var, nc_lidar[var].datatype,
    #                                     nc_lidar[var].dimensions,
    #                                     fill_value=-999.0)
    #         dict_var = nc_lidar[var].__dict__
    #         nc_lidar_new[var].setncatts(
    #             {k: dict_var[k] for k in dict_var.keys() if
    #              not k == '_FillValue'})
    #         nc_lidar_new[var][:] = nc_lidar[var][:].data
    #     except:
    #         wn.warn(var + ' dictionary could not be copied')

    # variables assign new
    u_nc = nc_lidar_new.createVariable('u', np.float32,
                                       ('time', 'height'),
                                       fill_value=-999.0)
    u_nc.long_name = 'eastward wind'
    u_nc.units = 'm s-1'
    u_nc.comments = 'Eastward wind, i.e. zonal component of wind'
    v_nc = nc_lidar_new.createVariable('v', np.float32,
                                       ('time', 'height'),
                                       fill_value=-999.0)
    v_nc.long_name = 'northward wind'
    v_nc.units = 'm s-1'
    v_nc.comments = 'Northward wind, i.e. meridional component of wind'
    w_nc = nc_lidar_new.createVariable('w', np.float32,
                                       ('time', 'height'),
                                       fill_value=-999.0)
    w_nc.long_name = 'upward air velocity'
    w_nc.units = 'm s-1'
    w_nc.comments = 'Upward air velocity, i.e. wind component away from ' \
                    'surface'
    covariance_matrix_nc = nc_lidar_new.createVariable(
        'covariances', np.float32, ('time', 'height', "row", "col"),
        fill_value=-999.0)
    covariance_matrix_nc.long_name = 'covariances of wind'
    covariance_matrix_nc.comments = 'Covariance matrix of wind vector'
    covariance_matrix_nc.units = 'm^2 s^-2'
    abs_vh_nc = nc_lidar_new.createVariable('wspeed', np.float32,
                                            ('time', 'height'),
                                            fill_value=-999.0)
    abs_vh_nc.long_name = 'wind speed'
    abs_vh_nc.units = 'm s-1'
    abs_vh_nc.comments = 'Absolute horizontal wind speed'
    vh_dir_nc = nc_lidar_new.createVariable('wdir', np.float32,
                                            ('time', 'height'),
                                            fill_value=-999.0)
    vh_dir_nc.long_name = 'wind direction'
    vh_dir_nc.units = '°'
    vh_dir_nc.comments = 'Wind from North = 0°, from East = 90°, asf.'
    sd_abs_vh_nc = nc_lidar_new.createVariable('sd_wspeed', np.float32,
                                               ('time', 'height'),
                                               fill_value=-999.0)
    sd_abs_vh_nc.long_name = 'standard deviation of wind speed'
    sd_abs_vh_nc.units = 'm s-1'
    sd_abs_vh_nc.comments = 'Standard deviation of wind speed derived from ' \
                            'the covariance matrix'
    nvrad_nc = nc_lidar_new.createVariable('nvrad', np.float32,
                                           ('time', 'height'),
                                           fill_value=-999.0)
    nvrad_nc.long_name = 'number of radial velocities'
    nvrad_nc.units = '1'
    nvrad_nc.comments = 'Number of involved radial velocities in the ' \
                      'wind vector fit'
    r2_nc = nc_lidar_new.createVariable('r2', np.float32,
                                        ('time', 'height'),
                                        fill_value=-999.0)
    r2_nc.long_name = 'coefficient of determination'
    r2_nc.units = '1'
    r2_nc.comments = 'Pearsons coefficient of determination ' \
                     'calculated for the wind vector fit'
    cn_nc = nc_lidar_new.createVariable('cn', np.float32,
                                        ('time', 'height'),
                                        fill_value=-999.0)
    cn_nc.long_name = 'condition number'
    cn_nc.units = '1'
    cn_nc.comments = 'Condition number of wind vector fit'
    time_bnds_nc = nc_lidar_new.createVariable('time_bnds', np.float64,
                                               ('time', 'nv'),
                                               fill_value=-999.0)
    time_bnds_nc.long_name = 'time boundaries'
    time_bnds_nc.units = 'seconds since 1970-01 00:00:00'
    i_h_new = np.repeat(-999, heights.size)
    for i in range(heights.size):
        i_h_new[i] = np.where(nc_lidar_new['height'][:].data[:] ==
                              heights[i])[0][0]

    # loop over heights
    t_lidar = nc_lidar['time'][:].data
    if not np.unique(np.sort(t_lidar) == t_lidar)[0]:
        nc_lidar_new.close()
        nc_lidar.close()
        system('rm ' + path_folder_out + 'tmp_' + nc_file_out)
        wn.warn('Input error: The input netcdf-file ' + nc_file_in + ' has ' +
                'no well defined time variable and for some i t[i] > t[i+1]')
        return

    if circ:
        if nc_description['mode'] == 'DBS':
            i_t_starts = np.arange(0, t_lidar.size - 2, 4)
            i_t_ends = np.append(i_t_starts[1:], t_lidar.size - 1)

        else:
            i_t_circ = t_indices_circuit_borders(nc_lidar, i_t_start=0,
                                                 i_t_end=t_lidar.size - 1)
            i_t_starts = i_t_circ['i_t_circ_start']
            i_t_ends = i_t_circ['i_t_circ_end']

        t_nc[:] = t_lidar[i_t_starts]

    else:
        tt = datetime.datetime.utcfromtimestamp(nc_lidar['time'][:].data[0])
        tt = datetime.datetime(tt.year, tt.month, tt.day)
        tt = (tt - datetime.datetime(1970, 1, 1)).total_seconds()
        t_nc[:] = np.arange(tt, tt + 60 * 60 * 24, duration)
        i_t_starts = np.repeat(-999, t_nc.size)
        i_t_ends = np.repeat(-999, t_nc.size)
        for i_t in range(t_nc.size - 1):
            i_next = np.where(t_lidar > t_nc[:].data[i_t])[0]
            i_prev = np.where(t_lidar < t_nc[:].data[i_t + 1])[0]
            if i_next.size > 0 and i_prev.size > 0:
                i_t_starts[i_t] = i_next[0]
                i_t_ends[i_t] = i_prev[-1] + 1

        i_next = np.where(t_lidar > t_nc[:].data[-1])[0]
        if i_next.size > 0:
            i_t_starts[-1] = i_next[0]
            i_t_ends[-1] = i_next[-1]

    time_bnds_nc[:, 0] = nc_lidar['time'][:].data[i_t_starts]
    time_bnds_nc[:, 1] = nc_lidar['time'][:].data[i_t_ends - 1]
    for i_t in range(t_nc.size):
        if not i_t_starts[i_t] == -999 and not i_t_ends[i_t] == -999 and \
                i_t_ends[i_t] - i_t_starts[i_t] > 2:
            uvw1 = uvw1_retrieval_vertical(nc_lidar=nc_lidar,
                                           i_t_start=i_t_starts[i_t],
                                           i_t_end=i_t_ends[i_t],
                                           heights=heights,
                                           quality_control_snr=
                                           quality_control_snr,
                                           lowest_frac=lowest_frac,
                                           highest_allowed_sigma=
                                           highest_allowed_sigma,
                                           iteration_stopping_sigma=
                                           iteration_stopping_sigma,
                                           n_ef=n_ef)
            u_nc[i_t, i_h_new] = uvw1['u']
            v_nc[i_t, i_h_new] = uvw1['v']
            w_nc[i_t, i_h_new] = uvw1['w']
            covariance_matrix_nc[i_t, i_h_new, :, :] = \
                uvw1['covariance_matrix']
            abs_vh_nc[i_t, i_h_new] = uvw1['abs_vh']
            vh_dir_nc[i_t, i_h_new] = uvw1['vh_dir']
            sd_abs_vh_nc[i_t, i_h_new] = uvw1['sd_abs_vh']
            nvrad_nc[i_t, i_h_new] = uvw1['nvrad']
            r2_nc[i_t, i_h_new] = uvw1['r2']
            cn_nc[i_t, i_h_new] = uvw1['cn']
            if i_t > 0 and nc_lidar_new['time_bnds'][:].data[i_t, 0] < \
                    nc_lidar_new['time'][:].data[i_t - 1]:
                nc_lidar_new['time_bnds'][i_t, 0] = \
                    nc_lidar_new['time'][:].data[i_t]
            elif nc_lidar_new['time_bnds'][:].data[i_t, 0] > \
                    nc_lidar_new['time'][:].data[i_t]:
                nc_lidar_new['time_bnds'][i_t, 0] = \
                    nc_lidar_new['time'][:].data[i_t]

            if nc_lidar_new['time_bnds'][:].data[i_t, 1] < \
                    nc_lidar_new['time'][:].data[i_t]:
                nc_lidar_new['time_bnds'][i_t, 1] = \
                    nc_lidar_new['time'][:].data[i_t]
            elif i_t < nc_lidar_new['time'][:].shape[0] - 1 and \
                    nc_lidar_new['time_bnds'][:].data[i_t, 1] > \
                    nc_lidar_new['time'][:].data[i_t + 1]:
                nc_lidar_new['time_bnds'][i_t, 1] = \
                    nc_lidar_new['time'][:].data[i_t]

            print('Process evolution:',
                  str(np.round((i_t + 1) / t_nc.size * 100)),
                  ' % for ', nc_file_out)

    nc_lidar_new.close()
    nc_lidar.close()
    system('mv ' + path_folder_out + 'tmp_' + nc_file_out + ' ' +
           path_folder_out + nc_file_out)
    print('Runtime of uvw2_time_series_netcdf: %s s'
          % (np.round(time.time() - start_time, 1)))


def heights_with_heights_of_corg(nc_lidar, heights=[90.3]):
    """ Retrieve heights of center of range gate corresponding (corg) to
        most often used zenith angle in degree with additional interpolated
        heights

    Args:
      nc_lidar: A loaded netcdf file with lidar data.
      heights: Further heights.
    Returns:
      heights_all: array of extended heights with heights of corg
                   (rounded on .1).
      heights_corg: bool array indication if interpolation height (0) or
                    corg (1).
    """

    z = most_zenith(nc_lidar)
    heights_l = nc_lidar['range'][:].data[:] * \
                np.cos(z / 360 * 2 * np.pi)
    heights_l = np.around(heights_l, 1)
    heights_all = np.unique(np.concatenate((heights, heights_l), 0), 1)[0]
    heights_all = [float(np.format_float_positional(i, precision=1)) for i in
                   heights_all]
    heights_l = [float(np.format_float_positional(i, precision=1)) for i in
                 heights_l]
    heights_corg = [i in heights_l for i in heights_all]
    return np.array(heights_all), np.array(heights_corg)


def uvw3_retrievals(nc_file_in, path_folder_in, path_folder_out,
                    duration=600, circ=True, heights_fix=np.array([90.3]),
                    quality_control_snr=False, check_exist=True,
                    lowest_frac=0.5, highest_allowed_sigma=3,
                    iteration_stopping_sigma=1, n_ef=12):
    """ Wrapping function for uvw2_time_series_netcdf

    Args:
      nc_file_in: Filename of nc-file with the lidar data located in folder in.
      path_folder_in: The path to folder containing the lidar data nc-file.
      path_folder_out: Path to folder where the retrieval nc file is saved.
      duration: Duration window with consecutive beams inside in s.
      circ: True if a retrieval per each lidar circuit should be proceeded.
      heights_fix: Heights for retrieval (linear interpolation of tilted
                   beams).
      quality_control_snr: If True, and if in nc_file_in qdv = 0 these dvs are
                           not used (because of bad SNR value).
      check_exist: If True then approach is canceled when output file is
                   already existing (without checking weather it contains the
                   same heights).
      lowest_frac: Lowest allowed fraction of useful beams to input number of
                   beams.
      highest_allowed_sigma: Assume wind field is isotropic gaussian; highest
                             allowed sigma to retrieve than v.
      iteration_stopping_sigma: Assume wind field is isotropic gaussian;
                               ending criterion for iteration that improve v.
      n_ef: Effective degrees of freedom to scale the covariance matrix.
    Returns:
    """

    name_control = nc_file_in.split('_')
    if not name_control[-1][-3:] == '.nc':
        wn.warn('Wrong input: The file ' + nc_file_in + ' is no netcdf')
        return

    mode = name_control[name_control.__len__() - 5][4:6]
    if mode in ['ST', 'PP', 'RH']:
        wn.warn('Wrong input: The file ' + nc_file_in + ' represents Stare, ' +
                'PPI or RHI scanning mode and this retrieval is not ' +
                'applicable')
        return

    if not nc_name_check_level(nc_file_in) == 1:
        wn.warn('Wrong input: The file ' + nc_file_in + ' is not of level 1')
        return

    if not isdir(path_folder_out):
        try:
            makedirs(path_folder_out)
        except:
            wn.warn(path_folder_out + ' simultaniously created!')

    if circ:
        try:
            nc_file_out = name_nc_file_out(nc_file_in=nc_file_in, circ=True,
                                           duration=np.nan,
                                           quality_control_snr=
                                           quality_control_snr)
            nc_lidar = nc.Dataset(path_folder_in + nc_file_in)
            heights, heights_corg = heights_with_heights_of_corg(
                nc_lidar=nc_lidar,
                heights=heights_fix)
            nc_lidar.close()
            uvw2_time_series_netcdf(nc_file_in=nc_file_in,
                                    path_folder_in=path_folder_in,
                                    path_folder_out=path_folder_out,
                                    duration=duration, heights=heights,
                                    heights_corg=heights_corg, circ=True,
                                    quality_control_snr=quality_control_snr,
                                    check_exist=check_exist,
                                    lowest_frac=lowest_frac,
                                    highest_allowed_sigma=
                                    highest_allowed_sigma,
                                    iteration_stopping_sigma=
                                    iteration_stopping_sigma, n_ef=n_ef)
        except Exception as e:
            system('rm ' + path_folder_out + 'tmp_' + nc_file_out)
            print(e, file=open(path_folder_out + 'ERROR_' +
                               nc_file_out[0:-3] + '.txt', 'w+'))
            wn.warn('Error: Some error occurred while proceeding ' +
                    'with circularily ' + nc_file_in + ' and circulation')
    else:
        if np.isnan(duration):
            return

        try:
            nc_file_out = name_nc_file_out(nc_file_in=nc_file_in,
                                           circ=False,
                                           duration=duration,
                                           quality_control_snr=
                                           quality_control_snr)
            nc_lidar = nc.Dataset(path_folder_in + nc_file_in)
            heights, heights_corg = heights_with_heights_of_corg(
                nc_lidar=nc_lidar,
                heights=heights_fix)
            nc_lidar.close()

            uvw2_time_series_netcdf(nc_file_in=nc_file_in,
                                    path_folder_in=path_folder_in,
                                    path_folder_out=path_folder_out,
                                    duration=duration, heights=heights,
                                    heights_corg=heights_corg, circ=False,
                                    quality_control_snr=
                                    quality_control_snr,
                                    check_exist=check_exist,
                                    lowest_frac=lowest_frac,
                                    highest_allowed_sigma=
                                    highest_allowed_sigma,
                                    iteration_stopping_sigma=
                                    iteration_stopping_sigma, n_ef=n_ef)
        except Exception as e:
            system('rm ' + path_folder_out + 'tmp_' + nc_file_out)
            print(e, file=open(path_folder_out + 'ERROR_' +
                               nc_file_out[0:-3] + '.txt', 'w+'))
            wn.warn('Error: Some mysterious error occurred while proceeding ' +
                    'with' + nc_file_in + ' and duration ' + str(duration) + 's')


###############################################################################
# STEP C: Wind product                                                        #
#                                                                             #
# Create end-product netcdf which contains both 600s winds and 600s gust      #
# peaks.                                                                      #
###############################################################################


def wind_and_gust_netcdf(nc_file_in_mean, path_folder_in_mean,
                         nc_file_in_circ, path_folder_in_circ,
                         path_folder_out, circulations_fraction=0.5,
                         check_exist=True, max_out=1, out_str='wind-gust'):
    """ Produce netcdf-files with mean wind and corresponding gust peak

    Args:
      nc_file_in_mean: Filename of nc-file with the lidar mean wind level 2
                       data located in path_folder_in.
      path_folder_in_mean: The path to folder containing the lidar data file
                           nc_file_in_mean.
      nc_file_in_circ: Filename of nc-file with the lidar circulation wind
                       level 2 data located in path_folder_in.
      path_folder_in_circ: The path to folder containing the lidar data file
                           nc_file_in_circ.
      path_folder_out: Path to folder where the output netcdf file is saved.
      circulations_fraction: Fraction of useful circulation retrievals
                             at least necessary for the unfiltered gust peak.
      check_exist: If True then approach is canceled when output file is
                   already existing.
      max_out: Outlier detection: This is the Maximum allowed outlier-distance
               [m/s] in set of uvw-circs compared to next lower (or higher)
               value.
      out_str: string for SAMD convention following file name.
    Returns:
    """

    if not isfile(path_folder_in_mean + nc_file_in_mean):
        wn.warn(nc_file_in_mean + ' does not exist where you think it is')
        return

    if not isfile(path_folder_in_circ + nc_file_in_circ):
        wn.warn(nc_file_in_circ + ' does not exist where you think it is')
        return

    nc_file_in_new = nc_name_raise_version(nc_file_in_circ).replace('wind-circ',
                                                                    out_str)
    if not isdir(path_folder_out):
        try:
            makedirs(path_folder_out)
        except:
            wn.warn(path_folder_out + ' simultaniously created!')

    if check_exist:
        if isfile(path_folder_out + nc_file_in_new):
            print('Consider: ' + nc_file_in_new +
                  ' is existing and not again calculated. Please set '
                  '>check_exist< to FALSE, if recalculation is needed')
            return

    print(nc_file_in_circ + ' to process for wind gust peaks')

    nc_lidar_circ = nc.Dataset(path_folder_in_circ + nc_file_in_circ)
    nc_lidar_mean = nc.Dataset(path_folder_in_mean + nc_file_in_mean)
    nc_lidar_new = nc.Dataset(path_folder_out + 'tmp_' + nc_file_in_new,
                              'w', format='NETCDF4_CLASSIC')
    # global attributes
    nc_lidar_new.setncatts(nc_lidar_mean.__dict__)
    nc_lidar_new.Title = 'Wind (mean, peak, minimum) derived from' \
                         ' Doppler wind lidars'
    nc_lidar_new.History = 'Combined wind product from mean-wind-file and ' + \
                           'circulation-wind-file: ' + nc_file_in_mean + \
                           ' and ' + nc_file_in_circ
    nc_lidar_new.Processing_date = time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.gmtime()) + '(UTC)'
    nc_lidar_new.Author = 'Julian Steinheuer, Julian.Steinheuer@uni-koeln.de'
    # copy dimensions
    for dname, the_dim in nc_lidar_mean.dimensions.items():
        nc_lidar_new.createDimension(dname, len(
            the_dim))

    # variables copy
    for var in ['lat', 'lon', 'time_bnds', 'zsl', 'height_corg']:
        try:
            nc_lidar_new.createVariable(var, nc_lidar_mean[var].datatype,
                                        nc_lidar_mean[var].dimensions,
                                        fill_value=-999.0)
            dict_var = nc_lidar_mean[var].__dict__
            nc_lidar_new[var].setncatts(
                {k: dict_var[k] for k in dict_var.keys() if
                 not k == '_FillValue'})
            nc_lidar_new[var][:] = nc_lidar_mean[var][:].data
        except:
            wn.warn(var + ' dictionary could not be copied')

    # dim variables copy
    for var in ['height', 'time']:
        try:
            nc_lidar_new.createVariable(var, nc_lidar_mean[var].datatype,
                                        nc_lidar_mean[var].dimensions)
            nc_lidar_new[var][:] = nc_lidar_mean[var][:].data
            nc_lidar_new[var].setncatts(nc_lidar_mean[var].__dict__)
        except:
            wn.warn(var + ' dictionary could not be copied')

    for var in ['wspeed', 'sd_wspeed', 'covariances',
                'u', 'v', 'w', 'wdir']:
        nc_lidar_new.createVariable(var, nc_lidar_mean[var].datatype,
                                    nc_lidar_mean[var].dimensions,
                                    fill_value=-999.0)
        dict_var = nc_lidar_mean[var].__dict__
        nc_lidar_new[var].setncatts(
            {k: dict_var[k] for k in dict_var.keys() if not k == '_FillValue'})
        nc_lidar_new[var][:] = nc_lidar_mean[var][:].data

    for var in ['wspeed', 'sd_wspeed', 'covariances',
                'u', 'v', 'w', 'wdir']:
        nc_lidar_new.createVariable(var + '_max',
                                    nc_lidar_mean[var].datatype,
                                    nc_lidar_mean[var].dimensions,
                                    fill_value=-999.0)
        dict_var = nc_lidar_circ[var].__dict__
        nc_lidar_new[var + '_max'].setncatts(
            {k: dict_var[k] for k in dict_var.keys() if not k == '_FillValue'})
        nc_lidar_new[var + '_max'].long_name = \
            nc_lidar_circ[var].long_name + ' of gust peak'
        nc_lidar_new[var + '_max'].comments = \
            nc_lidar_circ[var].comments + ' of gust peak'

    for var in ['wspeed', 'sd_wspeed', 'covariances',
                'u', 'v', 'w', 'wdir']:
        nc_lidar_new.createVariable(var + '_min',
                                    nc_lidar_mean[var].datatype,
                                    nc_lidar_mean[var].dimensions,
                                    fill_value=-999.0)
        dict_var = nc_lidar_circ[var].__dict__
        nc_lidar_new[var + '_min'].setncatts(
            {k: dict_var[k] for k in dict_var.keys() if not k == '_FillValue'})
        nc_lidar_new[var + '_min'].long_name = \
            nc_lidar_circ[var].long_name + ' of weakest wind'
        nc_lidar_new[var + '_min'].comments = \
            nc_lidar_circ[var].comments + ' of weakest wind'

    abs_vh_mean_nc = nc_lidar_mean.variables['wspeed'][:].data
    abs_vh_circ_nc = nc_lidar_circ.variables['wspeed'][:].data
    for i_t in range(nc_lidar_new.variables['time'].size):
        # circulation has to be started in window
        i_t_circ_window = (nc_lidar_circ.variables['time_bnds'][:, 0] <=
                           nc_lidar_mean.variables['time_bnds'][i_t, 1]) * \
                          (nc_lidar_circ.variables['time_bnds'][:, 0] >=
                           nc_lidar_mean.variables['time_bnds'][i_t, 0])
        frac_denominator = np.count_nonzero(i_t_circ_window == True)
        if not any(i_t_circ_window):
            continue

        for i_h in range(nc_lidar_new.variables['height'].size):
            if np.isnan(abs_vh_mean_nc[i_t, i_h]):
                continue

            if np.isnan(abs_vh_circ_nc[i_t_circ_window, i_h]).all():
                continue

            i_non_nan = ~np.isnan(abs_vh_circ_nc[i_t_circ_window, i_h])
            circ_frac = np.count_nonzero(i_non_nan == True) / frac_denominator
            if circ_frac <= circulations_fraction or sum(i_t_circ_window) < 2:
                wn.warn('Less than ' +
                        str(int(np.round(circulations_fraction * 100))) +
                        '% useful circulation retrievals in time window')
                continue

            abs_vh_circ_i_t_i_h = abs_vh_circ_nc[i_t_circ_window, i_h]
            max_i_t_s = np.argsort(-abs_vh_circ_i_t_i_h, axis=0)
            while abs_vh_circ_i_t_i_h[max_i_t_s[0]] - \
                    abs_vh_circ_i_t_i_h[max_i_t_s[1]] > max_out and \
                    sum(~np.isnan(abs_vh_circ_i_t_i_h)) > 2:
                abs_vh_circ_i_t_i_h[max_i_t_s[0]] = np.nan
                max_i_t_s = np.argsort(-abs_vh_circ_i_t_i_h, axis=0)
                circ_frac = circ_frac - 1 / frac_denominator

            min_i_t_s = np.argsort(abs_vh_circ_i_t_i_h, axis=0)
            while abs_vh_circ_i_t_i_h[min_i_t_s[1]] - \
                    abs_vh_circ_i_t_i_h[min_i_t_s[0]] > max_out and \
                    sum(~np.isnan(abs_vh_circ_i_t_i_h)) > 2:
                abs_vh_circ_i_t_i_h[min_i_t_s[0]] = np.nan
                min_i_t_s = np.argsort(abs_vh_circ_i_t_i_h, axis=0)
                circ_frac = circ_frac - 1 / frac_denominator

            if circ_frac <= circulations_fraction:
                wn.warn('Less than ' +
                str(int(np.round(circulations_fraction * 100))) +
                        '% useful circulation retrievals in time window')
                continue

            max_i_t = max_i_t_s[0]
            min_i_t = min_i_t_s[0]
            for var in ['wspeed', 'sd_wspeed', 'u', 'v', 'w', 'wdir']:
                nc_lidar_new[var + '_max'][i_t, i_h] = \
                    nc_lidar_circ.variables[var][
                        i_t_circ_window, i_h][
                        max_i_t]
                nc_lidar_new[var + '_min'][i_t, i_h] = \
                    nc_lidar_circ.variables[var][
                        i_t_circ_window, i_h][
                        min_i_t]

            nc_lidar_new['covariances_max'][i_t, i_h, :, :] = \
                nc_lidar_circ.variables['covariances'][
                i_t_circ_window, i_h, :, :][
                max_i_t, :, :]
            nc_lidar_new['covariances_min'][i_t, i_h, :, :] = \
                nc_lidar_circ.variables['covariances'][
                i_t_circ_window, i_h, :, :][
                min_i_t, :, :]

        print('Process evolution:',
              str(np.round((i_t + 1) / nc_lidar_new.variables['time'].size
                           * 100)), ' % for ', nc_file_in_new)

    nc_lidar_circ.close()
    nc_lidar_mean.close()
    nc_lidar_new.close()
    system('mv ' + path_folder_out + 'tmp_' + nc_file_in_new + ' ' +
           path_folder_out + nc_file_in_new)

###############################################################################
# STEP D: w-correction                                                        #
#                                                                             #
# w-values from the CSM have an turn-direction-dependent offset, likely the   #
# velocity of the fast rotation DWL-head, which needs to be corrected for.    #
###############################################################################

def w_correction(nc_file_in, path_folder_in, path_folder_out, check_exist=True,
                 fh_correction=0.135, hh_correction=-0.135):
    """ Produce netcdf-files with corrected w-values

    Args:
      nc_file_in: Filename of nc-file with lidar wind level 2 data located
                  in path_folder_in.
      path_folder_in: The path to folder containing the lidar data file
                      nc_file_in.
      check_exist: If True then approach is canceled when output file is
                   already existing.
      fh_correction: Value to add to w-values in the 30 min after the full
                     hour (fh), i.e., XX:00 UTC to XX:30 UTC.
      hh_correction: Value to add to w-values in the 30 min after the half
                     hour (hh), i.e., XX:30 UTC to XY:00 UTC.
    Returns:
    """

    if not isfile(path_folder_in + nc_file_in):
        wn.warn(nc_file_in + ' does not exist where you think it is')
        return

    nc_file_out = nc_name_raise_version(nc_file_in)
    if not isdir(path_folder_out):
        try:
            makedirs(path_folder_out)
        except:
            wn.warn(path_folder_out + ' simultaniously created!')

    if check_exist:
        if isfile(path_folder_out + nc_file_out):
            print('Consider: ' + nc_file_out +
                  ' is existing and not again calculated. Please set '
                  '>check_exist< to FALSE, if recalculation is needed')
            return

    print(nc_file_in + ' to process for w-correction')
    system('cp ' + path_folder_in + nc_file_in + ' ' + path_folder_out +
           'tmp_' + nc_file_out)
    nc_lidar = nc.Dataset(path_folder_out + 'tmp_' + nc_file_out, 'r+')
    t_nc = nc_lidar['time'][:].data
    t = [dt.datetime.utcfromtimestamp(ts) for ts in t_nc]
    t_min = np.array([t[i].minute for i in range(len(t))])
    i_fh = (t_min < 30)
    w_nc = nc_lidar['w'][:].data
    w_nc[i_fh, :] = w_nc[i_fh, :] + fh_correction
    w_nc[~i_fh, :] = w_nc[~i_fh, :] + hh_correction
    w_nc[w_nc < -990] = -999.0
    nc_lidar['w'][:] = w_nc
    try:
        w_nc = nc_lidar['w_max'][:].data
        w_nc[i_fh, :] = w_nc[i_fh, :] + fh_correction
        w_nc[~i_fh, :] = w_nc[~i_fh, :] + hh_correction
        w_nc[w_nc < -990] = -999.0
        nc_lidar['w_max'][:] = w_nc
        w_nc = nc_lidar['w_min'][:].data
        w_nc[i_fh, :] = w_nc[i_fh, :] + fh_correction
        w_nc[~i_fh, :] = w_nc[~i_fh, :] + hh_correction
        w_nc[w_nc < -990] = -999.0
        nc_lidar['w_min'][:] = w_nc
    except:
            wn.warn(nc_file_in + ' has no w_max and w_min')


    nc_lidar.Comments = nc_lidar.Comments + '; halfhourly w-correction is ' \
                        'done with ' + str(fh_correction) + ' and ' \
                        + str(hh_correction) + ' m/s.'
    nc_lidar.close()
    system('mv ' + path_folder_out + 'tmp_' + nc_file_out + ' ' +
           path_folder_out + nc_file_out)

###############################################################################
# STEP E: Quicklooks                                                          #
#                                                                             #
# Plot routines for lidar-netcdf's l2 files which contains horizontal wind    #
# information (u, v) or wind gust information (if gust=True, u_gust_peak,     #
# v_gust_peak).                                                               #
###############################################################################


def lidar_quicklook(nc_file_in, path_folder_in, path_folder_out,
                    name_prefix='quicklook_', gust=False, h_mask=True,
                    omit_lowest=True, str_out='png', arrowsize=6,
                    top_height=np.nan, top_wind=25,
                    check_exist=True, width=21, height=14, bar=True):
    """ Produce Quicklook-plot of lidar netcf level2 file

    Args:
      nc_file_in: Filename of nc-file with the lidar level 2 data located in
                  path_folder_in.
      path_folder_in: The path to folder containing the lidar data nc-file_in.
      path_folder_out: Path to folder where the quicklook file is saved.
      name_prefix: The name of the file will be
                   'name_prefix'_'ns_file_in'.'str_out'.
      gust: If True instead of u and v, u_gust_peak and v_gust_peak are plotted
            (if available).
      h_mask: If True only the heights of corg will be plotted.
      omit_lowest: oOmit the lowest corg, if True.
      str_out: Output string (png or pdf).
      arrowsize: Wind barbs size.
      top_height: If wanted the top plotted height can be assigned.
      top_wind: wind speed were all values above get same color.
      check_exist: If True then approach is canceled when output file is
                   already existing.
      width: Width of plot.
      height: Height of plot.
      bar: If True, a colour-bar bar is added to the left.
    Returns:
    """

    try:
        if not nc_name_check_level(nc_file_in) == 2:
            wn.warn(nc_file_in + ' is not of level 2')
            return

        if check_exist:
            if isfile(path_folder_out + name_prefix +
                      nc_file_in[:-3] + '.' + str_out):
                print('Consider: ' + name_prefix +
                      nc_file_in[:-3] + '.' + str_out +
                      ' is existing and not again calculated. Please set '
                      '>check_exist< to FALSE, if recalculation is needed')
                return
        print('Start: ' + name_prefix +
              nc_file_in[:-3] + '.' + str_out)
        nc_lidar = nc.Dataset(path_folder_in + nc_file_in)
        t = nc_lidar['time'][:].data
        t = [datetime.datetime.utcfromtimestamp(ts) for ts in t]
        h = nc_lidar['height'][:].data
        if gust:
            u = nc_lidar['u_max'][:].data
            v = nc_lidar['v_max'][:].data
        else:
            u = nc_lidar['u'][:].data
            v = nc_lidar['v'][:].data
        if h_mask:
            u = u[:, nc_lidar['height_corg'][:].data == 1]
            v = v[:, nc_lidar['height_corg'][:].data == 1]
            h = h[nc_lidar['height_corg'][:].data == 1]

        if omit_lowest:
            u = u[:, 1:]
            v = v[:, 1:]
            h = h[1:]

    except:
        print('Error in quicklook of ' + nc_file_in)
        wn.warn(
            'No valid input file (' + nc_file_in + ') or no gusts available')
        return

    u[np.where(u == -999.)] = np.nan
    v[np.where(v == -999.)] = np.nan
    t_2D = np.repeat(np.array(t)[None, :], u.shape[1], axis=0).transpose()
    h_2D = np.repeat(np.array(h)[None, :], u.shape[0], axis=0)
    abs_vh = np.sqrt(u ** 2 + v ** 2)
    fontsize = 30
    ticksize = fontsize * 3 / 4
    cmap = plt.cm.jet
    bounds = [i for i in range(top_wind + 2)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.barbs(t_2D, h_2D, u, v, abs_vh, length=arrowsize, cmap=cmap,
                    pivot='middle', norm=norm)
    label = 'wind speed [m/s]'
    if gust:
        label = 'wind gust peak [m/s]'

    if bar:
        cbar = plt.colorbar(img, boundaries=bounds,
                            ticks=[i for i in range(0, top_wind + 2, 5)],
                            extend='max',
                            label=label)
        cbar.ax.tick_params(labelsize=ticksize)
        cbar.set_label(label, fontsize=fontsize)

    # xy-axis
    plt.xlabel('UTC [mm-dd hh]', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.ylabel('height [m]', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    today = datetime.datetime(t[int((t.__len__()) / 2)].year,
                              t[int((t.__len__()) / 2)].month,
                              t[int((t.__len__()) / 2)].day)
    tomorrow = today + datetime.timedelta(days=1)
    if np.isnan(top_height):
        top_height = max(plt.axis.__call__()[3], 2010)

    plt.axis([today, tomorrow, -1, top_height])
    plt.gcf().set_size_inches([width, height])
    nc_lidar.close()
    if not isdir(path_folder_out):
        try:
            makedirs(path_folder_out)
        except:
            wn.warn(path_folder_out + ' simultaniously created!')

    plt.savefig(
        path_folder_out + name_prefix + nc_file_in[:-3] + '.' + str_out,
        bbox_inches='tight')
    plt.close()
    print(name_prefix + nc_file_in[:-3] + '.' + str_out + ' done!')
