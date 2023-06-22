# DWL-retrieval
This repository contains the code to calculate 10 minutes mean wind and wind 
gust peaks from daily Doppler wind lidar files (netcdfs in SAMD convention) 
as described in

**Steinheuer, Detring, Beyrich, Löhnert, Friederichs, and Fiedler (2022):
A new scanning scheme and flexible retrieval to derive both mean winds and
gusts from Doppler lidar measurements, Atmos. Meas. Tech
DOI: https://doi.org/10.5194/amt-15-3243-2022**

AND:

**Steinheuer, Vertical wind gust profiles, Dissertation, University of Cologne,
URL: https://kups.ub.uni-koeln.de/65655/**


**How to use**:
Download the data set related to the paper. Run the _main.py_ to reproduce the 
level2 gust files or modify for other lidar data. The functions that calculate
the outcomes are contained in _DWL_retrieval.py_

Two example days are given:
- The 29th September 2019 in a 24Beam configuration (due to size reasons). 
  _/data_ provides the level1 data and main_testday calculates level2 gust 
  (and 600s, and circ) outcomes for the day
- The 29th June 2021 in quck continuous scanning mode (too big for git, so 
  please download here: https://doi.org/10.25592/uhhfdm.11227, i.e., the example is: https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/fesstval/wind_and_gust/falkenberg_dlidcsm/level1/csm02/2021/catalog.html?dataset=ftpthreddsscan/fesstval/wind_and_gust/falkenberg_dlidcsm/level1/csm02/2021/sups_rao_dlidCSM02_l1_any_v00_20210629.nc). 
  _/data_ provides the level1 data and main_testday_qCSM calculates level2 gust 
  (and 600s, and circ) outcomes for the day with included correction of w

The repository includes the dependencies in "requirements.txt".

**Additional features**:
A simple quicklook plotting routine with is provided in _DWL_retrieval.py_

_version: 1.1_

_June 22, 2023_

This program is a free software distributed under the terms of the GNU General 
Public License as published by the Free Software Foundation, version 3 
(GNU-GPLv3). You can redistribute and/or modify by citing the mentioned 
publication, but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
