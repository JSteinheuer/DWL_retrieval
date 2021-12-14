# DWL-retrieval
This repository contains the code to calculate 10 minutes mean wind and wind  gust peaks from daily Doppler wind lidar files (netcdfs in SAMD convention)  as described in

**Steinheuer, Detring, Beyrich, Löhnert, Friederichs, and Fiedler (2021), A new scanning scheme and flexible retrieval to derive both mean winds and gusts from Doppler lidar measurements, Atmos. Meas. Tech. DOI: TBD**


**How to use**:
Download the data set related to the paper. Run the _main.py_ to reproduce the level2 gust files or modify for other lidar data. The functions that calculates the outcomes are contained in _DWL_retrieval.py_

On example day is given:
- The 29th September 2019 in a 24Beam conviguration (due to size reasons). _/data_ provides the level1 data and main_testday calculates level2 gust (and 600s, and circ) outcomes for the day

The repository includes the dependencies in "requirements.txt".

**Additional features**:
A simple quicklook plotting routine with is provided in _DWL_retrieval.py_



This program is a free software distributed under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 (GNU-GPLv3). You can redistribute and/or modify by citing the mentioned publication, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
