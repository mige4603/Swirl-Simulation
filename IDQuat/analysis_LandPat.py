#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:20:01 2018

@author: michael
"""

import analysis as anal

shared = raw_input('Directory: ')
file_data = raw_input('Data File: ')
file_land = raw_input('Landing Pattern File: ')
file_corr = raw_input('Correlation Pattern File: ')

vec_avg = anal.landing_pattern(shared+file_data, shared+file_land)
anal.correlation_pattern(vec_avg, shared+file_corr)