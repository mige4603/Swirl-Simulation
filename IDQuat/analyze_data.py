#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:20:01 2018

@author: michael
"""

import analysis as anal
from global_var import fileNames

vec_avg = anal.landing_pattern(fileNames[0], fileNames[2])
anal.correlation_pattern(vec_avg, fileNames[3])