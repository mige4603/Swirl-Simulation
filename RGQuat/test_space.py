# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:38:54 2020

@author: micha
"""

#import time
import numpy as np
import track_DustGrains as tDG

nGrains = 100

grain_inCon = np.empty((nGrains, 14))
grain_finCon = np.empty((nGrains, 7))
grain_finCon[:] = np.nan

for i in range(nGrains):
    grain = tDG.dust_grain()
    grain_inCon[i] = grain.InCon
    grain.track_phases()
    results = grain.sim_results
    if results.all():
        grain_finCon[i] = results
    
#grain.track_phases()
#results = grain.sim_results