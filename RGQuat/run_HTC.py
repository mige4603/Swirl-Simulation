# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:38:54 2020

@author: micha
"""

import os
import sys
import time
import numpy as np
#import h5py as hf

import analysis as an
#import global_var as gVar
import track_DustGrains as tDG

import multiprocessing as mp

def run_sim(grain):
    grain.track_phases()
    return grain.sim_results, grain.counts

def main(num_of_procs, num_of_grains, namelist):
    ### Instantiate Grains ###
    grains = np.array([tDG.dust_grain(namelist) for i in range(num_of_grains)])
    
    ### Run Simulation ###
    time_beg = time.time()
    
    p = mp.Pool(num_of_procs)
    results = p.map(run_sim, grains)
    
    time_end = time.time()
    time_data = time_end - time_beg
    
    ### Perform Counts ###
    counts = {'success' : 0,
              'rise fail' : 0,
              'fall fail' : 0,
              'impact fail' : 0,
              'collide fail' : 0,
              'lift fail' : 0,
              'never fail' : 0}
    
    for idx, res in enumerate(results):
        for key in res[1]:
            counts[key] = counts[key] + res[1][key]
        print('{0} {1} {2} {3} {4} {5} {6}'.format(res[0][0,0],res[0][0,1],res[0][0,2],res[0][0,3],res[0][0,4],res[0][0,5],res[0][0,6]))

    an.create_meta(counts, os.path.join(os.getcwd(), 'meta_data.txt'), time_data, namelist)
    
if __name__=='__main__':    
    num_of_procs = int(sys.argv[1])
    num_of_grains = int(sys.argv[2])
    namelist = sys.argv[3]
    main(num_of_procs, num_of_grains, namelist)
