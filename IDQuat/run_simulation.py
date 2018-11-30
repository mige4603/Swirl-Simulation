#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 00:36:07 2018

@author: michael
"""

import global_var as var
import analysis as anal
import track_DustGrains as tDG

import threading
import multiprocessing as mp
import time
import numpy as np

track = int(var.data_cap_inc/var.Im)
rate = []
run_num = 15

while True:
    alpha = 2*np.pi*np.random.random()
    theta = np.pi*np.random.random()
    vector = np.array([np.sin(theta)*np.cos(alpha),
                       np.sin(theta)*np.sin(alpha),
                       np.cos(theta)])
    var.mag = var.mag_mom*vector
    
    var.fileNames[0] = '{0}_{1}.h5'.format(var.shared_Name, run_num)
    var.fileNames[1] = '{0}_Meta_{1}.txt'.format(var.shared_Name, run_num)
    
    succ_queue = mp.JoinableQueue(var.Im)
    fail_queue = mp.JoinableQueue(var.Im)
    
    save_thread = threading.Thread(target=anal.save_data, args=(succ_queue, var.fileNames[0], ))
    fail_thread = threading.Thread(target=anal.save_meta, args=(fail_queue, var.fileNames[1],))
    
    save_thread.start()
    fail_thread.start()
    
    while True:
        start_time = time.time()
        
        proc = {}
        for i in range(var.nproc):
            proc_key = 'proc_{}'.format(i+1)
            
            proc[proc_key] = mp.Process(target=tDG.tracking_DustGrains, args=(var.mult, succ_queue, fail_queue))
            proc[proc_key].start()
            
        for i in range(var.nproc):
            key = 'proc_{}'.format(i+1)
            proc[key].join()
        
        run_time = time.time() - start_time 
        
        rate.append( round(var.Im/run_time,2) )
        
        rate_avg = round( np.mean(rate), 2 )
        rate_std = round( np.std(rate), 2 )
        print "\nCurrent rate : {} part/sec (+/-) {}".format( rate_avg, rate_std )
        
        counts = anal.get_counts(var.fileNames[1])
        num_grains = counts[3]-counts[4]
        if num_grains >= var.data_cap:
            run_num+=1
            
            del save_thread
            del fail_thread
            
            del succ_queue
            del fail_queue
            
            break
    
    
