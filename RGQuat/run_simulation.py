#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:10:11 2019

@author: michael
"""
import global_var as var
import analysis as anal
import track_DustGrains as tDG

import threading
import multiprocessing as mp
import time
import numpy as np

def initialize_grains(num_of_grains, num_of_procs, queues):
    for i in range(num_of_grains):
        grain = tDG.dust_grain()
    
        q_key = i % num_of_procs
        queue_key = 'queue_{}'.format(q_key+1)
            
        queues[queue_key].put(grain)
        
def track_dust_grains(numpts, i_queue, s_queue, f_queue):
    for i in range(numpts):
        grain = i_queue.get()
        grain.track_phases()
        if not isinstance(grain.sim_results, (bool,)):
            s_queue.put(grain.sim_results)
        f_queue.put(grain.counts)
        i_queue.task_done()
    
while True:
    start_time = time.time()
    
    s_queue = mp.JoinableQueue(var.Im)
    f_queue = mp.JoinableQueue(var.Im)
    
    save_thread = threading.Thread(target=anal.save_data, args=(s_queue, var.fileNames[0],))
    fail_thread = threading.Thread(target=anal.save_meta, args=(f_queue, var.fileNames[1],))
    
    queues = {}
    
    for i in range(var.nproc):
        queue_key = 'queue_{}'.format(i+1)
        queues[queue_key] = mp.JoinableQueue(var.mult)
    
    procs = {}
    
    procs['proc_init'] = threading.Thread(target=initialize_grains, args=(var.Im, var.nproc, queues,))
    procs['proc_init'].start()
    
    for i in range(var.nproc):
        proc_key = 'proc_{}'.format(i+1)
        queue_key = 'queue_{}'.format(i+1)
        
        procs[proc_key] = mp.Process(target=track_dust_grains, args=(var.mult, queues[queue_key], s_queue, f_queue,))
        procs[proc_key].start()
    
    procs['proc_init'].join()
    
    save_thread.start()
    fail_thread.start()

    for i in range(var.nproc):
        key = 'proc_{}'.format(i+1)
        procs[key].join()
        
    s_queue.put(None)
    f_queue.put(None)
    
    save_thread.join()
    fail_thread.join()
    
    run_time = time.time() - start_time 
    
    var.rate.append( round(var.Im/run_time,2) )
    rate_avg = round( np.mean(var.rate), 2 )
    rate_std = round( np.std(var.rate), 2 )
    print "\tCurrent rate : {} part/sec (+/-) {}".format( rate_avg, rate_std )
    