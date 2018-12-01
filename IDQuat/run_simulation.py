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

def initialize_grains(num_of_grains, num_of_procs, queues):
    for i in range(num_of_grains):
        grain = tDG.dust_grain()
    
        q_key = i % num_of_procs
        queue_key = 'queue_init_{}'.format(q_key+1)
            
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
    alpha = 2*np.pi*np.random.random()
    theta = np.pi*np.random.random()
    vector = np.array([np.sin(theta)*np.cos(alpha),
                       np.sin(theta)*np.sin(alpha),
                       np.cos(theta)])
    var.mag = var.mag_mom*vector
    
    var.fileNames[0] = '{0}_{1}.h5'.format(var.name, var.run_num)
    var.fileNames[1] = '{0}_Meta_{1}.txt'.format(var.name, var.run_num)
    
    s_queue = mp.JoinableQueue(var.Im)
    f_queue = mp.JoinableQueue(var.Im)
    
    save_thread = threading.Thread(target=anal.save_data, args=(s_queue, var.fileNames[0],))
    fail_thread = threading.Thread(target=anal.save_meta, args=(f_queue, var.fileNames[1],))
    
    save_thread.start()
    fail_thread.start()
    
    while True:
        start_time = time.time()
        
        queue_init = {}
        
        for i in range(var.nproc):
            queue_key = 'queue_init_{}'.format(i+1)
            queue_init[queue_key] = mp.JoinableQueue(var.mult)
        
        proc = {}
        
        proc['proc_init'] = threading.Thread(target=initialize_grains, args=(var.Im, var.nproc, queue_init,))
        proc['proc_init'].start()
        
        for i in range(var.nproc):
            proc_key = 'proc_{}'.format(i+1)
            queue_key = 'queue_init_{}'.format(i+1)
            
            proc[proc_key] = mp.Process(target=track_dust_grains, args=(var.mult, queue_init[queue_key], s_queue, f_queue,))
            proc[proc_key].start()
        
        proc['proc_init'].join()

        for i in range(var.nproc):
            key = 'proc_{}'.format(i+1)
            proc[key].join()
        
        run_time = time.time() - start_time 
        
        var.rate.append( round(var.Im/run_time,2) )
        rate_avg = round( np.mean(var.rate), 2 )
        rate_std = round( np.std(var.rate), 2 )
        print "\nCurrent rate : {} part/sec (+/-) {}".format( rate_avg, rate_std )
        
        counts = anal.get_counts(var.fileNames[1])
        num_grains = counts['success']
        if num_grains >= var.data_cap:
            var.run_num+=1
            
            s_qSize = s_queue.qsize()
            f_qSize = f_queue.qsize()
            while (s_qSize != 0) and (f_qSize != 0):
                s_qSize = s_queue.qsize()
                f_qSize = f_queue.qsize()
            
            del save_thread
            del fail_thread
            
            del s_queue
            del f_queue
            
            break
    
    
