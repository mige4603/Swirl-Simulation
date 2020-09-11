# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:38:54 2020

@author: micha
"""

#import time
import numpy as np
import track_DustGrains as tDG

import multiprocessing as mp

def run_risingPhase(queue_in, queue_out):
    for que in queue_in:
        grain = que.get()
        grain.rising_phase()
        queue_out.put(grain)
   
def run_fallingPhase(queue_in, que_chk):    
    while que_chk.qsize() > 0 or queue_in.qsize() != 0:
        grain = queue_in.get()
        if grain.grav.successful():
            grain.falling_phase()
            if grain.grav.successful():
                grain.impacting_phase()
                if grain.grav.successful():
                    grain.colliding_phase()
                    print(grain.sim_results)
                else:
                    print('Impact')
            else:
                print('Falling')
        else:
            print('Rising')
            
def main():            
    nGrains = 1
    
    grain_inCon = np.empty((nGrains, 14))
    grain_finCon = np.empty((nGrains, 7))
    grain_finCon[:] = np.nan
    
    rising_que = mp.Queue()
    falling_que = mp.Queue()
    
    for i in range(nGrains):
        grain = tDG.dust_grain()
        grain_inCon[i] = grain.InCon
        rising_que.put(grain)
    
    rising_proc = mp.Process(target=run_risingPhase, args=(rising_que, falling_que,))
    rising_proc.start()
    grain = falling_que.get()
    
    rising_proc.join()
    print(grain.grav)

if __name__=='__main__':
    main()


#print(falling_que.get())
#falling_proc = mp.Process(target=run_fallingPhase, args=(falling_que, rising_que,))

#rising_proc.start()
#falling_proc.start()

#rising_proc.join()
#falling_proc.join()
        
#grain.track_phases()
#results = grain.sim_results