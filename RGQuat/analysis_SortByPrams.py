#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:11:40 2019

@author: michael
"""

import functions as fun
import global_var as var

import h5py as h5
import numpy as np

dirc_name = var.sim_path
file_name = dirc_name+'RGQuat_Loft.h5'
file_name_new = dirc_name+'RGQuat_Loft_PramData.h5'

data = h5.File(file_name, 'r')

data_set = data['data'][:]
numpts = len(data_set)

print '\nSort Parameters'
B_Norm_Log = []
B_Dot = []
E_Dot = []

cnt = 1
trk = round(.01*numpts)
pct = (trk * 100) / numpts
for i in range(numpts):
    
    if i == cnt*trk:
        print '{} % Complete'.format(pct*cnt)
        cnt+=1
        
    sim = data_set[i]

    x, y = sim[0:2]
    z = np.sqrt(var.r_m**2 - (x**2 + y**2))
    r = np.array([x,y,z])
    r_unit = r / np.linalg.norm(r)
    
    e = sim[2:5]
    h = sim[5]
    
    B = fun.B_field(r)
    B_norm = np.linalg.norm(B)
    B_norm_log = np.log10(B_norm)
    
    B_DOT = np.dot(r_unit, B)
    B_dot = B_DOT / B_norm
    
    B = B - B_DOT * r_unit
    B = B/np.linalg.norm(B)
    
    e = e - np.dot(e, r_unit) * r_unit
    if not all (e==0):
        e = e/np.linalg.norm(e)
        
        E_dot = np.dot(e, B)
        
        B_Norm_Log.append(B_norm_log)
        B_Dot.append(B_dot)
        E_Dot.append(E_dot)
    
    else:
        continue
            
data.close()     

print '\nCreate New Dataset'
data_new = h5.File(file_name_new, 'w')

data_new.create_dataset('Field Magnitude', data=B_Norm_Log)
data_new.create_dataset('Field Orientation', data=B_Dot)
data_new.create_dataset('Grain Orientation', data=E_Dot)

data_new.close()