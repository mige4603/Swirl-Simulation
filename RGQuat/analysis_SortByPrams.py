#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:11:40 2019

@author: michael
"""

import functions as fun
import global_var as gVar

import os
import h5py as h5
import numpy as np


path = os.path.join(os.getcwd(),'monteCarlo_dipoles')
for pathDip in [f.name for f in os.scandir(path) if f.is_dir()]:
    print('Working in '+pathDip)
    path_dip = os.path.join(path, pathDip)
            
    meta_path = os.path.join(path_dip,'input.namelist')
    data_path = os.path.join(path_dip, 'data.h5')
    sort_path = os.path.join(path_dip, 'data_sort_prams.h5')
    
    if os.path.isfile(sort_path):
        print('   '+sort_path+' already exists!')
        continue
    
    params = gVar.variables(meta_path)

    with h5.File(data_path,'r') as hf_file:
        data_set = hf_file['data'][:]
    
    size = data_set.shape[0]
    
    B_Norm_Log = []
    B_Dot = []
    E_Dot = []
    
    for i in range(size):
        sim = data_set[i]
    
        r = sim[0:3]
        r_unit = r / np.linalg.norm(r)
        
        e = sim[3:6]
        h = sim[6]
        
        B = fun.B_field(r, params)
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
    
    with h5.File(sort_path,'w') as hf_file:    
        hf_file.create_dataset('Field Magnitude', data=B_Norm_Log)
        hf_file.create_dataset('Field Orientation', data=B_Dot)
        hf_file.create_dataset('Grain Orientation', data=E_Dot)
