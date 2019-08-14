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
#files = sorted( fun.list_files(dirc_name, 'h5') )

files = np.arange(201, 231)

for f in files:
    file = 'Dipole_Field_Sim_{}.h5'.format(f)
    print(file)
    
    meta_file_name = file[0:17]+'Meta_'+file[17:-2]+'txt'
    meta_file = open(dirc_name+meta_file_name, 'r')
    line = meta_file.readlines()
    meta_file.close()
    
    mag_locate_str = line[3]
    mag_locate = np.array([0, 0, float(mag_locate_str[23:-5]) ])
    var.dipole_position = mag_locate
    
    mag_orient_str = line[4]
    mag_orient_str = mag_orient_str[11:-7].split(', ')
    mag_orient = np.array([float(mag_orient_str[0]), float(mag_orient_str[1]), float(mag_orient_str[2])])
    var.dipole_moment = mag_orient
    
    data = h5.File(dirc_name+file, 'r')
    data_set = data['data'][:]
    data.close()
    
    size = len(data_set)
    
    B_Norm_Log = []
    B_Dot = []
    E_Dot = []
    
    for i in range(size):
        sim = data_set[i]
    
        r = sim[0:3]
        r_unit = r / np.linalg.norm(r)
        
        e = sim[3:6]
        h = sim[6]
        
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
        
    data_new = h5.File(dirc_name+'Sorted_Pram/'+file[0:-3]+'_PramSort.h5', 'w')
    
    data_new.create_dataset('Field Magnitude', data=B_Norm_Log)
    data_new.create_dataset('Field Orientation', data=B_Dot)
    data_new.create_dataset('Grain Orientation', data=E_Dot)
    
    data_new.close()
    
