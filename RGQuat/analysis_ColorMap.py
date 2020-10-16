#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:27:56 2019

@author: michael
"""

import functions as fun

import os, sys
import h5py as h5
import numpy as np


B_Norm_Bin = int(sys.argv[1])
B_Dot_Bin = int(sys.argv[2])
N_bin = B_Norm_Bin*B_Dot_Bin

save_name = os.path.join(os.getcwd(),'monteCarlo_dipoles','field_map_{0}-{1}.vtk'.format(B_Norm_Bin, B_Dot_Bin))

path = os.path.join(os.getcwd(),'monteCarlo_dipoles')
dipoles = [f.name for f in os.scandir(path) if f.is_dir()]

mag_min = -9
mag_max = -2
B_Norm_Log_Space = np.linspace(mag_min, mag_max, B_Norm_Bin)
norm_spc = (B_Norm_Log_Space[1] - B_Norm_Log_Space[0])

B_Dot_Space = np.linspace(-1, 1, B_Dot_Bin)
dot_spc = B_Dot_Space[1] - B_Dot_Space[0]

E_Dot_Space = {}
for dipole in dipoles:
    print('Working in '+dipole)
    path_dip = os.path.join(path,dipole,'data_sort_prams.h5')

    with h5.File(path_dip, 'r') as hf_file:
        B_Norm_Log = hf_file['Field Magnitude'][:]
        B_Dot = hf_file['Field Orientation'][:]
        E_Dot = hf_file['Grain Orientation'][:]
    
    numpts = len(E_Dot)
    
    for i in range(numpts):
        avg = E_Dot[i]
        
        norm = B_Norm_Log[i]
        norm_idx = fun.find_nearest(B_Norm_Log_Space, norm)
        
        dot = B_Dot[i]
        dot_idx = fun.find_nearest(B_Dot_Space, dot)
        
        key = '{}-{}'.format(norm_idx, dot_idx)
        if key in E_Dot_Space:
            E_Dot_Space[key].append(avg)
        else:
            E_Dot_Space[key] = [avg]
    
for key in E_Dot_Space:
    E_Dot_Space[key] = np.mean(E_Dot_Space[key])

print('\nWrite VTK file')
file = open(save_name, 'w')
file.write('# vtk DataFile Version 1.0\n'
           'B Field from Parsek\nASCII\n'
           'DATASET STRUCTURED_POINTS\n'
           'DIMENSIONS {0} {1} 1\n'
           'ORIGIN {2} -1 0\n'
           'SPACING {3} {4} 0.1\n'
           'POINT_DATA {5}\n'
           'VECTORS B float\n'.format(B_Norm_Bin, B_Dot_Bin, mag_min, norm_spc, dot_spc, N_bin))
for y in range(B_Dot_Bin):
    for x in range(B_Norm_Bin):
        key = '{}-{}'.format(x, y)
        if key in E_Dot_Space and not np.isnan(E_Dot_Space[key]):
            file.write('{} {} {}\n'.format(E_Dot_Space[key], 0, 0))
        else:
            file.write('{} {} {}\n'.format(0, 0, 0))
file.close() 
