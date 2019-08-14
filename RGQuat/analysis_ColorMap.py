#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:27:56 2019

@author: michael
"""

import functions as fun
import global_var as var

import h5py as h5
import numpy as np

dirc_name = var.sim_path+'Sorted_Pram/'
viz_name = var.sim_path+'Visuals/'
files = sorted( fun.list_files(dirc_name, 'h5') )

B_Norm_Bin = 1001
B_Dot_Bin = 1001
N_bin = B_Norm_Bin*B_Dot_Bin

mag_min = -9
mag_max = -2
B_Norm_Log_Space = np.linspace(mag_min, mag_max, B_Norm_Bin)
norm_spc = (B_Norm_Log_Space[1] - B_Norm_Log_Space[0])

B_Dot_Space = np.linspace(-1, 1, B_Dot_Bin)
dot_spc = B_Dot_Space[1] - B_Dot_Space[0]

print( '\nSorting E_Dot' )
E_Dot_Space = {}

for file in files:
    print(file)

    data = h5.File(dirc_name+file, 'r')
    B_Norm_Log = data['Field Magnitude'][:]
    B_Dot = data['Field Orientation'][:]
    E_Dot = data['Grain Orientation'][:]
    data.close()
    
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
file = open('{0}RGQuat_ColorMap_Bin_{1}.vtk'.format(var.sim_path, B_Norm_Bin), 'w')
file.write('# vtk DataFile Version 1.0\n'
           'B Field from Parsek\nASCII\n'
           'DATASET STRUCTURED_POINTS\n'
           'DIMENSIONS {0} {0} 1\n'
           'ORIGIN {1} -1 0\n'
           'SPACING {2} {3} 0.1\n'
           'POINT_DATA {4}\n'
           'VECTORS B float\n'.format(B_Norm_Bin, mag_min, norm_spc, dot_spc, N_bin))
for y in range(B_Dot_Bin):
    for x in range(B_Norm_Bin):
        key = '{}-{}'.format(x, y)
        if key in E_Dot_Space:
            file.write('{} {} {}\n'.format(E_Dot_Space[key], 0, 0))
        else:
            file.write('{} {} {}\n'.format(0, 0, 0))
file.close() 
