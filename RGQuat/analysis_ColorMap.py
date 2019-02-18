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

dirc_name = var.sim_path
file_name = dirc_name+'RGQuat_Loft_PramData.h5'

data = h5.File(file_name, 'r')
B_Norm_Log = data['Field Magnitude'][:]
B_Dot = data['Field Orientation'][:]
E_Dot = data['Grain Orientation'][:]
data.close()

numpts = int( .5* len(E_Dot) )

B_Norm_Bin = 501
B_Dot_Bin = 501
N_bin = B_Norm_Bin*B_Dot_Bin

mag_min = np.min(B_Norm_Log)
mag_max = np.max(B_Norm_Log)
B_Norm_Log_Space = np.linspace(mag_min, mag_max, B_Norm_Bin)
B_Dot_Space = np.linspace(-1, 1, B_Dot_Bin)

print '\nSorting E_Dot'
E_Dot_Space = {}

cnt = 1
trk = round(.01*numpts)
pct = round((trk * 100) / numpts)
for i in range(numpts):

    if i == cnt * trk:
        print '{} % Complete'.format(pct*cnt)
        cnt+=1
    
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
file = open('{}IDQuat_ColorMap_halfData.vtk'.format(dirc_name+'Visuals/'), 'w')
file.write('# vtk DataFile Version 1.0\n'
           'B Field from Parsek\nASCII\n'
           'DATASET STRUCTURED_POINTS\n'
           'DIMENSIONS {0} {0} 1\n'
           'ORIGIN 0 0 0\n'
           'SPACING {1} {1} {1}\n'
           'POINT_DATA {2}\n'
           'VECTORS B float\n'.format(B_Norm_Bin, 0.011976, N_bin))
for y in range(B_Dot_Bin):
    for x in range(B_Norm_Bin):
        key = '{}-{}'.format(x, y)
        if key in E_Dot_Space:
            file.write('{} {} {}\n'.format(E_Dot_Space[key], 0, 0))
        else:
            file.write('{} {} {}\n'.format(0, 0, 0))
file.close() 
