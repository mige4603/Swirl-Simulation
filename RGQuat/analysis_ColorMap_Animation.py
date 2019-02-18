#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:40:40 2019

@author: michael
"""

import functions as fun
import global_var as var

import h5py as h5
import numpy as np

dirc_name = var.sim_path
file_name = dirc_name+'RGQuat_Loft_PramData.h5'
anim_name = dirc_name+'Visuals/IDQuat_ColorMap_Animate/'

data = h5.File(file_name, 'r')
B_Norm_Log = data['Field Magnitude'][:]
B_Dot = data['Field Orientation'][:]
E_Dot = data['Grain Orientation'][:]
data.close()

num_frames = 1000
num_points =  len(E_Dot)

frames = np.linspace(0, num_points, num_frames, dtype=np.intp)

B_Norm_Bin = 501
B_Dot_Bin = 501
N_bin = B_Norm_Bin*B_Dot_Bin

mag_min = np.min(B_Norm_Log)
mag_max = np.max(B_Norm_Log)
B_Norm_Log_Space = np.linspace(mag_min, mag_max, B_Norm_Bin)
B_Dot_Space = np.linspace(-1, 1, B_Dot_Bin)

print '\nSorting E_Dot'
E_Dot_Space = {}
E_Dot_Space_Avg = {}
for i in range(num_frames-1):

    print '   frame {0} of {1}'.format(i+1, num_frames-1)
        
    ind_space = np.arange(frames[i], frames[i+1])
    for ind in ind_space:
        avg = E_Dot[ind]
        
        norm = B_Norm_Log[ind]
        norm_idx = fun.find_nearest(B_Norm_Log_Space, norm)
        
        dot = B_Dot[ind]
        dot_idx = fun.find_nearest(B_Dot_Space, dot)
        
        key = '{}-{}'.format(norm_idx, dot_idx)
        if key in E_Dot_Space:
            E_Dot_Space[key].append(avg)
        else:
            E_Dot_Space[key] = [avg]

    for key in E_Dot_Space:
        E_Dot_Space_Avg[key] = np.mean(E_Dot_Space[key])
        
    file = open('{0}frame_{1}.vtk'.format(anim_name, i+1), 'w')
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
            if key in E_Dot_Space_Avg:
                file.write('{} {} {}\n'.format(E_Dot_Space_Avg[key], 0, 0))
            else:
                file.write('{} {} {}\n'.format(0, 0, 0))
    file.close() 