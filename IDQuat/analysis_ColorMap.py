#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:56:49 2018

@author: michael
"""

import functions as fun
import global_var as var

import h5py as h5
import numpy as np

dirc_name = var.path+'Random/Condensed_DATA/'
file_name = dirc_name+'IDQuat_Random.h5'
data = h5.File(file_name, 'r')

mag_set = data['dipole']
for key in data:
    print key
'''
numpts = len(mag_set)

B_Norm_Log = np.array([])
B_Dot = np.array([])
E_Dot = np.array([])
for i in range(numpts):
    var.mag = mag_set[i]
    
    set_key = 'set %s' % i
    sim_set = data[set_key]
    npts = len(sim_set)
    
    B_Norm_Log = np.append(B_Norm_Log, np.zeros(npts-1))
    B_Dot = np.append(B_Dot, np.zeros(npts-1))
    E_Dot = np.append(E_Dot, np.zeros(npts-1))
    for j in range(1, npts):
        sim = sim_set[j]
        
        if all(sim!=0):
            r = sim[0:3]
            e = sim[3:6]
            h = sim[6]
            
            B = fun.B_field(r)
            B_norm = np.linalg.norm(B)
            B_norm_log = np.log10(B_norm)
            
            B_dot = B[2]/B_norm
            
            B[2] = 0
            B = B/np.linalg.norm(B)
            e[2] = 0
            e = e/np.linalg.norm(e)
            E_dot = np.dot(e, B)
            
            B_Norm_Log[j-1] = B_norm_log
            B_Dot[j-1] = B_dot
            E_Dot[j-1] = E_dot
            

numpts = len(E_Dot)

B_Norm_Bin = 501
B_Dot_Bin = 501
N_bin = B_Norm_Bin*B_Dot_Bin

mag_min = -7.05799192094
mag_max = -0.756961960822
B_Norm_Log_Space = np.linspace(mag_min, mag_max, B_Norm_Bin)
B_Dot_Space = np.linspace(-1, 1, B_Dot_Bin)

E_Dot_Space = {}
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
file = open('{}/IDQuat_ColorMap.vtk'.format(dirc_name), 'w')
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
'''