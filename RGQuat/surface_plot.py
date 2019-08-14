#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:38:11 2019

@author: michael
"""

import global_var as var
import functions as fun

import numpy as np

fileName = '/home/michael/Documents/LASP/Swirl-Simulation/RGQuat/Field_Data/Visuals/E_Surface_Field.vtk'

domain = np.linspace(-var.Dia, var.Dia, var.X_bin)

file = open(fileName, 'w')
file.write('# vtk DataFile Version 1.0\n'
           'B Field from Parsek\nASCII\n'
           'DATASET STRUCTURED_POINTS\n'
           'DIMENSIONS {0} {0} 1\n'
           'ORIGIN 0 0 0\n'
           'SPACING {1} {1} {1}\n'
           'POINT_DATA {2}\n'
           'VECTORS B float\n'.format(var.X_bin, var.spc, var.N_bin))

x_vec = np.array([1,0,0])
for y in domain:
    print( 'Y = %s' % str(round(y)) )
    for x in domain:
        z = np.sqrt(var.r_m**2 - (x**2 + y**2))
        r = np.array([x,y,z])
        r_unit = r / np.linalg.norm(r)
        
        x_Tan = x_vec - np.dot(x_vec, r_unit) * r_unit
        x_tan = x_Tan / np.linalg.norm(x_Tan)
        y_tan = np.cross(r_unit, x_tan)
        
        B = fun.E_field(r)
        B_normal = np.dot(B, r_unit)*r_unit
        B_tangent = B - B_normal
        
        B_tan_x = np.dot(B, x_tan)
        B_tan_y = np.dot(B, y_tan)
        file.write('{0} {1} {2}\n'.format(B_tan_x, B_tan_y, 0.0))
    
file.close()        