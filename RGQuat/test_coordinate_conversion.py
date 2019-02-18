#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:02:44 2019

@author: michael
"""

import functions as fun
import global_var as var
import numpy as np

xPts = 501
yPts = 501
nPts = xPts * yPts

x_dom = np.linspace(-var.Dia, var.Dia, xPts)
y_dom = np.linspace(-var.Dia, var.Dia, yPts)

file = open(var.field_path+'B_Surface_Dipole_Field.vtk', 'w')
file.write('# vtk DataFile Version 1.0\n'
           'B Field from Parsek\nASCII\n'
           'DATASET STRUCTURED_POINTS\n'
           'DIMENSIONS {0} {1} 1\n'
           'ORIGIN 0 0 0\n'
           'SPACING {2} {2} {2}\n'
           'POINT_DATA {3}\n'
           'VECTORS B float\n'.format(xPts, yPts, 0.01, nPts))

for ind_y, y in enumerate(y_dom):
    print 'Y : %s' % str(ind_y)
    for x in x_dom:
        z = np.sqrt(var.r_m**2 - (x**2 + y**2))
        r = np.array([x,y,z])
        
        r_unit = r/np.linalg.norm(r)
        
        B = fun.B_field(r)
        B_surf =  B - np.dot(B, r_unit)*r_unit
        
        file.write('{0} {1} {2}\n'.format(B_surf[0], B_surf[1], B_surf[2]))
        
file.close()
'''
SW_r, SW_mag = fun.dipole_moment_conversion(7.5, 58.4, 1.3, -11., 11100, 11.3*10**12)
NE_r, NE_mag = fun.dipole_moment_conversion(10.9, 55.3, 3.7, -68.7, 6700, 3.5*10**12)  
print NE_r
print NE_mag   
'''  