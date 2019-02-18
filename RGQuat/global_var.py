#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:33:51 2019

@author: michael
"""

import numpy as np
import h5py as h5

import functions as fun

run_num = 0
rate = []

path = '/home/michael/Documents/LASP/Swirl-Simulation/RGQuat/'
sim_path = path+'Sim_Data/'
field_path = path+'Field_Data/'
name = sim_path+'Dipole_Field_Sim'

# Field Data for Reiner Gamma Swirl
'''
field_data = h5.File(field_path+'RG_Full_Field_Data.h5', 'r')
B_data = field_data['B_field']
E_data = field_data['E_field']
'''
SW_r = np.array([15055.59399353, 19522.97153838, 1725823.91415573])
SW_mag = np.array([-1.12822658e+13, -6.07455748e+11,  -1.77414475e+11])

NE_r = np.array([-87674.4142417, 124611.72907353, 1723678.99391525])
NE_mag = np.array([-1.30337601e+12, -3.24501798e+12, -1.45152530e+11])

# Gravity on Moon surface [m/s^2]
g = 1.62519    

# Radius of Moon [m]
r_m = 1.7371e6

# Charge on Dust Grain (number of electrons)
q_min = 100
q_max = 1000
q_pop = np.linspace(q_min, q_max, 1000)   

# Range of Landing Site -Dia to Dia [m]    
Dia = 325000
zettaMAX = np.arcsin(Dia/r_m)

# Density Range [g/m^3]
den_min = 2.33e6
den_max = 4.37e6
den_pop = np.linspace(den_min, den_max, 10000)
 
# Range of Grain Diameters [m]
h_min = 0.1e-6  
h_max = 10.0e-6 
h_pop = np.linspace(h_min, h_max, 1000)
    
# Range of Magnetic Moments in Grains [Am^2/g]
FeByWght_min = .00062
FeByWght_max = .0012
FeByWght_pop = np.linspace(FeByWght_min, FeByWght_max, 100)

m_mom_min = 686160*FeByWght_min*den_min*((4/3)*np.pi*((h_min/2.0)**3))
m_mom_max = 686160*FeByWght_max*den_max*((4/3)*np.pi*((h_max/2.0)**3))

# Range of Velocities [m/s]    
V_min = 0.1
V_max = 6.0
V_pop = np.linspace(V_min, V_max, 10000)
    
# Range of Angular Velocities [rad/s]
Om_min = np.pi
Om_max = 2*np.pi
Om_pop = np.linspace(Om_min, Om_max, 10000)  

# Vertical Ejector Angle angle
thetaMAX = (np.pi*17.)/36. 

# Division of Processes
nproc = 2
Im = 100000
mult = Im/nproc

data_cap = 100000
data_cap_inc = data_cap

run_count = int( data_cap/Im )

fileNames = ['{}.h5'.format(name), 
             '{}_Meta.txt'.format(name),
             '{}_Landing.vtk'.format(name), 
             '{}_Correlation.vtk'.format(name)]

# Data Analysis
X_bin = 501
Y_bin = X_bin
N_bin = X_bin*Y_bin

spc = Dia/X_bin

X_dim = (2*Dia)/(X_bin-1)
Y_dim = (2*Dia)/(Y_bin-1)
