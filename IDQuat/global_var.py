#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 19:09:35 2018

@author: michael
"""
import numpy as np

run_num = 0
rate = []

dirc = 'Random'
path = '/home/michael/Documents/LASP/Swirl-Simulation/IDQuat/DATA/'+dirc+'/'
name = path+'IDQuat_'+dirc

# Magnetic moment of 1cm^3 Neodymium magnet (Am^2)
#mag_mom = 0.875   
mag_mom = .875
if dirc == 'Horizontal':
    mag = np.array([0,mag_mom,0])
elif dirc == 'Vertical':
    mag = np.array([0,0,mag_mom])
    
# Gravity on Earth surface [m/s^2]
g = 9.8       

# Charge on Dust Grain (number of electrons)
q_min = 1000
q_max = 10000
q_pop = np.linspace(q_min, q_max, 1000)   

# Range of Landing Site -Dia to Dia [m]    
Dia = 1.

# Density Range [g/m^3]
den_min = 2.3e6
den_max = 3.2e6
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
Om_min = 2*np.pi
Om_max = 4*np.pi
Om_pop = np.linspace(Om_min, Om_max, 10000)  # dAng/dt (rad/sec)
thetaMAX = (np.pi*17.)/36. # Vertical Ejector Angle angle

# Division of Processes
nproc = 2
Im = 1000
mult = Im/nproc

data_cap = 100000
data_cap_inc = data_cap

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