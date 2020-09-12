#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:33:51 2019

@author: michael
"""

import numpy as np
import h5py as h5

    
def variables():
    inputs = {}
    with open('input.namelist','r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0]!='#':
                line = line.strip()
                line = line.split()
                if len(line) > 0:
                    inputs[line[0]] = line[2]   
    '''
    run_num = 200
    rate = []
    
    path = '/home/cu-pwfa/Documents/Michael/Swirl-Simulation/RGQuat/'
    sim_path = path+'Sim_Data/'
    field_path = path+'Field_Data/'
    name = sim_path+'Dipole_Field_Sim'
    #name = sim_path+'Jan_Field_Sim'
    '''
    # Gravity surface [m/s^2]
    if 'grav' in inputs: 
        g = float(inputs['grav'])
    else:
        g = 1.62519    
    
    # Radius of Body [m]
    if 'radius' in inputs:
        r_m = float(inputs['radius'])
    else:
        r_m = 1.7371e6
    
    # Range of Landing Site -Dia to Dia [m]    
    if 'landRange' in inputs:
        Dia = float(inputs['landRange'])
    else:
        Dia = 130000
    zettaMAX = np.arcsin(Dia/r_m)
    
    # Charge on Dust Grain (number of electrons)
    if 'chargeMin' in inputs:
        q_min = float(inputs['chargeMin'])
        q_max = float(inputs['chargeMax'])
        q_pts = float(inputs['chargePts'])
    else:
        q_min = 100
        q_max = 1000
        q_pts = 1000
    q_pop = np.linspace(q_min, q_max, q_pts)   
    
    # Density Range [g/m^3]
    if 'denMin' in inputs:
        den_min = float(inputs['denMin'])
        den_max = float(inputs['denMax'])
        den_pop = float(inputs['denPts'])
    else:
        den_min = 2.33e6
        den_max = 4.37e6
        den_pop = 10000
    den_pop = np.linspace(den_min, den_max, den_pop)
     
    # Range of Grain Diameters [m]
    if 'bodyMin' in inputs:
        h_min = float(inputs['bodyMin'])
        h_max = float(inputs['bodyMax'])
        h_pts = float(inputs['bodyPts'])
    else:
        h_min = .1e-6  
        h_max = 10.0e-6
        h_pts = 1000
    h_pop = np.linspace(h_min, h_max, h_pts)
        
    # Range of Magnetic Moments in Grains [Am^2/g]
    if 'magMin' in inputs:
        FeByWght_min = float(inputs['magMin'])
        FeByWght_max = float(inputs['magMax'])
        FeByWght_pop = float(inputs['magPts'])
    else:
        FeByWght_min = .00062
        FeByWght_max = .0012
        FeByWght_pop = 100
    FeByWght_pop = np.linspace(FeByWght_min, FeByWght_max, FeByWght_pop)
    
    m_mom_min = FeByWght_min*den_min*((4/3)*np.pi*((h_min/2.0)**3))
    m_mom_max = FeByWght_max*den_max*((4/3)*np.pi*((h_max/2.0)**3))
    
    # Range of Velocities [m/s]
    if 'velMin' in inputs:
        V_min = float(inputs['velMin'])
        V_max = float(inputs['velMax'])
        V_pts = float(inputs['velPts'])
    else:
        V_min = 0.1
        V_max = 6.0
        V_pts = 10000
    V_pop = np.linspace(V_min, V_max, V_pts)
        
    # Range of Angular Velocities [rad/s]
    if 'omMin' in inputs:
        Om_min = float(inputs['omMin'])
        Om_max = float(inputs['omMax'])
        Om_pop = float(inputs['omPts'])
    else:
        Om_min = 0
        Om_max = 2 * np.pi * np.sqrt(3)
        Om_pop = 10000
    Om_pop = np.linspace(Om_min, Om_max, Om_pop)  
    
    # Vertical Ejector Angle angle
    thetaMAX = (np.pi*17.)/36. 
    '''
    # Division of Processes
    nproc = 20
    Im = 10000
    mult = int( Im/nproc )
    
    data_cap = 100000
    data_cap_inc = data_cap
    
    run_count = int( data_cap/Im )
    
    fileNames = ['{}.h5'.format(name), 
                 '{}_Meta.txt'.format(name),
                 '{}_Landing.vtk'.format(name), 
                 '{}_Correlation.vtk'.format(name)]
    '''
    if inputs['fieldType'] == 'JanDeca':
        # Field Data for Reiner Gamma Swirl
        field_data = h5.File('RG_Full_Field_Data.h5', 'r')
        B_data = field_data['B_field']
        E_data = field_data['E_field']
    
    elif inputs['fieldType'] == 'TwoDipoles':
        SW_r = np.array([15055.59399353, 19522.97153838, 1725823.91415573])
        SW_mag = np.array([-1.12822658e+13, -6.07455748e+11,  -1.77414475e+11])
    
        NE_r = np.array([-87674.4142417, 124611.72907353, 1723678.99391525])
        NE_mag = np.array([-1.30337601e+12, -3.24501798e+12, -1.45152530e+11])
    
    elif inputs['fieldType'] == 'OneDipole':
        depth = float(inputs['depth'])
        momX = float(inputs['momentX'])
        momY = float(inputs['momentY'])
        momZ = float(inputs['momentZ'])
        
        dipole_position = np.array([0,0,r_m - depth]) #np.linspace(1, 1000, 10000) * 1000
        dipole_moment = np.array([momX, momY, momZ]) #np.linspace(100, 1000, 1000) * 1e11
        
    else:
        raise KeyError(inputs['field type']+' is not a valid field type.')
    
    # Data Analysis
    X_bin = 501
    Y_bin = X_bin
    N_bin = X_bin*Y_bin
    
    spc = Dia/X_bin
    
    X_dim = (2*Dia)/(X_bin-1)
    Y_dim = (2*Dia)/(Y_bin-1)
    
    var_dict = {'Dia' : Dia,
                'g' : g,
                'r_m' : r_m,
                'V_min' : V_min,
                'V_max' : V_max,
                'V_pop' : V_pop,
                'Om_min' : Om_min,
                'Om_max' : Om_max,
                'Om_pop' : Om_pop,
                'h_min' : h_min,
                'h_max' : h_max,
                'h_pop' : h_pop,
                'q_min' : q_min,
                'q_max' : q_max,
                'q_pop' : q_pop,
                'den_min' : den_min,
                'den_max' : den_max,
                'den_pop' : den_pop,
                'FeByWght_min' : FeByWght_min,
                'FeByWght_max' : FeByWght_max,
                'FeByWght_pop' : FeByWght_pop,
                'thetaMAX' : thetaMAX,
                'zettaMAX' : zettaMAX,
                'm_mom_min' : m_mom_min,
                'm_mom_max' : m_mom_max,
                'dipole_moment' : dipole_moment,
                'dipole_position' : dipole_position}
    
    return var_dict
