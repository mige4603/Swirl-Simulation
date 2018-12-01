#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 00:59:03 2018

@author: michael
"""

import global_var as var
import numpy as np
import h5py as h5
from os.path import isfile 
from os import remove

def get_number(line, cnt=0):
    # Reads a line from .txt file and returns the first float in that line
    for j in line:
        if not j == ' ':
            begin = cnt
            end = begin+1
            while not line[end] == ' ':
                end += 1
            return float(line[begin:end])
        cnt += 1

def print_outcome(count, rate):
    """ Print the outcome of a set of dust grains.
    
    Parameters
    ----------
    count : array
        A set of the number of failed runs in each phase
        
    rate : float
        Number of seconds per dust grain
    """
    fail_sum = np.sum(count[0:3])
    
    print '\n'+str(fail_sum)+' Particles Failed'
    if np.sum(fail_sum) > 0:
        print '    '+str(count[0])+' Falling Phase'
        print '    '+str(count[1])+' Impact Phase'
        print '    '+str(count[2])+' Collision Phase'
    
    print '\n'+str(count[3])+' Particles Flatten'
    print '    '+str(count[3] - count[4])+' Tor_grav > Tor_field'
    print '    '+str(count[4])+' Tor_grav < Tor_field'
    
    print '\n'+str(count[5])+' Particles Never Flatten'
    
    print '\nPrevious Rate: '+str(rate)+' part/sec'
    
def merge_datasets(file_sub_1, file_sub_2, file_name, delete_old=False):
    """ Merge two data sets into a single file.
    
    Parameters
    ----------
    file_sub_1 : str
        Name of file where first dataset is stored
        
    file_sub_2 : str
        Name of file where second dataset is stored
        
    file_name : str
        Name of new file constructed with both datasets
    """
    hf_sub_2 = h5.File(file_sub_2, 'r')
    data_sub_2 = hf_sub_2['data']
    data_sub_2_lgth = len(data_sub_2)
    hf_sub_2.close()
    
    hf_sub_1 = h5.File(file_sub_1, 'a')
    data_sub_1 = hf_sub_1['data']
    
    data_sub_1.resize(data_sub_1.shape[0] + data_sub_2_lgth, axis=0)
    data_sub_1[-data_sub_2_lgth:] = data_sub_2
    new_data = data_sub_1
    hf_sub_1.close()
    
    hf = h5.File(file_name, 'w')
    hf.create_dataset('data', (len(new_data),7), maxshape=(None, 7), chunks=(1,7))
    hf['data'] = new_data
    hf.close()
    
    if delete_old:
        remove(file_sub_1)
        remove(file_sub_2)
    
def save_data(s_queue, file_data):
    """ Save data pushed to the queue."""
    while True:
        data = s_queue.get()
        if isfile(file_data):
            append_data(data, file_data)            
        else:
            create_data(data, file_data)
        s_queue.task_done()

def append_data(data, file_data):
    """ Append a .h5 data file and .txt meta data file.
    
    Parameters
    ----------
    data : array
        Array of successful dust grains
        
    file_data : str
        Name of .h5 file where the data will be save
    """
    hf = h5.File(file_data, 'a')
    data_old = hf['data']
    data_old.resize(data_old.shape[0]+1, axis=0)
    data_old[-1:] = data
    hf.close()

def create_data(sim_data, file_data):
    """ Create a .h5 data file and .txt meta data file.
    
    Parameters
    ----------
    data : array
        Array of successful dust grains
        
    file_data : str
        Name of .h5 file where the data will be save
    """
    hf = h5.File(file_data, 'w')
    hf.create_dataset('data', data=sim_data, maxshape=(None, 7), chunks=(1,7))
    hf.close()
    
def save_meta(s_queue, file_meta):
    """Save or update meta data from run.
    
    Parameters
    ----------
    count : array
        A set of the number of failed runs in each phase
        
    file_meta : str
        Name of the .txt file where the meta data will be saved
    """
    while True:
        fail=False
        count = s_queue.get()
        for key in count:
            if count[key] != 0:
                fail=True
                break
        if fail:
            if isfile(file_meta):
                append_meta(count, file_meta)
            else:
                create_meta(count, file_meta)            
        else:
            if isfile(file_meta):
                append_meta_header(file_meta)
            else:
                create_meta_header(file_meta)
        s_queue.task_done()

def append_meta(count, file_meta):
    """ Append a .h5 data file and .txt meta data file.
    
    Parameters
    ----------
    count : array
        A set of the number of failed runs in each phase
        
    file_meta : str
        Name of the .txt file where the meta data will be saved
    """
    count_prev = get_counts(file_meta)
    for key in count_prev:
        count[key] = count[key] + count_prev[key]
    
    fail_sum = count['fall fail'] + count['impact fail'] + count['collide fail']
    
    file = open(file_meta, 'w')
    
    file.write('Ideal Dipole\n\n'
               'Magnetic Field Location : (0, 0, -.01) \n'
               'Magnetic Dipole Strength : ('+str(var.mag[0])+', '+str(var.mag[1])+', '+str(var.mag[2])+') Am^2 \n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+')m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+')Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+')m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+')rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+')m^2\n'
               '\n'+str(count['success'])+' Individual Grains\n'
               '\n'
               +str(fail_sum)+' Particles Failed \n'
               '    '+str(count['fall fail'])+' Falling Phase\n'
               '    '+str(count['impact fail'])+' Impact Phase\n'
               '    '+str(count['collide fail'])+' Collision Phase\n'
               '\n'
               +str(count['success']+count['lift fail'])+' Particles Flatten\n'
               '    '+str(count['success'])+' Tor_grav > Tor_field\n'
               '    '+str(count['lift fail'])+' Tor_grav < Tor_field\n'
                '\n'
                +str(count['never fail'])+' Particles Never Flatten\n')
    file.close()
    
def append_meta_header(file_name):
    """ Change the meta header.
    
    Parameter
    ---------
    file_name : str
        Name of meta data file
    """
    count = get_counts(file_name)
    count['success']+=1
        
    fail_sum = count['fall fail'] + count['impact fail'] + count['collide fail']
    
    file = open(file_name, 'w')
    
    file.write('Ideal Dipole\n\n'
               'Magnetic Field Location : (0, 0, -.01) \n'
               'Magnetic Dipole Strength : ('+str(var.mag[0])+', '+str(var.mag[1])+', '+str(var.mag[2])+') Am^2 \n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+')m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+')Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+')m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+')rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+')m^2\n'
               '\n'+str(count['success'])+' Individual Grains\n'
               '\n'
               +str(fail_sum)+' Particles Failed \n'
               '    '+str(count['fall fail'])+' Falling Phase\n'
               '    '+str(count['impact fail'])+' Impact Phase\n'
               '    '+str(count['collide fail'])+' Collision Phase\n'
               '\n'
               +str(count['success']+count['lift fail'])+' Particles Flatten\n'
               '    '+str(count['success'])+' Tor_grav > Tor_field\n'
               '    '+str(count['lift fail'])+' Tor_grav < Tor_field\n'
                '\n'
                +str(count['never fail'])+' Particles Never Flatten\n')
    file.close()
    
    
    
def create_meta(count, file_meta):
    """ Create a .h5 data file and .txt meta data file.
    
    Parameters
    ----------
    count : array
        A set of the number of failed runs in each phase
        
    file_meta : str
        Name of the .txt file where the meta data will be saved
    """
    fail_sum = count['fall fail'] + count['impact fail'] + count['collide fail']
    
    file = open(file_meta, 'w')
    
    file.write('Ideal Dipole\n\n'
               'Magnetic Field Location : (0, 0, -.01) \n'
               'Magnetic Dipole Strength : ('+str(var.mag[0])+', '+str(var.mag[1])+', '+str(var.mag[2])+') Am^2 \n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+')m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+')Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+')m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+')rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+')m^2\n'
               '\n0 Individual Grains\n'
               '\n'
               +str(fail_sum)+' Particles Failed \n'
               '    '+str(count['fall fail'])+' Falling Phase\n'
               '    '+str(count['impact fail'])+' Impact Phase\n'
               '    '+str(count['collide fail'])+' Collision Phase\n'
               '\n'
               +str(count['success']+count['lift fail'])+' Particles Flatten\n'
               '    '+str(count['success'])+' Tor_grav > Tor_field\n'
               '    '+str(count['lift fail'])+' Tor_grav < Tor_field\n'
                '\n'
                +str(count['never fail'])+' Particles Never Flatten\n')
    
    file.close()
    
def create_meta_header(file_name):
    """ Write the meta file header.
    
    Paraview
    --------
    file_name : str
        name of meta data file
    """
    file = open(file_name, 'w')
    
    file.write('Ideal Dipole\n\n'
               'Magnetic Field Location : (0, 0, -.01) \n'
               'Magnetic Dipole Strength : ('+str(var.mag[0])+', '+str(var.mag[1])+', '+str(var.mag[2])+') Am^2 \n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+')m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+')Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+')m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+')rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+')m^2\n'
               '\n1 Individual Grains\n'
               '\n'
               '0 Particles Failed \n'
               '    0 Falling Phase\n'
               '    0 Impact Phase\n'
               '    0 Collision Phase\n'
               '\n'
               '0 Particles Flatten\n'
               '    0 Tor_grav > Tor_field\n'
               '    0 Tor_grav < Tor_field\n'
                '\n'
                '0 Particles Never Flatten\n')
    
    file.close()

def get_counts(file_name):
    """ Return failure counts from meta data. 
    
    Parameters
    ----------
    file_name : str
        name of meta data file
    """
    count = {}
    fp = open(file_name)
    for i, line in enumerate(fp):
        if i == 14:
            count['fall fail'] = get_number(line)
        elif i == 15:
            count['impact fail'] = get_number(line)
        elif i == 16:
            count['collide fail'] = get_number(line)
        elif i == 19:
            count['success'] = get_number(line)
        elif i == 20:
            count['lift fail'] = get_number(line)
        elif i == 22:
            count['never fail'] = get_number(line)
        elif i >  22:
            break
    fp.close
    
    return count

def get_mag(file_name):
    """ Return the magnetic field from a data set's meta file.
    
    Parameters
    ----------
    file_name : str
        Meta file name
    """
    mag = False
    
    fp = open(file_name)
    for i, line in enumerate(fp):
        if i == 3:
            mag_string = line[28:len(line)-8]        
            mag = np.empty(3)
            for i, t in enumerate( mag_string.split() ):
                if t[-1] == ',':
                    t = t[0:-1]
                try:
                    mag[i] = float(t)
                except ValueError:
                    mag = False
                    break
            break
    
    return mag
                    
def landing_pattern(file_data, file_lan_pat):
    """ Construct the landing pattern from data.
    
    Parameters
    ----------
    file_data : str
        data file name
        
    file_lan_pat : str
        landing pattern file name
    """
    hf = h5.File(file_data, 'r')
    data = hf['data']
    data_length = len(data)
    
    e_data = np.zeros((var.N_bin, 2))

    for i in range(1, data_length):
        r = data[i][0:3]
        e_vec = data[i][3:6]
        h = data[i][6]
        
        x, y, z = r
        e1, e2, e3 = e_vec
        
        rad = np.sqrt(x*x+y*y)
        e_vec_reduced = np.array([e1, e2])
        if (rad<=var.Dia) and not all(e_vec_reduced==0):
            e_vec_reduced = e_vec_reduced/np.linalg.norm(e_vec_reduced)
            
            Xdim = int(round((x+var.Dia)/var.X_dim))
            Ydim = int(round((y+var.Dia)/var.Y_dim))
            
            N = Ydim*var.X_bin + Xdim
            e_data[N] = np.add(e_data[N],h*e_vec_reduced)
            
    vec_avg = np.zeros((var.N_bin,2))
    
    for j in range(var.N_bin):
        if not all (e_data[j]==0):
            vec_avg[j] = e_data[j]/np.linalg.norm(e_data[j])
            
    file = open(file_lan_pat, 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} 1\n'
               'ORIGIN 0 0 0\n'
               'SPACING {1} {1} {1}\n'
               'POINT_DATA {2}\n'
               'VECTORS B float\n'.format(var.X_bin, var.spc, var.N_bin))
    for t in range(var.N_bin):
        x, y = vec_avg[t]
        file.write(str(x) + ' ' + str(y) + ' ' + str(0.0) + '\n')
    file.close() 
    
    return vec_avg
    
def correlation_pattern(vec_avg, file_cor_pat):
    """ Construct a correlation pattern with the landing data.
    
    Parameters
    ----------
    vec_avg : array
        averaged landing vector from data set
        
    file_cor_pat : str
        correlation pattern file name
    """
    cross_cor = np.zeros((var.N_bin, 2))
    
    for q in range(var.N_bin):
            
        if not all (vec_avg[q]==0):
            vec = vec_avg[q]
            sca = []   
            
            rem_1 = q % var.X_bin
            rem_2 = (q+1) % var.X_bin
            
            q_2 = q+var.X_bin
            if q_2 > 0 and q_2 < var.N_bin:
                vec_2 = vec_avg[q_2]
                if not all (vec_2==0):
                    sca.append(np.dot(vec_2,vec))
                        
            q_7 = q-var.X_bin
            if q_7 > 0 and q_7 < var.N_bin:
                vec_7 = vec_avg[q_7]
                if not all (vec_7==0):
                    sca.append(np.dot(vec_7,vec)) 
            
            if not (rem_1==0):
                q_1 = q+var.X_bin-1
                if q_1 > 0 and q_1 < var.N_bin:
                    vec_1 = vec_avg[q_1]
                    if not all (vec_1==0):
                        sca.append(np.dot(vec_1,vec)) 
                        
                q_4 = q-1
                if q_4 > 0 and q_4 < var.N_bin:
                    vec_4 = vec_avg[q_4]
                    if not all (vec_4==0):
                        sca.append(np.dot(vec_4,vec)) 
                    
                q_6 = q-var.X_bin-1
                if q_6 > 0 and q_6 < var.N_bin:
                    vec_6 = vec_avg[q_6]
                    if not all (vec_6==0):
                        sca.append(np.dot(vec_6,vec)) 
                        
            if not (rem_2==0):
                q_3 = q+var.X_bin+1
                if q_3 > 0 and q_3 < var.N_bin:
                    vec_3 = vec_avg[q_3]
                    if not all (vec_3==0):
                        sca.append(np.dot(vec_3,vec)) 
                        
                q_5 = q+1
                if q_5 > 0 and q_5 < var.N_bin:
                    vec_5 = vec_avg[q_5]
                    if not all (vec_5==0):
                        sca.append(np.dot(vec_5,vec)) 
            
                q_8 = q-var.X_bin+1
                if q_8 > 0 and q_8 < var.N_bin:
                    vec_8 = vec_avg[q_8]
                    if not all (vec_8==0):
                        sca.append(np.dot(vec_8,vec)) 
            
            if sca:            
                cross_cor[q] = np.array([np.mean(sca), np.std(sca)])
                
    file = open(file_cor_pat, 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} 1\n'
               'ORIGIN 0 0 0\n'
               'SPACING {1} {1} {1}\n'
               'POINT_DATA {2}\n'
               'VECTORS B float\n'.format(var.X_bin, var.spc, var.N_bin))
    for t in range(var.N_bin):            
        scalar_avg, scalar_std = cross_cor[t]
        file.write(str(scalar_avg) + ' ' + str(scalar_std) + ' ' + str(0.0) + '\n')
    file.close() 
