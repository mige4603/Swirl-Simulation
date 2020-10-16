#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:13:49 2019

@author: michael
"""

import global_var as gVar
import numpy as np
import h5py as h5

import os, sys
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
    
    print( '\n'+str(fail_sum)+' Particles Failed')
    if np.sum(fail_sum) > 0:
        print( '    '+str(count[0])+' Falling Phase' )
        print( '    '+str(count[1])+' Impact Phase' )
        print( '    '+str(count[2])+' Collision Phase' )
    
    print( '\n'+str(count[3])+' Particles Flatten' )
    print( '    '+str(count[3] - count[4])+' Tor_grav > Tor_field' )
    print( '    '+str(count[4])+' Tor_grav < Tor_field' )
    
    print( '\n'+str(count[5])+' Particles Never Flatten' )
    
    print( '\nPrevious Rate: '+str(rate)+' part/sec' )
    
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
        if data is None:
            break
        else:
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
        if count is None:
            break
        else:
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
    
    fail_sum = count['rise fail'] + count['fall fail'] + count['impact fail'] + count['collide fail']
    part_sum = fail_sum + count['success'] + count['lift fail'] + count['never fail']
    
    file = open(file_meta, 'w')
    
    file.write('Reiner Gamma\n'
               '\n'
               'Magnetic Field: Single Dipole \n'
               '\t Position: ('+str(var.dipole_position[0])+', '+str(var.dipole_position[1])+', '+str(var.dipole_position[2])+') m\n'
               '\t Moment: ('+str(var.dipole_moment[0])+', '+str(var.dipole_moment[1])+', '+str(var.dipole_moment[2])+') Am^2\n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+') m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+') Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+') m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+') rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+') m^2\n'
               '\n'+str(part_sum)+' Individual Grains\n'
               '\n'
               +str(fail_sum)+' Particles Failed \n'
               '    '+str(count['rise fail'])+' Rising Phase\n'
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
        
    fail_sum = count['rise fail'] + count['fall fail'] + count['impact fail'] + count['collide fail']
    part_sum = fail_sum + count['success'] + count['lift fail'] + count['never fail']
    
    file = open(file_name, 'w')
    
    file.write('Reiner Gamma\n'
               '\n'
               'Magnetic Field: Single Dipole \n'
               '\t Position: ('+str(var.dipole_position[0])+', '+str(var.dipole_position[1])+', '+str(var.dipole_position[2])+') m\n'
               '\t Moment: ('+str(var.dipole_moment[0])+', '+str(var.dipole_moment[1])+', '+str(var.dipole_moment[2])+') Am^2\n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+') m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+') Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+') m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+') rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+') m^2\n'
               '\n'+str(part_sum)+' Individual Grains\n'
               '\n'
               +str(fail_sum)+' Particles Failed \n'
               '    '+str(count['rise fail'])+' Rising Phase\n'
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
    
    
    
def create_meta(count, file_meta, time, namelist):
    """ Create a .h5 data file and .txt meta data file.
    
    Parameters
    ----------
    count : array
        A set of the number of failed runs in each phase
        
    file_meta : str
        Name of the .txt file where the meta data will be saved
    """
    var = gVar.variables(namelist)
    
    fail_sum = count['rise fail'] + count['fall fail'] + count['impact fail'] + count['collide fail']
    part_sum = fail_sum + count['success'] + count['lift fail'] + count['never fail']
    
    #file = open(file_meta, 'w')
    
    print('\n\nMagnetic Field: Single Dipole \n'
          '\t Position: ('+str(var['dipole_position'][0])+', '+str(var['dipole_position'][1])+', '+str(var['dipole_position'][2])+') m\n'
          '\t Moment: ('+str(var['dipole_moment'][0])+', '+str(var['dipole_moment'][1])+', '+str(var['dipole_moment'][2])+') Am^2\n'
          'Length of Grains : ('+str(var['h_min'])+' to '+str(var['h_max'])+') m\n'
          'Magnetic Moment of Grains : ('+str(var['m_mom_min'])+' to '+str(var['m_mom_max'])+') Am^2\n' 
          'Charge on Grains : (-'+str(var['q_min'])+' to -'+str(var['q_max'])+') x 10e-19 C\n'
          'Initial Linear Velocity : ('+str(var['V_min'])+' to '+str(var['V_max'])+') m/s\n'
          'Initial Angular Velocity : ('+str(var['Om_min'])+' to '+str(var['Om_max'])+') rad/s\n'
          'Landing Area : ('+str(2*var['Dia'])+' x '+str(2*var['Dia'])+') m^2\n'
          '\n'+str(part_sum)+' Individual Grains\n'
          '\n'
          +str(fail_sum)+' Particles Failed \n'
          '    '+str(count['rise fail'])+' Rising Phase\n'
          '    '+str(count['fall fail'])+' Falling Phase\n'
          '    '+str(count['impact fail'])+' Impact Phase\n'
          '    '+str(count['collide fail'])+' Collision Phase\n'
          '\n'
          +str(count['success']+count['lift fail'])+' Particles Flatten\n'
          '    '+str(count['success'])+' Tor_grav > Tor_field\n'
          '    '+str(count['lift fail'])+' Tor_grav < Tor_field\n'
          '\n'
          +str(count['never fail'])+' Particles Never Flatten\n'
          '\nTotal Simulation Time : '+str(time)+' sec\n'
          'Time per grain : {0} sec\n'.format(time/part_sum))
    
    #file.close()
    
def create_meta_header(file_name):
    """ Write the meta file header.
    
    Paraview
    --------
    file_name : str
        name of meta data file
    """
    file = open(file_name, 'w')
    
    file.write('Reiner Gamma\n'
               '\n'
               'Magnetic Field: Single Dipole \n'
               '\t Position: ('+str(var.dipole_position[0])+', '+str(var.dipole_position[1])+', '+str(var.dipole_position[2])+') m\n'
               '\t Moment: ('+str(var.dipole_moment[0])+', '+str(var.dipole_moment[1])+', '+str(var.dipole_moment[2])+') Am^2\n'
               'Length of Grains : ('+str(var.h_min)+' to '+str(var.h_max)+') m\n'
               'Magnetic Moment of Grains : ('+str(var.m_mom_min)+' to '+str(var.m_mom_max)+') Am^2\n' 
               'Charge on Grains : (-'+str(var.q_min)+' to -'+str(var.q_max)+') x 10e-19 C\n'
               'Initial Linear Velocity : ('+str(var.V_min)+' to '+str(var.V_max)+') m/s\n'
               'Initial Angular Velocity : ('+str(var.Om_min)+' to '+str(var.Om_max)+') rad/s\n'
               'Landing Area : ('+str(2*var.Dia)+' x '+str(2*var.Dia)+') m^2\n'
               '\n1 Individual Grains\n'
               '\n'
               '0 Particles Failed \n'
               '    0 Rising Phase\n'
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
        if i == 15:
            count['rise fail'] = get_number(line)
        elif i == 16:
            count['fall fail'] = get_number(line)
        elif i == 17:
            count['impact fail'] = get_number(line)
        elif i == 18:
            count['collide fail'] = get_number(line)
        elif i == 21:
            count['success'] = get_number(line)
        elif i == 22:
            count['lift fail'] = get_number(line)
        elif i == 24:
            count['never fail'] = get_number(line)
        elif i >  25:
            break
    fp.close
    
    return count
                    
def landing_pattern(file_data, file_lan_pat, params):
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
    
    e_data = np.zeros((params['N_bin'], 2))

    for i in range(data_length):
        r = data[i][0:3]
        r_unit = r / np.linalg.norm( r )
        
        r_tan_x = np.array([1,0,0]) - r_unit[0] * r_unit
        r_tan_x = r_tan_x / np.linalg.norm( r_tan_x )
        
        r_tan_y = np.cross(r_unit, r_tan_x)
        r_tan_y = r_tan_y / np.linalg.norm( r_tan_y )
        
        e_vec = data[i][3:6]        
    
        h = data[i][6]
        
        rad = np.sqrt( r[0]**2 + r[1]**2 )
        e_vec_reduced = np.array([ np.dot(e_vec, r_tan_x) , np.dot(e_vec, r_tan_y) ])#np.array([e1, e2])
        if (rad<=params['Dia']) and not all(e_vec_reduced==0):
            e_vec_reduced = e_vec_reduced / np.linalg.norm( e_vec_reduced )
            
            Xdim = int( round( ( r[0] + params['Dia'] ) / params['X_dim'] ) )
            Ydim = int( round( ( r[1] + params['Dia'] ) / params['Y_dim'] ) )
            
            N = Ydim * params['X_bin'] + Xdim
            e_data[N] = np.add( e_data[N], h * e_vec_reduced )
            
    vec_avg = np.zeros((params['N_bin'], 2))
    
    for j in range(params['N_bin']):
        if not all (e_data[j]==0):
            vec_avg[j] = e_data[j] / np.linalg.norm( e_data[j] )
            
    file = open(file_lan_pat, 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} 1\n'
               'ORIGIN 0 0 0\n'
               'SPACING {1} {1} {1}\n'
               'POINT_DATA {2}\n'
               'VECTORS B float\n'.format(params['X_bin'], params['spc'], params['N_bin']))
    for t in range(params['N_bin']):
        x, y = vec_avg[t]
        file.write(str(x) + ' ' + str(y) + ' ' + str(0.0) + '\n')
    file.close() 
    
    return vec_avg
    
def correlation_pattern(vec_avg, file_cor_pat, params):
    """ Construct a correlation pattern with the landing data.
    
    Parameters
    ----------
    vec_avg : array
        averaged landing vector from data set
        
    file_cor_pat : str
        correlation pattern file name
    """
    cross_cor = np.zeros((params['N_bin'], 2))
    
    for q in range(params['N_bin']):
            
        if not all (vec_avg[q]==0):
            vec = vec_avg[q]
            sca = []   
            
            rem_1 = q % params['X_bin']
            rem_2 = (q+1) % params['X_bin']
            
            q_2 = q+params['X_bin']
            if q_2 > 0 and q_2 < params['N_bin']:
                vec_2 = vec_avg[q_2]
                if not all (vec_2==0):
                    sca.append(np.dot(vec_2,vec))
                        
            q_7 = q-params['X_bin']
            if q_7 > 0 and q_7 < params['N_bin']:
                vec_7 = vec_avg[q_7]
                if not all (vec_7==0):
                    sca.append(np.dot(vec_7,vec)) 
            
            if not (rem_1==0):
                q_1 = q+params['X_bin']-1
                if q_1 > 0 and q_1 < params['N_bin']:
                    vec_1 = vec_avg[q_1]
                    if not all (vec_1==0):
                        sca.append(np.dot(vec_1,vec)) 
                        
                q_4 = q-1
                if q_4 > 0 and q_4 < params['N_bin']:
                    vec_4 = vec_avg[q_4]
                    if not all (vec_4==0):
                        sca.append(np.dot(vec_4,vec)) 
                    
                q_6 = q-params['X_bin']-1
                if q_6 > 0 and q_6 < params['N_bin']:
                    vec_6 = vec_avg[q_6]
                    if not all (vec_6==0):
                        sca.append(np.dot(vec_6,vec)) 
                        
            if not (rem_2==0):
                q_3 = q+params['X_bin']+1
                if q_3 > 0 and q_3 < params['N_bin']:
                    vec_3 = vec_avg[q_3]
                    if not all (vec_3==0):
                        sca.append(np.dot(vec_3,vec)) 
                        
                q_5 = q+1
                if q_5 > 0 and q_5 < params['N_bin']:
                    vec_5 = vec_avg[q_5]
                    if not all (vec_5==0):
                        sca.append(np.dot(vec_5,vec)) 
            
                q_8 = q-params['X_bin']+1
                if q_8 > 0 and q_8 < params['N_bin']:
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
               'VECTORS B float\n'.format(params['X_bin'], params['spc'], params['N_bin']))
    for t in range(params['N_bin']):            
        scalar_avg, scalar_std = cross_cor[t]
        file.write(str(scalar_avg) + ' ' + str(scalar_std) + ' ' + str(0.0) + '\n')
    file.close() 


### New Analysis Scripts ###
    
def read_dataOut(dataPath, readHead=False, nGrains=10000):
    meta = {}
    data = np.empty((nGrains, 7))
    with open(dataPath, 'r') as file:
        lines = file.readlines()
        
        for beg_idx, line in enumerate(lines):
            try:
                line = line.strip().split()
                err_chk = float(line[0])
                break
            
            except ValueError:
                continue
        
        for i, line in enumerate(lines[beg_idx:beg_idx+nGrains]):
            line = line.strip()
            line = line.split()
            for j, val in enumerate(line):
                data[i,j] = float(val)
        
        if readHead:
            header = []
            for line in lines[beg_idx+nGrains+2:beg_idx+nGrains+12]:
                header.append(line)
            meta['header'] = header
            
        for line in lines[beg_idx+nGrains+12::]:
            line = line.strip()
            line = line.split()
            if len(line) > 0:
                if line[1] == 'Rising':
                    meta['rise fail'] = int(line[0])
                elif line[1] == 'Falling':
                    meta['fall fail'] = int(line[0])
                elif line[1] == 'Impact':
                    meta['impact fail'] = int(line[0])
                elif line[1] == 'Collision':
                    meta['collide fail'] = int(line[0])
                elif line[1] == 'Tor_grav' and line[2] == '>':
                    meta['success'] = int(line[0])
                elif line[1] == 'Tor_grav' and line[2] == '<':
                    meta['lift fail'] = int(line[0])
                elif line[1] == 'Particles' and line[2] == 'Never':
                    meta['never fail'] = int(line[0])
                elif line[0] == 'Total':
                    meta['total time'] = float(line[4])
                elif line[0] == 'Time':
                    meta['grain time'] = float(line[4])
    
    return data, meta


def reformat_dipole_data(path, nGrains=10000, nCPU=2):
    from shutil import rmtree
    
    dipoles = [f.name for f in os.scandir(path) if f.is_dir()]
    for dipole in dipoles:
        print('Working on '+dipole)
        pathDip = os.path.join(path, dipole)
        runs = [f.name for f in os.scandir(pathDip) if f.is_dir()]
        if len(runs) > 0:
            keys = ['rise fail','fall fail','impact fail','collide fail','success','lift fail','never fail','total time','grain time']
            metaDip = {}
            for key in keys:
                metaDip[key] = 0    
                dataDip = np.empty((len(runs)*nGrains, 7))   
	
            try:
                read_chk = False
                data, meta = read_dataOut(os.path.join(pathDip, runs[0], 'data.out'), readHead=True, nGrains=nGrains)
                read_chk = True
            except ValueError:
                print('   Read Check failed for '+runs[0])
	
            if read_chk:
                dataDip[0:nGrains] = data
                for key in meta:
                    if key == 'total time' or key =='grain time':
                        metaDip[key] = np.r_[metaDip[key], meta[key]]
                    elif key == 'header':
                        metaDip[key] = meta[key]
                    else:
                        metaDip[key] = metaDip[key] + meta[key]
	    
            with open(os.path.join(pathDip,runs[0],'input.namelist'),'r') as file:
                lines = file.readlines()
            with open(os.path.join(pathDip,'input.namelist'),'w') as file:
                for line in lines:
                    file.write(line)
	            
            for idx, run in enumerate(runs[1::]):
                idx+=1
                pathRun = os.path.join(pathDip, run, 'data.out')
                try:
                    read_chk = False
                    data, meta = read_dataOut(pathRun, nGrains=nGrains)
                    read_chk = True
                except ValueError:
                    print('   Read Check failed for '+run)
    	            
                if read_chk:
                    dataDip[idx*nGrains:(idx+1)*nGrains] = data
                    for key in meta:
                        if key == 'total time' or key =='grain time':
                            metaDip[key] = np.r_[metaDip[key], meta[key]]
                        else:
                            metaDip[key] = metaDip[key] + meta[key]
    	
                    metaPath = os.path.join(pathDip,'meta_data.txt')
                    dataPath = os.path.join(pathDip,'data.h5')
                    with open(metaPath,'w') as file:
                        header = metaDip['header']
                        for head in header:
                            file.write(head)
    	        
                        fail_sum = metaDip['rise fail'] + metaDip['fall fail'] + metaDip['impact fail'] + metaDip['collide fail']
                        part_sum = fail_sum + metaDip['success'] + metaDip['lift fail'] + metaDip['never fail']
                        time_avg = (float(nCPU)/3600) * np.sum(metaDip['total time'][1::])
                        grain_avg = nCPU * np.mean(metaDip['grain time'][1::])
            	                
                        file.write(str(part_sum)+' Individual Grains\n\n'
                                   +str(fail_sum)+' Particles Failed \n'
                                   '    '+str(metaDip['rise fail'])+' Rising Phase\n'
                                   '    '+str(metaDip['fall fail'])+' Falling Phase\n'
                                   '    '+str(metaDip['impact fail'])+' Impact Phase\n'
                                   '    '+str(metaDip['collide fail'])+' Collision Phase\n\n'
                                   +str(metaDip['success']+metaDip['lift fail'])+' Particles Flatten\n'
                                   '    '+str(metaDip['success'])+' Tor_grav > Tor_field\n'
                                   '    '+str(metaDip['lift fail'])+' Tor_grav < Tor_field\n\n'
            	                    +str(metaDip['never fail'])+' Particles Never Flatten\n'
                                   '\nTotal Simulation Time : '+str(time_avg)+' cpu hours\n'
                                   'Time per grain : '+str(grain_avg)+' sec')
    	
            with h5.File(dataPath,'w') as hf_file:
                hf_file.create_dataset('data', data=dataDip)
            	
            for runDir in [f.name for f in os.scandir(pathDip) if f.is_dir()]:
                runPath = os.path.join(pathDip, runDir)
                rmtree(runPath)
                	    
            os.remove(os.path.join(pathDip, 'run_exec.sh'))
            os.remove(os.path.join(pathDip, 'run_list.txt'))
            os.remove(os.path.join(pathDip, 'submit.sub'))
            
        else:
            print('   No Runs.')

