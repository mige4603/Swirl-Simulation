#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:10:00 2018

@author: michael
"""

import global_var as var
import functions as fun
import analysis as anal

import h5py as h5

first_set = 10
dirc_name = var.path+'Random/'

meta_files = sorted( fun.list_files(dirc_name, 'txt') )
data_files = sorted( fun.list_files(dirc_name, 'h5') )

meta_length = len(meta_files)
data_length = len(data_files)

if meta_length <= data_length:
    numpts = meta_length
else:
    numpts = data_length

data_set = h5.File(dirc_name+'Condensed_DATA/IDQuat_Random.h5', 'r+')
mag_set = data_set['dipole']
for i in range(numpts):
    mag_new = anal.get_mag(dirc_name+meta_files[i])
    mag_set.resize(mag_set.shape[0]+1, axis=0)
    mag_set[-1:] = mag_new
    
    data_i = h5.File(dirc_name+data_files[i], 'r')
    
    name = 'set {}'.format(first_set+i)
    data_set.create_dataset(name, data=data_i['data'])
    
data_set.close()
