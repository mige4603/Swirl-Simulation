# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:35:40 2016

@author: mige4603
"""

import time
start_time = time.time()

from math import *
import numpy as np
import numpy.linalg as la
import SwirlSim_QuatGlobal as sg
import SwirlSim_QuatPrams as sp
import SwirlSim_QuatInt as sdi
from SwirlSim_QuatFunc import getNumb
import multiprocessing
from os.path import isfile 

def worker(part, que):
    results = {}
    
    chk = len(part) * 0.01
    pnt = (chk/len(part))*100
    trk = 1.0
    
    for i in part:
        mu = Particles[i][6]
        length = Particles[i][5]
        vol = .01*pi*length**3  
        m = 5745*vol  # Mass [kg]    
        I_x = (m*length**2)/12.0 
        
        the = ((35.0*pi)/72.0)*np.random.random()
        phi = 2*pi*np.random.random()
        V = np.random.choice(sp.V_pop)*np.array([sin(the)*cos(phi), sin(the)*sin(phi), cos(the)])
        R = np.array([Particles[i][0], Particles[i][1], 0])
        inPos = np.insert(V, 0, R)
        
        e_3 = Particles[i][2:5]
        E_1 = np.array([-e_3[2], 0, e_3[1]])
        e_1 = E_1 / la.norm(E_1)
        dAng = np.random.choice(sp.Om_pop)
        AngMom = sp.I_x*dAng*e_1
        inRot = np.insert(AngMom, 0, 0)
        
        inCon = np.insert(inRot, 0, inPos)
        params = [m, I_x, mu, e_3]        
        
        P = Particles[i]
        P.getFiCon()
        F = P.fiCon
        R = P.tRack        
        results[i] = np.array([F[0], F[1], F[2], F[3], R[0], R[1], R[2], R[3], R[4], R[5]])
        
        if part[0] == 0:
            if i == round(chk*trk):
                print '    Process 1 : '+str(round(pnt*trk))+' % Complete'
                trk += 1
        if part[0] > 0:
            if (i-part[0]) == round(chk*trk):
                print '    Process 2 : '+str(round(pnt*trk))+' % Complete'
                trk += 1
            
    que.put(results)        

sg.init()
h_min, h_max, m_mom_min, m_mom_max, q, V_min, V_max, Om_min, Om_max, Dia = sg.h_min, sg.h_max, sg.m_mom_min, sg.m_mom_max, sg.q, sg.V_min, sg.V_max, sg.Om_min, sg.Om_max, sg.Dia

print '\nImport Particles'

nproc = 2
Im = 10000
mult = Im/nproc
ind = 3
fileNames = ['HZD_fullQUAT_TST_{}.txt'.format(ind),'HZC_fullQUAT_TST_{}.txt'.format(ind)]

myData = np.genfromtxt('SimQuat_Data.txt',skip_header=24)
myData_len = len(myData)
partPop = range(myData)

Particles = []
I = []
for i in range(Im):
    ind = np.random.choice(partPop)
    partPop.remove(ind)
    Particles.append(myData[ind])
    I.append(ind)

queue = multiprocessing.Queue()
procs = []

for i in range(nproc):
    p = multiprocessing.Process(target=worker, args=(range(mult*i,mult*(i+1)), queue))
    procs.append(p)
    p.start()

resultFull = {}
for i in range(nproc):
    resultFull.update(queue.get())

for p in procs:
    p.join()

track = np.zeros(6)
cTrol = np.empty((Im, 5))
fiCon = np.empty((Im, 4))
for i in range(Im):
    P = Particles[i]
    inCon = P.cTrol
    length = P.length
    fiCon[i] = resultFull[i][0:4]
    track = np.add(track, resultFull[i][4:10])
    cTrol[i] = [length, inCon[0,0], inCon[0,1], inCon[1,0], inCon[1,1]]

print '\n'+str(np.sum(track[0:3]))+' Particles Failed'
if np.sum(track[0:3]) > 0:
    print '    '+str(track[0])+' Proximity Phase'
    print '    '+str(track[1])+' Pre Impact Phase'
    print '    '+str(track[2])+' Post Impact Phase'

print '\n'+str(np.sum(track[3:5]))+' Particles Flatten'
print '    '+str(track[3])+' Tor_grav > Tor_field'
print '    '+str(track[4])+' Tor_grav < Tor_field'

print '\n'+str(track[5])+' Particles Never Flatten'

print '\nWritting Data File'

if isfile(fileNames[0]):
   
    track_rd = []
    fp = open(fileNames[0])
    for i, line in enumerate(fp):
        if i == 3:
            Parts = getNumb(line)
        elif i == 12:
            track_rd.append(getNumb(line))
        elif i == 13:
            track_rd.append(getNumb(line))
        elif i == 14:
            track_rd.append(getNumb(line))
        elif i == 17:
            track_rd.append(getNumb(line))
        elif i == 18:
            track_rd.append(getNumb(line))
        elif i == 20:
            track_rd.append(getNumb(line))
        elif i > 21:
            break
    fp.close

    track = np.add(track_rd, track)
    Parts = Parts + Im
    
    with open(fileNames[0], 'r') as file:
        data = file.readlines()
        
    data[3] = str(round(Parts))+' Individual Grains\n'
    data[11] = str(round(sum(track[0:3])))+' Particles Failed\n'
    data[12] = '    '+str(round(track[0]))+' Proximity Phase\n'
    data[13] = '    '+str(round(track[1]))+' Pre Impact Phase\n'
    data[14] = '    '+str(round(track[2]))+' Post Impact Phase\n'
    data[16] = str(round(sum(track[3:5])))+' Particles Flatten\n'
    data[17] = '    '+str(round(track[3]))+' Tor_grav > Tor_field\n'
    data[18] = '    '+str(round(track[4]))+' Tor_grav < Tor_field\n'
    data[20] = str(round(track[5]))+' Particles Never Flatten\n'
    
    with open(fileNames[0], 'w') as file:
        file.writelines(data)
    
    file = open(fileNames[0], 'a')
    for i in range(Im):
        file.write(str(fiCon[i,0])+' '+str(fiCon[i,1])+' '+str(fiCon[i,2])+' '+str(fiCon[i,3])+' '+str(cTrol[i,0])+'\n')
    file.close()
    
else:
    file = open(fileNames[0], 'w')
    
    file.write('Horizontal Dipole\n'
               'Magnetic Field Location : B_Field_Horz.txt \n'
               'Magnetic Dipole Strength : B_Field_Horz.txt \n'
               +str(Im)+' Individual Grains\n'
               '('+str(h_min)+' to '+str(h_max)+')m length of grains\n'
               '('+str(m_mom_min)+' to '+str(m_mom_max)+')Am^2 magnetic moment of grains\n' 
               +str(q)+' C charge on grains \n'
               '('+str(V_min)+' to '+str(V_max)+')m/s initial linear velocity \n'
               '('+str(Om_min)+' to '+str(Om_max)+')rad/s initial angular velocity \n'
               '('+str(Dia)+' x '+str(Dia)+')m^2 Landing Area \n'
               '\n'
               +str(np.sum(track[0:3]))+' Particles Failed \n'
               '    '+str(track[0])+' Proximity Phase \n'
               '    '+str(track[1])+' Pre Impact Phase \n'
               '    '+str(track[2])+' Post Impact Phase \n'
               '\n'
               +str(np.sum(track[3:5]))+' Particles Flatten \n'
               '    '+str(track[3])+' Tor_grav > Tor_field \n'
               '    '+str(track[4])+' Tor_grav < Tor_field \n'
               '\n'
               +str(track[5])+' Particles Never Flatten \n'
               '\n'
               'Processing Time : \n'
               '\n')   
    
    for i in range(Im):
        file.write(str(fiCon[i,0])+' '+str(fiCon[i,1])+' '+str(fiCon[i,2])+' '+str(fiCon[i,3])+' '+str(cTrol[i,0])+'\n')
    file.close()

if isfile(fileNames[1]):
    file = open(fileNames[1], 'a')
    for i in range(Im):
        file.write(str(cTrol[i,1])+' '+str(cTrol[i,2])+' '+str(cTrol[i,3])+' '+str(cTrol[i,4])+'\n')
    file.close()
    
else:
    file = open(fileNames[1], 'w')
    file.write('Horizontal Dipole\n'
               'Magnetic Field Location : B_Field_Horz.txt\n'
               'Magnetic Dipole Strength : B_Field_Horz.txt\n'
               +str(Im)+' Individual Grains\n'
               '('+str(h_min)+' to '+str(h_max)+')m length of grains\n'
               '('+str(m_mom_min)+' to '+str(m_mom_max)+')Am^2 magnetic moment of grains\n' 
               +str(q)+' C charge on grains \n'
               '('+str(V_min)+' to '+str(V_max)+')m/s initial linear velocity \n'
               '('+str(Om_min)+' to '+str(Om_max)+')rad/s initial angular velocity \n'
               '('+str(Dia)+' x '+str(Dia)+')m^2 Landing Area \n'
               '\n')
    for i in range(Im):
        file.write(str(cTrol[i,1])+' '+str(cTrol[i,2])+' '+str(cTrol[i,3])+' '+str(cTrol[i,4])+'\n')
    file.close()

print '\nProcessing Time : '+str(round(time.time() - start_time, 1))+' Seconds'