#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:51:00 2019

@author: michael
"""

#import global_var as var
import numpy as np
from os import listdir

def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def mult_Q(P, R, coeff=False):
    """ Performs quaternionic multiplicatione between P and R. """
    
    if coeff == False:
        coeff = 1.0
    
    u0 = coeff*(P[0]*R[0] - np.dot(P[1:4],R[1:4]))
    u = coeff*np.add(np.add(P[0]*R[1:4],R[0]*P[1:4]),np.cross(P[1:4],R[1:4]))
    
    Q = np.array([u0, u[0], u[1], u[2]])
    return Q
    
def conj_Q(Q):
    """ Returns the complex conjugate of Q."""
    Q_new = np.array([Q[0], -Q[1], -Q[2], -Q[3]])
    return Q_new    

def B_field(R, var):
    """ Returns B_field at location R as described by Reiner Gamma field data."""
    '''
    r_SW = R - var['SW_r']
    r_SW_norm = np.linalg.norm(r_SW)
    B_SW_1 = (3*np.dot(var['SW_mag'],r_SW)*r_SW)/(r_SW_norm**5)
    B_SW_2 = var['SW_mag']/(r_SW_norm**3)
    B_SW = np.subtract(B_SW_1, B_SW_2)*10**-7
    
    r_NE = R - var['NE_r']
    r_NE_norm = np.linalg.norm(r_NE)
    B_NE_1 = (3*np.dot(var['NE_mag'],r_NE)*r_NE)/(r_NE_norm**5)
    B_NE_2 = var['NE_mag']/(r_NE_norm**3)
    B_NE = np.subtract(B_NE_1, B_NE_2)*10**-7
    
    return B_SW + B_NE
    '''
    r = R - var['dipole_position']
    r_norm = np.linalg.norm(r)
    B = 10**-7 * ( ( ( 3 * np.dot( var['dipole_moment'], r) * r ) / ( r_norm**5 ) ) - (var['dipole_moment'] / ( r_norm**3 ) ) ) 

    return B
    '''    
    R_o = np.array([-325000, -325000, 1898000])
    x, y, z = np.subtract(R, R_o)
    
    if (-678.497 < x < 650678) and (-678.497 < y < 650678) and (-194729 < z < 678.497):
        i = int(abs(x) / 1357)
        j = int(abs(y) / 1357)
        k = int(abs(z) / 1357)
        
        return np.array( var['B_data'][k,j,i] )
        
    elif (-678.497 < x < 650678) and (-678.497 < y < 650678) and (z <= -194729):
        i = int(abs(x) / 1357)
        j = int(abs(y) / 1357)
        k = 143
        
        return np.array( var['B_data'][k,j,i] )
    
    else:
        return np.array([0, 0, 0])  
    '''
    
def E_field(R, var):
    """ Returns E_field at location R as described by Reiner Gamma field data."""
    '''
    R_o = np.array([-325000, -325000, 1898000])
    x, y, z = np.subtract(R, R_o)
    
    if (-678.497 < x < 650678) and (-678.497 < y < 650678) and (-194729 < z < 678.497):
        i = int(abs(x) / 1357)
        j = int(abs(y) / 1357)
        k = int(abs(z) / 1357)
        
        return np.array( var['E_data'][k, j, i] )
        
    elif (-678.497 < x < 650678) and (-678.497 < y < 650678) and (z <= -194729):
        i = int(abs(x) / 1357)
        j = int(abs(y) / 1357)
        k = 143
        
        return np.array( var['E_data'][k, j, i] )
        
    else:
        return np.array([0, 0, 0])  
    '''
    return np.zeros(3)
    
def integrate_flight(t, InCon, Prams):
    """ Track grain through phase space during flight.

    Parameters
    ----------
    InCon : array
        phase space of grain
        -position : {0:3} 
        -velocity : {3:6}
        -quaternion : {6:10}
        -quaternion derivative : {10:14}
        
    Prams : dict
        system parameters
        'length'   : length of grain
        'mass'     : mass of grain  
        'charge'   : charge on grain
        'magnetic' : magnetic moment of grain
        'momentx'  : moment of inertia in X
        'momentz'  : moment of inertia in Z
    """
    r = InCon[0:3]
    r_nit = r/np.linalg.norm(r)
    v = InCon[3:6]
    quat = InCon[6:10]
    quat_norm = np.linalg.norm(quat)
    quat = quat/quat_norm    
    quat_deriv = InCon[10:14]
    
    B = B_field(r, Prams['var'])
    E = E_field(r, Prams['var']) 
    
    # Forces acting on the center of mass
    accel = -Prams['var']['g']*r_nit
    accel = accel + (Prams['charge']/Prams['mass']) * np.cross(v, B)
    accel = accel + (Prams['charge']/Prams['mass'])*E

    # Rotation
    B_body = mult_Q(conj_Q(quat), mult_Q(np.array([0.0, B[0], B[1], B[2]]),quat))[1:4]
    torq = Prams['magnetic']*np.array([-B_body[1], B_body[0], 0])
    
    angVel = 2*mult_Q(conj_Q(quat), quat_deriv)[1:4]
    angMom = np.array([Prams['momentx']*angVel[0], Prams['momentx']*angVel[1], Prams['momentz']*angVel[2]])
    angRot = np.cross(angVel, angMom)
    
    tRot = np.subtract(torq, angRot)
    moment_inv = 1./Prams['momentx']
    tRot_Q = np.array([0.0, moment_inv*tRot[0], moment_inv*tRot[1], tRot[2]/Prams['momentz']])
    
    ddQ_1 = mult_Q(quat_deriv, mult_Q(conj_Q(quat),quat_deriv))
    ddQ_2 = 0.5*mult_Q(quat, tRot_Q)
    ddQ = np.add(ddQ_1, ddQ_2)
    
    # Change rates
    deriv = [quat_deriv, ddQ]
    deriv = np.insert(deriv, 0, accel)
    deriv = np.insert(deriv, 0, v)  
    return deriv

def integrate_impact(t, InCon, Prams):
    """ Track grain through phase space during its impact with the surface.
    
    Parameters
    ----------
    t : float
        time step used by integrator (defined in SciPy module)
    
    InCon : array
        phase space of grain
        -quaternion = {0:4}
        -quaternion derivative = {4:8}
        
    Prams : dict
        system parameters
        'momentx'   :   moment of inertia in x
        'momentz'   :   moment of inertia in z
        'accel'     :   acceleration on center of mass
    """
    
    quat = InCon[0:4]
    quat_norm = np.linalg.norm(quat)
    quat = quat/quat_norm    
    quat_deriv = InCon[4:8]    
    
    accel_body = mult_Q(conj_Q(quat), mult_Q(Prams['accel'], quat))
    torq = np.array([-accel_body[2], accel_body[1], 0.0])
    
    angVel = 2*mult_Q(conj_Q(quat), quat_deriv)[1:4]
    angMom = np.array([Prams['momentx']*angVel[0], Prams['momentx']*angVel[1], Prams['momentz']*angVel[2]])
    angRot = np.cross(angVel, angMom)
    
    tRot = np.subtract(torq, angRot)
    momentx_inv = 1./Prams['momentx']
    tRot_Q = np.array([0.0, momentx_inv*tRot[0], momentx_inv*tRot[1], tRot[2]/Prams['momentz']])
    
    quat_deriv_deriv = mult_Q(quat_deriv, mult_Q(conj_Q(quat),quat_deriv))
    quat_deriv_deriv = quat_deriv_deriv + 0.5*mult_Q(quat, tRot_Q)
    
    return np.insert(quat_deriv_deriv, 0, quat_deriv)  

def dipole_moment_conversion(lat, lon, inc, dec, depth, moment, var):
    """ Conversts selenographic coordinates and angles of magnetic 
        inclination/declination to simulation coordinates for the 
        location and orientation of a dipolar magnetic moment.
        
    Parameters
    ----------
    lat : float
        Angle of lattitude
            90 deg north = 90
            90 deg south = -90
        
    lon : float
        Angle of longitude
            90 deg west = 90
            90 deg east = -90
            
    inc : float
        Angle of inclination, down from horizontal plane is positive (deg).
    
    dec : float
        Angle of declination, east of north is positive (deg).
        
    depth : float
        Depth of dipole below lunar surface (m).
        
    moment : float
        Strength of dipole (Am^2).
        
    r_sim : array
        Location of dipole in simulation coordinates
        
    mag_sim : array
        Orientation of magnetic moment in simulation coordinates
    """     
    inv_180 = 1./180
    
    sin_172 = np.sin(172*np.pi*inv_180)   
    cos_172 = np.cos(172*np.pi*inv_180)
    
    sin_32 = np.sin(32*np.pi*inv_180)
    cos_32 = np.cos(32*np.pi*inv_180)
    
    sin_82 = np.sin(82*np.pi*inv_180)
    cos_82 = np.cos(82*np.pi*inv_180)
    
    sin_122 = np.sin(122*np.pi*inv_180)
    cos_122 = np.cos(122*np.pi*inv_180)
    
    M = np.array([[sin_172*cos_32, sin_172*sin_32, cos_172],
                  [sin_82*cos_122, sin_82*sin_122, cos_82],
                  [sin_82*cos_32, sin_82*sin_32, cos_82]])
    
    phi = (90.-lat) * (np.pi/180)
    theta = (90.-lon) * (np.pi/180)
    
    r = np.array([np.sin(phi)*np.cos(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(phi)])
    
    r_sim = np.dot(M, r)
    r_sim = (r_sim/np.linalg.norm(r_sim)) * (var['r_m'] - depth)
    
    z_tan = np.array([0,0,1]) - r[2]*r
    z_tan = z_tan/np.linalg.norm(z_tan)
    
    x_tan = np.cross(z_tan, r)
    
    alpha = (90 + inc) * (np.pi / 180)
    beta = (90 - dec) * (np.pi / 180)
    
    mag = x_tan * (np.sin(alpha) * np.cos(beta)) + z_tan * (np.sin(alpha) * np.sin(beta)) + r * np.cos(alpha)
    
    mag_sim = np.dot(M, mag)
    mag_sim = (mag_sim/np.linalg.norm(mag_sim)) * moment
    
    return r_sim, mag_sim

