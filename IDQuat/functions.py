#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 18:56:02 2018

@author: michael
"""

import global_var as var
import numpy as np

def B_field(R):
    """ Returns B_field at location R as described an idealized magnetic dipole."""
    R_o = np.array([0, 0, -.01])
    r = np.subtract(R, R_o)
    r_norm = np.linalg.norm(r)
    B_1 = (3*np.dot(var.mag,r)*r)/(r_norm**5)
    B_2 = var.mag/(r_norm**3)
    B = np.subtract(B_1, B_2)*10**-7
    
    return B

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
    v = InCon[3:6]
    quat = InCon[6:10]
    quat_norm = np.linalg.norm(quat)
    if (quat_norm > 1.00001) or (quat_norm < 0.99999):
        quat = quat/quat_norm    
    quat_deriv = InCon[10:14]
    
    B = B_field(r)

    # Forces acting on the center of mass
    Acc_g = np.array([0,0,-var.g])
    Acc_b = (Prams['charge']/Prams['mass']) * np.cross(v, B)
    Acc = np.add(Acc_g, Acc_b)

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
    deriv = np.insert(deriv, 0, Acc)
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
    if (quat_norm > 1.00001) or (quat_norm < 0.99999):
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
    
    q1 = mult_Q(quat_deriv, mult_Q(conj_Q(quat),quat_deriv))
    q2 = 0.5*mult_Q(quat, tRot_Q)
    quat_deriv_deriv = np.add(q1, q2)
    
    return np.insert(quat_deriv_deriv, 0, quat_deriv)        