#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:23:16 2018

@author: michael
"""
import global_var as var
import functions as fun

import scipy.integrate as sp
#import threading
#import multiprocessing as mp

import numpy as np

class tracking_DustGrains():
    """Generate a duct grain object."""
    def __init__(self, num_of_grains, succ_queue, fail_queue):
        self.s_queue = succ_queue
        self.f_queue = fail_queue
        self.num = num_of_grains
        
        for i in range(num_of_grains):
            self.succ = 0
            self.fall_fail = 0
            self.impact_fail = 0
            self.collide_fail = 0
            self.lift_fail = 0
            self.never_fail = 0    
            
            
            self.initial_conditions()
            self.track_phases()
            
            self.fail_count = np.array([self.fall_fail,
                                        self.impact_fail,
                                        self.collide_fail,
                                        self.succ+self.lift_fail,
                                        self.lift_fail,
                                        self.never_fail])
            self.f_queue.put(self.fail_count)

    def initial_conditions(self, ell_mode=False):       
        """Generate a randomized set of initial conditions.
        
        Parameters
        ----------
        InCon : array
            phase space of grain
            -position : {0:3} 
            -velocity : {3:6}
            -quaternion : {6:9}
            -quaternion derivative : {9:13}
            
        Prams : array
            system parameters
            'length'   : length of grain
            'mass'     : mass of grain  
            'charge'   : charge on grain
            'magnetic' : magnetic moment of grain
            'momentx'  : moment of inertia in X
            'momentz'  : moment of inertia in Z
        """
        radial_landing_space = var.Dia*np.random.random()
        angle = 2*np.pi*np.random.random()
        
        x = radial_landing_space*np.cos(angle)
        y = radial_landing_space*np.sin(angle)   
        z = 1.
        r = np.array([x, y, z])
        
        InCon = r
        
        vx = 0
        vy = 0
        vz = 0
        v = np.array([vx, vy, vz])
        
        InCon = np.append(InCon, v)
    
        alpha = np.pi*np.random.random()
        theta = 2*np.pi*np.random.random()
        quaternion_vector = np.sin(alpha/2)*np.array([0,np.cos(theta),np.sin(theta),0])
        quaternion_scalar = np.array([np.cos(alpha/2), 0.0, 0.0, 0.0])  
        quaternion = np.add(quaternion_vector, quaternion_scalar)
        
        InCon = np.append(InCon, quaternion)
        
        quaternion_derivative = np.zeros(4)
        
        InCon = np.append(InCon, quaternion_derivative)
        
        diameter = np.random.choice(var.h_pop)   
        density = np.random.choice(var.den_pop) 
        Fe_by_weight = np.random.choice(var.FeByWght_pop)
        
        if ell_mode:
            """ Ellipsoid Particles (AspectRatio==0.5) """
            length = (2.**(2./3))*(.5*var.diameter)
            vol = (1./3)*np.pi*(length)**3 
            mass = density*vol  # Mass [g]      
            I_x = ((.5)**(1./3))*mass*(.5*diameter)*(.5*diameter)
            I_z = ((2.**(1./3))/5)*mass*(.5*diameter)*(.5*diameter)
        else:
            """ Needle Particles (AspectRatio==0.1) """
            length = 4.0548*diameter
            vol = (4.0/3.0)*np.pi*(diameter/2.0)**3     
            mass = density*vol  # Mass [g]      
            I_x = (403./4800)*mass*length*length  
            I_z = (1./800)*mass*length*length
        
        magnetic = Fe_by_weight*mass
        charge = (-1.6e-19)*np.random.choice(var.q_pop) 
        
        B = fun.B_field(r)
        B_body = fun.mult_Q(quaternion, fun.mult_Q(np.array([0,B[0],B[1],B[2]]),fun.conj_Q(quaternion)))[1:4]
        PE = -magnetic*B_body[2]
        En = PE
        
        Prams = {'length'   :   length, 
                 'mass'     :   mass,
                 'charge'   :   charge,
                 'magnetic' :   magnetic,
                 'momentx'  :   I_x,
                 'momentz'  :   I_z,
                 'energy'   :   En}
        
        self.InCon = InCon
        self.Prams = Prams
            
    def track_phases(self, integrate_mode='vode', nstp=5000):
        """Track grain through phase space across all flight phases."""
        self.grav = sp.ode(fun.integrate_flight).set_integrator(integrate_mode,nsteps=nstp)
        self.grav.set_f_params(self.Prams).set_initial_value(self.InCon, 0.0)
        
        self.falling_phase()
        if not self.grav.successful():
            self.fall_fail+=1
            
        self.impacting_phase()
        if not self.grav.successful():
            self.impact_fail+=1  
            
        self.colliding_phase(integrate_mode, nstp)
    
    def falling_phase(self):
        """ Track dust grain to within ten body lengths of the surface. """    
        div = 1.0
        dt_base = np.sqrt(1.8/var.g)
       
        self.grav.integrate(self.grav.t + dt_base)
        
        if self.grav.successful():
            magnetic_field = fun.B_field(self.grav.y[0:3])
            a = (self.Prams['charge']/self.Prams['mass']) * np.cross(self.grav.y[3:6], magnetic_field)[2]
            a = a - var.g     
            while self.grav.y[2] >= (10*self.Prams['length']) and self.grav.successful():     
                dt = dt_base/div
                dr = self.grav.y[5]*dt+.5*a*dt*dt
                while self.grav.y[2] >= (self.Prams['length']-dr) and self.grav.successful():
                    self.grav.integrate(self.grav.t + 0.999*dt)
                    magnetic_field = fun.B_field(self.grav.y[0:3])
                    a = (self.Prams['charge']/self.Prams['mass']) * np.cross(self.grav.y[3:6], magnetic_field)[2]
                    a = a - var.g
                div = div*10.0
            
    def impacting_phase(self):
        """ Track dust grain until it collides with the surface. """
        r = self.grav.y[0:3]
        v = self.grav.y[3:6]
        
        magnetic_field = fun.B_field(r)
        a = (self.Prams['charge']/self.Prams['mass']) * np.cross(v, magnetic_field)[2]
        a = a - var.g
        dr = self.Prams['length']-r[2]
        
        if dr < 0:
            dt = (-v[2]-np.sqrt(v[2]*v[2]+2*dr*a))/a
            self.grav.integrate(self.grav.t+dt)
        
        quat = self.grav.y[6:10]/np.linalg.norm(self.grav.y[6:10])
        orient = fun.mult_Q(quat,fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(quat)))[1:4]
        r_low = self.grav.y[2]-(.5*self.Prams['length']*abs(orient[2]))
        dt = (-self.grav.y[5]-np.sqrt(self.grav.y[5]*self.grav.y[5]-0.1*self.Prams['length']*a))/a
        while r_low >= 0 and self.grav.successful():
            self.grav.integrate(self.grav.t + dt)
            quat = self.grav.y[6:10]/np.linalg.norm(self.grav.y[6:10])
            orient = fun.mult_Q(quat,fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(quat)))[1:4]
            r_low = self.grav.y[2]-(.5*self.Prams['length']*abs(orient[2]))

    def colliding_phase(self, integrate_mode, nstp):
        """ Track dust grain until it lies flatt on the surface."""
        r = self.grav.y[0:3]
        v = self.grav.y[3:6]
        quat = self.grav.y[6:10]
        quat_deriv = self.grav.y[10:14]
        
        orient = fun.mult_Q(quat,fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(quat)))[1:4]
        
        Om_space = 2*fun.mult_Q(quat_deriv, fun.conj_Q(quat))[1:4]
        sign = orient[2]/abs(orient[2])
        
        B = fun.B_field(r)
        F_grav = np.array([0,0,-var.g*self.Prams['mass']])
        accel = np.add(self.Prams['magnetic']*B, .5*self.Prams['length']*sign*F_grav)
        accel = np.insert(accel, 0, 0)
        
        # Define Body axes in space frame (e_1, e_2, e_3)
        E_3 = fun.mult_Q(quat, fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(quat)))[1:4]
        e_3 = E_3/np.linalg.norm(E_3) 
        E_2 = fun.mult_Q(quat, fun.mult_Q(np.array([0,0,1,0]), fun.conj_Q(quat)))[1:4]
        e_2 = E_2/np.linalg.norm(E_2)
        e_1 = np.cross(e_2, e_3)
        
        # Include the dust grain's linear velocity with the new angular
        # velocity, then use the new angular velocity to calcuate the 
        # initial dQ
        Om_1 = (-2.0/self.Prams['length'])*sign*np.dot(v, e_2)
        Om_2 = (2.0/self.Prams['length'])*sign*np.dot(v, e_1)
        Om_space = np.add(Om_space, np.add(Om_1*e_1, Om_2*e_2))
        quat_deriv = 0.5*fun.mult_Q(np.array([0,Om_space[0],Om_space[1],Om_space[2]]), quat)
        
        impact_incon = np.insert(quat_deriv, 0, quat)
        
        Prams_new = {}
        Prams_new['momentz'] = self.Prams['momentz']
        Prams_new['momentx'] = self.Prams['momentx'] + .25*self.Prams['mass']*self.Prams['length']*self.Prams['length']
        Prams_new['accel'] = accel
        
        lim = 0
        e3_chg_prev = -1.0
        e3_prev = abs(e_3[2])
        
        # Call Collision Integrator
        Om_body = 2*fun.mult_Q(fun.conj_Q(quat),quat)[1:4]
        V_rem = np.subtract(v, np.dot(v,e_3)*e_3)
        EnLIN = .5*self.Prams['mass']*np.dot(V_rem,V_rem)
        EnTOT = self.Prams['energy'] + EnLIN
        dt = (np.arccos(e3_prev-.1)-np.arccos(e3_prev))*np.sqrt(self.Prams['momentx']/(2*EnTOT-Prams_new['momentz']*Om_body[2]*Om_body[2]))
        
        self.impact = sp.ode(fun.integrate_impact).set_integrator(integrate_mode,nsteps=nstp)
        self.impact.set_f_params(Prams_new).set_initial_value(impact_incon, 0.0)
        while (e3_prev >= 0.1) and (lim < 5) and self.impact.successful():
            self.impact.integrate(self.impact.t + dt)
            Q = self.impact.y[0:4]/np.linalg.norm(self.impact.y[0:4])
            e_3 = fun.mult_Q(Q, fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(Q)))[1:4]
            
            # Check the paritcle for signs of infinite oscillation
            e3_2 = abs(e_3[2])
            e3_chg = (e3_2 - e3_prev)/abs(e3_2 - e3_prev)
            e3_prev = e3_2
            if not e3_chg == e3_chg_prev:
                e3_chg_prev = e3_chg
                lim += 1
                
        if self.impact.successful() and (lim < 5):
            # The dust grain may only have flattened because of its linear momentum.
            # Therefore, we must check to see if the torque due to its magnetic moment
            # might be sufficiently strong to overcome graity and lift it off the lunar surface
            T_nit = np.cross(e_3, np.array([0,0,1]))
            t_nit = T_nit/np.linalg.norm(T_nit)
            T_grav = -0.5*self.Prams['length']*self.Prams['mass']*var.g*t_nit
            T_mag = self.Prams['magnetic']*np.cross(e_3, B)
            T_net = np.add(T_grav, np.dot(T_mag, t_nit)*t_nit)
            lft_check = np.dot(T_net, t_nit)
            if lft_check <= 0.0:
                data_new = r
                data_new = np.append(data_new, e_3)
                data_new = np.append(data_new, self.Prams['length'])
                
                self.s_queue.put(data_new)
                self.succ+=1
                
            else:
                self.lift_fail+=1
                
        elif self.impact.successful() and not (lim < 5):
            # Dust Grain Never Flatened Out
            self.never_fail+=1
            
        else:
            # Post-Collison Phase Failure 
            self.collide_fail+=1
            