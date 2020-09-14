#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:58:34 2019

@author: michael
"""

import global_var as gVar
import functions as fun

import scipy.integrate as sp
import numpy as np

class dust_grain():
    def __init__(self, namelist):
        self.var = gVar.variables(namelist)
        self.counts = {'success' : 0,
                       'rise fail' : 0,
                       'fall fail' : 0,
                       'impact fail' : 0,
                       'collide fail' : 0,
                       'lift fail' : 0,
                       'never fail' : 0}
        
        self.initial_conditions()
        

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
            
        Prams : dictionary
            system parameters
            'length'   : length of grain
            'mass'     : mass of grain  
            'charge'   : charge on grain
            'magnetic' : magnetic moment of grain
            'momentx'  : moment of inertia in X
            'momentz'  : moment of inertia in Z
        """        
        zetta = self.var['zettaMAX']*np.random.random()
        alpha = 2*np.pi*np.random.random()
        
        x = np.sin(zetta)*np.cos(alpha)
        y = np.sin(zetta)*np.sin(alpha)    
        z = np.cos(zetta)
        
        r = self.var['r_m'] * np.array([x, y, z])
        
        InCon = r
        
        theta = self.var['thetaMAX']*np.random.random()
        phi = 2*np.pi*np.random.random()
        
        r_unit = r / np.linalg.norm(r)
        
        X_Tan = np.array([1,0,0]) - r_unit[0] * r_unit
        x_tan = X_Tan / np.linalg.norm(X_Tan)
        
        Y_Tan = np.cross(r_unit, x_tan)
        y_tan = Y_Tan / np.linalg.norm(Y_Tan)
        
        vx = np.cos(phi)*np.sin(theta) * x_tan
        vy = np.sin(phi)*np.sin(theta) * y_tan
        vz = np.cos(theta) * r_unit
        v_norm = np.random.choice(self.var['V_pop'])
        
        v = v_norm * (vx + vy + vz)
        
        InCon = np.append(InCon, v)
        
        gamma = 2*np.pi*np.random.random()
        quaternion = (1/np.sqrt(2))*np.array([1.0, -np.sin(gamma), np.cos(gamma), 0.0])
        
        InCon = np.append(InCon, quaternion)
        
        theta = 2*np.pi*np.random.random()
        phi = np.pi*np.random.random()
        
        omega_x = np.sin(phi) * np.cos(theta)
        omega_y = np.sin(phi) * np.sin(theta)
        omega_z = np.cos(phi)
        omega_norm = np.random.choice(self.var['Om_pop'])
        
        omega = omega_norm * np.array([0.0, omega_x, omega_y, omega_z])
        quaternion_derivative = 0.5*fun.mult_Q(omega, quaternion) 
        
        InCon = np.append(InCon, quaternion_derivative)
        
        diameter = np.random.choice(self.var['h_pop'])   
        density = np.random.choice(self.var['den_pop']) 
        Fe_by_weight = np.random.choice(self.var['FeByWght_pop'])
        
        if ell_mode:
            """ Ellipsoid Particles (AspectRatio==0.5) """
            length = (2.**(2./3))*(.5*diameter)
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
        charge = (-1.6e-19)*np.random.choice(self.var['q_pop']) 
        
        B = fun.B_field(r, self.var)
        B_body = fun.mult_Q(quaternion, fun.mult_Q(np.array([0,B[0],B[1],B[2]]),fun.conj_Q(quaternion)))[1:4]
        PE = -magnetic*B_body[2]
        En = PE
        
        Prams = {'length'   :   length, 
                 'mass'     :   mass,
                 'charge'   :   charge,
                 'magnetic' :   magnetic,
                 'momentx'  :   I_x,
                 'momentz'  :   I_z,
                 'energy'   :   En,
                 'var'      :   self.var}
        
        self.InCon = InCon
        self.Prams = Prams
        
            
    def track_phases(self, integrate_mode='vode', nstp=5000):
        """Track grain through phase space across all flight phases."""
        self.grav = sp.ode(fun.integrate_flight).set_integrator(integrate_mode,nsteps=nstp)
        self.grav.set_f_params(self.Prams).set_initial_value(self.InCon, 0.0)
        
        self.rising_phase()
        if not self.grav.successful():
            self.counts['rise fail'] += 1
            self.sim_results = np.full(7,np.nan).reshape(1,7)
        else:
            self.falling_phase()
            if not self.grav.successful():
                self.counts['fall fail']+=1
                self.sim_results = np.full(7,np.nan).reshape(1,7)
            else:
                self.impacting_phase()
                if not self.grav.successful():
                    self.counts['impact fail']+=1 
                    self.sim_results = np.full(7,np.nan).reshape(1,7)
                else:
                    self.colliding_phase(integrate_mode, nstp)
                    
    def rising_phase(self):
        """Track dust grain past its maximum displacement from lunar surface. """
        r = self.grav.y[0:3]
        r_unit = r/np.linalg.norm(r)
        v = self.grav.y[3:6]
        v_radial = np.dot(v, r_unit)
        
        B = fun.B_field(r, self.var)
        E = fun.E_field(r, self.var)
        
        a = - self.var['g'] * r_unit + (self.Prams['charge']/self.Prams['mass']) * (np.cross(v, B) + E)
        a_radial = np.dot(a, r_unit)
        
        self.dt_base = -.5 * ( v_radial / a_radial )
        r_norm_prev = self.var['r_m']
        while self.grav.successful():
            self.grav.integrate(self.grav.t + self.dt_base)
            r_norm = np.linalg.norm(self.grav.y[0:3])
            if r_norm > r_norm_prev:
                r_norm_prev = r_norm
            else:
                break     
    
    def falling_phase(self):
        """ Track dust grain to within ten body lengths of the surface. """    
        div = 1.0
        
        r = self.grav.y[0:3]
        r_norm = np.linalg.norm(r)
        r_unit = r/r_norm 
         
        v = self.grav.y[3:6]
        v_radial = np.dot(v, r_unit)
        
        B = fun.B_field(r, self.var)
        E = fun.E_field(r, self.var)
        
        a = - self.var['g'] * r_unit + (self.Prams['charge']/self.Prams['mass']) * (np.cross(v, B) + E)
        a_radial = np.dot(a, r_unit)
        
        while (r_norm) >= (self.var['r_m'] + 10*self.Prams['length']) and self.grav.successful():     
            dt = self.dt_base/div
            dr = v_radial * dt + .5 * a_radial*(dt**2)
            while r_norm >= (self.var['r_m'] + self.Prams['length']-dr) and self.grav.successful():
                self.grav.integrate(self.grav.t + 0.999*dt)
                
                r = self.grav.y[0:3]
                r_norm = np.linalg.norm(r)
                r_unit = r/r_norm 
                 
                v = self.grav.y[3:6]
                v_radial = np.dot(v, r_unit)
                
                B = fun.B_field(r,self.var)
                E = fun.E_field(r,self.var)
                
                a = - self.var['g'] * r_unit + (self.Prams['charge']/self.Prams['mass']) * (np.cross(v, B) + E)
                a_radial = np.dot(a, r_unit)
                
            div = div * 10.
            
    def impacting_phase(self):
        """ Track dust grain until it collides with the surface. """
        r = self.grav.y[0:3]
        r_norm = np.linalg.norm(r)
        r_unit = r/r_norm 
         
        v = self.grav.y[3:6]
        v_radial = np.dot(v, r_unit)
        
        B = fun.B_field(r,self.var)
        E = fun.E_field(r,self.var)
        
        a = - self.var['g'] * r_unit + (self.Prams['charge']/self.Prams['mass']) * (np.cross(v, B) + E)
        a_radial = np.dot(a, r_unit)
        
        dr = (self.var['r_m'] + self.Prams['length']) - r_norm
        
        if dr < 0:
            dt = (-v_radial - np.sqrt( v_radial**2 + 2 * dr * a_radial ) ) / a_radial
            self.grav.integrate(self.grav.t + dt)
            r_norm = np.linalg.norm(self.grav.y[0:3])
            
        quat = self.grav.y[6:10] / np.linalg.norm(self.grav.y[6:10])
        orient = fun.mult_Q(quat, fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(quat) ) )[1:4]
        r_low = r_norm - (.5 * self.Prams['length'] * abs( np.dot(r_unit, orient) ) )
        dt = (-v_radial - np.sqrt( v_radial**2 - 0.1 * self.Prams['length'] * a_radial ) ) / a_radial
        while r_low >= self.var['r_m'] and self.grav.successful():
            self.grav.integrate(self.grav.t + dt)
            quat = self.grav.y[6:10] / np.linalg.norm(self.grav.y[6:10])
            orient = fun.mult_Q(quat, fun.mult_Q( np.array([0,0,0,1]), fun.conj_Q(quat) ) )[1:4]
            r_norm = np.linalg.norm( self.grav.y[0:3] )
            r_low = r_norm - ( .5 * self.Prams['length'] * abs( np.dot(r_unit, orient) ) )
            
    def colliding_phase(self, integrate_mode, nstp):
        """ Track dust grain until it lies flatt on the surface.""" 
        r = self.grav.y[0:3]
        r_unit = r / np.linalg.norm(r)
        v = self.grav.y[3:6]
        quat = self.grav.y[6:10] / np.linalg.norm(self.grav.y[6:10])
        quat_deriv = self.grav.y[10:14]
        
        orient = fun.mult_Q( quat, fun.mult_Q( np.array([0,0,0,1]), fun.conj_Q(quat) ) )[1:4]
        sign = orient[2]/abs(orient[2])
        
        B = fun.B_field(r,self.var)
        E = fun.E_field(r,self.var)
        
        F_grav = - self.Prams['mass'] * self.var['g'] * r_unit
        accel = self.Prams['magnetic'] * B
        accel = accel + .5*self.Prams['length']*sign*(F_grav + self.Prams['charge']*E)
        accel = np.insert(accel, 0, 0)
        
        # Define Body axes in space frame (e_1, e_2, e_3)
        E_3 = fun.mult_Q(quat, fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(quat)))[1:4]
        e_3 = E_3/np.linalg.norm(E_3) 
        E_2 = fun.mult_Q(quat, fun.mult_Q(np.array([0,0,1,0]), fun.conj_Q(quat)))[1:4]
        e_2 = E_2/np.linalg.norm(E_2)
        e_1 = np.cross(e_2, e_3)
        
        # Include the dust grain's linear velocity with the new angular velocity
        Om_space = 2*fun.mult_Q(quat_deriv, fun.conj_Q(quat))[1:4]
        
        Om_1 = (-2.0/self.Prams['length'])*sign*np.dot(v, e_2)
        Om_2 = (2.0/self.Prams['length'])*sign*np.dot(v, e_1)
        Om_space = Om_space + Om_1*e_1 + Om_2*e_2
        quat_deriv = 0.5*fun.mult_Q(np.array([0,Om_space[0],Om_space[1],Om_space[2]]), quat)
        
        impact_incon = np.insert(quat_deriv, 0, quat)
        
        self.Prams_new = {}
        self.Prams_new['momentz'] = self.Prams['momentz']
        self.Prams_new['momentx'] = self.Prams['momentx'] + .25*self.Prams['mass']*self.Prams['length']*self.Prams['length']
        self.Prams_new['accel'] = accel
        
        lim = 0
        e3_chg_prev = -1.0
        e3_prev = abs(np.dot(e_3, r_unit))
        
        # Call Collision Integrator
        Om_body = 2*fun.mult_Q(fun.conj_Q(quat), quat)[1:4]
        V_rem = np.subtract(v, np.dot(v,e_3)*e_3)
        EnLIN = .5*self.Prams['mass']*np.dot(V_rem,V_rem)
        EnTOT = self.Prams['energy'] + EnLIN
        dt = (np.arccos(e3_prev-.1)-np.arccos(e3_prev))*np.sqrt(self.Prams['momentx']/(2*EnTOT-self.Prams_new['momentz']*Om_body[2]*Om_body[2]))
        
        impact = sp.ode(fun.integrate_impact).set_integrator(integrate_mode,nsteps=nstp)
        impact.set_f_params(self.Prams_new).set_initial_value(impact_incon, 0.0)        
        while (e3_prev >= 0.1) and (lim < 5) and impact.successful():
            impact.integrate(impact.t + dt)
            Q = impact.y[0:4]/np.linalg.norm(impact.y[0:4])
            e_3 = fun.mult_Q(Q, fun.mult_Q(np.array([0,0,0,1]), fun.conj_Q(Q)))[1:4]
            
            # Check the paritcle for signs of infinite oscillation
            e3_2 = abs( np.dot(e_3, r_unit) )
            e3_chg = (e3_2 - e3_prev)/abs(e3_2 - e3_prev)
            e3_prev = e3_2
            if not e3_chg == e3_chg_prev:
                e3_chg_prev = e3_chg
                lim += 1
                
        if impact.successful() and (lim < 6):
            # The dust grain may only have flattened because of its linear momentum.
            # Therefore, we must check to see if the torque due to its magnetic moment
            # might be sufficiently strong to overcome graity and lift it off the lunar surface
            T_nit = np.cross(e_3, r_unit)
            t_nit = T_nit / np.linalg.norm(T_nit)
            T_grav = - 0.5 * self.Prams['length'] * self.Prams['mass'] * self.var['g'] * t_nit
            T_mag = self.Prams['magnetic'] * np.cross(e_3, B)
            T_net = np.add(T_grav, np.dot(T_mag, t_nit) * t_nit)
            lft_check = np.dot(T_net, t_nit)
            if lft_check <= 0.0:
                self.counts['success']+=1
                
                data_new = r
                data_new = np.append(data_new, e_3)
                data_new = np.append(data_new, self.Prams['length'])
                
                self.sim_results = np.empty((1, 7))
                self.sim_results[0] = data_new
                
            else:
                self.counts['lift fail']+=1
                self.sim_results = np.full(7,np.nan).reshape(1,7)
                
        elif impact.successful() and not (lim < 6):
            # Dust Grain Never Flatened Out
            self.counts['never fail']+=1
            self.sim_results = np.full(7,np.nan).reshape(1,7)
            
        else:
            # Post-Collison Phase Failure 
            self.counts['collide fail']+=1
            self.sim_results = np.full(7,np.nan).reshape(1,7)
            