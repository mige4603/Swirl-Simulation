# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:19:39 2017

@author: mige4603
"""
# I just want to see what this does.  Also, I erased "import time" directly below this comment.
import numpy as np
import scipy.integrate as sp
import multiprocessing
from os.path import isfile 
import matplotlib.pyplot as plt

def mult_Q(P, R, coeff=False):
    
    if coeff == False:
        coeff = 1.0
    
    u0 = coeff*(P[0]*R[0] - np.dot(P[1:4],R[1:4]))
    u = coeff*np.add(np.add(P[0]*R[1:4],R[0]*P[1:4]),np.cross(P[1:4],R[1:4]))
    
    Q = np.array([u0, u[0], u[1], u[2]])
    return Q
    
def conj_Q(Q):
    Q_new = np.array([Q[0], -Q[1], -Q[2], -Q[3]])
    return Q_new    

def InCon():
    diam = np.random.choice(h_pop)    
    length = 4.0548*diam
    
    vol = (4.0/3.0)*np.pi*(diam/2.0)**3  
    dens = np.random.choice(den_pop) # Desnity [g/m^3]
    m = dens*vol  # Mass [g]      
    I_x = (m*(length**2))/12.0      
    FeByWght = np.random.choice(FeByWght_pop)
    mu = FeByWght*m
    
    q = (-1.6e-19)*np.random.choice(q_pop)    # Charge on dust grains [Coulomb]
    
    # Initial Position
    zetta = zettaMAX*np.random.random()
    alpha = 2*np.pi*np.random.random()
    
    x = r_m*np.sin(zetta)*np.cos(alpha)
    y = r_m*np.sin(zetta)*np.sin(alpha)    
    z = r_m*np.cos(zetta)
    
    Rl = np.array([x, y, z])
    
    # Initial Velocity
    theta = thetaMAX*np.random.random()
    phi = 2*np.pi*np.random.random()
    
    V = np.random.choice(V_pop)
    Vl = V*np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    
    # Initial Conditions for trajectory
    IP = np.insert(Vl, 0, Rl)
    
    # Initial Orientation (e_1 x e_2 = e_3)
    theta = 2*np.pi*np.random.random()
    Q = (1/np.sqrt(2))*np.array([1.0, -np.sin(theta), np.cos(theta), 0.0])
    
    Omx = 2*np.pi*np.random.random()
    Omy = 2*np.pi*np.random.random()
    Omz = 2*np.pi*np.random.random()
    Om = np.array([0.0, Omx, Omy, Omz])
    dQ = 0.5*mult_Q(Om, Q)  
    
    IR = np.insert(dQ, 0, Q)
    
    # Total Energy
    Om_body = mult_Q(Q,mult_Q(Om, conj_Q(Q)))[1:4]
    KE = .5*I_x*(Om_body[0]*Om_body[0]+Om_body[1]*Om_body[1])
    B = B_field(Rl)
    B_body = mult_Q(Q, mult_Q(np.array([0,B[0],B[1],B[2]]),conj_Q(Q)))[1:4]
    PE = -mu*B_body[2]
    En = KE+PE
    
    return [length, m, q, I_x, mu, En], np.insert(IR, 0, IP)

def land(t, InCon, params):
    # Used to calculate position, velocity, rotatation and 
    # rotational velocity of dust grains about their own center of mass
    # as the particle's respond to a variable gravity force, Lorentz forces
    # from the magnetic dipole, electrostatic forces due to a localized 
    # electric field, and the torque on the magnetic moment of the particles
    # as they move through the magnetic field.    
    h, m, q, I_x, mu = params
    x, y, z, Vx, Vy, Vz, Q0, Q1, Q2, Q3, dQ0, dQ1, dQ2, dQ3 = InCon

    Q = np.array([Q0, Q1, Q2, Q3])
    Q_norm = np.linalg.norm(Q)
    if (Q_norm > 1.00001) or (Q_norm < 0.99999):
        Q = Q/Q_norm
    
    dQ = np.array([dQ0, dQ1, dQ2, dQ3])       
    
    r = np.array([x, y, z])
    r_nit = r/np.linalg.norm(r)
    V = np.array([Vx, Vy, Vz])
    B = B_field(r)
    E = E_field(r) 
    
    # Forces acting on the center of mass
    Acc_g = -g*r_nit
    Acc_b = (q/m) * np.cross(V, B)
    Acc_e = (q/m)*E
    Acc_gb = np.add(Acc_g, Acc_b)
    Acc = np.add(Acc_gb, Acc_e)
    
    # Rotation
    B_body = mult_Q(conj_Q(Q), mult_Q(np.array([0.0, B[0], B[1], B[2]]),Q))[1:4]
    torq = mu*np.array([-B_body[1], B_body[0], 0])
    
    angVel = 2*mult_Q(conj_Q(Q), dQ)[1:4]
    angMom = I_x*np.array([angVel[0], angVel[1], 0.0])
    angRot = np.cross(angVel, angMom)
    
    tRot = np.subtract(torq, angRot)
    tRot_Q = (1.0/I_x)*np.array([0.0, tRot[0], tRot[1], 0.0])
    
    ddQ_1 = mult_Q(dQ, mult_Q(conj_Q(Q),dQ))
    ddQ_2 = 0.5*mult_Q(Q, tRot_Q)
    ddQ = np.add(ddQ_1, ddQ_2)
    
    # Change rates
    deriv = [dQ, ddQ]
    deriv = np.insert(deriv, 0, Acc)
    deriv = np.insert(deriv, 0, V)   
    return deriv
   
def impact(t, InCon, params):
    # Calculates the effect of the grain's collision with the lunar surface, 
    # tracking the particle's rotation about its collision point.
    I_x, I_z, A = params
    Q0, Q1, Q2, Q3, dQ0, dQ1, dQ2, dQ3 = InCon

    Q = np.array([Q0, Q1, Q2, Q3])
    Q_norm = np.linalg.norm(Q)
    if (Q_norm > 1.00001) or (Q_norm < 0.99999):
        Q = Q/Q_norm
    
    dQ = np.array([dQ0, dQ1, dQ2, dQ3])
    
    # Torque
    A_body = mult_Q(conj_Q(Q), mult_Q(np.array([0,A[0],A[1],A[2]]), Q))
    torq = np.array([-A_body[2], A_body[1], 0.0])
    
    angVel = 2*mult_Q(conj_Q(Q), dQ)[1:4]
    angMom = np.array([I_x*angVel[0], I_x*angVel[1], I_z*angVel[2]])
    angRot = np.cross(angVel, angMom)
    
    tRot = np.subtract(torq, angRot)
    tRot_Q = np.array([0.0, tRot[0]/I_x, tRot[1]/I_x, tRot[2]/I_z])
    
    ddQ_1 = mult_Q(dQ, mult_Q(conj_Q(Q),dQ))
    ddQ_2 = 0.5*mult_Q(Q, tRot_Q)
    ddQ = np.add(ddQ_1, ddQ_2)
    
    return np.insert(ddQ, 0, dQ)        
    
def Integrate(inCon, params):
    
    h, m, q, I_x, mu, En = params
    
    # Brings the dust grain past its maximum 
    # displacement from lunar surface (Rising Phase)
    r = inCon[0:3]
    r_nit = r/np.linalg.norm(r)
    V = inCon[3:6]
    V_r = np.dot(V, r_nit)

    B = B_field(r)
    E = E_field(r)
    Acc_g = -g*r_nit
    Acc_b = (q/m) * np.cross(V, B)
    Acc_e = (q/m)*E
    Acc_gb = np.add(Acc_g, Acc_b)
    Acc = np.add(Acc_gb, Acc_e)
    A_r = np.dot(Acc, r_nit)

    div = 1.0
    dt_base = -.5*(V_r/A_r)
    dt = dt_base
    r_cm_prev = r_m
    nstp = 5000
    paramGRAV = [h,m,q,I_x,mu]
    Grav = sp.ode(land).set_integrator('vode',nsteps=nstp)
    Grav.set_f_params(paramGRAV).set_initial_value(inCon, 0.0)
    while Grav.successful():
        Grav.integrate(Grav.t + dt)
        r_cm = np.linalg.norm(Grav.y[0:3])
        if r_cm > r_cm_prev:
            r_cm_prev = r_cm
        else:
            break     

    if Grav.successful():
        # Brings the dust grains within ten body    
        # lengths of the surface. (Proximity Phase) 
        r = Grav.y[0:3]
        r_nit = r/r_cm           
        V = Grav.y[3:6]
        B = B_field(r)
        E = E_field(r)
        Acc_g = -g*r_nit
        Acc_b = (q/m) * np.cross(V, B)
        Acc_e = (q/m)*E
        Acc_gb = np.add(Acc_g, Acc_b)
        Acc = np.add(Acc_gb, Acc_e)        
        V_r = np.dot(V,r_nit)
        A_r = np.dot(Acc,r_nit)
        while r_cm >= (r_m+10*h) and Grav.successful():     
            dt = dt_base/div
            dr = V_r*dt+.5*A_r*dt*dt
            while r_cm >= (r_m - dr + h) and Grav.successful():
                Grav.integrate(Grav.t + 0.999*dt)
                r = Grav.y[0:3]
                r_cm = np.linalg.norm(r)
                r_nit = r/r_cm
                V = Grav.y[3:6]
                B = B_field(r)
                E = E_field(r)
                Acc_g = -g*r_nit
                Acc_b = (q/m) * np.cross(V, B)
                Acc_e = (q/m)*E
                Acc_gb = np.add(Acc_g, Acc_b)
                Acc = np.add(Acc_gb, Acc_e)
                V_r = np.dot(V,r_nit)
                A_r = np.dot(Acc,r_nit)
            div = div*10.0

        if Grav.successful():
            # Lowest point on the dust grain collides 
            # with the lunar surface (Pre-Impact Phase)
            dr = (r_m+h)-r_cm
            if dr < 0:
                dt = (-V_r-np.sqrt(V_r*V_r+2*dr*A_r))/A_r
                Grav.integrate(Grav.t+dt)
                r_cm = np.linalg.norm(Grav.y[0:3])
            
            Q = Grav.y[6:10]/np.linalg.norm(Grav.y[6:10])
            Orient = mult_Q(Q,mult_Q(np.array([0,0,0,1]), conj_Q(Q)))[1:4]
            r_low = r_cm-(.5*h*abs(np.dot(r_nit,Orient)))
            dt = (-V_r-np.sqrt(V_r*V_r-0.1*h*A_r))/A_r
            while r_low >= r_m and Grav.successful():
                Grav.integrate(Grav.t + dt)
                r_cm = np.linalg.norm(Grav.y[0:3])
                Q = Grav.y[6:10]/np.linalg.norm(Grav.y[6:10])
                Orient = mult_Q(Q,mult_Q(np.array([0,0,0,1]), conj_Q(Q)))[1:4]
                r_low = r_cm-(.5*h*abs(np.dot(r_nit,Orient)))

            if Grav.successful():
                # After the lowest point of the duest grain has made contact
                # with the lunar surface, that point becomes the new point of
                # rotation.  We then must include the grain's linear velocity
                # with its previous angular velocity to determine its new
                # initial angular velocity for the integrator (Post-Impact Phase).
                Om_space = 2*mult_Q(Grav.y[10:15], conj_Q(Q))[1:4]
                
                r = Grav.y[0:3]
                r_nit = r/np.linalg.norm(r)
                B = B_field(r)
                E = E_field(r) 
                F_grav = -g*m*r_nit
                A = np.add(mu*B, .5*h*np.add(F_grav, q*E))
                
                # Define Body axes in space frame (e_1, e_2, e_3)
                E_1 = mult_Q(Q, mult_Q(np.array([0,1,0,0]), conj_Q(Q)))[1:4]
                e_1 = E_1/np.linalg.norm(E_1)
                E_2 = mult_Q(Q, mult_Q(np.array([0,0,1,0]), conj_Q(Q)))[1:4]
                e_2 = E_2/np.linalg.norm(E_2)
                e_3 = np.cross(e_1, e_2)
                
                # Include the dust grain's linear velocity with the new angular
                # velocity, then use the new angular velocity to calcuate the 
                # initial dQ
                V_cm = Grav.y[3:6]
                Om_1 = (-2.0/h)*np.dot(V_cm, e_2)
                Om_2 = (2.0/h)*np.dot(V_cm, e_1)
                Om_space = np.add(Om_space, np.add(Om_1*e_1, Om_2*e_2))
                dQ = 0.5*mult_Q(np.array([0,Om_space[0],Om_space[1],Om_space[2]]), Q)
                
                inCon_new = np.insert(dQ, 0, Q)
                Iz = .25*m*h*h
                Ix = I_x + Iz
                params_new = [Ix, Iz, A]
                
                # These conditions are used to check whether the particle will
                # oscillate around the magnetic field line forever, never 
                # flattening out.
                lim = 0
                e3_chg_prev = -1.0
                e3_prev = abs(np.dot(e_3,r_nit))
                
                # Call Collision Integrator
                Om_body = 2*mult_Q(conj_Q(Q),Q)[1:4]
                V_rem = np.subtract(V_cm, np.dot(V_cm,e_3)*e_3)
                EnLIN = .5*m*np.dot(V_rem,V_rem)
                EnTOT = En + EnLIN
                dt = (np.arccos(e3_prev-.1)-np.arccos(e3_prev))*np.sqrt(I_x/(2*EnTOT-Iz*Om_body[2]*Om_body[2]))
                Imp = sp.ode(impact).set_integrator('vode',nsteps=nstp)
                Imp.set_f_params(params_new).set_initial_value(inCon_new, 0.0)
                while (e3_prev >= 0.1) and (lim < 5) and Imp.successful():
                    Imp.integrate(Imp.t + dt)
                    Q = Imp.y[0:4]/np.linalg.norm(Imp.y[0:4])
                    e_3 = mult_Q(Q, mult_Q(np.array([0,0,0,1]), conj_Q(Q)))[1:4]
                    
                    # Check the paritcle for signs of infinite oscillation
                    e3_2 = abs(np.dot(e_3,r_nit))
                    e3_chg = (e3_2 - e3_prev)/abs(e3_2 - e3_prev)
                    e3_prev = e3_2
                    if not e3_chg == e3_chg_prev:
                        e3_chg_prev = e3_chg
                        lim += 1

                if Imp.successful() and (lim < 5):
                    # The dust grain may only have flattened because of its linear momentum.
                    # Therefore, we must check to see if the torque due to its magnetic moment
                    # might be sufficiently strong to overcome graity and lift it off the lunar surface
                    T_nit = np.cross(e_3, r_nit)
                    t_nit = T_nit/np.linalg.norm(T_nit)
                    T_grav = -0.5*h*m*g*t_nit
                    T_mag = mu*np.cross(e_3, B)
                    T_net = np.add(T_grav, np.dot(T_mag, t_nit)*t_nit)
                    lft_check = np.dot(T_net, t_nit)
                    if lft_check <= 0.0:
                        # Dust Grain Remains Flat
                        return [r[0], r[1], e_3[0], e_3[1], e_3[2]], [0,0,0,0,1,0,0]
                    else:
                        # Dust Grain Lifted Off Surface
                        return [r[0],r[1],0,0,0], [0,0,0,0,0,1,0]
                
                elif Imp.successful() and not (lim < 5):
                    # Dust Grain Never Flatened Out
                    return [r[0],r[1],0,0,0], [0,0,0,0,0,0,1]
                else:
                    # Post-Collison Phase Failure 
                    return np.zeros(5), [0,0,0,1,0,0,0]
            else:
                # Pre-Collison Phase Failure 
                return np.zeros(5), [0,0,1,0,0,0,0]
        else:
            # Proximity Phase Failure 
            return np.zeros(5), [0,1,0,0,0,0,0]
    else:
        # Rising Phase Failure 
        return np.zeros(5), [1,0,0,0,0,0,0]
    
def B_field(R):
    # Returns B_field at location R as described by imported field data.
    # The data is the from the Jan Deca and is the infered field data for
    # the Reiner Gamma swirl environment
    
    R_o = np.array([-325000, -325000, 1898000])
    x, y, z = np.subtract(R, R_o)
    
    if (-678.497 < x < 650678) and (-678.497 < y < 650678) and (-194729 < z < 678.497):
        i = round(abs(x) / 1357)
        j = round(abs(y) / 1357)
        k = round(abs(z) / 1357)
        
        Bx, By, Bz = B_data[i*144*480 + j*144 + k]
        return np.array([Bz, By, -Bx])*(6e-9/.0016)
        
    elif (-678.497 < x < 650678) and (-678.497 < y < 650678) and (z <= -194729):
        i = round(abs(x) / 1357)
        j = round(abs(y) / 1357)
        k = 143
        
        Bx, By, Bz = B_data[i*144*480 + j*144 + k]
        return np.array([Bz, By, -Bx])*(6e-9/.0016)
        
    else:
        return np.array([0, 0, 0])  
    
def E_field(R):
    # Returns E_field at location R as described by imported field data.
    # The data is the from the Jan Deca and is the infered field data for
    # the Reiner Gamma swirl environment
    
    R_o = np.array([-325000, -325000, 1898000])
    x, y, z = np.subtract(R, R_o)
    
    if (-678.497 < x < 650678) and (-678.497 < y < 650678) and (-194729 < z < 678.497):
        i = round(abs(x) / 1357)
        j = round(abs(y) / 1357)
        k = round(abs(z) / 1357)
        
        Ex, Ey, Ez = E_data[i*144*480 + j*144 + k]
        return np.array([Ez, Ey, -Ex])*(2.1/2.72e-5)*0.001
     
    elif (-678.497 < x < 650678) and (-678.497 < y < 650678) and (z <= -194729):
        i = round(abs(x) / 1357)
        j = round(abs(y) / 1357)
        k = 143
        
        Ex, Ey, Ez = E_data[i*144*480 + j*144 + k]
        return np.array([Ez, Ey, -Ex])*(2.1/2.72e-5)*0.001
        
    else:
        return np.array([0, 0, 0]) 
    
def getNumb(line, cnt=0):
    # Reads a line from .txt file and returns the first float in that line
    for j in line:
        if not j == ' ':
            begin = cnt
            end = begin+1
            while not line[end] == ' ':
                end += 1
            return float(line[begin:end])
        cnt += 1
        
def worker(part, que):
    results = {}
    
    chk = len(part) * 0.1
    pnt = (chk/len(part))*100
    trk = 1.0
    
    for i in part:
        
        P = Particles[i]
        P.getFiCon()
        results[i] = [P.fiCon, P.tRack]
        
        if part[0] == 0:
            if i == round(chk*trk):
                print '    Process 1 : '+str(round(pnt*trk))+' % Complete'
                trk += 1
        if part[0] > 0:
            if (i-part[0]) == round(chk*trk):
                print '    Process 2 : '+str(round(pnt*trk))+' % Complete'
                trk += 1
       
    que.put(results)       
    
class Particle(object):

    def __init__(self):
        self.params, self.inCon = InCon()
    
    def getFiCon(self):
        self.fiCon, self.tRack = Integrate(self.inCon, self.params)

G = 6.67408e-11   # Gravitational Constant [m^3 kg^-1 s^-2[]
M = 7.3477e22     # Mass of moon [kg]
r_m = 1.7371e6    # Radius of moon [m]
g = 1.62519       # Gravity on moon surgace [m/s^2]
'''
print '\nImporting Data'
beg_time = time.time()
B_data = np.loadtxt('B_Jan_cycle12000.txt', skiprows=9)
E_data = np.loadtxt('E_Jan_cycle12000.txt', skiprows=9)
print '    '+str(round((time.time()-beg_time)/60.0, 1))+' minutes'
'''
# Charge on Dust Grain
q_min = 100
q_max = 1000
q_pop = np.linspace(q_min, q_max, 1000)   

# Range of Landing Site -Dia to Dia [m]    
Dia = 325000
zettaMAX = np.arcsin(Dia/r_m)

# Density Range [g/m^3]
den_min = 2.33e6
den_max = 4.37e6
den_pop = np.linspace(den_min, den_max, 10000)
    
# Range of Grain Diameters [m]
h_min = 9.0e-6  
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

nproc = 2
Im = 10000
mult = Im/nproc

shared_Name = 'RGQuat'
ind = int(h_max/1.0e-6)
data_CAP = 1000000
'''
# Test single set of particles instead of cycling through different sizes
print '\nCreate Particles'
start_time = time.time()
Particles = []
for count in range(Im):
    P = Particle()
    Particles.append(P)

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

Time = time.time() - start_time       

track = np.zeros(7)
fiCon = np.empty((Im, 6))
for i in range(Im):
    P = Particles[i]
    h_len = P.params[0]
    fiCon[i] = np.insert(resultFull[i][0], 5, h_len)
    track = np.add(track, resultFull[i][1])

print '\n'+str(np.sum(track[0:4]))+' Particles Failed'
if np.sum(track[0:4]) > 0:
    print '    '+str(track[0])+' Rising Phase'
    print '    '+str(track[1])+' Proximity Phase'
    print '    '+str(track[2])+' Pre Impact Phase'
    print '    '+str(track[3])+' Post Impact Phase'

print '\n'+str(np.sum(track[4:6]))+' Particles Flatten'
print '    '+str(track[4])+' Tor_grav > Tor_field'
print '    '+str(track[5])+' Tor_grav < Tor_field'

print '\n'+str(track[6])+' Particles Never Flatten'

print '\n'+str(round(Time/Im,2))+' sec/part'
'''
# Iterative Simulations with Analysis over decreasing grain size
while True:
    fileNames = ['{0}_{1}.txt'.format(shared_Name, ind), '{0}PatView_{1}.vtk'.format(shared_Name, ind), '{0}DotView_{1}.vtk'.format(shared_Name, ind), '{}FieldView.vtk'.format(shared_Name), '{}FullFieldView.vtk'.format(shared_Name)]
    note = 0
    Rate = []
    while True:
        start_time = time.time()    
        
        print '\nCreate Particles'
        Particles = []
        for count in range(Im):
            P = Particle()
            Particles.append(P)
        
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
        
        Time = time.time() - start_time
        Rate.append(Im/Time)        
        
        track = np.zeros(7)
        fiCon = np.empty((Im, 6))
        for i in range(Im):
            P = Particles[i]
            h_len = P.params[0]
            fiCon[i] = np.insert(resultFull[i][0], 5, h_len)
            track = np.add(track, resultFull[i][1])
        
        print '\n'+str(np.sum(track[0:4]))+' Particles Failed'
        if np.sum(track[0:4]) > 0:
            print '    '+str(track[0])+' Rising Phase'
            print '    '+str(track[1])+' Proximity Phase'
            print '    '+str(track[2])+' Pre Impact Phase'
            print '    '+str(track[3])+' Post Impact Phase'
        
        print '\n'+str(np.sum(track[4:6]))+' Particles Flatten'
        print '    '+str(track[4])+' Tor_grav > Tor_field'
        print '    '+str(track[5])+' Tor_grav < Tor_field'
        
        print '\n'+str(track[6])+' Particles Never Flatten'
        
        print '\nProcessing Time : ('+str(np.round(np.average(Rate), 1))+' +/- '+str(np.round(np.std(Rate), 1))+') particles per second'
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
                elif i == 15:
                    track_rd.append(getNumb(line))
                elif i == 18:
                    track_rd.append(getNumb(line))
                elif i == 19:
                    track_rd.append(getNumb(line))
                elif i == 21:
                    track_rd.append(getNumb(line))
                elif i > 21:
                    break
            fp.close
        
            track = np.add(track_rd, track)
            
            Parts = Parts + Im
            if Parts >= data_CAP:
                note += 1
            
            with open(fileNames[0], 'r') as file:
                data = file.readlines()
                
            data[3] = str(round(Parts))+' Individual Grains\n'
            data[11] = str(round(sum(track[0:4])))+' Particles Failed\n'
            data[12] = '    '+str(round(track[0]))+' Rising Phase\n'
            data[13] = '    '+str(round(track[1]))+' Proximity Phase\n'
            data[14] = '    '+str(round(track[2]))+' Pre Impact Phase\n'
            data[15] = '    '+str(round(track[3]))+' Post Impact Phase\n'
            data[17] = str(round(sum(track[4:6])))+' Particles Flatten\n'
            data[18] = '    '+str(round(track[4]))+' Tor_grav > Tor_field\n'
            data[19] = '    '+str(round(track[5]))+' Tor_grav < Tor_field\n'
            data[21] = str(round(track[6]))+' Particles Never Flatten\n'
            data[23] = 'Processing Time : ('+str(np.round(np.average(Rate), 1))+' +/- '+str(np.round(np.std(Rate), 1))+') part/sec\n'
            
            with open(fileNames[0], 'w') as file:
                file.writelines(data)
            
            file = open(fileNames[0], 'a')
            for i in range(Im):
                file.write(str(fiCon[i,0])+' '+str(fiCon[i,1])+' '+str(fiCon[i,2])+' '+str(fiCon[i,3])+' '+str(fiCon[i,4])+' '+str(fiCon[i,5])+'\n')
            file.close()
            
        else:
            file = open(fileNames[0], 'w')
            
            file.write('Horizontal Dipole\n'
                       'Magnetic Field Location : B_Field_Horz.txt \n'
                       'Magnetic Dipole Strength : B_Field_Horz.txt \n'
                       +str(Im)+' Individual Grains\n'
                       '('+str(h_min)+' to '+str(h_max)+')m length of grains\n'
                       '('+str(m_mom_min)+' to '+str(m_mom_max)+')Am^2 magnetic moment of grains\n' 
                       '(-'+str(q_min)+' to -'+str(q_max)+') x 10e-19 C charge on grains \n'
                       '('+str(V_min)+' to '+str(V_max)+')m/s initial linear velocity \n'
                       '('+str(Om_min)+' to '+str(Om_max)+')rad/s initial angular velocity \n'
                       '('+str(2*Dia)+' x '+str(2*Dia)+')m^2 Landing Area \n'
                       '\n'
                       +str(np.sum(track[0:4]))+' Particles Failed \n'
                       '    '+str(track[0])+' Rising Phase \n'
                       '    '+str(track[1])+' Proximity Phase \n'
                       '    '+str(track[2])+' Pre Impact Phase \n'
                       '    '+str(track[3])+' Post Impact Phase \n'
                       '\n'
                       +str(np.sum(track[4:6]))+' Particles Flatten \n'
                       '    '+str(track[4])+' Tor_grav > Tor_field \n'
                       '    '+str(track[5])+' Tor_grav < Tor_field \n'
                       '\n'
                       +str(track[6])+' Particles Never Flatten \n'
                       '\n'
                       'Processing Time : ('+str(np.round(np.average(Rate), 1))+' +/- '+str(np.round(np.std(Rate), 1))+') part/sec\n'
                       '\n')   
            
            for i in range(Im):
                file.write(str(fiCon[i,0])+' '+str(fiCon[i,1])+' '+str(fiCon[i,2])+' '+str(fiCon[i,3])+' '+str(fiCon[i,4])+' '+str(fiCon[i,5])+'\n')
            file.close()
            
            if Im >= data_CAP:
                note +=1
        
        if note > 0:
            break
    
    # Write Field Files
    if not isfile(fileNames[3]):  
        D_bin = 480
        E_bin = 26
        N = D_bin*D_bin        
        
        arcLEN = r_m*zettaMAX
        x_bin = np.linspace(-Dia, Dia, D_bin+1)
        
        Brad = np.zeros((N*E_bin,3))
        Bx = np.zeros(N)
        By = np.zeros(N)
        Bz = np.zeros(N)
        for i in range(D_bin):
            for j in range(D_bin):
                x = x_bin[j]
                y = x_bin[i]  
                z = np.sqrt(r_m*r_m-(x*x+y*y))
                r = np.array([x,y,z])
                B = B_field(r)
                Bx[i*D_bin + j] = B[0]
                By[i*D_bin + j] = B[1]
                Bz[i*D_bin + j] = B[2]
                arc = r_m*np.arcsin(np.sqrt(x*x+y*y)/r_m)
                if (arc <= arcLEN):
                    r_nit = r/np.linalg.norm(r)
                    Br = np.subtract(B,np.dot(B,r_nit)*r_nit)
                    k = round(abs(z-r_m)/1357)
                    Brad[k*N + i*D_bin + j] = Br
                                   
        file = open(fileNames[3], 'w')
        file.write('# vtk DataFile Version 1.0\n'
                   'B Field from Parsek\nASCII\n'
                   'DATASET STRUCTURED_POINTS\n'
                   'DIMENSIONS {0} {0} {1}\n'
                   'ORIGIN 0 0 0\n'
                   'SPACING 0.00390625 0.00390625 0.00390625\n'
                   'POINT_DATA {2}\n'
                   'VECTORS B float\n'.format(D_bin, E_bin, N*E_bin))
        for s in range(N*E_bin):
                file.write(str(Brad[s,0]) + ' ' + str(Brad[s,1]) + ' ' + str(Brad[s,2]) + '\n')
        file.close()
        
        file = open(fileNames[4], 'w')
        file.write('# vtk DataFile Version 1.0\n'
                   'B Field from Parsek\nASCII\n'
                   'DATASET STRUCTURED_POINTS\n'
                   'DIMENSIONS {0} {0} {1}\n'
                   'ORIGIN 0 0 0\n'
                   'SPACING 0.00390625 0.00390625 0.00390625\n'
                   'POINT_DATA {2}\n'
                   'VECTORS B float\n'.format(D_bin, 1, N))
        for t in range(N):
                file.write(str(Bx[t]) + ' ' + str(By[t]) + ' ' + str(Bz[t]) + '\n')
        file.close()                
    
    print '\nData Analysis'
    fill = np.genfromtxt(fileNames[0], skip_header=24)
    len_fill = len(fill)
    
    D_bin = 480
    E_bin = 26
    Dt_bin = 400
    
    arcLEN = r_m*zettaMAX
    h_bins = 21
    h_dist = np.linspace(0,arcLEN,h_bins)
    h_diff = (h_dist[1]-h_dist[0])/2.0
    
    # Modify Data
    dataPat = np.zeros((D_bin*D_bin*E_bin, 3))
    dataDot = np.zeros((len_fill, 3))
    dataSET = [[]for a in range(h_bins-1)]
    for i in range(len_fill):
        x, y, e_x, e_y, e_z = fill[i, 0:5]
        z = np.sqrt(r_m*r_m-(x*x+y*y))
        r = np.array([x,y,z])
        r_nit = r/np.linalg.norm(r)
        e_3 = np.array([e_x,e_y,e_z])
        E_vec = np.subtract(e_3,np.dot(e_3,r_nit)*r_nit)
        e_vec = E_vec/np.linalg.norm(E_vec)
        B = B_field(r)
        Br = np.subtract(B, np.dot(B,r_nit)*r_nit)
        arc = r_m*np.arcsin(np.sqrt(x*x+y*y)/r_m)
        if (arc <= arcLEN) and not all(e_3 == 0.):
            R_o = np.array([-325000, -325000, r_m])
            Xp, Yp, Zp = np.subtract(r, R_o)    
            if (-678.5 < Xp < 650678) and (-678.5 < Yp < 650678) and (-34478.5 < Zp < 678.5):
                I = round(abs(Xp) / 1357)
                J = round(abs(Yp) / 1357)
                K = round(abs(Zp) / 1357)
                dataPat[K*D_bin*D_bin + J*D_bin + I] = e_vec
                
            d = np.dot(e_vec, Br)/np.linalg.norm(Br)
            dataDot[i] = y, x, d
            
            chk = np.floor(arc/(h_diff*2))
            dataSET[int(chk)].append(d)
    
    # Making Scalar Product Plot 
    radius = np.empty(h_bins-1)
    average = np.empty(h_bins-1)
    StanDev = np.empty(h_bins-1)
    for j in range(h_bins-1):
        SET = dataSET[j]
        radius[j] = h_diff + 2*j*h_diff
        average[j] = np.average(SET)
        StanDev[j] = np.std(SET)  
        
    fig, axs = plt.subplots()      
    dotPLOT = plt.errorbar(radius, average, xerr=h_diff, yerr=StanDev, fmt='o', label='Mean Scalar Product')
    
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles, labels)
    plt.legend(handles=[dotPLOT])
    plt.title('Scalar Product vs. Radial Displacement')
    plt.xlabel('Radial Displacement (meters)')
    plt.ylabel('Scalar Product')
    
    plt.savefig('{0}_Plot_{1}.png'.format(shared_Name,ind))
    
    x_bin = np.linspace(-Dia, Dia, D_bin+1)
    d_bin = np.linspace(-1, 1, Dt_bin+1)
    
    # Constructing Histogram
    histDot, edgesDot = np.histogramdd(dataDot, bins=(x_bin, x_bin, d_bin))
    N = D_bin*D_bin    
    
    # Organising Data
    dotAvg = np.zeros(N)
    dotDev = np.zeros(N)
    untDev = np.zeros(N)
    
    cnt = D_bin * 0.1
    pnt = (cnt/D_bin)*100
    trk = 1
    
    ### Constructs Landing Pattern and Standard Deviation Pattern ###
    print '\nDot Product Analysis'
    for i in range(D_bin):
        
        if i == round(trk*cnt - 1):
            print '    '+str(trk*pnt)+'% Complete'
            trk += 1
            
        for j in range(D_bin):
            ## Data Analysis ##        
            numDot = np.zeros(Dt_bin)     
            denDot = np.zeros(Dt_bin)
            for q in range(Dt_bin):            
                dot = round(-0.9975 + 0.005*q, 2)
                n = histDot[i, j, q]
                numDot[q] = dot*n
                denDot[q] = n
            denDotAvg = np.sum(denDot)
            if not denDotAvg == 0:
                dot = np.sum(numDot) / denDotAvg
                dotVar = np.zeros(Dt_bin)
                untVar = np.zeros(Dt_bin)
                for p in range(Dt_bin):
                    n = denDot[p]
                    if not n == 0:
                        num = numDot[p]/n
                        dotVar[p] = n*(num - dot)**2
                        untVar[p] = n*(num - 1.0)**2
                dotAvg[i*D_bin + j] = dot
                dotDev[i*D_bin + j] = np.sqrt(np.sum(dotVar)/denDotAvg)
                untDev[i*D_bin + j] = np.sqrt(np.sum(untVar)/denDotAvg)
                
    print '\nWriting Files'
    
    print '    1'
    file = open(fileNames[1], 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} {1}\n'
               'ORIGIN 0 0 0\n'
               'SPACING 0.00390625 0.00390625 0.00390625\n'
               'POINT_DATA {2}\n'
               'VECTORS B float\n'.format(D_bin, E_bin, N*E_bin))
    for a in range(N*E_bin):
            file.write(str(dataPat[a,0]) + ' ' + str(dataPat[a,1]) + ' ' + str(dataPat[a,2]) + '\n')
    file.close()
    
    print '    2'
    file = open(fileNames[2], 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} {1}\n'
               'ORIGIN 0 0 0\n'
               'SPACING 0.00390625 0.00390625 0.00390625\n'
               'POINT_DATA {2}\n'
               'VECTORS B float\n'.format(D_bin, 1, N))
    for i in range(N):
            file.write(str(dotAvg[i]) + ' ' + str(dotDev[i]) + ' ' + str(untDev[i]) + '\n')
    file.close()
    break
    ind -= 1
    if ind == 0:
        break
    elif ind == 1:
        h_min = 0.1e-6
        h_max = 1.0e-6       
    else:   
        h_min = h_min - 1.0e-6
        h_max = h_max - 1.0e-6
