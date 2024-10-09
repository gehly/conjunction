import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import os

import TudatPropagator as prop


# Earth parameters
GME = 398600.4415*1e9  # m^3/s^2
J2E = 1.082626683e-3
Re = 6378137.0





def test_damico_tsx_tdx(datafile):
    
    # Constants
    Cd = 2.3                # unitless
    Cr = 1.3                # unitless
    A = 3.2                 # m^2
    m = 1238.                # kg
    
    # Chief Orbit Parameters
    a = 6892.945*1000.                  # m
    e = 1e-12
    i = 97.*np.pi/180.                  # rad
    RAAN = 0.                           # rad
    w = 270.*np.pi/180.                 # rad
    true_anom = 0.                      # rad
    P = 2.*np.pi*np.sqrt(a**3./GME)     # sec
    
    # Orbit Differences
    di_x = 0.
    di_y = -1000./a             # rad
    di = 0.                     # rad
    dRAAN = (di_y/np.sin(i))    # rad
    
    de_x = 0.
    de_y = 300./a                       # non-dim
    de = np.linalg.norm([de_x, de_y])   # non-dim
    
    # Compute initial relative angles (vector coordinates in Orbit Frame 1)
    # theta gives angle between N1 (asc node) and N12 (relative asc node)
    # phi gives angle between N1 and de vector
    theta0 = math.atan2(di_y, di_x)      # rad
    phi0 = math.atan2(de_y, de_x)        # rad
    
    # Compute mean rate of change of angles due to J2 (D'Amico Eqs 20-22)
    dphi_dt = 1.5*(np.pi/P)*(Re**2./a**2.)*J2E*(5.*np.cos(i)**2. - 1)
    diy_dt = -3.*(np.pi/P)*(Re**2./a**2.)*J2E*np.sin(i)**2.*di
    
    # Deputy Orbit
    a2 = a + 0.
    e2 = e + de
    i2 = i + di                         # rad
    RAAN2 = RAAN + dRAAN                # rad
    w2 = 90*np.pi/180.                  # rad
    true_anom2 = true_anom - np.pi      # rad
    
    # Rotation matrix from ECI to Orbit Frame 1
    R1 = compute_R1(i)    
    R3 = compute_R3(RAAN)
    OF1_ECI = R1 @ R3
    
    # Initial conditions
    X1 = kep2cart([a, e, i, RAAN, w, true_anom])
    X2 = kep2cart([a2, e2, i2, RAAN2, w2, true_anom2])
    
    # Check initial orbit energy and relative orbit params
    r1_vect = X1[0:3].reshape(3,1)
    v1_vect = X1[3:6].reshape(3,1)
    r2_vect = X2[0:3].reshape(3,1)
    v2_vect = X2[3:6].reshape(3,1)        
    
    r1 = np.linalg.norm(r1_vect)
    v1 = np.linalg.norm(v1_vect)
    r2 = np.linalg.norm(r2_vect)
    v2 = np.linalg.norm(v2_vect)
    
    energy1 = v1**2./2. - GME/r1
    energy2 = v2**2./2. - GME/r2
    
    # Cartesian relative orbit parameters 
    rho_eci = r2_vect - r1_vect
    drho_eci = v2_vect - v1_vect
    rho_ric = eci2ric(r1_vect, v1_vect, rho_eci)
    drho_ric = eci2ric_vel(r1_vect, v1_vect, rho_ric, drho_eci)
    
    # Relative E/I vectors
    de_vect_eci, di_vect_eci = inertial2relative_ei(r1_vect, v1_vect, r2_vect, v2_vect)
    
    # Alternate formulation of rotation matrix
    h1_vect = np.cross(r1_vect, v1_vect, axis=0)
    e1_vect = np.cross(v1_vect, h1_vect, axis=0)/GME - r1_vect/r1
    ih_1 = h1_vect/np.linalg.norm(h1_vect)
    ie_1 = e1_vect/np.linalg.norm(e1_vect)
    ip_1 = np.cross(ih_1, ie_1, axis=0)
        
    P_ECI = np.concatenate((ie_1, ip_1, ih_1), axis=1).T
    P_ECI_check = compute_R3(w) @ R1 @ R3
    
    R3w = compute_R3(-w)
    
    OF1_ECI_alt = R3w @ P_ECI
    
    
    # print('')
    
    # print('h1_vect', )
    
    # print('ie_1', ie_1)
    # print('ip_1', ip_1)
    # print('ih_1', ih_1)
    
    # print(OF1_ECI)
    # print(OF1_ECI_alt)
    
    # print('')
    # print(P_ECI)
    # print(P_ECI_check)
    
    print('')
    print('initial ECI states')
    print('X1', X1)
    print('X2', X2)
    
    
    # Check rotation matrices
    diff = np.max(abs(OF1_ECI_alt - OF1_ECI))
    print('check rotation matrices', diff)
    
    # Rotate e/i vectors to Orbit Frame 1
    de_vect_of = np.dot(OF1_ECI, de_vect_eci)
    di_vect_of = np.dot(OF1_ECI, di_vect_eci)
    angle = np.arccos(np.dot(de_vect_of.flatten(), di_vect_of.flatten())/(np.linalg.norm(de_vect_of)*np.linalg.norm(di_vect_of)))
    
    # Check relative e/i vectors
    de_check = np.array([[de_x], [de_y], [0.]])
    di_check = np.array([[di_x], [di_y], [0.]])
    
    print('de_vect_of0', de_vect_of)
    print('di_vect_of0', di_vect_of)
    print('check de', de_vect_of - de_check)
    print('check di', di_vect_of - di_check)
    
    print('')
    print('relative orbit params at t0')
    print('rho_ric', rho_ric)
    print('drho_ric', drho_ric)
    print('de', de_vect_of)
    print('di', di_vect_of)
    print('angle [deg]', angle*180./np.pi)
    
    plt.figure()
    plt.plot([0., de_x], [0., de_y], 'k')
    plt.xlabel('de[x]')
    plt.ylabel('de[y]')
    
    plt.figure()
    plt.plot([0., di_x], [0., di_y], 'k')
    plt.xlabel('di[x]')
    plt.ylabel('di[y]')
    
    
    # Propagate orbits
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    #bodies_to_create = ['Earth']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = m
    state_params['area'] = A
    state_params['Cd'] = Cd
    state_params['Cr'] = Cr
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20    
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    # int_params['tudat_integrator'] = 'rk4'
    # int_params['step'] = 20.
    
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 20.
    int_params['max_step'] = 20.
    int_params['min_step'] = 20.
    int_params['rtol'] = np.inf
    int_params['atol'] = np.inf
    
    tvec = np.array([0., 86400.*31])
    Xo = np.concatenate((X1, X2), axis=0)
    
    tout, Xout = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
    
    # Compute mean vectors and angles over time
    phi = phi0 + dphi_dt*tout
    dex_t = de*np.cos(phi)
    dey_t = de*np.sin(phi)
    
    dix_t = di_x*np.ones(len(tout,))
    diy_t = di_y + diy_dt*tout
    theta = np.asarray([math.atan2(yy, di_x) for yy in diy_t])
    
    
    
    pklFile = open( datafile, 'wb' )
    pickle.dump( [tout, Xout, phi, theta, dex_t, dey_t, dix_t, diy_t], pklFile, -1 )
    pklFile.close()
    
    
    return


def compute_angle_diff(a, b):
    
    diff = (a - b) % (2.*np.pi)
    if diff < -np.pi:
        diff += 2.*np.pi
    if diff > np.pi:
        diff -= 2.*np.pi    
    
    return abs(diff)


def plot_damico_tsx_tdx(datafile):
    
    # Load data
    pklFile = open(datafile, 'rb' )
    data = pickle.load( pklFile )
    tout = data[0]
    Xout = data[1]
    phi = data[2]
    theta = data[3]
    dex_t = data[4]
    dey_t = data[5]
    dix_t = data[6]
    diy_t = data[7]
    pklFile.close()
    
    
    # Compute mean angle
    mean_angle = [compute_angle_diff(pi,ti) for pi,ti in zip(phi, theta)]
    
    print('')
    print(len(tout))
    print(len(phi))
    print(len(theta))
    print(len(mean_angle))
    
    # Orbit elements
    Xo = Xout[0,0:6].reshape(6,1)
    elem = cart2kep(Xo)
    a = elem[0]
    P = 2*np.pi*np.sqrt(a**3/GME)

    # Compute relative e/i vectors and angle between them at each time
    thrs_plot = []
    de_x_plot = []
    de_y_plot = []
    di_x_plot = []
    di_y_plot = []
    de_x_mean = []
    de_y_mean = []
    di_x_mean = []
    di_y_mean = []
    angle_deg_plot = []
    mean_angle_plot = []
    a_plot = []
    e_plot = []
    i_plot = []
    rt_plot = []
    it_plot = []
    ct_plot = []
    radial_list = []
    intrack_list = []
    crosstrack_list = []
    danger_ind = np.nan
    angle_minimum = 5.*np.pi/180.
    danger_angle = np.inf
    for kk in range(len(tout)):
        r1_vect = Xout[kk,0:3].reshape(3,1)
        v1_vect = Xout[kk,3:6].reshape(3,1)
        r2_vect = Xout[kk,6:9].reshape(3,1)
        v2_vect = Xout[kk,9:12].reshape(3,1)
        X1 = Xout[kk,0:6].reshape(6,1)
        elem = cart2kep(X1)
        inc = float(elem[2,0])
        RAAN = float(elem[3,0])
        
        # Compute relative positions in rotating Hill frame
        rho_eci = r2_vect - r1_vect
        rho_ric = eci2ric(r1_vect, v1_vect, rho_eci)
        radial_list.append(float(rho_ric[0,0]))
        intrack_list.append(float(rho_ric[1,0]))
        crosstrack_list.append(float(rho_ric[2,0]))
        
        # Compute e/i vectors and rotate to Orbit Frame 1
        de_vect_eci, di_vect_eci = inertial2relative_ei(r1_vect, v1_vect, r2_vect, v2_vect)
        
        R1 = compute_R1(inc)    
        R3 = compute_R3(RAAN)
        OF1_ECI = R1 @ R3
        
        de_vect_of = np.dot(OF1_ECI, de_vect_eci)
        di_vect_of = np.dot(OF1_ECI, di_vect_eci)
        angle = np.arccos(np.dot(de_vect_of.flatten(), di_vect_of.flatten())/(np.linalg.norm(de_vect_of)*np.linalg.norm(di_vect_of)))
        
        # Check for date when angle is closest to perpendicular, if it exists
        # angle_diff = abs(angle - np.pi/2.)
        angle_diff = abs(mean_angle[kk] - np.pi/2.)
        if angle_diff < angle_minimum:
            angle_minimum = angle_diff
            # danger_angle = angle
            danger_angle = mean_angle[kk]
            danger_time = tout[kk]
            danger_ind = kk
            danger_de_x = float(de_vect_of[0,0])
            danger_de_y = float(de_vect_of[1,0])
            danger_di_x = float(di_vect_of[0,0])
            danger_di_y = float(di_vect_of[1,0])
            
        
        # Store data for plots
        if kk % 100 == 0:
            thrs_plot.append(tout[kk]/3600.)
            de_x_plot.append(float(de_vect_of[0,0]))
            de_y_plot.append(float(de_vect_of[1,0]))
            di_x_plot.append(float(di_vect_of[0,0]))
            di_y_plot.append(float(di_vect_of[1,0]))
            de_x_mean.append(dex_t[kk])
            de_y_mean.append(dey_t[kk])
            di_x_mean.append(dix_t[kk])
            di_y_mean.append(diy_t[kk])
            angle_deg_plot.append(angle*180./np.pi)
            mean_angle_plot.append(mean_angle[kk]*180./np.pi)
            a_plot.append(float(elem[0,0])/1000.)
            e_plot.append(float(elem[1,0]))
            i_plot.append(inc*180/np.pi)
        
            
    
    
        
        
    plt.figure()
    plt.plot([ti/24. for ti in thrs_plot], angle_deg_plot, 'k.')
    plt.plot([ti/24. for ti in thrs_plot], mean_angle_plot, 'b.')
    if not np.isnan(danger_ind):
        plt.plot(danger_time/86400., danger_angle*180/np.pi, 'ro')
    plt.xlabel('Time [days]')
    plt.ylabel('Separation Angle [deg]')
    
    plt.figure()
    plt.plot(de_x_plot, de_y_plot, 'k.')
    plt.plot(de_x_mean, de_y_mean, 'b.')
    plt.xlim([-6e-5, 6e-5])
    plt.ylim([-6e-5, 6e-5])    
    if not np.isnan(danger_ind):
        plt.plot([0., danger_de_x], [0., danger_de_y], 'r')
    
    # plt.axis('equal')
    plt.grid()
    plt.xlabel('de [x]')
    plt.ylabel('de [y]')
    
    plt.figure()
    plt.plot(di_x_plot, di_y_plot, 'k.')
    plt.plot(di_x_mean, di_y_mean, 'b.')
    plt.xlim([-6e-4, 6e-4])
    plt.ylim([-6e-4, 6e-4])
    if not np.isnan(danger_ind):
        plt.plot([0., danger_di_x], [0., danger_di_y], 'r')
        
    plt.grid()
    plt.xlabel('di [x]')
    plt.ylabel('di [y]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs_plot, a_plot, 'k.')
    plt.ylabel('SMA [km]')
    plt.subplot(3,1,2)
    plt.plot(thrs_plot, e_plot, 'k.')
    plt.ylabel('ECC')
    plt.subplot(3,1,3)
    plt.plot(thrs_plot, i_plot, 'k.')
    plt.ylabel('INC [deg]')
    plt.xlabel('Time [hours]')
    
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(tout/3600., radial_list, 'k.')
    # plt.ylabel('R [m]')
    # plt.subplot(3,1,2)
    # plt.plot(tout/3600., intrack_list, 'k.')
    # plt.ylabel('I [m]')
    # plt.subplot(3,1,3)
    # plt.plot(tout/3600., crosstrack_list, 'k.')
    # plt.ylabel('C [m]')
    # plt.xlabel('Time [hours]')
    
    
    
    # Plot relative distances for a complete orbit in 5 day increments
    day_list = [0, 5, 10, 15, 20, 25, 30]
    # day_list = [26.7]
    if not np.isnan(danger_ind):
        day_list.append(tout[danger_ind]/86400.)
        print('Danger Time [days]', tout[danger_ind]/86400.)
        print('check', danger_time/86400.)
        
    plt.figure()
    for day in day_list:
        ind1 = np.where(tout >= day*86400.-3*P)
        ind2 = np.where(tout < day*86400.+3*P+1000.)
        inds = list(set(ind1[0]).intersection(set(ind2[0])))
        
        r_plot = [radial_list[ii] for ii in inds]
        c_plot = [crosstrack_list[ii] for ii in inds]
        
        
        color = 'k'        
        if day == 0:
            linewidth=3.
            color = 'g'
        elif day == 30:
            linewidth=3.
            color = 'b'
        else:
            linewidth=1.
        
        
        if not np.isnan(danger_ind):
            if day == tout[danger_ind]/86400.:
                color = 'r'
                linewidth = 3.

                
        plt.plot(c_plot, r_plot, color=color, linewidth=linewidth)
        
    plt.ylabel('Radial [m]')
    plt.xlabel('Cross-Track [m]')
    plt.xlim([-1000, 1000])
    plt.ylim([-1000, 1000])
    plt.grid()        
    
    
    plt.show()
    
    return


###############################################################################
# Utilities
###############################################################################


def inertial2relative_ei(rc_vect, vc_vect, rd_vect, vd_vect, GM=GME):
    '''
    This function converts inertial Cartesian position and velocity vectors of 
    two space objects into relative eccentricity and inclination vectors, also
    expressed in the inertial coordinate frame.
    
    '''
    
    # Reshape inputs if needed
    rc_vect = np.reshape(rc_vect, (3,1))
    vc_vect = np.reshape(vc_vect, (3,1))
    rd_vect = np.reshape(rd_vect, (3,1))
    vd_vect = np.reshape(vd_vect, (3,1))
    
    # Compute angular momentum unit vectors
    hc_vect = np.cross(rc_vect, vc_vect, axis=0)
    hd_vect = np.cross(rd_vect, vd_vect, axis=0)
    ih_c = hc_vect/np.linalg.norm(hc_vect)
    ih_d = hd_vect/np.linalg.norm(hd_vect)
    
    # Compute eccentricity vectors
    ec_vect = np.cross(vc_vect, hc_vect, axis=0)/GM - rc_vect/np.linalg.norm(rc_vect)
    ed_vect = np.cross(vd_vect, hd_vect, axis=0)/GM - rd_vect/np.linalg.norm(rd_vect)
    
    # Compute relative ecc/inc vectors in inertial frame
    di_vect = np.cross(ih_c, ih_d, axis=0)
    de_vect = ed_vect - ec_vect    
    
    return de_vect, di_vect


def compute_R1(theta):
    
    R1 = np.array([[1.,              0.,             0.],
                   [0.,   np.cos(theta),  np.sin(theta)],
                   [0.,  -np.sin(theta),  np.cos(theta)]])
    
    
    return R1


def compute_R3(theta):
    
    R3 = np.array([[ np.cos(theta),  np.sin(theta),   0.],
                   [-np.sin(theta),  np.cos(theta),   0.],
                   [           0.,              0.,   1.]])
    
    return R3


def eci2ric(rc_vect, vc_vect, Q_eci=[]):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Q_eci (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_eci : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Q_eci as appropriate for vector or matrix
    if len(Q_eci) == 0:
        Q_ric = ON
    elif np.size(Q_eci) == 3:
        Q_eci = Q_eci.reshape(3,1)
        Q_ric = np.dot(ON, Q_eci)
    else:
        Q_ric = np.dot(np.dot(ON, Q_eci), ON.T)

    return Q_ric


def ric2eci(rc_vect, vc_vect, Q_ric=[]):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    Q_ric (vector or matrix) to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in ECI
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T

    # Rotate Qin as appropriate for vector or matrix
    if len(Q_ric) == 0:
        Q_eci = NO
    elif np.size(Q_ric) == 3:
        Q_eci = np.dot(NO, Q_ric)
    else:
        Q_eci = np.dot(np.dot(NO, Q_ric), NO.T)

    return Q_eci


def eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    relative velocity drho_eci to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    rho_ric : 3x1 numpy array
      relative position vector in RIC
    drho_eci : 3x1 numpy array
      relative velocity vector in ECI

    Returns
    ------
    drho_ric : 3x1 numpy array
      relative velocity in RIC

    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)
    rho_ric = rho_ric.reshape(3,1)
    drho_eci = drho_eci.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Compute angular velocity vector
    dtheta = h/rc**2.
    w = np.reshape([0., 0., dtheta], (3,1))
    
    # Compute relative velocity vector using kinematic identity
    drho_ric = np.dot(ON, drho_eci) - np.cross(w, rho_ric, axis=0)

    return drho_ric


def ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    relative velocity drho_ric to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    rho_ric : 3x1 numpy array
      relative position vector in RIC
    drho_ric : 3x1 numpy array
      relative velocity vector in RIC

    Returns
    ------
    drho_eci : 3x1 numpy array
      relative velocity in ECI

    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)
    rho_ric = rho_ric.reshape(3,1)
    drho_ric = drho_ric.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T
    
    # Compute angular velocity vector
    dtheta = h/rc**2.
    w = np.reshape([0., 0., dtheta], (3,1))
    
    # Compute relative velocity vector using kinematic identity
    drho_eci = np.dot(NO, (drho_ric + np.cross(w, rho_ric, axis=0))) 
    
    return drho_eci


def kep2cart(elem, GM=GME):
    '''
    This function converts a vector of Keplerian orbital elements to a
    Cartesian state vector in inertial frame.
    
    Parameters
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [m]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [rad]
    elem[3] : RAAN
      Right Asc Ascending Node    [rad]
    elem[4] : w
      Argument of Periapsis       [rad]
    elem[5] : theta
      True Anomaly                [rad]
      
      
    Returns
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [m]
    cart[1] : y
      Position in y               [m]
    cart[2] : z
      Position in z               [m]
    cart[3] : dx
      Velocity in x               [m/s]
    cart[4] : dy
      Velocity in y               [m/s]
    cart[5] : dz
      Velocity in z               [m/s]  
      
    '''
    
    # Retrieve input elements
    a = float(elem[0])
    e = float(elem[1])
    i = float(elem[2])
    RAAN = float(elem[3])
    w = float(elem[4])
    theta = float(elem[5])

    # Calculate h and r
    p = a*(1 - e**2)
    h = np.sqrt(GM*p)
    r = p/(1. + e*math.cos(theta))

    # Calculate r_vect and v_vect
    r_vect = r * \
        np.array([[math.cos(RAAN)*math.cos(theta+w) - math.sin(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(RAAN)*math.cos(theta+w) + math.cos(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(theta+w)*math.sin(i)]])

    vv1 = math.cos(RAAN)*(math.sin(theta+w) + e*math.sin(w)) + \
          math.sin(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv2 = math.sin(RAAN)*(math.sin(theta+w) + e*math.sin(w)) - \
          math.cos(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv3 = -(math.cos(theta+w) + e*math.cos(w))*math.sin(i)
    
    v_vect = -GM/h * np.array([[vv1], [vv2], [vv3]])

    cart = np.concatenate((r_vect, v_vect), axis=0)
    
    return cart


def cart2kep(cart, GM=GME):
    '''
    This function converts a Cartesian state vector in inertial frame to
    Keplerian orbital elements.
    
    Parameters
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [m]
    cart[1] : y
      Position in y               [m]
    cart[2] : z
      Position in z               [m]
    cart[3] : dx
      Velocity in x               [m/s]
    cart[4] : dy
      Velocity in y               [m/s]
    cart[5] : dz
      Velocity in z               [m/s]
      
    Returns
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [rad]
    elem[3] : RAAN
      Right Asc Ascending Node    [rad]
    elem[4] : w
      Argument of Periapsis       [rad]
    elem[5] : theta
      True Anomaly                [rad]    
      
    '''
    
    # Retrieve input cartesian coordinates
    r_vect = cart[0:3].reshape(3,1)
    v_vect = cart[3:6].reshape(3,1)

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    ir_vect = r_vect/r
    v2 = np.linalg.norm(v_vect)**2
    h_vect = np.cross(r_vect, v_vect, axis=0)
    h = np.linalg.norm(h_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)     # km
    
    # Calculate eccentricity
    e_vect = np.cross(v_vect, h_vect, axis=0)/GM - ir_vect
    e = np.linalg.norm(e_vect)

    # Calculate RAAN and inclination
    ih_vect = h_vect/h

    RAAN = math.atan2(ih_vect[0,0], -ih_vect[1,0])   # rad
    i = math.acos(ih_vect[2,0])   # rad
    if RAAN < 0.:
        RAAN += 2.*math.pi

    # Apply correction for circular orbit, choose e_vect to point
    # to ascending node
    if e != 0:
        ie_vect = e_vect/e
    else:
        ie_vect = np.array([[math.cos(RAAN)], [math.sin(RAAN)], [0.]])

    # Find orthogonal unit vector to complete perifocal frame
    ip_vect = np.cross(ih_vect, ie_vect, axis=0)

    # Form rotation matrix PN
    PN = np.concatenate((ie_vect, ip_vect, ih_vect), axis=1).T

    # Calculate argument of periapsis
    w = math.atan2(PN[0,2], PN[1,2])  # rad
    if w < 0.:
        w += 2.*math.pi

    # Calculate true anomaly
    cross1 = np.cross(ie_vect, ir_vect, axis=0)
    tan1 = np.dot(cross1.T, ih_vect).flatten()[0]
    tan2 = np.dot(ie_vect.T, ir_vect).flatten()[0]
    theta = math.atan2(tan1, tan2)    # rad
    
    # Update range of true anomaly for elliptical orbits
    if a > 0. and theta < 0.:
        theta += 2.*math.pi
    
    # # Convert angles to deg
    # i *= 180./math.pi
    # RAAN *= 180./math.pi
    # w *= 180./math.pi
    # theta *= 180./math.pi
    
    # Form output
    elem = np.array([[a], [e], [i], [RAAN], [w], [theta]])
      
    return elem



if __name__ == '__main__':
    
    plt.close('all')
    
    
    datafile = os.path.join('unit_test', 'damico_tsx_tdx_output.pkl')
    
    # test_damico_tsx_tdx(datafile)
    plot_damico_tsx_tdx(datafile)

