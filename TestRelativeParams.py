import numpy as np
import math


# Generic Values
arcsec2rad = np.pi/(3600.*180.)

# Earth parameters
wE = 7.2921158553e-5  # rad/s
GME = 398600.4415  # km^3/s^2
J2E = 1.082626683e-3

# WGS84 Data (Pratap and Misra P. 103)
Re = 6378.1370   # km
rec_f = 298.257223563




def test_damico_tsx_tdx():
    
    # Constants
    Cd = 2.3                # unitless
    Cr = 1.3                # unitless
    A = 3.2                 # m^2
    m = 1238                # kg
    
    # Chief Orbit Parameters
    a = 6892.945*1000.      # m
    e = 1e-12
    i = 97.*np.pi/180.      # rad
    RAAN = 0.               # rad
    w = 270.*np.pi/180.     # rad
    theta = 0.              # rad
    
    # Orbit Differences
    di_x = 0.
    di_y = -1./a                # rad
    di = 0.                     # rad
    dRAAN = (di_y/np.sin(i))    # rad
    
    de_x = 0.
    de_y = 0.3/a                        # non-dim
    psi = 90.*np.pi/180.                # rad
    de = np.linalg.norm([de_x, de_y])   # non-dim
    
    # Deputy Orbit
    a2 = a + 0.
    e2 = e + de
    i2 = i + di                 # rad
    RAAN2 = RAAN + dRAAN        # rad
    w2 = 90*np.pi/180.          # rad
    theta2 = theta - np.pi      # rad
    
    # Rotation matrix from ECI to Orbit Frame 1
    R1 = compute_R1(i)    
    R3 = compute_R3(RAAN)
    OF1_ECI = R1 @ R3
    
    # Initial conditions
    X1 = kep2cart([a, e, i, RAAN, w, theta])
    X2 = kep2cart([a2, e2, i2, RAAN2, w2, theta2])
    
    # Check initial orbit energy and relative orbit params
    r1_vect = X1[0:3].reshape(3,1)
    v1_vect = X1[3:6].reshape(3,1)
    r2_vect = X2[0:3].reshape(3,1)
    v2_vect = X2[0:3].reshape(3,1)        
    
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
    
    R3w = compute_R3(-w)
    
    OF1_ECI_alt = R3w @ P_ECI
    
    
    
    
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
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]
      
      
    Returns
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]  
      
    '''
    
    # Retrieve input elements, convert to radians
    a = float(elem[0])
    e = float(elem[1])
    i = float(elem[2]) * math.pi/180
    RAAN = float(elem[3]) * math.pi/180
    w = float(elem[4]) * math.pi/180
    theta = float(elem[5]) * math.pi/180

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
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]
      
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
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]    
      
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
    
    # Convert angles to deg
    i *= 180./math.pi
    RAAN *= 180./math.pi
    w *= 180./math.pi
    theta *= 180./math.pi
    
    # Form output
    elem = np.array([[a], [e], [i], [RAAN], [w], [theta]])
      
    return elem



