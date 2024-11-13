###############################################################################
# This file contains code developed from the open source NASA CARA Analysis 
# Tools, provided under the NASA Open Source Software Agreement.
#
# Copyright © 2020 United States Government as represented by the Administrator 
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Modified (port to python) by Steve Gehly Feb 27, 2024
#
# References:
#
#  [1] Denenberg, E., "Satellite Closest Approach Calculation Through 
#      Chebyshev Proxy Polynomials," Acta Astronautica, 2020.
#
#  [2] Hall, D.T., Hejduk, M.D., and Johnson, L.C., "Remediating Non-Positive
#      Definite State Covariances for Collision Probability Estimation," 2017.
#
#  [3] https://github.com/nasa/CARA_Analysis_Tools
#
#  [4] Foster, J., Estes, H., "A parametric analysis of orbital debris collision 
#      probability and maneuver rate for space vehicles," Tech Report, 1992.
#
#  [5] Alfano, S., "Review of Conjunction Probability Methods for Short-term 
#      Encounters," Advances in the Astronautical Sciences, Vol. 127, Jan 2007, 
#      pp 719-746.
#
#  [6] Alfano, S., "Satellite Conjuction Monte Carlo Analysis," AAS Spaceflight
#      Mechanics Meeting (AAS-09-233), 2009.
#  
#
#
###############################################################################

import numpy as np
import math
from datetime import datetime
from scipy.integrate import dblquad
from scipy.special import erfcinv
import pickle
import time
import pandas as pd
import os
import sys

from tudatpy.astro import time_conversion

import TudatPropagator as prop

metis_dir = r'C:\Users\sgehly\Documents\code\metis'
sys.path.append(metis_dir)

# import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
import dynamics.dynamics_functions as dyn
import sensors.measurement_functions as mfunc
# import sensors.sensors as sens
# import sensors.visibility_functions as visfunc
# from utilities import astrodynamics as astro
# from utilities import coordinate_systems as coord
from utilities import eop_functions as eop
# from utilities import time_systems as timesys
# from utilities import tle_functions as tle
from utilities.constants import GME


###############################################################################
# Basic I/O
###############################################################################

def read_catalog_file(rso_file):
    '''
    This function reads a pickle file containing data for Resident Space Objects
    (RSOs) and returns a dictionary containing the same data, indexed by 
    5 digit NORAD ID.
    
    Parameters
    ------
    rso_file : string
        path and filename of pickle file containing RSO data
    
    Returns
    ------
    rso_dict : dictionary
        RSO data indexed by 5 digit NORAD ID
        The following data are provided for each object:
            UTC : datetime object corresponding to state and covar 
            state : 6x1 numpy array, Cartesian position and velocity in ECI 
            covar : 6x6 numpy array, covariance matrix associated with state
            mass : float [kg]
            area : float [m^2]
            Cd : float, drag coefficient
            Cr : float, SRP coefficient
            
            Units of m and m/s for position/velocity
            
    '''

    # Load RSO dict
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()    
    
    return rso_dict


def read_cdm_file(cdm_file):
    '''
    This function reads a text file containing for a Conjunction Data Message
    (CDM) and returns a dictionary containing the same data, indexed by 
    5 digit NORAD ID.
    
    Parameters
    ------
    cdm_file : string
        path and filename of pickle file containing CDM data
    
    Returns
    ------
    cdm_data : dict
        extracted CDM data including TCA, miss distance, Pc [meters, seconds] 
    
    '''
    
    cdm_data = {}
    
    with open(cdm_file) as file:
        lines = [line.rstrip() for line in file]        
    
        
    for ii in range(len(lines)):
        line = lines[ii]
                
        try:
            key, value = [x.strip() for x in line.split('=')]
            

            
            if key == 'TCA':
                cdm_data['TCA_UTC'] = datetime.fromisoformat(value)
                
            if key == 'MISS_DISTANCE':
                md, unit = value.split()
                cdm_data['MISS_DISTANCE'] = float(md)
                
            if key == 'RELATIVE_SPEED':
                speed, unit = value.split()
                cdm_data['RELATIVE_SPEED'] = float(speed)
                
            if key == 'RELATIVE_POSITION_R':
                r_pos, unit = value.split()
                r_pos = float(r_pos)
                
            if key == 'RELATIVE_POSITION_T':
                t_pos, unit = value.split()
                t_pos = float(t_pos)
            
            if key == 'RELATIVE_POSITION_N':
                n_pos, unit = value.split()
                n_pos = float(n_pos)
                
            if key == 'RELATIVE_VELOCITY_R':
                r_vel, unit = value.split()
                r_vel = float(r_vel)
                
            if key == 'RELATIVE_VELOCITY_T':
                t_vel, unit = value.split()
                t_vel = float(t_vel)
                
            if key == 'RELATIVE_VELOCITY_N':
                n_vel, unit = value.split()
                n_vel = float(n_vel)
                
            if key == 'COLLISION_PROBABILITY':
                Pc = float(value)
                cdm_data['Pc'] = Pc
                
            if key == 'OBJECT' and value == 'OBJECT1':
                obj1_ind = ii

                
            if key == 'OBJECT' and value == 'OBJECT2':             
                obj2_ind = ii
                
        except:
            continue
        
    # Store output data 
    rtn_pos = np.reshape([r_pos, t_pos, n_pos], (3,1))
    rtn_vel = np.reshape([r_vel, t_vel, n_vel], (3,1))
    cdm_data['rtn_pos'] = rtn_pos
    cdm_data['rtn_vel'] = rtn_vel
    
    # Retrieve data for object 1
    lines_obj1 = lines[obj1_ind:obj2_ind] 
    obj1_id, X1, P1 = read_cdm_state_covar(lines_obj1)
    
    cdm_data['obj1'] = {}
    cdm_data['obj1']['obj_id'] = obj1_id
    cdm_data['obj1']['X'] = X1
    cdm_data['obj1']['P'] = P1
    
    # Retrieve data for object 2
    lines_obj2 = lines[obj2_ind:] 
    obj2_id, X2, P2 = read_cdm_state_covar(lines_obj2)
    
    cdm_data['obj2'] = {}
    cdm_data['obj2']['obj_id'] = obj2_id
    cdm_data['obj2']['X'] = X2
    cdm_data['obj2']['P'] = P2

    
    return cdm_data


def read_cdm_state_covar(lines):
    
    for ii in range(len(lines)):
        line = lines[ii]
                
        try:
            key, value = [x.strip() for x in line.split('=')]
            
            if key == 'OBJECT_DESIGNATOR':
                obj_id = int(value)
                        
            if key == 'X':
                x_pos, unit = value.split()
                x_pos = float(x_pos)
                if unit == '[km]':
                    x_pos *= 1000.
            
            if key == 'Y':
                y_pos, unit = value.split()
                y_pos = float(y_pos)
                if unit == '[km]':
                    y_pos *= 1000.
                    
            if key == 'Z':
                z_pos, unit = value.split()
                z_pos = float(z_pos)
                if unit == '[km]':
                    z_pos *= 1000.
                    
            if key == 'X_DOT':
                x_vel, unit = value.split()
                x_vel = float(x_vel)
                if unit == '[km/s]':
                    x_vel *= 1000.
                    
            if key == 'Y_DOT':
                y_vel, unit = value.split()
                y_vel = float(y_vel)
                if unit == '[km/s]':
                    y_vel *= 1000.
                    
            if key == 'Z_DOT':
                z_vel, unit = value.split()
                z_vel = float(z_vel)
                if unit == '[km/s]':
                    z_vel *= 1000.
            
            if key == 'CR_R':
                CR_R, unit = value.split()
                CR_R = float(CR_R)
                
            if key == 'CT_R':
                CT_R, unit = value.split()
                CT_R = float(CT_R)
                
            if key == 'CT_T':
                CT_T, unit = value.split()
                CT_T = float(CT_T)
                
            if key == 'CN_R':
                CN_R, unit = value.split()
                CN_R = float(CN_R)
                
            if key == 'CN_T':
                CN_T, unit = value.split()
                CN_T = float(CN_T)
                
            if key == 'CN_N':
                CN_N, unit = value.split()
                CN_N = float(CN_N)
                
            if key == 'CRDOT_R':
                CRDOT_R, unit = value.split()
                CRDOT_R = float(CRDOT_R)
                
            if key == 'CRDOT_T':
                CRDOT_T, unit = value.split()
                CRDOT_T = float(CRDOT_T)
                
            if key == 'CRDOT_N':
                CRDOT_N, unit = value.split()
                CRDOT_N = float(CRDOT_N)
                
            if key == 'CRDOT_RDOT':
                CRDOT_RDOT, unit = value.split()
                CRDOT_RDOT = float(CRDOT_RDOT)
                
            if key == 'CTDOT_R':
                CTDOT_R, unit = value.split()
                CTDOT_R = float(CTDOT_R)
                
            if key == 'CTDOT_T':
                CTDOT_T, unit = value.split()
                CTDOT_T = float(CTDOT_T)
            
            if key == 'CTDOT_N':
                CTDOT_N, unit = value.split()
                CTDOT_N = float(CTDOT_N)
                
            if key == 'CTDOT_RDOT':
                CTDOT_RDOT, unit = value.split()
                CTDOT_RDOT = float(CTDOT_RDOT)
                
            if key == 'CTDOT_TDOT':
                CTDOT_TDOT, unit = value.split()
                CTDOT_TDOT = float(CTDOT_TDOT)
                
            if key == 'CNDOT_R':
                CNDOT_R, unit = value.split()
                CNDOT_R = float(CNDOT_R)
                
            if key == 'CNDOT_T':
                CNDOT_T, unit = value.split()
                CNDOT_T = float(CNDOT_T)
                
            if key == 'CNDOT_N':
                CNDOT_N, unit = value.split()
                CNDOT_N = float(CNDOT_N)
                
            if key == 'CNDOT_RDOT':
                CNDOT_RDOT, unit = value.split()
                CNDOT_RDOT = float(CNDOT_RDOT)
                
            if key == 'CNDOT_TDOT':
                CNDOT_TDOT, unit = value.split()
                CNDOT_TDOT = float(CNDOT_TDOT)
                
            if key == 'CNDOT_NDOT':
                CNDOT_NDOT, unit = value.split()
                CNDOT_NDOT = float(CNDOT_NDOT)
            
        except:
            continue
        
    X = np.reshape([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel], (6,1))
    P = np.zeros((6,6))
    P[0,0] = CR_R
    P[1,1] = CT_T
    P[2,2] = CN_N
    P[3,3] = CRDOT_RDOT
    P[4,4] = CTDOT_TDOT
    P[5,5] = CNDOT_NDOT
    P[0,1] = P[1,0] = CT_R
    P[0,2] = P[2,0] = CN_R
    P[0,3] = P[3,0] = CRDOT_R
    P[0,4] = P[4,0] = CTDOT_R
    P[0,5] = P[5,0] = CNDOT_R
    P[1,2] = P[2,1] = CN_T
    P[1,3] = P[3,1] = CRDOT_T
    P[1,4] = P[4,1] = CTDOT_T
    P[1,5] = P[5,1] = CNDOT_T
    P[2,3] = P[3,2] = CRDOT_N
    P[2,4] = P[4,2] = CTDOT_N
    P[2,5] = P[5,2] = CNDOT_N
    P[3,4] = P[4,3] = CTDOT_RDOT
    P[3,5] = P[5,3] = CNDOT_RDOT
    P[4,5] = P[5,4] = CNDOT_TDOT
    
    return obj_id, X, P


def retrieve_conjunction_data_at_tca(cdm_file=''):
    '''
    This function retrieves the epoch and state vectors at TCA.
    
    Parameters
    ------
    cdm_file : string, optional
        path and filename of CDM data file (default='')
    
    '''
    
    # If a CDM is provided, parse and retrieve object data
    if len(cdm_file) > 0:
        cdm_data = read_cdm_file(cdm_file)
        TCA_UTC = cdm_data['TCA_UTC']
        X1 = cdm_data['obj1']['X']
        X2 = cdm_data['obj2']['X']
        
        # Convert TCA to seconds from epoch in TDB
        tudat_datetime_utc = time_conversion.datetime_to_tudat(TCA_UTC)
        time_scale_converter = time_conversion.default_time_scale_converter()
        TCA_epoch_utc = tudat_datetime_utc.epoch()
        TCA_epoch_tdb = time_scale_converter.convert_time(
                        input_scale = time_conversion.utc_scale,
                        output_scale = time_conversion.tdb_scale,
                        input_value = TCA_epoch_utc)
        
    # Otherwise, retrieve data from other source
        
    
    return TCA_epoch_tdb, X1, X2


###############################################################################
# Initialize Parameters
###############################################################################


def perturb_state_vector(Xo, P):
    
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6,))
    Xf = Xo + pert_vect.reshape(Xo.shape)
    
    return Xf


def filter_setup(Xo, tvec, noise):    
    
    # # Initialize default parameters
    # state_params, int_params, bodies = prop.initialize_tudat('rkdp87')
    
    # # Update for GPS measurement simulation
    # step = 30.
    # tol = np.inf
    # int_params['step'] = step
    # int_params['max_step'] = step
    # int_params['min_step'] = step
    # int_params['atol'] = tol
    # int_params['rtol'] = tol
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME*1e9
    
    # Define integrator parameters
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    

    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
    # Sensor and measurement parameters
    sensor_id = 'GPS'
    sensor_params = {}
    sensor_params[sensor_id] = {}
    
    meas_types = ['x', 'y', 'z']
    sensor_params[sensor_id]['meas_types'] = meas_types
    
    sigma_dict = {}
    sigma_dict['x'] = noise
    sigma_dict['y'] = noise
    sigma_dict['z'] = noise
    sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        
    # Propagate orbit
    # tout, X_truth = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
    tout, X_truth = dyn.general_dynamics(Xo, tvec, state_params, int_params)  
    
    
    truth_dict = {}
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    
    for kk in range(len(tout)):
        
        Xk = X_truth[kk,0:6].reshape(6,1)
        truth_dict[tout[kk]] = Xk
        
        Yk = Xk[0:3].reshape(3,1)
        for mtype in meas_types:
            ind = meas_types.index(mtype)
            Yk[ind] += np.random.randn()*sigma_dict[mtype]        
            
        meas_dict['tk_list'].append(tout[kk])
        meas_dict['Yk_list'].append(Yk)
        meas_dict['sensor_id_list'].append(sensor_id)            
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
    
    # Initial state for filter
    P = np.diag([1e6, 1e6, 1e6, 1., 1., 1.])
    state_dict = {}
    state_dict[tout[0]] = {}
    state_dict[tout[0]]['X'] = perturb_state_vector(Xo, P)
    state_dict[tout[0]]['P'] = P
    
    
    return state_dict, meas_dict, params_dict, truth_dict


def run_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    
    params_dict['int_params']['intfcn'] = dyn.ode_twobody_stm
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)    
    # analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict)
    
    t0 = sorted(list(state_dict.keys()))[0]
    Xo = filter_output[t0]['X']
    Po = filter_output[t0]['P']
    
    return Xo, Po


def initialize_covar(t0, Xo, thrs=3., interval=300., noise=1.):
        
    # Setup time vector for simulated GPS measurements
    tvec = np.arange(t0, t0 + (3600.*thrs), interval)
    
    # Initialize parameter dictionaries
    state_dict, meas_dict, params_dict, truth_dict = filter_setup(Xo, tvec, noise)
    
    # Run filter
    meas_fcn = mfunc.H_inertial_xyz
    Xo, Po = run_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    
    
    return Xo, Po


###############################################################################
# Compute Risk Metrics
###############################################################################


def compute_euclidean_distance(r_A, r_B):
    
    d = np.linalg.norm(r_A - r_B)
    
    return d


def compute_mahalanobis_distance(r_A, r_B, P_A, P_B):    
    
    Psum = P_A + P_B
    invP = np.linalg.inv(Psum)
    diff = r_A - r_B
    M = float(np.sqrt(np.dot(diff.T, np.dot(invP, diff)))[0,0])
    
    return M


def compute_RTN_posvel():
    
    return


###############################################################################
# Relative Ecc/Inc Vector Separation
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
    
    # Compute separation angle
    angle = np.arccos(np.dot(de_vect.flatten(), di_vect.flatten())/(np.linalg.norm(de_vect)*np.linalg.norm(di_vect)))
    
    return angle, de_vect, di_vect


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




def compute_angle_diff(a, b):
    
    diff = (a - b) % (2.*np.pi)
    if diff < -np.pi:
        diff += 2.*np.pi
    if diff > np.pi:
        diff -= 2.*np.pi    
    
    return abs(diff)


###############################################################################
# 2D Probability of Collision (Pc) Functions
###############################################################################

def Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=1e-8, HBR_type='circle'):
    '''
    This function computes the probability of collision (Pc) in the 2D 
    encounter plane following the method of Foster. The code has been ported
    from the MATLAB library developed by the NASA CARA team, listed in Ref 3.
    The function supports 3 types of hard body regions: circle, square, and 
    square equivalent to the area of the circle. The input covariance may be
    either 3x3 or 6x6, but only the 3x3 position covariance will be used in
    the calculation of Pc.
    
    
    Parameters
    ------
    X1 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 1 in ECI [m, m/s]
    P1 : 6x6 numpy array
        Estimated covariance of Object 1 in ECI [m^2, m^2/s^2]
    X2 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 2 in ECI [m, m/s]
    P2 : 6x6 numpy array
        Estimated covariance of Object 2 in ECI [m^2, m^2/s^2]
    HBR : float
        hard-body region (e.g. radius for spherical object) [m]
    rtol : float, optional
        relative tolerance for numerical quadrature (default=1e-8)
    HBR_type : string, optional
        type of hard body region ('circle', 'square', or 'squareEqArea')
        (default='circle')
    
    Returns
    ------
    Pc : float
        probability of collision
    
    '''
    
    # Retrieve and combine the position covariance
    Peci = P1[0:3,0:3] + P2[0:3,0:3]
    
    # Construct the relative encounter frame
    r1 = np.reshape(X1[0:3], (3,1))
    v1 = np.reshape(X1[3:6], (3,1))
    r2 = np.reshape(X2[0:3], (3,1))
    v2 = np.reshape(X2[3:6], (3,1))
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v, axis=0)
    
    # Unit vectors of relative encounter frame
    yhat = v/np.linalg.norm(v)
    zhat = h/np.linalg.norm(h)
    xhat = np.cross(yhat, zhat, axis=0)
    
    # Transformation matrix
    eci2xyz = np.concatenate((xhat.T, yhat.T, zhat.T))
    
    # Transform combined covariance to relative encounter frame (xyz)
    Pxyz = np.dot(eci2xyz, np.dot(Peci, eci2xyz.T))
    
    # 2D Projected covariance on the x-z plane of the relative encounter frame
    red = np.array([[1., 0., 0.], [0., 0., 1.]])
    Pxz = np.dot(red, np.dot(Pxyz, red.T))

    # Exception Handling
    # Remediate non-positive definite covariances
    Lclip = (1e-4*HBR)**2.
    Pxz_rem, Pxz_det, Pxz_inv, posdef_status, clip_status = remediate_covariance(Pxz, Lclip)
    
    
    # Calculate Double Integral
    x0 = np.linalg.norm(r)
    z0 = 0.
    
    # Set up quadrature
    atol = 1e-13
    Integrand = lambda z, x: math.exp(-0.5*(Pxz_inv[0,0]*x**2. + Pxz_inv[0,1]*x*z + Pxz_inv[1,0]*x*z + Pxz_inv[1,1]*z**2.))

    if HBR_type == 'circle':
        lower_semicircle = lambda x: -np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        upper_semicircle = lambda x:  np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'square':
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, z0-HBR, z0+HBR, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'squareEqArea':
        HBR_eq = HBR*np.sqrt(math.pi)/2.
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR_eq, x0+HBR_eq, z0-HBR_eq, z0+HBR_eq, epsabs=atol, epsrel=rtol)[0])
    
    else:
        print('Error: HBR type is not supported! Must be circle, square, or squareEqArea')
        print(HBR_type)
    
    return Pc



def remediate_covariance(Praw, Lclip, Lraw=[], Vraw=[]):
    '''
    This function provides a level of exception handling by detecting and 
    remediating non-positive definite covariances in the collision probability
    calculation, following the procedure in Hall et al. (Ref 2). This code has
    been ported from the MATLAB library developed by the NASA CARA team, 
    listed in Ref 3.
    
    The function employs an eigenvalue clipping method, such that eigenvalues
    below the specified Lclip value are reset to Lclip. The covariance matrix,
    determinant, and inverse are then recomputed using the original 
    eigenvectors and reset eigenvalues to ensure the output is positive (semi)
    definite. An input of Lclip = 0 will result in the output being positive
    semi-definite.
    
    Parameters
    ------
    Praw : nxn numpy array
        unremediated covariance matrix    
    
    Returns
    ------
    
    
    '''
    
    # Ensure the covariance has all real elements
    if not np.all(np.isreal(Praw)):
        print('Error: input Praw is not real!')
        print(Praw)
        return
    
    # Calculate eigenvectors and eigenvalues if not input
    if len(Lraw) == 0 and len(Vraw) == 0:
        Lraw, Vraw = np.linalg.eig(Praw)
        
    # Define the positive definite status of Praw
    posdef_status = np.sign(min(Lraw))
    
    # Clip eigenvalues if needed, and record clipping status
    Lrem = Lraw.copy()
    if min(Lraw) < Lclip:
        clip_status = True
        Lrem[Lraw < Lclip] = Lclip
    else:
        clip_status = False
        
    # Determinant of remediated covariance
    Pdet = np.prod(Lrem)
    
    # Inverse of remediated covariance
    Pinv = np.dot(Vraw, np.dot(np.diag(1./Lrem), Vraw.T))
    
    # Remediated covariance
    if clip_status:
        Prem = np.dot(Vraw, np.dot(np.diag(Lrem), Vraw.T))
    else:
        Prem = Praw.copy()
    
    
    return Prem, Pdet, Pinv, posdef_status, clip_status


###############################################################################
#
###############################################################################



###############################################################################
# Time of Closest Approach (TCA) Calculation
###############################################################################


def compute_TCA(X1, X2, trange, rso1_params, rso2_params, int_params, 
                bodies=None, rho_min_crit=0., N=16, subinterval_factor=0.5):
    '''
    This function computes the Time of Closest Approach using Chebyshev Proxy
    Polynomials. Per Section 3 of Denenberg (Ref 1), the function subdivides
    the full time interval specified by trange into units no more than half the 
    orbit period of the smaller orbit, which should contain at most one local
    minimum of the relative distance between orbits. The funtion will either
    output the time and Euclidean distance at closest approach over the full
    time interval, or a list of all close approaches under a user-specified
    input rho_min_crit.
    
    Parameters
    ------
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [m, m/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [m, m/s]
    trange : 2 element list or array [t0, tf]
        initial and final time for the full interval [sec since J2000]
    rso1_params : dictionary
        state_params of object 1
    rso2_params : dictionary
        state_params of object 2
    int_params : dictionary
        integration parameters
    bodies : tudat object, optional
        contains parameters for the environment bodies used in propagation        
    rho_min_crit : float, optional
        critical value of minimum distance (default=0.)
        if > 0, output will contain all close approaches under this distance
    N : int, optional
        order of the Chebyshev Proxy Polynomial approximation (default=16)
        default corresponds to recommended value from Denenberg Section 3
    subinterval_factor : float, optional
        factor to multiply smaller orbit period by (default=0.5)
        default corresponds to recommended value from Denenberg Section 3
        
    Returns
    ------
    T_list : list
        time in seconds since J2000 at which relative distance between objects
        is minimum or under rho_min_crit
    rho_list : list
        list of ranges between the objects
    
    '''
    
    # Setup Tudat propagation if needed
    if bodies is None:
        bodies = prop.tudat_initialize_bodies()
        
    # Setup first time interval
    subinterval = compute_subinterval(X1, X2, subinterval_factor)
    t0 = trange[0]
    a = trange[0]
    b = min(trange[-1], a + subinterval)
        
    # Compute interpolation matrix for Chebyshev Proxy Polynomials of order N
    # Note that this can be reused for all polynomial approximations as it
    # only depends on the order
    interp_mat = compute_interpolation_matrix(N)
    
    # Loop over times in increments of subinterval until end of trange
    T_list = []
    rho_list = []
    rho_min = np.inf
    tmin = 0.
    while b <= trange[1]:
        
        print('')
        print('current interval [t0, tf]')
        print(a, b)
        print('dt [sec]', b-a)
        print('dt total [hours]', (b-trange[0])/3600.)
    
        # Determine Chebyshev-Gauss-Lobato node locations
        tvec = compute_CGL_nodes(a, b, N)
        
        # Evaluate function at node locations
        gvec, dum1, dum2, dum3, X1out, X2out, ti_out =  gvec_tudat(t0, tvec, X1, X2, rso1_params, rso2_params, int_params, bodies)
        # print('gvec', gvec)
        
        
        # Find the roots of the relative range rate g(t)
        troots = compute_gt_roots(gvec, interp_mat, a, b)
        # print('troots', troots)
        
        # If this is first pass, include the interval endpoints for evaluation
        if np.isinf(rho_min):
            troots = np.concatenate((troots, np.array([trange[0], trange[-1]])))
            
                
        # Check if roots constitute a global minimum and/or are below the
        # critical threshold
        if len(troots) > 0:
            
            # dum, rvec, ivec, cvec = gvec_fcn(t0, troots, X1, X2, params)
            # dum, rvec, ivec, cvec =  gvec_tudat(t0, troots, X1, X2, rso1_params, rso2_params, int_params, bodies)
            dum1, rvec, ivec, cvec, dum2, dum3, dum4 = gvec_tudat(t0, troots, X1, X2, rso1_params, rso2_params, int_params, bodies)
            for ii in range(len(troots)):
                rho = np.sqrt(rvec[ii]**2 + ivec[ii]**2 + cvec[ii]**2)
                
                # print('ti', troots[ii])
                # print('rho', rho)
                
                # Store if below critical threshold
                if rho < rho_min_crit:
                    rho_list.append(rho)
                    T_list.append(troots[ii])
                
                # Update global minimum
                if rho < rho_min:
                    rho_min = rho
                    tmin = troots[ii]
            
        # Increment time interval
        if b == trange[-1]:
            break
        
        a = float(b)
        if b + subinterval <= trange[-1]:
            b += subinterval
        else:
            b = trange[-1]  
            
        # Update state vectors for next iteration
        X1 = X1out
        X2 = X2out
        t0 = ti_out
            
    # # Evaluate the relative range at the endpoints of the interval to ensure
    # # these are not overlooked
    # dum, rvec, ivec, cvec = gvec_fcn(trange, X1, X2, params)
    # rho0 = np.sqrt(rvec[0]**2 + ivec[0]**2 + cvec[0]**2)
    # rhof = np.sqrt(rvec[-1]**2 + ivec[-1]**2 + cvec[-1]**2)
    
    # # Store the global minimum and append to lists if others are below 
    # # critical threshold
    # rho_candidates = [rho_min, rho0, rhof]
    # global_min = min(rho_candidates)
    # global_tmin = [tmin, trange[0], trange[-1]][rho_candidates.index(global_min)]
    
    
    
    # if ((rho0 < rho_min_crit) and (rhof < rho_min_crit)) or (rho0 == rhof):
    #     T_list = [trange[0], trange[-1]]
    #     rho_list = [rho0, rhof]
    # elif rho0 < rhof:
    #     T_list = [trange[0]]
    #     rho_list = [rho0]
    # elif rhof < rho0:
    #     T_list = [trange[-1]]
    #     rho_list = [rhof]   
        
    # If a global minimum has been found, store output
    if rho_min < np.inf and tmin not in T_list:
        T_list.append(tmin)
        rho_list.append(rho_min)
    
    # # Otherwise, compute and store the minimum range and TCA using the 
    # # endpoints of the interval
    # else:
           
        
    # Sort output
    if len(T_list) > 1:
        sorted_inds = np.argsort(T_list)
        T_list = [T_list[ii] for ii in sorted_inds]
        rho_list = [rho_list[ii] for ii in sorted_inds]
    
    return T_list, rho_list


def gvec_tudat(t0, tvec, X1, X2, rso1_params, rso2_params, int_params, bodies):
    '''
    This function computes terms for the Denenberg TCA algorithm.
    
    '''
    
    # Compute function values to find roots of
    # In order to minimize rho, we seek zeros of first derivative
    # f(t) = dot(rho_vect, rho_vect)
    # g(t) = df/dt = 2*dot(drho_vect, rho_vect)
    gvec = np.zeros(len(tvec),)
    rvec = np.zeros(len(tvec),)
    ivec = np.zeros(len(tvec),)
    cvec = np.zeros(len(tvec),)
    jj = 0
    for ti in tvec:
        
        if ti == t0:
        # if ti == tvec[0]:
            X1_t = X1
            X2_t = X2
            
        else:
            tin = [t0, ti]
            # tin = [tvec[0], ti]
            tout1, Xout1 = prop.propagate_orbit(X1, tin, rso1_params, int_params, bodies)
            tout2, Xout2 = prop.propagate_orbit(X2, tin, rso2_params, int_params, bodies)
            
            X1_t = Xout1[-1,:]
            X2_t = Xout2[-1,:]        
        
        rc_vect = X1_t[0:3].reshape(3,1)
        vc_vect = X1_t[3:6].reshape(3,1)
        rd_vect = X2_t[0:3].reshape(3,1)
        vd_vect = X2_t[3:6].reshape(3,1)
        
        rho_eci = rd_vect - rc_vect
        drho_eci = vd_vect - vc_vect
        rho_ric = eci2ric(rc_vect, vc_vect, rho_eci)
        drho_ric = eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci)
        
        rho_ric = rho_ric.flatten()
        drho_ric = drho_ric.flatten()
        
        # print('')
        # print('ti', ti)
        # print('X1_t', X1_t)
        # print('X2_t', X2_t)
        
        gvec[jj] = float(2*np.dot(rho_ric.T, drho_ric))
        rvec[jj] = float(rho_ric[0])
        ivec[jj] = float(rho_ric[1])
        cvec[jj] = float(rho_ric[2])
        jj += 1    
    
    
    return gvec, rvec, ivec, cvec, X1_t, X2_t, ti


def compute_CGL_nodes(a, b, N):
    '''
    This function computes the location of the Chebyshev-Gauss-Lobatto nodes
    over the interval [a,b] given the order of the Chebyshev Proxy Polynomial 
    N. Per the algorithm in Denenberg, these nodes can be computed once and 
    used to approximate the derivative of the distance function, as well as the 
    relative distance components in RIC coordinates, for the same interval.
    
    Parameters
    ------
    a : float
        lower bound of interval
    b : float
        upper bound of interval
    N : int
        order of the Chebyshev Proxy Polynomial approximation
        
    Returns
    ------
    xvec : 1D (N+1) numpy array
        CGL node locations
    
    '''
    
    # Compute CGL nodes (Denenberg Eq 11)
    jvec = np.arange(0,N+1)
    xvec = ((b-a)/2.)*(np.cos(np.pi*jvec/N)) + ((b+a)/2.)
    
    return xvec


def compute_interpolation_matrix(N):
    '''
    This function computes the (N+1)x(N+1) interpolation matrix given the order
    of the Chebyshev Proxy Polynomial N. Per the algorithm in Denenberg, this 
    matrix can be computed once and reused to approximate the derivative of the
    distance function over multiple intervals, as well as to compute the 
    relative distance components in RIC coordinates.
    
    Parameters
    ------
    N : int
        order of the Chebyshev Proxy Polynomial approximation
    
    Returns
    ------
    interp_mat : (N+1)x(N+1) numpy array
        interpolation matrix
        
    '''
    
    # Compute values of pj (Denenberg Eq 13)
    pvec = np.ones(N+1,)
    pvec[0] = 2.
    pvec[N] = 2.
    
    # Compute terms of interpolation matrix (Denenberg Eq 12)
    # Set up arrays of j,k values and compute outer product matrix
    jvec = np.arange(0,N+1)
    kvec = jvec.copy()
    jk_mat = np.dot(jvec.reshape(N+1,1),kvec.reshape(1,N+1))
    
    # Compute cosine term and pj,pk matrix, then multiply component-wise
    Cmat = np.cos(np.pi/N*jk_mat)
    pjk_mat = (2./N)*(1./np.dot(pvec.reshape(N+1,1), pvec.reshape(1,N+1)))
    interp_mat = np.multiply(pjk_mat, Cmat)
    
    return interp_mat


def compute_gt_roots(gvec, interp_mat, a, b):
    
    # Order of approximation
    N = len(gvec) - 1
    
    # Compute aj coefficients (Denenberg Eq 14)
    aj_vec = np.dot(interp_mat, gvec.reshape(N+1,1))
    
    # Compute the companion matrix (Denenberg Eq 18)
    Amat = np.zeros((N,N))
    Amat[0,1] = 1.
    Amat[-1,:] = -aj_vec[0:N].flatten()/(2*aj_vec[N])
    Amat[-1,-2] += 0.5
    for jj in range(1,N-1):
        Amat[jj,jj-1] = 0.5
        Amat[jj,jj+1] = 0.5
        
    # Compute eigenvalues
    # TODO paper indicates some eigenvalues may have small imaginary component
    # but testing seems to show this is much more significant issue, needs
    # further analysis
    eig, dum = np.linalg.eig(Amat)
    eig_real = np.asarray([np.real(ee) for ee in eig if (np.isreal(ee) and ee >= -1. and ee <= 1.)])
    roots = (b+a)/2. + eig_real*(b-a)/2.

    return roots


def compute_subinterval(X1, X2, subinterval_factor=0.5, GM=3.986004415e14):
    '''
    This function computes an appropriate length subinterval of the specified
    (finite) total interval on which to find the closest approach. Per the
    discussion in Denenberg Section 3, for 2 closed orbits, there will be at
    most 4 extrema (2 minima) during one revolution of the smaller orbit. Use
    of a subinterval equal to half this time yields a unique (local) minimum
    over the subinterval and has shown to work well in testing.
    
    Parameters
    ------
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [m, m/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [m, m/s]
    subinterval_factor : float, optional
        factor to multiply smaller orbit period by (default=0.5)
    GM : float, optional
        gravitational parameter (default=GME) [m^3/s^2]
        
    Returns
    ------
    subinterval : float
        duration of appropriate subinterval [sec]
        
    '''
    
    # Compute semi-major axis
    a1 = compute_SMA(X1, GM)
    a2 = compute_SMA(X2, GM)
    
    # print('a1', a1)
    # print('a2', a2)
    
    # If both orbits are closed, choose the smaller to compute orbit period
    if (a1 > 0.) and (a2 > 0.):
        amin = min(a1, a2)
        period = 2.*np.pi*np.sqrt(amin**3./GM)
        
    # If one orbit is closed and the other is an escape trajectory, choose the
    # closed orbit to compute orbit period
    elif a1 > 0.:
        period = 2.*np.pi*np.sqrt(a1**3./GM)
    
    elif a2 > 0.:
        period = 2.*np.pi*np.sqrt(a2**3./GM)
        
    # If both orbits are escape trajectories, choose an arbitrary period 
    # corresponding to small orbit
    else:
        period = 3600.
        
    # print(period)

        
    # Scale the smaller orbit period by user input
    subinterval = period*subinterval_factor
    
    # print('subinterval', subinterval)

    
    return subinterval


def compute_SMA(cart, GM=3.986004415e14):
    '''
    This function computes semi-major axis given a Cartesian state vector in
    inertial coordinates.
    
    Parameters
    ------
    cart : 6x1 numpy array
        cartesian state vector in ECI [m, m/s]
    GM : float, optional
        gravitational parameter (default=GME) [m^3/s^2]
        
    Returns
    ------
    a : float
        semi-major axis [m]
    
    '''
    
    # Retrieve position and velocity vectors
    r_vect = cart[0:3].flatten()
    v_vect = cart[3:6].flatten()

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    v2 = np.dot(v_vect, v_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)        
    
    return a


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


def unit_test_tca():
    '''
    This function performs a unit test of the compute_TCA function. The object
    parameters are such that a collision is expected 30 minutes after the
    initial epoch (zero miss distance).
    
    The TCA function is run twice, using a fixed step RK4 and variable step
    RKF78 to compare the results.
    
    '''
    
    # Initial time and state vectors
    t0 = (datetime(2024, 3, 23, 5, 30, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    
    X1 = np.array([[ 3.75944379e+05],
                   [ 6.08137408e+06],
                   [ 3.28340214e+06],
                   [-5.32161464e+03],
                   [-2.32172417e+03],
                   [ 4.89152047e+03]])
    
    X2 = np.array([[ 3.30312011e+06],
                   [-2.69542170e+06],
                   [-5.71365135e+06],
                   [-4.06029364e+03],
                   [-6.22037456e+03],
                   [ 9.09217382e+02]])
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create) 
    
    rso1_params = {}
    rso1_params['mass'] = 260.
    rso1_params['area'] = 17.5
    rso1_params['Cd'] = 2.2
    rso1_params['Cr'] = 1.3
    rso1_params['sph_deg'] = 8
    rso1_params['sph_ord'] = 8    
    rso1_params['central_bodies'] = ['Earth']
    rso1_params['bodies_to_create'] = bodies_to_create
    
    rso2_params = {}
    rso2_params['mass'] = 100.
    rso2_params['area'] = 1.
    rso2_params['Cd'] = 2.2
    rso2_params['Cr'] = 1.3
    rso2_params['sph_deg'] = 8
    rso2_params['sph_ord'] = 8    
    rso2_params['central_bodies'] = ['Earth']
    rso2_params['bodies_to_create'] = bodies_to_create
    
    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 1.
    
    # Expected result
    TCA_true = 764445600.0  
    rho_true = 0.
    
    # Interval times
    tf = t0 + 3600.
    trange = np.array([t0, tf])
    
    # RK4 test
    start = time.time()
    T_list, rho_list = compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                    int_params, bodies)
    

    
    print('')
    print('RK4 TCA unit test runtime [seconds]:', time.time() - start)
    print('RK4 TCA error [seconds]:', T_list[0]-TCA_true)
    print('RK4 miss distance error [m]:', rho_list[0]-rho_true)
    
    
    # RK78 test
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    
    
    start = time.time()
    T_list, rho_list = compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                   int_params, bodies)
    

    print('')
    print('RK78 TCA unit test runtime [seconds]:', time.time() - start)
    print('RK78 TCA error [seconds]:', T_list[0]-TCA_true)
    print('RK78 miss distance error [m]:', rho_list[0]-rho_true)
    
    
    
    return



if __name__ == '__main__':
    
    # unit_test_tca()
    
    cdm_dir = r'data\cdm'
    fname = os.path.join(cdm_dir, '2024-09-13--00--31698-36605.1726187583000.cdm')
    read_cdm_file(fname)