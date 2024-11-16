import numpy as np
import math
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt

import EstimationUtilities as EstUtil
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil
import ConjunctionSimulator as ConjSim

# Earth parameters
GME = 398600.4415*1e9  # m^3/s^2
J2E = 1.082626683e-3
Re = 6378137.0

###############################################################################
# Basic I/O
###############################################################################

def unit_test_io():

    # Set directory for assignment data
    assignment_data_directory = 'data/unit_test'
    
    # Load RSO catalog file
    rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
    rso_dict = ConjUtil.read_catalog_file(rso_file)
    obj_id = 91260
    
    print('')
    print('RSO File contains', len(rso_dict), 'objects')
    print('rso_dict contains the following objects:')
    print(list(rso_dict.keys()))
    
    print('')
    print('Data for each object are stored as a dictionary and can be retrieved using the object ID')
    print('The following fields are available')
    print(list(rso_dict[obj_id].keys()))
    
    
    # Load truth data for estimation case
    truth_file = os.path.join(assignment_data_directory, 'test_sparse_truth_grav.pkl')
    t_truth, X_truth, state_params = EstUtil.read_truth_file(truth_file)
    
    print('')
    print('The truth data file contains the following:')
    print('t_truth shape', t_truth.shape)
    print('X_truth shape', X_truth.shape)
    print('X at t0', X_truth[0,:])
    print('state_params')
    print(state_params)
    
    
    # Load measurement data for estimation case
    meas_file = os.path.join(assignment_data_directory, 'test_sparse_radar_meas.pkl')
    state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)
    
    print('')
    print('The measurement file contains an augmented state params dictionary that' 
          'includes the initial state and covariance for the filter to use')
    print(state_params)
    print('meas_dict fields:')
    print(list(meas_dict.keys()))
    print('tk_list length', len(meas_dict['tk_list']))
    print('Y at t0', meas_dict['Yk_list'][0])
    print('sensor_params fields:')
    print(list(sensor_params.keys()))
    
    return


###############################################################################
# Compute TCA Example

# This code performs a unit test of the compute_TCA function. The object
# parameters are such that a collision is expected 30 minutes after the
# initial epoch (zero miss distance).

# The TCA function is run twice, using a fixed step RK4 and variable step
# RKF78 to compare the results.
#
###############################################################################

def unit_test_tca():

    print('\nBegin TCA Test')
    
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
    T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
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
    T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                            int_params, bodies)
    
    
    print('')
    print('RK78 TCA unit test runtime [seconds]:', time.time() - start)
    print('RK78 TCA error [seconds]:', T_list[0]-TCA_true)
    print('RK78 miss distance error [m]:', rho_list[0]-rho_true)

    return

###############################################################################
# Run Unscented Kalman Filter Example
#
# This code performs a unit test of the UKF function. It does not compute
# errors or generate plots, just ensures that everything is present to run
# the filter without error.
#
###############################################################################

# def unit_test_ukf():

#     print('\nBegin UKF Test')
    
#     # Load measurement data
#     meas_file = os.path.join(assignment_data_directory, 'test_sparse_radar_meas.pkl')
#     state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)
    
    
#     # For the assignment, you should always set the filter to zero out Drag and SRP
#     # In the case truth/measurements are generated with a higher fidelity force 
#     # model, the SNC parameters of the filter should be tuned to account for this
    
#     # Set the Drag and SRP coefficient to zero for the filter acceleration model
#     state_params['Cd'] = 0.
#     state_params['Cr'] = 0.
    
#     # Setup filter parameters such as process noise
#     Qeci = 1e-12*np.diag([1., 1., 1.])
#     Qric = 1e-12*np.diag([1., 1., 1.])
    
#     filter_params = {}
#     filter_params['Qeci'] = Qeci
#     filter_params['Qric'] = Qric
#     filter_params['alpha'] = 1.
#     filter_params['gap_seconds'] = 600.
    
#     # Choose integration parameters
#     # int_params = {}
#     # int_params['tudat_integrator'] = 'rk4'
#     # int_params['step'] = 10.
    
#     int_params = {}
#     int_params['tudat_integrator'] = 'rkf78'
#     int_params['step'] = 10.
#     int_params['max_step'] = 1000.
#     int_params['min_step'] = 1e-3
#     int_params['rtol'] = 1e-12
#     int_params['atol'] = 1e-12
    
#     # Initialize tudat bodies
#     bodies_to_create = ['Earth', 'Sun', 'Moon']
#     bodies = prop.tudat_initialize_bodies(bodies_to_create)
    
#     # Run filter
#     filter_output = EstUtil.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)

#     return


###############################################################################
# Propagation Tests
#
# This test evaluates the propagator/integrator settings required to reliably
# propagate a state vector backward and forward yielding a small numerical 
# error.
#
###############################################################################

def unit_test_forward_backprop():
    
    # Initial state vector
    cdm_dir = r'data\cdm'
    cdm_file = os.path.join(cdm_dir, '2024-09-13--00--31698-36605.1726187583000.cdm')
    TCA_epoch_tdb, X1, X2 = ConjUtil.retrieve_conjunction_data_at_tca(cdm_file)
    
    # Default integrator/propagator settings
    state_params, int_params, bodies = prop.initialize_propagator('rkdp87')
    
    # Update for backprop
    step = 30.
    tol = np.inf
    int_params['step'] = -step
    int_params['max_step'] = -step
    int_params['min_step'] = -step
    int_params['atol'] = tol
    int_params['rtol'] = tol
    
    t0 = TCA_epoch_tdb - 7*86400.
    tvec = np.array([TCA_epoch_tdb, t0])
    
    tback, Xback = prop.propagate_orbit(X1, tvec, state_params, int_params, bodies)
    
    # Update for forward propagation
    tvec = np.array([t0, TCA_epoch_tdb])
    Xo = Xback[0,:]
    
    int_params['step'] = step
    int_params['max_step'] = step
    int_params['min_step'] = step
    
    tfor, Xfor = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
    
    
    Xerr = Xfor - Xback
    pos_err = [np.linalg.norm(Xerr[ii,0:3]) for ii in range(len(tfor))]
    vel_err = [np.linalg.norm(Xerr[ii,3:6]) for ii in range(len(tfor))]
    thrs = (tfor - tfor[0])/3600.
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.semilogy(thrs, pos_err, 'k.')
    plt.ylabel('Pos Err [m]')
    plt.subplot(2,1,2)
    plt.semilogy(thrs, vel_err, 'k.')
    plt.ylabel('Vel Err [m]')
    plt.xlabel('Time [hours]')
    
    plt.show()
    
    return


def unit_test_initialize_covar():
    
    # Initial state vector
    cdm_dir = r'data\cdm'
    cdm_file = os.path.join(cdm_dir, '2024-09-13--00--31698-36605.1726187583000.cdm')
    TCA_epoch_tdb, X1, X2 = ConjUtil.retrieve_conjunction_data_at_tca(cdm_file)
    
    # Default integrator/propagator settings
    state_params, int_params, bodies = prop.initialize_propagator('rkdp87')
    
    # Update for backprop
    step = 30.
    tol = np.inf
    int_params['step'] = -step
    int_params['max_step'] = -step
    int_params['min_step'] = -step
    int_params['atol'] = tol
    int_params['rtol'] = tol
    
    t0 = TCA_epoch_tdb - 7*86400.
    tvec = np.array([TCA_epoch_tdb, t0])
    
    tback, Xback = prop.propagate_orbit(X1, tvec, state_params, int_params, bodies)
    
    # Retrieve for covariance setup
    tvec = np.array([t0, TCA_epoch_tdb])
    Xo = Xback[0,:].reshape(6,1)
    
    dum, Po = ConjUtil.initialize_covar(t0, Xo, thrs=3., interval=300., noise=1.)
    
    print('Xo', Xo)
    print('dum', dum)
    print('err', dum - Xo)
    print(Po)
    
    
    return


def damico_test_case_params():    
    
    # Constants
    Cd = 2.3                # unitless
    Cr = 1.3                # unitless
    A = 3.2                 # m^2
    m = 1238.                # kg
    
    # Chief Orbit Parameters
    a = 6892.945*1000.                  # m
    e = 1e-8
    i = 97.*np.pi/180.                  # rad
    RAAN = 0.                           # rad
    w = 270.*np.pi/180.                 # rad
    M = np.pi/2.                        # rad
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
    M2 = M + np.pi                      # rad
    
    osc_elem1 = [a, e, i, RAAN, w, M]
    osc_elem2 = [a2, e2, i2, RAAN2, w2, M2]
    
    X1 = ConjUtil.kep2cart(osc_elem1, GME)
    X2 = ConjUtil.kep2cart(osc_elem2, GME)
    
    return X1, X2


def unit_test_relative_ei_vector():
    
    damico = False
    
    if damico:
        
        # D'Amico TSX/TDX setup
        t0 = 0.
        X1_0, X2_0 = damico_test_case_params()
        
    else:
    
        # Initial state vector
        cdm_dir = r'data\cdm'
        cdm_file = os.path.join(cdm_dir, '2024-09-13--00--31698-36605.1726187583000.cdm')
        # cdm_file = os.path.join(cdm_dir, '2024-09-13--00--31698-36605.1726187776000.cdm')
        # cdm_file = os.path.join(cdm_dir, '2024-09-13--10--31698-36605.1726221677000.cdm')
        TCA_epoch_tdb, X1, X2 = ConjUtil.retrieve_conjunction_data_at_tca(cdm_file)
        
        # Default integrator/propagator settings
        state_params, int_params, bodies = prop.initialize_propagator('rkdp87')
        
        # Update for backprop
        step = 30.
        tol = np.inf
        int_params['step'] = -step
        int_params['max_step'] = -step
        int_params['min_step'] = -step
        int_params['atol'] = tol
        int_params['rtol'] = tol
        
        t0 = TCA_epoch_tdb - 1*86400.
        tvec = np.array([TCA_epoch_tdb, t0])
        
        tback1, Xback1 = prop.propagate_orbit(X1, tvec, state_params, int_params, bodies)
        tback2, Xback2 = prop.propagate_orbit(X2, tvec, state_params, int_params, bodies)
        
        # Inertial state vectors at t0
        X1_0 = Xback1[0,:].reshape(6,1)
        X2_0 = Xback2[0,:].reshape(6,1)
    
    
    
    
    # Compute initial orbits and separation
    d2 = np.linalg.norm(X1_0[0:3] - X2_0[0:3])    
    elem1 = ConjUtil.cart2kep(X1_0)
    elem2 = ConjUtil.cart2kep(X2_0)
    
    print('elem1', elem1)
    print('elem2', elem2)

    
    # Generate the initial covariances
    dum, P1_0 = ConjUtil.initialize_covar(t0, X1_0, thrs=3., interval=300., noise=10.)
    dum, P2_0 = ConjUtil.initialize_covar(t0, X2_0, thrs=3., interval=300., noise=10.)
    
    # Compute relative e/i vectors and separation angle for mean states
    angle, de_vect_of, di_vect_of = ConjUtil.inertial2relative_ei(X1_0, X2_0, GM=GME)
    angle_mean, de_vect_mean, di_vect_mean = ConjUtil.inertial2meanrelative_ei(X1_0, X2_0, GM=GME)
    
    print('')
    print('X1_0', X1_0)
    print('X2_0', X2_0)
    print('P1 std', np.sqrt(np.diag(P1_0)))
    print('P2 std', np.sqrt(np.diag(P2_0)))
    print('separation distance [m]', d2)
    print('e/i angle [deg]', angle*180/np.pi)
    print('de_vect_of', de_vect_of)
    print('di_vect_of', di_vect_of)
    
    # Generate samples and compute e/i vectors and separation angle
    N = 10000
    samples1 = np.random.default_rng().multivariate_normal(X1_0.flatten(), P1_0, N)
    samples2 = np.random.default_rng().multivariate_normal(X2_0.flatten(), P2_0, N)
    dex_list = []
    dey_list = []
    dix_list = []
    diy_list = []
    angle_list = []
    mean_dex_list = []
    mean_dey_list = []
    mean_dix_list = []
    mean_diy_list = []
    mean_angle_list = []
    for ii in range(N):
        s1_ii = samples1[ii,:].reshape(6,1)
        s2_ii = samples2[ii,:].reshape(6,1)
        
        angle_ii, de_vect_ii, di_vect_ii = ConjUtil.inertial2relative_ei(s1_ii, s2_ii, GM=GME)
        angle_mean_ii, de_vect_mean_ii, di_vect_mean_ii = ConjUtil.inertial2meanrelative_ei(s1_ii, s2_ii, GM=GME)
        
        # print(s1_ii)
        # print(s2_ii)
        # print(angle_ii*180/np.pi)
        # print(de_vect_ii)
        # print(di_vect_ii)
        
        # mistake
    
        dex_list.append(float(de_vect_ii[0,0]))
        dey_list.append(float(de_vect_ii[1,0]))
        dix_list.append(float(di_vect_ii[0,0]))
        diy_list.append(float(di_vect_ii[1,0]))
        angle_list.append(angle_ii*180./np.pi)
        
        mean_dex_list.append(float(de_vect_mean_ii[0,0]))
        mean_dey_list.append(float(de_vect_mean_ii[1,0]))
        mean_dix_list.append(float(di_vect_mean_ii[0,0]))
        mean_diy_list.append(float(di_vect_mean_ii[1,0]))
        mean_angle_list.append(angle_mean_ii*180./np.pi)
        
        
    print('')
    print('osculating elements results')
    print('mean angle [deg]', np.mean(angle_list))
    print('angle std [deg]', np.std(angle_list))
    
    print('')
    print('mean elements results')
    print('mean angle [deg]', np.mean(mean_angle_list))
    print('angle std [deg]', np.std(mean_angle_list))
    
    
    dex_lim = max([abs(dex) for dex in dex_list])
    dey_lim = max([abs(dey) for dey in dey_list])
    dix_lim = max([abs(dix) for dix in dix_list])
    diy_lim = max([abs(diy) for diy in diy_list])
    
    de_lim = max(dex_lim, dey_lim)*2
    di_lim = max(dix_lim, diy_lim)*2
    
    # Generate plots
    plt.figure()
    plt.plot(dex_list, dey_list, 'k.')
    plt.plot([0, de_vect_of[0,0]], [0, de_vect_of[1,0]], 'r')
    plt.xlim([-de_lim, de_lim])
    plt.ylim([-de_lim, de_lim])  
    plt.xlabel('de[x]')
    plt.ylabel('de[y]')
    plt.title('Osculating Elements')
    plt.grid()
    
    plt.figure()
    plt.plot(dix_list, diy_list, 'k.')
    plt.plot([0, di_vect_of[0,0]], [0, di_vect_of[1,0]], 'r')
    plt.xlim([-di_lim, di_lim])
    plt.ylim([-di_lim, di_lim])  
    plt.xlabel('di[x]')
    plt.ylabel('di[y]')
    plt.title('Osculating Elements')
    plt.grid()
    
    plt.figure()
    plt.hist(angle_list)
    plt.xlabel('Separation Angle [deg]')
    plt.ylabel('Occurrence')
    plt.title('Osculating Elements')
    
    plt.figure()
    plt.plot(mean_dex_list, mean_dey_list, 'k.')
    plt.plot([0, de_vect_of[0,0]], [0, de_vect_of[1,0]], 'r')
    plt.xlim([-de_lim, de_lim])
    plt.ylim([-de_lim, de_lim])  
    plt.xlabel('de[x]')
    plt.ylabel('de[y]')
    plt.title('Mean Elements')
    plt.grid()
    
    plt.figure()
    plt.plot(mean_dix_list, mean_diy_list, 'k.')
    plt.plot([0, di_vect_of[0,0]], [0, di_vect_of[1,0]], 'r')
    plt.xlim([-di_lim, di_lim])
    plt.ylim([-di_lim, di_lim])  
    plt.xlabel('di[x]')
    plt.ylabel('di[y]')
    plt.title('Mean Elements')
    plt.grid()
    
    plt.figure()
    plt.hist(mean_angle_list)
    plt.xlabel('Separation Angle [deg]')
    plt.ylabel('Occurrence')
    plt.title('Mean Elements')
    
    
    
    plt.show()
    
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    # unit_test_forward_backprop()
    
    # unit_test_initialize_covar()

    unit_test_relative_ei_vector()
    



