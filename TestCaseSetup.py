import numpy as np
import math
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import json
import pickle
import time

# Load tudatpy modules
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array


import ConjunctionUtilities as conj
import TudatPropagator as prop


# Load spice kernels
# spice.load_standard_kernels()

# Load spice kernels
# spice_interface.load_standard_kernels()


###############################################################################
# Tudat Functions
###############################################################################

def backprop_demo():    

    # Set simulation start and end epochs
    simulation_start_epoch = 0.0
    simulation_end_epoch = -2*constants.JULIAN_DAY
    
    # Initial state
    semi_major_axis=7500.0e3
    eccentricity=0.1
    inclination=np.deg2rad(85.3)
    argument_of_periapsis=np.deg2rad(235.7)
    longitude_of_ascending_node=np.deg2rad(23.4)
    true_anomaly=np.deg2rad(139.87)
    
    kep = np.array([7500.,
                   0.1,
                   85.3,
                   23.4,
                   235.7,
                   139.87])
    


    # Create default body settings for "Earth"
    bodies_to_create = ["Earth"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    # Create system of bodies (in this case only Earth)
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Add vehicle object to system of bodies
    bodies.create_empty_body("Delfi-C3")

    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi-C3"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]

    # Define accelerations acting on Delfi-C3
    acceleration_settings_delfi_c3 = dict(
        Earth=[propagation_setup.acceleration.point_mass_gravity()]
    )

    acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Set initial conditions for the satellite that will be
    # propagated in this simulation. The initial conditions are given in
    # Keplerian elements and later on converted to Cartesian elements
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=7500.0e3,
        eccentricity=0.1,
        inclination=np.deg2rad(85.3),
        argument_of_periapsis=np.deg2rad(235.7),
        longitude_of_ascending_node=np.deg2rad(23.4),
        true_anomaly=np.deg2rad(139.87),
    )
    
    print(earth_gravitational_parameter)


    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    fixed_step_size = -1.
    integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)
    # integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    #                         time_step = fixed_step_size,
    #                         coefficient_set = propagation_setup.integrator.rk_4 )


    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings
    )

    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)
    

    print(
        f"""
    Single Earth-Orbiting Satellite Example.
    The initial position vector of Delfi-C3 is [km]: \n{
        states[simulation_start_epoch][:3] / 1E3}
    The initial velocity vector of Delfi-C3 is [km/s]: \n{
        states[simulation_start_epoch][3:] / 1E3}
    \nAfter {simulation_end_epoch} seconds the position vector of Delfi-C3 is [km]: \n{
        states[simulation_end_epoch][:3] / 1E3}
    And the velocity vector of Delfi-C3 is [km/s]: \n{
        states[simulation_end_epoch][3:] / 1E3}
        """
    )
    
    

    # Define a 3D figure using pyplot
    fig = plt.figure(figsize=(6,6), dpi=125)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Delfi-C3 trajectory around Earth')

    # Plot the positional state history
    ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3], label=bodies_to_propagate[0], linestyle='-.')
    ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')

    # Add the legend and labels, then show the plot
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()
    
    
    return


def backprop_perturbed_orbit():
    
    # Set simulation start and end epochs
    simulation_start_epoch = 0.0
    simulation_end_epoch = -2*constants.JULIAN_DAY
    
    # Object parameters
    reference_area = 1.
    mass = 100.
    radiation_pressure_coefficient = 1.3
    drag_coefficient = 2.2

    # Create default body settings for "Earth"
    bodies_to_create = ["Earth", "Sun", "Moon"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    # Create system of bodies (in this case only Earth)
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Add vehicle object to system of bodies
    bodies.create_empty_body("Delfi-C3")
    bodies.get("Delfi-C3").mass = mass

    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi-C3"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]
    
    # Create radiation pressure settings, and add to vehicle
    # Radiation pressure is set up assuming tudatpy version >= 0.8
    # Code for earlier tudatpy is commented out below
    
    
    occulting_bodies_dict = dict()
    occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
    
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area, radiation_pressure_coefficient, occulting_bodies_dict )
    
    # Radiation pressure setup for tudatpy < 0.8
    # occulting_bodies = ["Earth"]
    # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    #     "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
    # )               

    # body_settings.get( "Satellite" ).radiation_pressure_target_settings = radiation_pressure_settings
    
    environment_setup.add_radiation_pressure_target_model(
        bodies, "Delfi-C3", radiation_pressure_settings)
    
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0, 0]
    )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "Delfi-C3", aero_coefficient_settings)

    # Define accelerations acting on Delfi-C3
    acceleration_settings_delfi_c3 = dict(
        Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(8,8),
                propagation_setup.acceleration.aerodynamic()],
        
        Sun=[propagation_setup.acceleration.point_mass_gravity(),
              propagation_setup.acceleration.radiation_pressure()],
        
        Moon=[propagation_setup.acceleration.point_mass_gravity()]
        
        # Earth=[propagation_setup.acceleration.point_mass_gravity()]
    )

    acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Set initial conditions for the satellite that will be
    # propagated in this simulation. The initial conditions are given in
    # Keplerian elements and later on converted to Cartesian elements
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=7000.0e3,
        eccentricity=0.01,
        inclination=np.deg2rad(85.3),
        argument_of_periapsis=np.deg2rad(235.7),
        longitude_of_ascending_node=np.deg2rad(23.4),
        true_anomaly=np.deg2rad(139.87),
    )

    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    fixed_step_size = -1.
    integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)
    # integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    #                         time_step = fixed_step_size,
    #                         coefficient_set = propagation_setup.integrator.rk_4 )


    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings
    )

    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)
    
    
    
    print(states_array.shape)
    print(states_array[0,:])
    print(states_array[-1,:])
    
    state_t_minus_48hrs = states_array[0,1:]
    state_t0 = states_array[-1,1:]
    
    # Run forward
    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_start_epoch)

    # Create numerical integrator settings
    fixed_step_size = 1.
    integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)
    # integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    #                         time_step = fixed_step_size,
    #                         coefficient_set = propagation_setup.integrator.rk_4 )


    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        state_t_minus_48hrs,
        simulation_end_epoch,
        integrator_settings,
        termination_settings
    )

    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Extract the resulting state history and convert it to an ndarray
    states2 = dynamics_simulator.state_history
    states_array2 = result2array(states2)
    
    
    print(states_array2.shape)
    print(states_array2[0,:])
    print(states_array2[-1,:])
    
    
    
    return



###############################################################################
# Create a simulated conjunction
###############################################################################


def build_and_test_conjuction(rso_file):
    
    
    # Load RSO dict
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    ###########################################################################
    # Starlink case
    ###########################################################################
    asset_id = 45551
    
    # Hit 1
    impactor_id = 90001
    impactor_mass = 100.
    impactor_area = 1.
    impactor_Cd = 2.2
    impactor_Cr = 1.3
    TCA_hrs = 36.
    rho_ric = np.array([0., 0., 0.]).reshape(3,1)
    drho_ric = np.array([100., -15300., 100.]).reshape(3,1)
    
    rso_dict = create_conjunction(rso_dict, asset_id, impactor_id, 
                                  impactor_mass, impactor_area, impactor_Cd,
                                  impactor_Cr, TCA_hrs, rho_ric, drho_ric)
    
    
    
    # Compute TCA to confirm
    X1 = rso_dict[asset_id]['state']    
    X2 = rso_dict[impactor_id]['state']
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create) 

    rso1_params = {}
    rso1_params['mass'] = rso_dict[asset_id]['mass']
    rso1_params['area'] = rso_dict[asset_id]['area']
    rso1_params['Cd'] = rso_dict[asset_id]['Cd']
    rso1_params['Cr'] = rso_dict[asset_id]['Cr']
    rso1_params['sph_deg'] = 8
    rso1_params['sph_ord'] = 8    
    rso1_params['central_bodies'] = ['Earth']
    rso1_params['bodies_to_create'] = bodies_to_create
    
    rso2_params = {}
    rso2_params['mass'] = rso_dict[impactor_id]['mass']
    rso2_params['area'] = rso_dict[impactor_id]['area']
    rso2_params['Cd'] = rso_dict[impactor_id]['Cd']
    rso2_params['Cr'] = rso_dict[impactor_id]['Cr']
    rso2_params['sph_deg'] = 8
    rso2_params['sph_ord'] = 8    
    rso2_params['central_bodies'] = ['Earth']
    rso2_params['bodies_to_create'] = bodies_to_create
    
    int_params = {}
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    
    t0 = (rso_dict[asset_id]['UTC'] - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    tf = t0 + TCA_hrs*3600. + 12.*3600
    TCA_true = t0 + TCA_hrs*3600.
    trange = np.array([t0, tf])
    
    T_list, rho_list = conj.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                        int_params, bodies)
    
    
    print('')
    print('TCA error [seconds]:', T_list[0] - TCA_true)
    print('miss distance error [m]:', rho_list[0] - np.linalg.norm(rho_ric))
    
    
    
    return  
    

def create_conjunction(rso_dict, asset_id, impactor_id, impactor_mass,
                       impactor_area, impactor_Cd, impactor_Cr, TCA_hrs,
                       rho_ric, drho_ric):
    
    # Asset data
    Xo_true = rso_dict[asset_id]['state']
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = rso_dict[asset_id]['mass']
    state_params['area'] = rso_dict[asset_id]['area']
    state_params['Cd'] = rso_dict[asset_id]['Cd']
    state_params['Cr'] = rso_dict[asset_id]['Cr']
    state_params['sph_deg'] = 8
    state_params['sph_ord'] = 8    
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 1.
    
    # int_params = {}
    # int_params['tudat_integrator'] = 'rkf78'
    # int_params['step'] = 10.
    # int_params['max_step'] = 1000.
    # int_params['min_step'] = 1e-3
    # int_params['rtol'] = 1e-12
    # int_params['atol'] = 1e-12
    
    # Integration times
    t0 = (rso_dict[asset_id]['UTC'] - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    tf = t0 + TCA_hrs*3600.
    tvec = np.array([t0, tf])
    
    print('prop')
    tout, Xout = prop.propagate_orbit(Xo_true, tvec, state_params, int_params, bodies)
    Xf_true = Xout[-1,:].reshape(6,1)
    
    
    # Compute impactor truth state
    rc_vect = Xf_true[0:3].reshape(3,1)
    vc_vect = Xf_true[3:6].reshape(3,1)
        
    rho_eci = conj.ric2eci(rc_vect, vc_vect, rho_ric)
    drho_eci = conj.ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric)
    r_eci = rc_vect + rho_eci
    v_eci = vc_vect + drho_eci
    
    Xf_imp_true = np.concatenate((r_eci, v_eci), axis=0)    
    kep_imp = cart2kep(Xf_imp_true)
    
    rp = float(kep_imp[0,0]*(1-kep_imp[1,0]))
    ra = float(kep_imp[0,0]*(1+kep_imp[1,0]))
    
    if rp < 6578000:
        print('Error: rp < 200 km')
        return rso_dict
    
    print('')
    # print('Xf_true', Xf_true)
    # print('Xf_imp_true', Xf_imp_true)
    # print('kep_imp', kep_imp)
    print('rp', rp)
    print('ra', ra)
    print('miss distance', np.linalg.norm(Xf_true[0:3] - Xf_imp_true[0:3]))
    print('')
    
            
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = impactor_mass
    state_params['area'] = impactor_area
    state_params['Cd'] = impactor_Cd
    state_params['Cr'] = impactor_Cr
    state_params['sph_deg'] = 8
    state_params['sph_ord'] = 8    
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = -1.
    
    # int_params['tudat_integrator'] = 'rkf78'
    # int_params['step'] = 10.
    # int_params['max_step'] = 1000.
    # int_params['min_step'] = 1e-3
    # int_params['rtol'] = 1e-12
    # int_params['atol'] = 1e-12
    
    # Integration times
    tvec = np.array([tf, t0])
    
    print('backprop impactor')
    tout2, Xout2 = prop.propagate_orbit(Xf_imp_true, tvec, state_params, int_params, bodies)
    
    Xo_imp_true = Xout2[0,:].reshape(6,1)
    Xf_imp_check = Xout2[-1,:].reshape(6,1)
    
    print('')
    print('Xo true (primary)\n', Xo_true)
    print('Xf true (primary)\n', Xf_true)
    print('Xf true (impactor)\n', Xf_imp_check)
    print('Xo true (impactor)\n', Xo_imp_true)
    
    
    # Add to output
    rso_dict[impactor_id] = {}
    rso_dict[impactor_id]['UTC'] = rso_dict[asset_id]['UTC']
    rso_dict[impactor_id]['state'] = Xo_imp_true
    rso_dict[impactor_id]['mass'] = impactor_mass
    rso_dict[impactor_id]['area'] = impactor_area
    rso_dict[impactor_id]['Cd'] = impactor_Cd
    rso_dict[impactor_id]['Cr'] = impactor_Cr
    
    
    return rso_dict


###############################################################################
# Utility functions
###############################################################################


def cart2kep(cart, GM=3.986004415e14):
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
      Semi-Major Axis             [m]
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





def compute_euclidean_distance(r_A, r_B):
    
    d = np.linalg.norm(r_A - r_B)
    
    return d


def compute_mahalanobis_distance(r_A, r_B, P_A, P_B):    
    
    Psum = P_A + P_B
    invP = np.linalg.inv(Psum)
    diff = r_A - r_B
    M = float(np.sqrt(np.dot(diff.T, np.dot(invP, diff)))[0,0])
    
    return M





if __name__ == '__main__':
    
    
    plt.close('all')
    
    # backprop_demo()
    
    # backprop_perturbed_orbit()
    

    rso_file = os.path.join('data', 'unit_test', 'asset_catalog_truth.pkl')
    build_and_test_conjuction(rso_file)
    
















