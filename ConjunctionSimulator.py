import numpy as np


from tudatpy.astro import time_conversion

import ConjunctionUtilities as cutil
import TudatPropagator as prop

###############################################################################
# Initialize Conjuction Data
###############################################################################

def retrieve_conjunction_data_at_tca(cdm_file=''):
    
    # If a CDM is provided, parse and retrieve object data
    if len(cdm_file) > 0:
        cdm_data = cutil.read_cdm_file(cdm_file)
        TCA_UTC = cdm_data['TCA_UTC']
        X1 = cdm_data['obj1']['X']
        X2 = cdm_data['obj2']['X']
        
        # Convert TCA to seconds from epoch in TDB
        tudat_datetime_UTC = time_conversion.datetime_to_tudat(TCA_UTC)
        time_scale_converter = time_conversion.default_time_scale_converter()
        TCA_epoch_utc = tudat_datetime_UTC.epoch()
        TCA_epoch_tdb = time_scale_converter.convert_time(
                        input_scale = time_conversion.utc_scale,
                        output_scale = time_conversion.tdb_scale,
                        input_value = TCA_epoch_utc)
        
    
    return TCA_epoch_tdb, X1, X2


def initialize_propagator(tudat_integrator):
    '''
    This function sets up default settings for the tudat integrator and 
    propagator. 
    
    Parameters
    ------
    tudat_integrator : string
        choose from 'rk4', 'rk78', 'rkdp87'
    
    '''
    
    # Default settings
    bodies = prop.tudat_initialize_bodies(['Earth', 'Sun', 'Moon'])
    state_params = {}
    state_params['central_bodies'] = ['Earth']
    state_params['mass'] = 100.
    state_params['area'] = 1.
    state_params['Cd'] = 2.2
    state_params['Cr'] = 1.3
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    
    # Choose integrator
    
    # RK4
    if tudat_integrator == 'rk4':        
        int_params = {}
        int_params['tudat_integrator'] = 'rk4'
        int_params['step'] = 1.
    
    # RK78 Variable
    elif tudat_integrator == 'rkf78': 
        int_params = {}
        int_params['tudat_integrator'] = 'rkf78'
        int_params['step'] = 10.
        int_params['max_step'] = 1000.
        int_params['min_step'] = 1e-3
        int_params['rtol'] = 1e-12
        int_params['atol'] = 1e-12
    
    # DP87 Variable
    elif tudat_integrator == 'rkdp87':
        int_params = {}
        int_params['tudat_integrator'] = 'rkdp87'
        int_params['step'] = 10.
        int_params['max_step'] = 1000.
        int_params['min_step'] = 1e-3
        int_params['rtol'] = 1e-12
        int_params['atol'] = 1e-12
        
    else:
        print('error: invalid input')
        print(tudat_integrator)
    
    
    return state_params, int_params