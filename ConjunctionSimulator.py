import numpy as np
from datetime import datetime, timedelta
import pickle

from tudatpy.astro import time_conversion

import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop



###############################################################################
# Initialize Conjuction Data
###############################################################################


def retrieve_and_save_tle_data(TDB0, obj_id_list, obj_params_file, true_catalog_file):
    '''
    This function retrieves TLE data for the provided object list and a time 
    window of several days before and after the specified initial TDB datetime.
    All objects are then propagated to TDB0 and data is saved to a file.

    Parameters
    ------
    TDB0 : datetime object
        desired epoch for catalog

    obj_id_list : list
        NORAD IDs of objects

    obj_params : dictionary
        propagator parameters, indexed by object ID as needed
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]

    true_catalog_file : string
        path and filename for output data file    

    '''

    # Retrieve TLE data and convert to ECI. For the purpose of data 
    # retrieval, UTC and TDB time don't need to be converted
    UTC_list = [TDB0 - timedelta(days=5.), TDB0 + timedelta(days=5.)]
    tle_dict = ConjUtil.get_spacetrack_tle_data(obj_id_list, UTC_list)
    state_dict = ConjUtil.tle2eci(tle_dict)

    # Propagate all state vectors to common epoch
    TDB_epoch = time_conversion.datetime_to_tudat(TDB0).epoch()
    obj_params = ConjUtil.read_object_params_file(obj_params_file)
    bodies_to_create = obj_params['bodies_to_create']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)
    true_catalog = ConjUtil.propagate_all_states(TDB_epoch, state_dict, obj_params, bodies=bodies)

    # Save file
    pklFile = open(true_catalog_file, 'wb')
    pickle.dump([true_catalog], pklFile, -1)
    pklFile.close()

    return


def build_catalog_from_tle(UTC0, obj_id_list):
    '''
    This function initializes a catalog of objects with conjuntions by retrieving
    TLE data for the primaries and user provided miss distance information for 
    the impactor.
    '''


    return


