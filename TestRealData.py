import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import sys

metis_dir = r'C:\Users\sgehly\Documents\code\metis'
sys.path.append(metis_dir)

from utilities import tle_functions as tle

GME = 398600.4415*1e9  # m^3/s^2


def inertial2relative_ei(rc_vect, vc_vect, rd_vect, vd_vect, GM):
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



def tsx_tdx_test_case():

    # TSX/TDX Real Data - 
    # CCSDS_CDM_VERS               = 1.0
    # CREATION_DATE                = 2024-08-09T17:57:16.643578
    # ORIGINATOR                   = Privateer Space
    # MESSAGE_ID                   = Privateer_2024-08-11T19:29:25.234659_31698_36605
    # TCA                          = 2024-08-11T19:29:25.234659
    # MISS_DISTANCE                = 132.99859609349443 [m]
    # RELATIVE_SPEED               = 0.3797270373181049 [m/s]
    # RELATIVE_POSITION_R          = -44.61287375944862 [m]
    # RELATIVE_POSITION_T          = 90.68495579935542 [m]
    # RELATIVE_POSITION_N          = -86.45551948506773 [m]
    # RELATIVE_VELOCITY_R          = 0.3537287325842616 [m/s]
    # RELATIVE_VELOCITY_T          = 0.0520987770244437 [m/s]
    # RELATIVE_VELOCITY_N          = -0.1278840257705187 [m/s]
    # START_SCREEN_PERIOD          = 2024-08-08T12:00:00.000
    # STOP_SCREEN_PERIOD           = 2024-08-13T12:00:00.000
    # COLLISION_PROBABILITY        = 4.834309499782247E-03
    # COLLISION_PROBABILITY_METHOD = HALL-2021
    # COMMENT COLLISION_PROBABILITY        = 3.482041198720882E-03
    # COMMENT COLLISION_PROBABILITY_METHOD = ELROD-2019
    
    
    
    
    X                            = 2151.688944903789 
    Y                            = 1334.179725739959 
    Z                            = -6413.14999437017 
    X_DOT                        = -4.232774734985 
    Y_DOT                        = -5.736530531004 
    Z_DOT                        = -2.6141275475213 
    
    Xo_tsx = 1000.*np.reshape([X, Y, Z, X_DOT, Y_DOT, Z_DOT], (6,1))
    
    X                            = 2151.690980995972 
    Y                            = 1334.048466062339 
    Z                            = -6413.128654784359 
    X_DOT                        = -4.232595004923 
    Y_DOT                        = -5.736581489607 
    Z_DOT                        = -2.614458142361 
    
    Xo_tdx = 1000.*np.reshape([X, Y, Z, X_DOT, Y_DOT, Z_DOT], (6,1))
    
    de_vect, di_vect = inertial2relative_ei(Xo_tsx[0:3], Xo_tsx[3:6], Xo_tdx[0:3], Xo_tdx[3:6], GME)
    
    angle = np.arccos(np.dot(de_vect.flatten(), di_vect.flatten())/(np.linalg.norm(de_vect)*np.linalg.norm(di_vect)))


    print(angle*180/np.pi)
    
    return


def tsx_tdx_tle_analysis():
    
    # Retrieve all TLEs for 3 months, propagate to nearest day
    # Need to propagate to ensure both TSX and TDX at same time
    obj_id_list = [31698, 36605]
    UTC0 = datetime(2024, 7, 1)
    UTC_list = [UTC0 + timedelta(days=ii/10.) for ii in range(1000)]
    
    
    output_state = tle.propagate_TLE(obj_id_list, UTC_list)
    
    # Loop over times
    de_plot = np.zeros((2,len(UTC_list)))
    di_plot = np.zeros((2,len(UTC_list)))
    angle_plot = np.zeros((len(UTC_list),))
    for ii in range(len(UTC_list)):
        
        # Retrieve GCRF states and convert to meters
        r1_GCRF = output_state[obj_id_list[0]]['r_GCRF'][ii]*1000.
        v1_GCRF = output_state[obj_id_list[0]]['v_GCRF'][ii]*1000.
        r2_GCRF = output_state[obj_id_list[1]]['r_GCRF'][ii]*1000.
        v2_GCRF = output_state[obj_id_list[1]]['v_GCRF'][ii]*1000.
        
        # Compute e/i vectors
        de_vect, di_vect = inertial2relative_ei(r1_GCRF, v1_GCRF, r2_GCRF, v2_GCRF, GME)
        
        # Compute angle separation
        angle = np.arccos(np.dot(de_vect.flatten(), di_vect.flatten())/(np.linalg.norm(de_vect)*np.linalg.norm(di_vect)))
        
        # Store data for plot
        de_plot[:,ii] = de_vect.flatten()[0:2]
        di_plot[:,ii] = di_vect.flatten()[0:2]
        angle_plot[ii] = angle*180/np.pi
    
    
    # Plot over time
    plt.figure()
    plt.plot(de_plot[0,:], de_plot[1,:], 'k.')
    # plt.xlim([-6e-5, 6e-5])
    # plt.ylim([-6e-5, 6e-5])    
    
    # plt.axis('equal')
    plt.grid()
    plt.xlabel('de [x]')
    plt.ylabel('de [y]')
    
    plt.figure()
    plt.plot(di_plot[0,:], di_plot[1,:], 'k.')
    # plt.xlim([-6e-4, 6e-4])
    # plt.ylim([-6e-4, 6e-4])
        
    plt.grid()
    plt.xlabel('di [x]')
    plt.ylabel('di [y]')
    
    
    plt.figure()
    plt.plot(UTC_list, angle_plot, 'ko-')
    plt.ylabel('Separation Angle [deg]')
    
    
    plt.show()
    
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    tsx_tdx_tle_analysis()
    
    
    