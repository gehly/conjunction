import numpy as np




def test_damico_tsx_tdx():
    
    # Constants
    Re = 6378.1363      # km
    GM = 3.986004e5     # km^3/s^2
    J2 = 0.001082       # [unitless]
    Cd = 0.             # [unitless]
    
    # Chief Orbit Parameters
    a = 6892.945        # km
    e = 1e-12
    i = 97.             # deg
    p = a*(1-e**2)      # km
    w0 = 270.           # deg
    
    # dw = .75*J2*sqrt(mu/a^3)*(Re/p)^2*(5*cosd(i)^2 - 1);    %rad/s
    # dw = dw*180/pi; %deg/sec
    # dw = dw*86400;  %deg/day
    # %Calculate orbit period
    # P = 2*pi*sqrt(a^3/mu);      %sec
    # P = P/86400;    %days
    
    
    
    
    
    return