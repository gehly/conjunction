clear
close all

%Constants
J2 = 0.001082;      %[unitless]
Cd = 2.3;             %[unitless]
A = 3.2;    %m^2
m = 1238;   %kg
A_m = (A/1e6)/m;    %km^2/kg

%Given data
%Values from gravity model GGM02C.sha
Re = 6.378136300000000e+003;    %km
mu = 3.986004415000000e+005;    %km^3/s^2

%Other given data
theta0 = 14*pi/180;     % initial earth angle, rad
dtheta = 7.292115e-5;   % earth rotation rate, rad/s
density0 = 3.614e-4;        % reference density, kg/km^3
ro = 700.000 + Re;   %reference altitude, km
H = 88.667;                 %scale height, km
Cr = 1.3;

%Given Data
%Chief Orbit
a = 6892.945;    %km
e = 1e-12;
i = 97;     %deg
RAAN = 0;
w = 270;
Mo = 0;

%Orbit Differences
di_x = 0;
di_y = -1/a; %rad
di = 0;     %deg
dRAAN = (di_y/sind(i))*180/pi;   %deg

de_x = 0;
de_y = .3/a;    %non-dim
psi = 90;   %deg
de = norm([de_x de_y]);  %non-dim

%dw = acosd((de^2 - e^2 - (e+de)^2)/(-2*e*(e+de)));
%dw = 90;

%Deputy Orbit
a2 = a + 0;
e2 = e + de;
i2 = i + di;    %deg
RAAN2 = RAAN + dRAAN;   %deg
w2 = 90;  %deg
Mo2 = Mo - 180;    %deg

%Rotation Matrix from ECI to Orbit Frame 1
R1 = [1   0        0;
      0  cosd(i)  sind(i);
      0 -sind(i)  cosd(i)];
  
R3 = [cosd(RAAN)  sind(RAAN)  0;
     -sind(RAAN)  cosd(RAAN)  0;
      0           0           1];

OF1_ECI = R1*R3;


%Input initial conditions
%Convert to cartesian position and velocity
x_in = [1/a e i RAAN w Mo]';
x_out = element_conversion(x_in,mu,0,0,1);
r1_vect0 = x_out(1:3)
v1_vect0 = x_out(4:6)

x_in = [1/a2 e2 i2 RAAN2 w2 Mo2]';
x_out = element_conversion(x_in,mu,0,0,1);
r2_vect0 = x_out(1:3)
v2_vect0 = x_out(4:6)

%Calculate initial energy and angular momentum
E1_0 = dot(v1_vect0,v1_vect0)/2 - mu/norm(r1_vect0);
h1_vect0 = cross(r1_vect0,v1_vect0);
ih1_0 = h1_vect0/norm(h1_vect0);

E2_0 = dot(v2_vect0,v2_vect0)/2 - mu/norm(r2_vect0);
h2_vect0 = cross(r2_vect0,v2_vect0);
ih2_0 = h2_vect0/norm(h2_vect0);

%Calculate initial eccentricity and inclination vectors
%Calculate eccentricity unit vector
e1_vect0 = cross(v1_vect0,h1_vect0)/mu - r1_vect0/norm(r1_vect0);
e2_vect0 = cross(v2_vect0,h2_vect0)/mu - r2_vect0/norm(r2_vect0);
de_vect0_eci = e2_vect0 - e1_vect0;     %ECI coordinates

%Calculate rotation matrix to Orbit 1 frame
%Perifocal frame defined by e, h
ie1_0 = e1_vect0/norm(e1_vect0);
ip1_0 = cross(ih1_0,ie1_0);

P_ECI = [ie1_0 ip1_0 ih1_0]';
R3 = [cosd(w) -sind(w)  0;
      sind(w)  cosd(w)  0;
      0        0        1];
OF1_ECI = R3*P_ECI;


%Rotate de and di vectors to OF1
de_vect0 = OF1_ECI*de_vect0_eci;

di_vect0_eci = cross(ih1_0,ih2_0);  %ECI coordinates
di_vect0 = OF1_ECI*di_vect0_eci;

%Print results for t0
de_vect0
de_vect0_check = [de_x de_y 0]'
di_vect0
di_vect0_check = [di_x di_y 0]'

rho0_eci = r2_vect0 - r1_vect0
rho0_of1 = OF1_ECI*rho0_eci
rho0_p = P_ECI*rho0_eci
drho0_eci = v2_vect0 - v1_vect0
drho0_of1 = OF1_ECI*drho0_eci
drho0_p = P_ECI*drho0_eci
rho0 = norm(rho0_eci)
drho0 = norm(drho0_eci)

%Set up call to numerical integration routine 
%TurboProp rk45
delta_t = 20;
odefun = 'TwoBody_drag_grav_SRP_ls';
options = [delta_t 1e-12];
extras.mu = mu;  %km^3/s^2
extras.radius = Re;     %km
extras.degord = [200 200];
extras.degordstm = [200 200];
extras.gravfile = which('GGM02C.sha');
extras.reftime = 2455563.0; %Julian Date 1/1/11 12:00 UTC
extras.A_m = A_m;   %km^2/kg
extras.EOP = [theta0 dtheta];   %rad, rad/s
extras.atmos = [density0 ro H];     %km and kg

tin = (0:20:30*86400)';
int0 = [r1_vect0;v1_vect0;Cd;Cr];

tic

[tout,intout] = rk45_mex(odefun,tin,int0,options,extras);

toc

%Final values out
r1_vect = intout(:,1:3)';
v1_vect = intout(:,4:6)';

%Re-initialize and repeat for r2,v2
extras.A_m = 1.02*A_m;  %km^2/kg

int0 = [r2_vect0;v2_vect0;Cd;Cr];

tic

[tout,intout] = rk45_mex(odefun,tin,int0,options,extras);

toc

%Final values out
r2_vect = intout(:,1:3)';
v2_vect = intout(:,4:6)';


t_days = tout/86400;
for i = 1:31
    ind1 = find(t_days <= i-1 & t_days >= i-1.01);    
    ind(i) = ind1(end);
end

t_wholedays = t_days(ind);
r1_vect_whole = r1_vect(:,ind);
v1_vect_whole = v1_vect(:,ind);
r2_vect_whole = r2_vect(:,ind);
v2_vect_whole = v2_vect(:,ind);

ind2 = round(linspace(1,length(r1_vect),10000));
t_partdays = t_days(ind2);
r1_vect_part = r1_vect(:,ind2);
v1_vect_part = v1_vect(:,ind2);
r2_vect_part = r2_vect(:,ind2);
v2_vect_part = v2_vect(:,ind2);

save turboprop_drag_grav_SRP_ls.mat t_wholedays r1_vect_whole v1_vect_whole r2_vect_whole v2_vect_whole...
    t_partdays r1_vect_part v1_vect_part r2_vect_part v2_vect_part
