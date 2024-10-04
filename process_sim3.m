clear
close all

%Constants
Re = 6378.1363;     %km
mu = 3.986004e5;    %km^3/s^2
J2 = 0.001082;      %[unitless]
Cd = 0;             %[unitless]

%Chief Orbit Parameters
a = 6892.945;    %km
e = 1e-12;
i = 97;     %deg
p = a*(1-e^2);  %km
w0 = 270;    %deg
dw = .75*J2*sqrt(mu/a^3)*(Re/p)^2*(5*cosd(i)^2 - 1);    %rad/s
dw = dw*180/pi; %deg/sec
dw = dw*86400;  %deg/day
%Calculate orbit period
P = 2*pi*sqrt(a^3/mu);      %sec
P = P/86400;    %days

loadfile = 'turboprop_drag_grav_SRP_ls2.mat'
%loadfile = 'ode45_j2_nodrag4.mat'
load(loadfile)

flag = 2;

if flag == 0 
    t = t_wholedays;
    r1_vect = r1_vect_whole;
    v1_vect = v1_vect_whole;
    r2_vect = r2_vect_whole;
    v2_vect = v2_vect_whole;
elseif flag == 1
    t = t_wholedays;
    r1_vect = r1_vect_plot;
    v1_vect = v1_vect_plot;
    r2_vect = r2_vect_plot;
    v2_vect = v2_vect_plot;    
else
    t = t_partdays;
    r1_vect = r1_vect_part;
    v1_vect = v1_vect_part;
    r2_vect = r2_vect_part;
    v2_vect = v2_vect_part;
end

%For each time, calculate de and di vectors
dotprod = 1;
for i = 1:length(t)
    %Calculate angular momentum unit vectors
    h1_vect = cross(r1_vect(:,i),v1_vect(:,i));
    ih1 = h1_vect/norm(h1_vect);
    h2_vect = cross(r2_vect(:,i),v2_vect(:,i));
    ih2 = h2_vect/norm(h2_vect);
    
    %Calculate di vector
    di_vect_eci = cross(ih1,ih2);  %ECI coordinates
    
    %Calculate eccentricity unit vectors    
    e1_vect = cross(v1_vect(:,i),h1_vect)/mu - r1_vect(:,i)/norm(r1_vect(:,i));
    e2_vect = cross(v2_vect(:,i),h2_vect)/mu - r2_vect(:,i)/norm(r2_vect(:,i));
    de_vect_eci = e2_vect - e1_vect;     %ECI coordinates

    %Calculate rotation matrix to Orbit 1 frame
    %Perifocal frame defined by e1, h1
    ie1 = e1_vect/norm(e1_vect);
    ip1 = cross(ih1,ie1);

    P_ECI = [ie1 ip1 ih1]';
    
    w = w0 + dw*t(i);
    
    R3 = [cosd(w) -sind(w)  0;
          sind(w)  cosd(w)  0;
          0        0        1];
    
    OF1_ECI = R3*P_ECI;

    %Rotate de and di vectors to OF1
    de_vect(:,i) = OF1_ECI*de_vect_eci;    
    di_vect(:,i) = OF1_ECI*di_vect_eci;    
    
    %Calculate rotation matrix to Hill Frame O
    or = r1_vect(:,i)/norm(r1_vect(:,i));
    oh = ih1;
    ot = cross(oh,or);
    ON = [or ot oh]';
    
    %Calculate rho in ECI frame, then rotate to Hill
    rho_eci = r2_vect(:,i) - r1_vect(:,i);
    rho(:,i) = ON*rho_eci;     
    
    %Find date when de and di vectors are smallest    
    if abs(dot(de_vect(:,i),di_vect(:,i))) < dotprod + 1e-12
        dot_ind = i;
        dotprod = abs(dot(de_vect(:,i),di_vect(:,i)));
    end
  
end
t(dot_ind)

%Edit rho matrix to include times around 5 day increments
ind1 = find(mod(t,30)<=3*P);
ind1 = ind1(1:end-1);
rho1 = rho(:,ind1);
x1 = rho1(1,:)*1000;  %meters
z1 = rho1(3,:)*1000;  %meters

ind2 = find(mod(t,30)<=(5+3*P) & mod(t,30)>=5);
rho2 = rho(:,ind2);
x2 = rho2(1,:)*1000;  %meters
z2 = rho2(3,:)*1000;  %meters

ind3 = find(mod(t,30)<=(10+3*P) & mod(t,30)>=10);
rho3 = rho(:,ind3);
x3 = rho3(1,:)*1000;  %meters
z3 = rho3(3,:)*1000;  %meters

ind4 = find(mod(t,30)<=(15+3*P) & mod(t,30)>=15);
rho4 = rho(:,ind4);
x4 = rho4(1,:)*1000;  %meters
z4 = rho4(3,:)*1000;  %meters

ind5 = find(mod(t,30)<=(20+3*P) & mod(t,30)>=20);
rho5 = rho(:,ind4);
x5 = rho5(1,:)*1000;  %meters
z5 = rho5(3,:)*1000;  %meters

ind6 = find(mod(t,30)<=(25+3*P) & mod(t,30)>=25);
rho6 = rho(:,ind6);
x6 = rho6(1,:)*1000;  %meters
z6 = rho6(3,:)*1000;  %meters

ind7 = find(mod(t,30)<=(t(dot_ind)+3*P) & mod(t,30)>=t(dot_ind));
rho7 = rho(:,ind7);
x7 = rho7(1,:)*1000;  %meters
z7 = rho7(3,:)*1000;  %meters

ind8 = find(mod(t,30)>=(30-3*P));
rho8 = rho(:,ind8);
x8 = rho8(1,:)*1000;  %meters
z8 = rho8(3,:)*1000;  %meters


%Separate ecc and inc plots
%Create vector for perpendicular ecc
x_plot = [0 de_vect(1,dot_ind)]';
y_plot = [0 de_vect(2,dot_ind)]';
figure
plot(de_vect(1,:),de_vect(2,:),'k.'),axis equal
hold on
plot(x_plot,y_plot,'k','LineWidth',2)
xlim([-6e-5 6e-5]),ylim([-6e-5 6e-5]),grid
xlabel('ex axis')
ylabel('ey axis')

x_plot = [0 di_vect(1,dot_ind)]';
y_plot = [0 di_vect(2,dot_ind)]';
figure
plot(di_vect(1,:),di_vect(2,:),'k.'),axis equal
hold on
plot(x_plot,y_plot,'k','LineWidth',2)
xlim([-6e-4 6e-4]),ylim([-6e-4 6e-4]),grid
xlabel('ix axis')
ylabel('iy axis')


%Plot ecc and inc together
x_plot = [0 de_vect(1,dot_ind)]';
y_plot = [0 de_vect(2,dot_ind)]';
figure
plot(de_vect(1,:),de_vect(2,:),'k.'),axis equal
hold on
plot(x_plot,y_plot,'k','LineWidth',2)
xlim([-6e-5 6e-5]),ylim([-20e-5 6e-5]),grid

x_plot = [0 di_vect(1,dot_ind)]';
y_plot = [0 di_vect(2,dot_ind)]';
plot(di_vect(1,:),di_vect(2,:),'k.'),axis equal
plot(x_plot,y_plot,'k','LineWidth',2)

%Radial and Cross-track separation
figure
plot(z1,x1,'k','LineWidth',3),axis equal, grid
hold on
plot(z2,x2,'k')
plot(z3,x3,'k')
plot(z4,x4,'k')
plot(z5,x5,'k')
plot(z6,x6,'k')
plot(z7,x7,'k')
plot(z8,x8,'k','LineWidth',3)
xlabel('Cross-track separation, m')
ylabel('Radial separation, m')







