#--------------------------------------------------------------------------








#----------------------------------------------
#------------código a ser executado-----------
%-------------------------------------------
% Program: deformation_vep2d.m
% 2D viscoelastoplastic plane strain deformation
% solid mechanics rate formulation
% 9-node quadrilaterals
%-----------------------------------------
import numpy as np

# physical parameters
lx = 4  # length of x domain (m)
lz = 1  # length of z domain (m)
emod_v = np.array([1, 1, 1, 1]) * 1e11  # Young's modulus (Pa)
pois_v = np.array([0.3, 0.3, 0.3, 0.3])  # Poisson's ratio
smod_v = emod_v / (2 * (1 + pois_v))  # shear modulus (Pa)
bmod_v = emod_v / (3 * (1 - 2 * pois_v))  # bulk modulus (Pa)
visc_v = np.array([1, 1, 100, 100]) * 1e18  # shear viscosity (Pas)
coh_v = 20 * np.array([1, 1, 1, 0.9]) * 1e6  # cohesion (Pa)
phi_v = np.array([1, 1, 1, 1]) * 30 * np.pi / 180  # friction angle(radians)
psi_v = np.array([0, 0, 0, 0])  # dilation angle (radians)
grav_v = np.array([2700, 2700, 2700, 2700]) * 9.8  # density*gravity
seconds_per_year = 60 * 60 * 24 * 365
bvel = 5e-3 / seconds_per_year  # boundary velocity (m/s)
edot = bvel / lx  # initial strain rate (1/s)
plasticity = False  # include/ignore plasticity (true/false)
nst = 3  # number of stress/strain components

# characteristic scales
length_scale = lz  # length scale
stress_scale = max(smod_v)  # stress_scale
time_scale = 1 / edot  # time_scale

#----------------------------------------------------------------------UNDER REVIEW----------------------
% rescaled parameters
emod_v = emod_v/stress_scale ;
smod_v = smod_v/stress_scale ;
bmod_v = bmod_v/stress_scale ;
visc_v = visc_v/(stress_scale*time_scale) ;
coh_v = coh_v/stress_scale ;
grav_v = grav_v*length_scale/stress_scale ;
lx = lx/length_scale ;
lz = lz/length_scale ;
Deformation of Earth’s Crust 199
bvel = bvel/length_scale*time_scale ;
edot = edot*time_scale ;
layer_top = 0.55 ; % z coord. (nondimensional) at top of layer
layer_bot = 0.45 ; % z coord. (nondimensional) at bottom of layer
% numerical parameters
ntime = 100 ; % number of time steps to perform
nxe = 40 ; % n elements in x-direction
nze = 40 ; % n elements in z-direction
nels = nxe*nze ; % total number of elements
nx = 2*nxe+1 ; % n of nodes in x-direction
nz = 2*nze+1 ; % n of nodes in z-direction
nn = nx*nz ; % total number of nodes
dx = lx/nxe ; % element size in x-direction
dz = lz/nze ; % element size in z-direction
nod = 9 ; % number of nodes
ndof = 2 ; % number of degrees of freedom
ntot = nod + nod ; % total degrees of freedom in an element
nip = 9 ; % number of integration points in an element
eps = 1e-3 ; % convergence tolerance
limit = 50 ; % maximum number of iterations
dt = 0.01 ; % time step (non dimensional)
%-----------------------------------------
% generate coordinates and nodal numbering
%-----------------------------------------
% computational mesh
% define mesh (numbering in z direction)
g_coord = zeros(2,nn) ;
n=1;
for i = 1:nx
for j=1:nz
g_coord(1,n) = (i-1)*dx/2 ;
g_coord(2,n) = (j-1)*dz/2 ;
n=n+1;
end
end
% establish node numbering numbering
gnumbers = reshape(1:nn,[nz nx]) ;
iel = 1 ;
for i=1:2:nx-1
for j=1:2:nz-1
g_num(1,iel) = gnumbers(j,i) ; % node 1
g_num(2,iel) = gnumbers(j+1,i) ; % node 2
g_num(3,iel) = gnumbers(j+2,i) ; % node 3
g_num(4,iel) = gnumbers(j+2,i+1) ; % node 4
g_num(5,iel) = gnumbers(j+2,i+2) ; % node 5
g_num(6,iel) = gnumbers(j+1,i+2) ; % node 6
g_num(7,iel) = gnumbers(j,i+2) ; % node 7
g_num(8,iel) = gnumbers(j,i+1) ; % node 8
g_num(9,iel) = gnumbers(j+1,i+1) ; % node 9
iel = iel + 1 ; % increment the element number
end
end
%-----------------------------------------
200 Applications of the Finite Element Method in Earth Science
% create global-local connection arrays
%-----------------------------------------
sdof = 0 ; % system degrees of freedom
nf = zeros(ndof,nn) ; % node degree of freedom array
for n = 1:nn % loop over all nodes
for i=1:ndof % loop over each degree of freedom
sdof = sdof + 1 ; % increment total number of equations
nf(i,n) = sdof ; % record the equation number for each node
end
end
% equation numbering for each element
g = zeros(ntot,1) ;
g_g = zeros(ntot,nels);
for iel=1:nels ;
num = g_num(:,iel) ; % extract nodes for the element
inc=0 ;
for i=1:nod ;
for k=1:2 ;
inc=inc+1 ;
g(inc)=nf(k,num(i)) ;
end
end
g_g(:,iel) = g ;
end
%-----------------------------------------
% define boundary conditions
%-----------------------------------------
% boundary nodes
bx0 = find(g_coord(1,:)==0) ;
bxn = find(g_coord(1,:)==lx) ;
bz0 = find(g_coord(2,:)==0) ;
bzn = find(g_coord(2,:)==lz) ;
% fixed boundary equations along with their values
bcdof = [ nf(1,bx0) nf(1,bxn) nf(2,bz0) ] ;
bcval = [ zeros(1,length(bx0)) -bvel*ones(1,length(bxn)) zeros(1,length(bz0)) ] ;
%------------------------------------------------------------
% establish phases
%-----------------------------------------------------------
phase = ones(1,nels) ; % top
phase(g_coord(2,g_num(9,:))<layer_bot)=2 ; % bottom
phase(find(g_coord(2,g_num(9,:))≤layer_top & g_coord(2,g_num(9,:))≥layer_bot))=3 ; ...
% layer
phase(nze*nxe/2-nze/2)=4 ; % inclusion
%-------------------------------------------
% perturb boundaries of layer
%-------------------------------------------
iit=find(g_coord(2,:)==layer_bot);% nodes on top of layer
iib=find(g_coord(2,:)==layer_top);% nodes on bottom of layer
ii = [iit iib]; % combined nodes
g_coord(2,ii)=g_coord(2,ii)+dz*0.05*(rand(1,length(ii))-0.5) ;
%-----------------------------------------
% integration data and shape functions
%----------------------------------------
Deformation of Earth’s Crust 201
% local coordinates of Gauss integration points for nip=3x3
points(1:3:7,1) = -sqrt(0.6);
points(2:3:8,1) = 0;
points(3:3:9,1) = sqrt(0.6);
points(1:3,2) = sqrt(0.6);
points(4:6,2) = 0 ;
points(7:9,2) = -sqrt(0.6);
% Gauss weights for nip=3x3
w = [ 5./9. 8./9. 5./9.] ;
v = [ 5./9.*w ; 8./9.*w ; 5./9.*w ] ;
wts = v(:) ;
% evaluate shape functions and their derivatives
% at integration points and save the results
for k = 1:nip
xi = points(k,1);
eta = points(k,2);
etam = eta - 1; etap = eta+1;
xim = xi - 1 ; xip = xi + 1 ;
x2p1 = 2*xi+1 ; x2m1 = 2*xi-1 ;
e2p1 = 2*eta+1 ; e2m1 = 2*eta-1 ;
% shape functions
fun= [ .25*xi*xim*eta*etam -.5*xi*xim*etap*etam ...
.25*xi*xim*eta*etap -.5*xip*xim*eta*etap ...
.25*xi*xip*eta*etap -.5*xi*xip*etap*etam ...
.25*xi*xip*eta*etam -.5*xip*xim*eta*etam xip*xim*etap*etam ] ;
% first derivatives of shape functions
der(1,1) = 0.25*x2m1*eta*etam ; %dN1dxi
der(1,2) =-0.5*x2m1*etap*etam ; %dN2dxi, etc
der(1,3) = 0.25*x2m1*eta*etap ;
der(1,4) = -xi*eta*etap ;
der(1,5) = 0.25*x2p1*eta*etap ;
der(1,6) =-0.5*x2p1*etap*etam ;
der(1,7) = 0.25*x2p1*eta*etam ;
der(1,8) = -xi*eta*etam ;
der(1,9) = 2*xi*etap*etam ;
der(2,1) = 0.25*xi*xim*e2m1 ; %dN1deta
der(2,2) =-xi*xim*eta ;%dN2deta, etc
der(2,3) = 0.25*xi*xim*e2p1 ;
der(2,4) =-0.5*xip*xim*e2p1 ;
der(2,5) = 0.25*xi*xip*e2p1 ;
der(2,6) =-xi*xip*eta ;
der(2,7) = 0.25*xi*xip*e2m1 ;
der(2,8) =-0.5*xip*xim*e2m1 ;
der(2,9) = 2*xip*xim*eta ;
fun_s(k,:) = fun ; % save shape functions
der_s(:,:,k) = der ; % save derivatives
end
%-----------------------------------------
% initialisation
%-----------------------------------------
lhs = sparse(sdof,sdof) ; % global stiffness matrix
b = zeros(sdof,1) ; % global rhs vector
displ = zeros(sdof,1) ; % solution vector (velocities)
tensor = zeros(nst,nip,nels) ; % stresses at integration points
tensor0 = zeros(nst,nip,nels) ; % old stresses
202 Applications of the Finite Element Method in Earth Science
%----------------------------------------------------
% loading loop
%----------------------------------------------------
time = 0 ; % initialise time
for n=1:ntime
time = time + dt ; % increment time
%------------------------------------------
% iterations
%------------------------------------------
error = eps*2 ; % initialise error
iters = 0 ; % initialise iteration counter
displ = zeros(sdof,1) ; % initialise solution vector
while error > eps & iters < limit
iters = iters + 1 % increment iteration counter
lhs = sparse(sdof,sdof); % initialised global stiffness matrix
b = zeros(sdof,1) ; % initialised global load vector
%-----------------------------------------
% element integration and assembly
%-----------------------------------------
for iel=1:nels % sum over elements
num = g_num(:,iel) ; % list of element nodes
g = g_g(:,iel) ; % element equation numbers
coord = g_coord(:,num)' ; % nodal coordinates
KM = zeros(ntot,ntot) ; % initialise stiffness matrix
R = zeros(ntot,1) ; % initialise stress load vector
uv = displ(g) ; % current nodal velocities
% retieve material properties for the current element
smod = smod_v(phase(iel)) ; % shear modulus
K = bmod_v(phase(iel)) ; % bulk modulus
mu = visc_v(phase(iel)) ; % viscosity
phi = phi_v(phase(iel)) ; % friction coeff.
psi = psi_v(phase(iel)) ; % dilation angle
coh = coh_v(phase(iel)) ; % cohesion
rhog = grav_v(phase(iel)) ; % rho x gravity
% viscoelastic material matrix
di = dt*(3*mu*K + 3*dt*smod*K + 4*smod*mu) ;
od = dt*(-2*smod*mu + 3*mu*K + 3*dt*smod*K);
d = 3*(mu + dt*smod) ;
ed = 2*mu*dt*smod/(2*mu + dt*smod) ;
dee = [di/d od/d 0 ; od/d di/d 0 ; 0 0 ed];
% stress matrix
di = 3*mu + dt*smod ;
od = dt*smod ;
ed = 2 * mu / (2 * mu + dt * smod) ;
dees = [di/d od/d 0 ; od/d di/d 0 ; 0 0 ed] ;
for k = 1:nip % integration loop
fun = fun_s(:,k) ; % shape functions
fune = zeros(1,ntot) ; % extended shape function array
fune(2:2:ntot) = fun ; % shape function for z-dof only
Deformation of Earth’s Crust 203
der = der_s(:,:,k) ; % shape functions in local coords
jac = der*coord ; % jacobian matrix
detjac = det(jac) ; % det. of the Jacobian
dwt = detjac*wts(k) ; % detjac x weight
invjac = inv(jac) ; % inverse of the Jacobian
deriv = invjac*der ; % shape functions in physical coords
bee = zeros(nst,ntot); % kinematic matrix
bee(1,1:2:ntot-1) = deriv(1,:) ;
bee(2,2:2:ntot) = deriv(2,:) ;
bee(3,1:2:ntot-1) = deriv(2,:) ;
bee(3,2:2:ntot) = deriv(1,:) ;
strain_rate = bee*uv ; % strain rates
stress = dee*strain_rate + dees*tensor0(:,k,iel) ; % stresses
if plasticity % do only if plasticity included
tau = (1/4*(stress(1)-stress(2))ˆ2+stress(3)ˆ2)ˆ(1/2); % tau star
sigma = 1/2*(stress(1)+stress(2)); % sigma star
F = tau + sigma*sin(phi)-coh*cos(phi); % plastic yield function
if F>0 % return stresses to yield surface
if (sigma≤coh/tan(phi)) % 'normal' case
beta = abs(coh*cos(phi)-sin(phi)*sigma)/tau ;
sxx_new = sigma + beta*(stress(1)-stress(2))/2 ;
szz_new = sigma - beta*(stress(1)-stress(2))/2 ;
sxz_new = beta*stress(3) ;
else % special treatment for corners of yield surface
sxx_new = coh/tan(phi) ;
szz_new = coh/tan(phi) ;
sxz_new = 0 ;
end
stress(1) = sxx_new ; stress(2)=szz_new ; stress(3)=sxz_new ;
end % end of stress return algorithm
end % end of plasticity
if iters==1 % normal rhs stress
stress_rhs = dees*tensor0(:,k,iel) ;
else % total stress
stress_rhs = stress ;
end
KM = KM + bee'*dee*bee*dwt ; % element stiffness matrix
R = R + (bee'*stress_rhs + fune'*rhog)*dwt ; % load vector
tensor(:,k,iel) = stress ; % stresses at integration point
end
% assemble global stiffness matrix and rhs vector
lhs(g,g) = lhs(g,g) + KM ; % stiffness matrix
b(g) = b(g) - R ; % load vector
end
%-------------------------------------------------
% implement boundary conditions and solve system
%-------------------------------------------------
% apply boundary conditions
lhs(bcdof,:) = 0 ;
tmp = spdiags(lhs,0) ; tmp(bcdof)=1 ; lhs=spdiags(tmp,0,lhs);
if iters==1
b(bcdof) = bcval ;
else
b(bcdof) = 0 ;
end
displ_inc = lhs \ b ; % solve for velocity increment
204 Applications of the Finite Element Method in Earth Science
displ = displ + displ_inc ; % update total velocity
error = max(abs(displ_inc))/max(abs(displ)) % estimate error
end
%----------------------------------------------
% end of iterations
%------------------------------------------------
% update mesh coordinates
g_coord(1,:) = g_coord(1,:) + dt*displ(nf(1,:))' ;
g_coord(2,:) = g_coord(2,:) + dt*displ(nf(2,:))' ;
tensor0 = tensor ; % save stresses
shortening_percent = (1 - (max(g_coord(1,:))-min(g_coord(1,:)))/lx)*100
%---------------------------------
% plotting of results
%---------------------------------
xgrid = reshape(g_coord(1,:),nz,nx)*length_scale ;% m
zgrid = reshape(g_coord(2,:),nz,nx)*length_scale ;% m
c = length_scale/time_scale*seconds_per_year*1e3 ;
u_solution = reshape(displ(nf(1,:)),nz,nx)*c ;% (mm/yr)
v_solution = reshape(displ(nf(2,:)),nz,nx)*c ;% (mm/yr)
tensor_scaled = tensor*stress_scale/1e6 ; % MPa
figure(1) , clf % mesh
plot(xgrid,zgrid,'b')
hold on
plot(xgrid',zgrid','b')
plot(g_coord(1,iit),g_coord(2,iit),'r')
plot(g_coord(1,iib),g_coord(2,iib),'r')
hold off
axis equal
title(['Deformed mesh after ',num2str(shortening_percent), ' % shortening'])
figure(2) , clf % velocity field
quiver(xgrid,zgrid,u_solution,v_solution)
axis equal
drawnow
title('Velocity vector field')
figure(3) , clf % x-velocity
pcolor(xgrid,zgrid,u_solution)
colormap(jet)
colorbar
shading interp
axis equal
title('x-velocity (mm/yr)')
figure(4) , clf % z-velocity
pcolor(xgrid,zgrid,v_solution)
colormap(jet)
colorbar
shading interp
axis equal
title('z-velocity (mm/yr)')
Deformation of Earth’s Crust 205
figure(5) , clf % plot negative mean stress
for iel=1:nels
num = g_num(:,iel) ;
coord = g_coord(:,num(1:8))' ;
means = (tensor(1,:,iel)+tensor(2,:,iel))/2;
nodevalues = fun_s\means' ;
h = fill(coord(:,1),coord(:,2),-nodevalues(1:8)) ;
set(h,'linestyle','none')
hold on
end
plot(xgrid(:,1:8:end),zgrid(:,1:8:end),'Color',[0.8,0.8,0.8])
plot(xgrid(1:8:end,:)',zgrid(1:8:end,:)','Color',[0.8,0.8,0.8])
hold off
axis equal
colorbar
title('Mean stress (MPa)')
%------------------------------------------------------
end
%-----------------------------------------------------
% end of time integration
%----------------------------------------------------
