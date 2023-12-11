#PENDENCIAS
#
#
#
#
#
#
#
#
#
#
#
#
#




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CÓDIGO BASE - DOGBONE FEM
#EXTRAÍDO DE: https://polymerfem.com/full-finite-element-solver-in-200-lines-of-python/
#



import sys
import numpy as np
import math
from matplotlib import pyplot as plt
## Input file syntax:
##    *Node
##    1, 0.0, 0.0
##    2, 0.0, 1.0
##    3, 1.0, 1.0
##    4, 1.0, 0.0
##    *Element
##    1, 1, 2, 3, 4
##    *Step
##    *Boundary
##    1, 1, 2, 0.0          # nodeId, dof1, dof2, value
##    2, 1, 1, 0.0
##    3, 1, 1, 0.01
##    4, 1, 1, 0.01
##    4, 2, 2, 0.0
def shape(xi):
	"""Shape functions for a 4-node, isoparametric element
		N_i(xi,eta) where i=[1,2,3,4]
		Input: 1x2,  Output: 1x4"""
	xi,eta = tuple(xi)
	N = [(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta), (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)]
	return 0.25 * np.array(N)
def gradshape(xi):
	"""Gradient of the shape functions for a 4-node, isoparametric element.
		dN_i(xi,eta)/dxi and dN_i(xi,eta)/deta
		Input: 1x2,  Output: 2x4"""
	xi,eta = tuple(xi)
	dN = [[-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
		  [-(1.0-xi), -(1.0+xi), (1.0+xi),  (1.0-xi)]]
	return 0.25 * np.array(dN)
def local_error(str):
	print("*** ERROR ***")
	print(str)
	sys.exit(3)
def read_inp_file(inpFileName, nodes, conn, boundary):
	print('\n** Read input file')
	inpFile = open(inpFileName, 'r')
	lines = inpFile.readlines()
	inpFile.close()
	state = 0
	for line in lines:
		line = line.strip()
		if len(line) <= 0: continue
		if line[0] == '*':
			state = 0
		if line.lower() == "*node":
			state = 1
			continue
		if line.lower() == "*element":
			state = 2
			continue
		if line.lower() == "*boundary":
			state = 3
			continue
		if state == 0:
			continue
		if state == 1:
			# read nodes
			values = line.split(",")
			if len(values) != 3:
				local_error("A node definition needs 3 values")
			nodeNr = int(values[0]) - 1  # zero indexed
			xx = float(values[1])
			yy = float(values[2])
			nodes.append([xx,yy])   # assume the nodes are ordered 1, 2, 3...
			continue
		if state == 2:
			# read elements
			values = line.split(",")
			if len(values) != 5:
				local_error("An element definition needs 5 values")
			elemNr = int(values[0])
			n1 = int(values[1]) - 1  # zero indexed
			n2 = int(values[2]) - 1
			n3 = int(values[3]) - 1
			n4 = int(values[4]) - 1
			#conn.append([n1, n2, n3, n4]) # assume elements ordered 1, 2, 3
			conn.append([n1, n4, n3, n2]) # assume elements ordered 1, 2, 3
			continue
		if state == 3:
			# read displacement boundary conditions
			values = line.split(",")
			if len(values) != 4:
				local_error("A displacement boundary condition needs 4 values")
			nodeNr = int(values[0]) - 1  # zero indexed
			dof1 = int(values[1])
			dof2 = int(values[2])
			val = float(values[3])
			if dof1 == 1:
				boundary.append([nodeNr,1,val])
			if dof2 == 2:
				boundary.append([nodeNr,2,val])
			continue

def main():
	##
	## Main Program
	##
	nodes = []
	conn = []
	boundary = []
	if len(sys.argv) <= 1: local_error('No input file provided.')
	print('Input file:', sys.argv[1])
	read_inp_file(sys.argv[1], nodes, conn, boundary)
	nodes = np.array(nodes)
	num_nodes = len(nodes)
	print('   number of nodes:', len(nodes))
	print('   number of elements:', len(conn))
	print('   number of displacement boundary conditions:', len(boundary))

	###############################
	# Plane-strain material tangent (see Bathe p. 194)
	# C is 3x3
	E = 100.0
	v = 0.3
	C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v, v, 0.0], [v, 1.0-v, 0.0], [0.0, 0.0, 0.5-v]])
	###############################
	# Make stiffness matrix
	# if N is the number of DOF, then K is NxN
	K = np.zeros((2*num_nodes, 2*num_nodes))    # square zero matrix
	# 2x2 Gauss Quadrature (4 Gauss points)
	# q4 is 4x2
	q4 = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]) / math.sqrt(3.0)
	print('\n** Assemble stiffness matrix')
	# strain in an element: [strain] = B    U
	#                        3x1     = 3x8  8x1
	#
	# strain11 = B11 U1 + B12 U2 + B13 U3 + B14 U4 + B15 U5 + B16 U6 + B17 U7 + B18 U8
	#          = B11 u1          + B13 u1          + B15 u1          + B17 u1
	#          = dN1/dx u1       + dN2/dx u1       + dN3/dx u1       + dN4/dx u1
	B = np.zeros((3,8))
	# conn[0] is node numbers of the element
	for c in conn:     # loop through each element
		# coordinates of each node in the element
		# shape = 4x2
		# for example:
		#    nodePts = [[0.0,   0.0],
		#               [0.033, 0.0],
		#               [0.033, 0.066],
		#               [0.0,   0.066]]
		nodePts = nodes[c,:]
		Ke = np.zeros((8,8))	# element stiffness matrix is 8x8
		for q in q4:			# for each Gauss point
			# q is 1x2, N(xi,eta)
			dN = gradshape(q)       # partial derivative of N wrt (xi,eta): 2x4
			J  = np.dot(dN, nodePts).T # J is 2x2
			dN = np.dot(np.linalg.inv(J), dN)    # partial derivative of N wrt (x,y): 2x4
			# assemble B matrix  [3x8]
			B[0,0::2] = dN[0,:]
			B[1,1::2] = dN[1,:]
			B[2,0::2] = dN[1,:]
			B[2,1::2] = dN[0,:]
			# element stiffness matrix
			Ke += np.dot(np.dot(B.T,C),B) * np.linalg.det(J)
		# Scatter operation
		for i,I in enumerate(c):
			for j,J in enumerate(c):
				K[2*I,2*J]     += Ke[2*i,2*j]
				K[2*I+1,2*J]   += Ke[2*i+1,2*j]
				K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1]
				K[2*I,2*J+1]   += Ke[2*i,2*j+1]
	###############################
	# Assign nodal forces and boundary conditions
	#    if N is the number of nodes, then f is 2xN
	f = np.zeros((2*num_nodes))          # initialize to 0 forces
	# How about displacement boundary conditions:
	#    [k11 k12 k13] [u1] = [f1]
	#    [k21 k22 k23] [u2]   [f2]
	#    [k31 k32 k33] [u3]   [f3]
	#
	#    if u3=x then
	#       [k11 k12 k13] [u1] = [f1]
	#       [k21 k22 k23] [u2]   [f2]
	#       [k31 k32 k33] [ x]   [f3]
	#   =>
	#       [k11 k12 k13] [u1] = [f1]
	#       [k21 k22 k23] [u2]   [f2]
	#       [  0   0   1] [u3]   [ x]
	#   the reaction force is
	#       f3 = [k31 k32 k33] * [u1 u2 u3]
	for i in range(len(boundary)):  # apply all boundary displacements
		nn  = boundary[i][0]
		dof = boundary[i][1]
		val = boundary[i][2]
		j = 2*nn
		if dof == 2: j = j + 1
		K[j,:] = 0.0
		K[j,j] = 1.0
		f[j] = val
	###############################
	print('\n** Solve linear system: Ku = f')	# [K] = 2N x 2N, [f] = 2N x 1, [u] = 2N x 1
	u = np.linalg.solve(K, f)
	###############################
	print('\n** Post process the data')
	# (pre-allocate space for nodal stress and strain)
	node_strain = []
	node_stress = []
	for ni in range(len(nodes)):
		node_strain.append([0.0, 0.0, 0.0])
		node_stress.append([0.0, 0.0, 0.0])
	node_strain = np.array(node_strain)
	node_stress = np.array(node_stress)

	print(f'   min displacements: u1={min(u[0::2]):.4g}, u2={min(u[1::2]):.4g}')
	print(f'   max displacements: u1={max(u[0::2]):.4g}, u2={max(u[1::2]):.4g}')
	emin = np.array([ 9.0e9,  9.0e9,  9.0e9])
	emax = np.array([-9.0e9, -9.0e9, -9.0e9])
	smin = np.array([ 9.0e9,  9.0e9,  9.0e9])
	smax = np.array([-9.0e9, -9.0e9, -9.0e9])
	for c in conn:	# for each element (conn is Nx4)
										# c is like [2,5,22,53]
		nodePts = nodes[c,:]			# 4x2, eg: [[1.1,0.2], [1.2,0.3], [1.3,0.4], [1.4, 0.5]]
		for q in q4:					# for each integration pt, eg: [-0.7,-0.7]
			dN = gradshape(q)					# 2x4
			J  = np.dot(dN, nodePts).T			# 2x2
			dN = np.dot(np.linalg.inv(J), dN)	# 2x4
			B[0,0::2] = dN[0,:]					# 3x8
			B[1,1::2] = dN[1,:]
			B[2,0::2] = dN[1,:]
			B[2,1::2] = dN[0,:]

			UU = np.zeros((8,1))				# 8x1
			UU[0] = u[2*c[0]]
			UU[1] = u[2*c[0] + 1]
			UU[2] = u[2*c[1]]
			UU[3] = u[2*c[1] + 1]
			UU[4] = u[2*c[2]]
			UU[5] = u[2*c[2] + 1]
			UU[6] = u[2*c[3]]
			UU[7] = u[2*c[3] + 1]
			# get the strain and stress at the integration point
			strain = B @ UU		# (B is 3x8) (UU is 8x1) 		=> (strain is 3x1)
			stress = C @ strain	# (C is 3x3) (strain is 3x1) 	=> (stress is 3x1)
			emin[0] = min(emin[0], strain[0][0])
			emin[1] = min(emin[1], strain[1][0])
			emin[2] = min(emin[2], strain[2][0])
			emax[0] = max(emax[0], strain[0][0])
			emax[1] = max(emax[1], strain[1][0])
			emax[2] = max(emax[2], strain[2][0])

			node_strain[c[0]][:] = strain.T[0]
			node_strain[c[1]][:] = strain.T[0]
			node_strain[c[2]][:] = strain.T[0]
			node_strain[c[3]][:] = strain.T[0]
			node_stress[c[0]][:] = stress.T[0]
			node_stress[c[1]][:] = stress.T[0]
			node_stress[c[2]][:] = stress.T[0]
			node_stress[c[3]][:] = stress.T[0]
			smax[0] = max(smax[0], stress[0][0])
			smax[1] = max(smax[1], stress[1][0])
			smax[2] = max(smax[2], stress[2][0])
			smin[0] = min(smin[0], stress[0][0])
			smin[1] = min(smin[1], stress[1][0])
			smin[2] = min(smin[2], stress[2][0])
	print(f'   min strains: e11={emin[0]:.4g}, e22={emin[1]:.4g}, e12={emin[2]:.4g}')
	print(f'   max strains: e11={emax[0]:.4g}, e22={emax[1]:.4g}, e12={emax[2]:.4g}')
	print(f'   min stress:  s11={smin[0]:.4g}, s22={smin[1]:.4g}, s12={smin[2]:.4g}')
	print(f'   max stress:  s11={smax[0]:.4g}, s22={smax[1]:.4g}, s12={smax[2]:.4g}')
	###############################
	print('\n** Plot displacement')
	xvec = []
	yvec = []
	res  = []
	plot_type = 'e11'
	for ni,pt in enumerate(nodes):
		xvec.append(pt[0] + u[2*ni])
		yvec.append(pt[1] + u[2*ni+1])
		if plot_type=='u1':  res.append(u[2*ni])				# x-disp
		if plot_type=='u2':  res.append(u[2*ni+1])				# y-disp
		if plot_type=='s11': res.append(node_stress[ni][0])		# s11
		if plot_type=='s22': res.append(node_stress[ni][1])		# s22
		if plot_type=='s12': res.append(node_stress[ni][2])		# s12
		if plot_type=='e11': res.append(node_strain[ni][0])		# e11
		if plot_type=='e22': res.append(node_strain[ni][1])		# e22
		if plot_type=='e12': res.append(node_strain[ni][2])		# e12
	tri = []
	for c in conn:
		tri.append( [c[0], c[1], c[2]] )
		tri.append( [c[0], c[2], c[3]] )
	t = plt.tricontourf(xvec, yvec, res, triangles=tri, levels=14, cmap=plt.cm.jet)
	#plt.scatter(xvec, yvec, marker='o', c='b', s=0.5) # (plot the nodes)
	plt.grid()
	plt.colorbar(t)
	plt.title(plot_type)
	plt.axis('equal')
	plt.show()
	print('Done.')
if __name__ == '__main__':
	main()
 #------------------------------------------------------------------------------------------------------------------------------------
 # CÓDIGOS DO LIVRO GUY SIMPSON - CAPÍTULO 12
##POLIR E MODULARIZAR ETAPAS DA PROGRAMAÇÃO DO CÓDIGO
##The viscoelastoplastic model presented in Section 12.4 is straightforward to implement in a Maltab
#finite element code. In the following text, we present a script to show how this is performed in practice. Two problems are investigated to illustrate how the model can be applied (Figure 12.3). The
#first problem treats the folding of a layered viscoelastic material (neglecting plasticity), while the second case deals with strain localization in a viscoelastoplastic material. The basic steps that must be
#performed in both cases can be summarized as follows:
#
#Figure 12.3 Setup for folding and shear localization numerical simulations (see results in Figures 12.4 and 12.5).
#Numbers indicate the lithological unit (termed the “phase” in the Matlab script). The folding experiment was
#performed with a viscoelastic material where the central layer (i.e., phases 3 and 4) has a shear viscosity (1020 Pa s) that
#is 100 times greater than that of the surrounding matrix (1018 Pa s). Other parameters are as follows: density = 2700 kg
#m−3, gravity = 9.8 m s−2, Young’s modulus = 1011 Pa, Poisson’s ratio = 0.3 (all considered uniform throughout),
#boundary velocity = 5 mm year−1, and nxe = nze = 80. The shear localization experiment was performed with a
#viscoelastoplastic material with the following material properties: Young’s modulus for layer = 1010 Pa, Young’s
#modulus for matrix = 0.5 × 1010 Pa, Poisson’s ratio = 0.3, shear viscosity = 1022 Pa s, cohesion = 20 MPa, cohesion for
#phase 4 = 18 MPa, friction angle 30∘, dilatancy angle 0∘, boundary velocity = 5 mm year−1, and nxe = nze = 200.



#1) Assign all material properties (e.g., shear and bulk moduli, viscosity, and friction coefficient) as
#vectors, along with other physical parameters such as the length and depth of the model domain.
#This can be done, for example, with lines such as
#lx = 4 ; % Length in x-direction (m)
#lz = 1 ; % Length in z direction (m)
#visc_v = [ 1 1 100 100]*1e20 ; % Viscosity (Pa s)
#coh_v = 20*[1 1 1 0.9 ]*1e6 ; % cohesion (Pa)
#phi_v = ([1 1 1 1])*30*pi/180 ; % friction angle (radians)
#In the case of the viscosity vector visc_v, note that the phases (or units) 1 and 2 have shear
#viscosities of 1020 Pa s, while phases 3 and 4 have viscosities of 1022 Pa s.



#2) Nondimensionalize all physical parameters using characteristic scales for length (taken as lz, the
#initial depth extent of the model domain), stress (the maximum shear modulus), and time (inverse
#of the initially imposed horizontal strain rate, edot). Note that a similar approach was also applied
#and discussed in slightly more detail in Chapter 11. The following snippet shows an example of
#how this is performed:
#% Define characteristic scales
#length_scale = lz ;
#stress_scale = max(smod_v) ;
#time_scale = 1/edot ;
#% Non-dimensional scaling
#lx = lx/length_scale ;
#lz = lz/length_scale ;
#visc_v = visc_v/(stress_scale*time_scale) ;
#coh_v = coh_v/stress_scale ;

#3) Define all numerical parameters (e.g., number of elements in each direction, time step, and number of integration points).


#4) Define the mesh. For example, for the nine-node mesh this can be achieved with the following
#Matlab snippet:
#% Mesh coordinates
#g_coord = zeros(2,nn) ; % storage for node coordinates
#n=1; % initialise node counter
#for i = 1:nx % loop over nodes in x-direction
#for j=1:nz % loop over nodes in z-direction
#g_coord(1,n) = (i-1)*dx/2 ;
#g_coord(2,n) = (j-1)*dz/2 ;
#n=n+1; % increment node counter
#end
#end

#5) Recall that discretization of the governing equation was performed here with nine-node quadrilaterals (see Section 12.3). The array defining the nodes for each element (g_num) can be formed
with the following snippet:
#% establish node numbering for each element
#gnumbers = reshape(1:nn,[ny nx]) ; % grid of node numbers
#g_num = zeros(nod,nels);
iel = 1 ; % intialise element number
for i=1:2:nx-1 % loop over x-nodes
for j=1:2:nz-1 % loop over z-nodes
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

#6) The arrays containing the equation numbers for each node (nf) and the equation numbers for
#each element (g_g) can be formed as done for previous coupled problems. For example, recalling
#that each node has two degrees of freedom (vx and vz) and that the degrees of freedom for each
#element are ordered as vx1
, vz1
, vx2
, vz2
,…, vx9
, vz9
, these two arrays can be constructed with the
following snippet:
% Define equation numbering on nodes
sdof = 0 ; % system degrees of freedom
nf = zeros(ndof,nn) ; % node degree of freedom array
for n = 1:nn % loop over all nodes
for i=1:ndof % loop over each degree of freedom
sdof = sdof + 1 ; % increment total number of equations
nf(i,n) = sdof ; % save equation number for each node
end
end
% Equation numbering for each element
g = zeros(ntot,1) ;
g_g = zeros(ntot,nels);
for iel=1:nels ;
num = g_num(:,iel) ; % extract nodes for the element
Deformation of Earth’s Crust 193
inc=0 ;
for i=1:nod
for k=1:2
inc=inc+1 ;
g(inc) = nf(k,num(i)) ;
end
end
g_g(:,iel) = g ;
end
Note that this numbering scheme differs from the nine-node quadrilateral mesh illustrated in
Figure 7.12 where the two degrees of freedom were ordered as p1, p2, ... p9, c1, c2,…, c9. The actual
ordering scheme is arbitrary; but once one is chosen, it’s important to maintain consistency
throughout the program.

#7) The boundary conditions to be imposed in both experiments are illustrated in Figure 12.3. The
arrays containing the boundary equations (bcdof) and their fixed values (bcval) can be generated
with the following snippet:
% boundary nodes
bx0 = find(g_coord(1,:)==0) ;
bxn = find(g_coord(1,:)==lx) ;
bz0 = find(g_coord(2,:)==0) ;
bzn = find(g_coord(2,:)==lz) ;
% Fixed boundary equations (vx = dof 1, vz = dof2)
% along with their values
bcdof = [nf(1,bx0) nf(1,bxn) nf(2,bz0) ] ;
bcval = [zeros(1,length(bx0)) -bvel*ones(1,length(bxn)) zeros(1,length(bz0)) ] ;
Here, bvel is the imposed velocity at the x = lx boundary. Note that the z = 0 boundary is located
at the base of the model, not at the surface (Figure 12.3).

#8) As noted already, each element is assigned a phase number to incorporate different material properties in different parts of the model. The models here include four different phases, though it’s
straightforward to include more. The phase numbers for the domain illustrated in Figure 12.3 can
be created with the following lines:
% Establish phases
phase = ones(1,nels) ; % upper part of domain
phase(g_coord(2,g_num(9,:))<layer_bot)=2 ; % lower part of domain
phase(find(g_coord(2,g_num(9,:))≤layer_top & ...
g_coord(2,g_num(9,:))≥layer_bot))=3 ; % middle layer
phase(nze*nxe/2-nze/2)=4 ; % central inclusion
Here, the variables layer_bot and layer_top are the z-coordinates at the base (z = 0.45) and
top (z = 0.55) of the central layer and the fourth phase is assigned to a single element located at
the center of the model domain (Figure 12.3).

#9) A small random perturbation is added to the mesh that serves to facilitate development of deformation (folding or shear) instabilities. This is done by randomly shifting the vertical coordinates
at the boundaries of the central layer by a magnitude of up to 5% of the vertical element spacing
dz. This can be achieved with the following lines:
% Perturb boundaries of central layer
ii=find(g_coord(2,:)==layer_bot | g_coord(2,:)==layer_top) ;
g_coord(2,ii)=g_coord(2,ii) + dz*0.05*(rand(1,length(ii))-0.5) ;
194 Applications of the Finite Element Method in Earth Science

#10) The element stiffness matrix and load vectors (see Equations 12.25–12.27) are evaluated in the
presented program by Gauss–Legendre Quadrature using nine integration points (three in each
direction). Before the integrals are evaluated, the locations of the integration points and the
weights must be provided, which can be achieved with the following lines (e.g., see Table 4.1):
% Local coordinates of Gauss integration points for nip=3x3
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

#11) The shape functions and shape functions derivatives that correspond to the nine-node quadrilateral elements can be defined, evaluated at the integration points and saved for later use with the
following snippet (see also Sections 7.3 and 10.2):
% Evaluate shape functions and their derivatives
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
% derivatives of shape functions
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
der(2,2) =-xi*xim*eta ; %dN2deta, etc
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

#12) Once the various system matrices and load vectors have been initialized, it is necessary to perform the element integration and assembly before solving for the nodal velocities. This must be
done within a time loop (since the matrices will generally vary in time), and it must also be done
within a loop involving multiple iterations at each time level, because the load vector contains
loads generated by plastic deformation (see Equation 12.32) that make the problem nonlinear.
The general structure of a nonlinear program is illustrated in Figure 7.13. Within these various
loops, the following major tasks must be performed:
• In a loop over all elements, retrieve the material properties for the current element. This can be
performed using lines such as the following:
smod = smod_v(phase(iel)) ; % shear modulus
K = bmod_v(phase(iel)) ; % bulk modulus
Here, smod_v and bmod_v are vectors containing the shear and bulk moduli with values for
each unit (phase) and phase is an array that contains the phase number (in this case, an integer between 1 and 4; see Figure 12.3). Once the material properties have been obtained, the
viscoelastic material matrix D̃ (dee, Equation 12.45) and the stress matrix Ds (dees, Equation
12.46) can be formed.
• In a loop over integration points, form the kinematic (B) matrix (Equation 12.28) with the lines
% kinematic matrix
bee = zeros(nst,ntot) ;
bee(1,1:2:ntot-1) = deriv(1,:) ;
bee(2,2:2:ntot) = deriv(2,:) ;
bee(3,1:2:ntot-1) = deriv(2,:) ;
bee(3,2:2:ntot) = deriv(1,:) ;
where deriv contains the first spatial derivatives of the shape functions in physical coordinates.
Once B has been constructed, the strain rates and stresses at the current integration point can
be computed with the snippet
strain_rate = bee*uv ; % strain rates
stress = dee*strain_rate+dees*tensor0(:,k,iel) ; % stresses
which follow directly from Equations 12.16 and 12.19, respectively. The vector uv contains the
latest velocity estimation (i.e., vi+1 in Equation 12.31) for the nodes of the current element.
• If plasticity is included, compute the plastic yield function (Equation 12.47). If F > 0, return
stresses to the yield surface using Equations 12.50, 12.51, and 12.52. These various tasks can be
performed with the following snippet:
if plasticity % do only if plasticity is included
tau=(1/4*(stress(1)-stress(2))ˆ2+stress(3)ˆ2)ˆ(1/2); % tau
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
196 Applications of the Finite Element Method in Earth Science
stress(1) = sxx_new ;
stress(2) = szz_new ;
stress(3) = sxz_new ;
end % end of stress return algorithm
end % end of plasticity
• Perform the operations required to integrate the stiffness matrix KM and the right-hand-side
load vector R according to Equations 12.25 and 12.32 with the following lines:
if iters==1 % normal rhs stress
stress_rhs = dees*tensor0(:,k,iel) ;
else % total stress
stress_rhs = stress ;
end
fun = fun_s(:,k) ; % shape functions
fune = zeros(1,ntot) ; % extended shape function array
fune(2:2:ntot) = fun ; % shape function copied into z-dof
dwt = detjac*wts(k) ; % multiplier
KM = KM + bee'*dee*bee*dwt ; % element stiffness matrix
R = R + (bee'*stress_rhs+fune'*rhog)*dwt ; % load vector
Note here that the stress appearing in the load vector (i.e., stress_rhs is computed as Ds𝜎0
for the first iteration (see Equation 12.27), whereas it is the newly computed stresses (i.e., 𝝈, see
Equation 12.32) for all iterations thereafter. Note also that the variable fune used to compute
the gravity loads contains zeros in the positions linked to the x degree of freedom (since gravity
doesn’t act in this direction), whereas it contains the shape functions in positions linked to the
z degree of freedom.
• The final task within the integration loop is to save the integration point stresses, which is
performed with the following line:
tensor(:,k,iel) = stress ; % save stresses at int. point
• After the integration loop has been completed, the left-hand global matrix and the global
right-hand-side load vector are assembled, as done in previous programs. This marks the end of
the element loop.
• After the end of the element loop, boundary conditions must be imposed. With the incremental
formulation presented, the boundary values must be set to the applied boundary velocities
during the first iteration but to the change in boundary velocities (i.e., to zero) during all future
iterations. In Matlab, these tasks are achieved with the following lines:
% apply boundary conditions
lhs(bcdof,:) = 0 ;
tmp = spdiags(lhs,0) ;
tmp(bcdof)=1 ;
lhs=spdiags(tmp,0,lhs);
if iters==1
b(bcdof) = bcval ;
else
b(bcdof) = 0 ;
end
• Compute the new velocity increment by solving Equation 12.30 and update the total velocity
according to Equation 12.31. These tasks can be performed with the following lines:
Deformation of Earth’s Crust 197
Figure 12.4 Folding in a layered viscoelastic material after 40% shortening. The central layer has a viscosity 100
higher than the surrounding matrix. The shaded colors represent mean stress, while the grid shows finite deformation
(which is not the computational mesh). Parameter values are listed in the caption of Figure 12.3.
displ_inc = lhs \ b ; % solve for change in velocity
displ = displ + displ_inc ; % update total velocity
• To evaluate whether the solution has converged, and therefore whether the iteration loop can be
exited, it is necessary to estimate how much the solution changes from one iteration to the next.
A simple means of checking convergence can be obtained by calculating the largest absolute
velocity increment normalized to the total maximum absolute velocity, that is,
error = max(abs(displ_inc))/max(abs(displ)) % error estimate
If this “error” exceeds a certain tolerance, a new iteration must be computed (i.e., return to
point 2); otherwise, the iteration loop can be exited.
13) The final tasks that must be performed before a new time step can be computed are to advect the
mesh coordinates using the newly computed velocity field and to save a copy of the old stresses
needed for the next time step. These tasks can be achieved with the following lines:
% Advect mesh
g_coord(1,:)=g_coord(1,:)+dt*displ(nf(1,:))';% update x-coords
g_coord(2,:)=g_coord(2,:)+dt*displ(nf(2,:))';% update z-coords
tensor0 = tensor ; % save 'old' stresses for next time step
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##

##-----------------------------------------ANOTAÇÕES-------------------------------------------------------------------
##PROCURAR NO CAP.12 AS SEGUINTES SEÇÕES DOS CÓDIGOS:
##
##computacional
##bound
##boundary nodes
##------------------------------------------BOUNDARY NODES----------------------------------------------------------------
##% boundary nodes
bx0 = find(g_coord(1,:)==0) ;
bxn = find(g_coord(1,:)==lx) ;
bz0 = find(g_coord(2,:)==0) ;
bzn = find(g_coord(2,:)==lz) ;
% Fixed boundary equations (vx = dof 1, vz = dof2)
% along with their values
bcdof = [nf(1,bx0) nf(1,bxn) nf(2,bz0) ] ;
bcval = [zeros(1,length(bx0)) -bvel*ones(1,length(bxn)) zeros(1,length(bz0)) ] ;
##-----------------------------------------------------------------------------------------------------------------

##------------------------------------------FIXED BOUNDARIES----------------------------------------------------------------
##% fixed boundary equations along with their values
bcdof = [ nf(1,bx0) nf(1,bxn) nf(2,bz0) ] ;
bcval = [ zeros(1,length(bx0)) -bvel*ones(1,length(bxn)) zeros(1,length(bz0)) ] ;
##
##-----------------------------------------------------------------------------------------------------------------
##ploting of results
##--------------------------------PLOTING OF RESULTS---------------------------------------------------------------
##% plotting of results
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
title('Mean stress (MPa)'
##---------------------------------------------------------------------------
##dogbone -> figura 12.5
##phy parameters
##----------------------------------------------------------------------------
##%---------------------------------------------
% Program: deformation_vep2d.m
% 2D viscoelastoplastic plane strain deformation
% solid mechanics rate formulation
% 9-node quadrilaterals
%-----------------------------------------
clear
seconds_per_year = 60*60*24*365 ;
% physical parameters
lx = 4 ; % length of x domain (m)
lz = 1 ; % length of z domain (m)
emod_v = [1 1 1 1 ]*1e11 ; % Young's modulus (Pa)
pois_v = [0.3 0.3 0.3 0.3 ] ; % Poisson's ratio
smod_v = emod_v./(2*(1+pois_v)) ; % shear modulus (Pa)
bmod_v = emod_v./(3*(1-2*pois_v)) ; % bulk modulus (Pa)
visc_v = [ 1 1 100 100]*1e18 ; % shear viscosity (Pas)
coh_v = 20*[1 1 1 0.9 ]*1e6 ; % cohesion (Pa)
phi_v = ([1 1 1 1])*30*pi/180 ; % friction angle(radians)
psi_v = [0 0 0 0] ; % dilation angle (radians)
grav_v = [ 2700 2700 2700 2700 ]*9.8 ; % density*gravity
bvel = 5e-3/seconds_per_year ; % boundary velocity (m/s)
edot = bvel/lx ; % initial strain rate (1/s)
plasticity= logical(0) ; % include/ignore plasticity (true/false)
nst = 3 ; % number of stress/strain components
##
##
##
##----------------------------------------------------------------------------
##