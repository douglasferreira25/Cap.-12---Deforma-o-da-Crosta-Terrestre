#--------------------------------------------------------------------------








#----------------------------------------------
#------------código a ser executado-----------
# import preprocess, process, posprocess
# from preprocess import main
# ndim, nod, nodes, conn, boundary = preprocess.main("geofem.txt")
#
# # analysis = "geomechanics"
# #
# # preprocess.read_asc_file("dem.asc")
# #
# # if analysis == "flowaroundfaults" :
# #     process.flowaroundfaults(ndim, nod, nodes, conn, boundary)
# #
# # if analysis == "thermalanalysis" :
# #     process.thermalanalysis(ndim, nod, nodes, conn, boundary)
# #
# # if analysis == "geomechanics" :
# #     process.geomechanics(ndim, nod, nodes, conn, boundary)
#
#
# #posprocess.main(ndim, nod, nodes, conn, boundary)
#
#
#
# #--------------------------------------------------------------------------
#
#
#






#----------------------------------------------
#------------código a ser executado-----------
#-------------------------------------------
# Program: deformation_vep2d.m
#% 2D viscoelastoplastic plane strain deformation
# solid mechanics rate formulation
# 9-node quadrilaterals
#-----------------------------------------
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

# rescaled parameters
emod_v = emod_v/stress_scale
smod_v = smod_v/stress_scale
bmod_v = bmod_v/stress_scale
visc_v = visc_v/(stress_scale*time_scale)
coh_v = coh_v/stress_scale
grav_v = grav_v*length_scale/stress_scale
lx = lx/length_scale
lz = lz/length_scale
bvel = bvel/length_scale*time_scale
edot = edot*time_scale
layer_top = 0.55  # z coord. (nondimensional) at top of layer
layer_bot = 0.45  # z coord. (nondimensional) at bottom of layer
# numerical parameters
ntime = 100  # number of time steps to perform
nxe = 40  # n elements in x-direction
nze = 40  # n elements in z-direction
nels = nxe*nze  # total number of elements
nx = 2*nxe+1  # n of nodes in x-direction
nz = 2*nze+1  # n of nodes in z-direction
nn = nx*nz  # total number of nodes
dx = lx/nxe  # element size in x-direction
dz = lz/nze  # element size in z-direction
nod = 9  # number of nodes
ndof = 2  # number of degrees of freedom
ntot = nod * ndof  # total degrees of freedom in an element
nip = 9  # number of integration points in an element
eps = 1e-3  # convergence tolerance
limit = 50  # maximum number of iterations
dt = 0.01  # time step (non dimensional)
#-----------------------------------------
# generate coordinates and nodal numbering
#-----------------------------------------
# computational mesh
# define mesh (numbering in z direction)
g_coord = [[(i-1)*dx/2, (j-1)*dz/2] for i in range(1, nx+1) for j in range(1, nz+1)]
# establish node numbering numbering
gnumbers = [[i for i in range(1, nx+1)] for j in range(1, nz+1)]
iel = 1
g_num = []
for i in range(1, nx, 2):
    for j in range(1, nz, 2):
        g_num.append([
            gnumbers[j-1][i-1], #node 1
            gnumbers[j][i-1], # node 2
            gnumbers[j+1][i-1], # node 3
            gnumbers[j+1][i], # node 4
            gnumbers[j+1][i+1], # node 5
            gnumbers[j][i+1], # node 6
            gnumbers[j-1][i+1], # node 7
            gnumbers[j-1][i], # node 8
            gnumbers[j][i] # node 9
            ])
        iel += 1
#-----------------------------------------
# create global-local connection arrays
#-----------------------------------------
sdof = 0  # system degrees of freedom
nf = [[0 for i in range(nn)] for j in range(ndof)]  # node degree of freedom array
for n in range(nn):  # loop over all nodes
    for i in range(ndof):  # loop over each degree of freedom
        sdof += 1  # increment total number of equations
        nf[i][n] = sdof  # record the equation number for each node
# equation numbering for each element
g = [0 for i in range(ntot)]
g_g = np.array([[0 for i in range(nels)] for j in range(ntot)])
for iel in range(nels):
    num = g_num[iel]  # extract nodes for the element
    inc = 0
    for i in range(nod):
        for k in range(2):
            inc += 1
            g[inc-1] = nf[k][num[i]-1]
#    g_g[:, iel] = g
#-----------------------------------------
# define boundary conditions
#-----------------------------------------
# boundary nodes
bx0 = [i for i in range(nn) if g_coord[i][0] == 0]
bxn = [i for i in range(nn) if g_coord[i][0] == lx]
bz0 = [i for i in range(nn) if g_coord[i][1] == 0]
bzn = [i for i in range(nn) if g_coord[i][1] == lz]
# fixed boundary equations along with their values
bcdof = [nf[0][i] for i in bx0] + [nf[0][i] for i in bxn] + [nf[1][i] for i in bz0]
bcval = [0 for i in bx0] + [-bvel for i in bxn] + [0 for i in bz0]
print(g_coord)
#------------------------------------------------------------
# establish phases
#-----------------------------------------------------------
phase = [1 for i in range(nels)]  # top
for i in range(nels):
    if g_coord[g_num[i][8]-1][1] < layer_bot:
        phase[i] = 2  # bottom
    elif g_coord[g_num[i][8]-1][1] <= layer_top and g_coord[g_num[i][8]-1][1] >= layer_bot:
        phase[i] = 3  # layer
phase[nze*nxe//2-nze//2] = 4  # inclusion
#-------------------------------------------
# perturb boundaries of layer
#-------------------------------------------
iit = [i for i in range(nn) if g_coord[i][1] == layer_bot]  # nodes on top of layer
iib = [i for i in range(nn) if g_coord[i][1] == layer_top]  # nodes on bottom of layer
ii = iit + iib  # combined nodes
import random
g_coord = [[g_coord[i][0], g_coord[i][1]+dz*0.05*(random.random()-0.5)] if i in ii else g_coord[i] for i in range(nn)]
#-----------------------------------------
# integration data and shape functions
#----------------------------------------
# local coordinates of Gauss integration points for nip=3x3
import math
points = np.array([[-math.sqrt(0.6), math.sqrt(0.6), -math.sqrt(0.6)], [0, 0, math.sqrt(0.6)]])
#print(points.shape)
points = np.tile(points,(1,3))
print(points.shape)
#erro detectado - só ha 2 vetores - é preciso 9 vetores
# Gauss weights for nip=3x3
w = [5/9, 8/9, 5/9]
v = [[5/9*w[i] for i in range(3)], [8/9*w[i] for i in range(3)], [5/9*w[i] for i in range(3)]]
wts = [v[i][j] for i in range(3) for j in range(3)]
# evaluate shape functions and their derivatives
# at integration points and save the results
fun_s = np.zeros((nip,nod))
#print(nod)
der_s = np.zeros((ndof,nip,nod))
for k in range(nod):
    #print(nip, points)
    xi = points[0][k]
    eta = points[1][k]
    etam = eta - 1
    etap = eta + 1
    xim = xi - 1
    xip = xi + 1
    x2p1 = 2*xi + 1
    x2m1 = 2*xi - 1
    e2p1 = 2*eta + 1
    e2m1 = 2*eta - 1
    # shape functions
    fun = [0.25*xi*xim*eta*etam, -0.5*xi*xim*etap*etam, 0.25*xi*xim*eta*etap, -0.5*xip*xim*eta*etap, 0.25*xi*xip*eta*etap, -0.5*xi*xip*etap*etam, 0.25*xi*xip*eta*etam, -0.5*xip*xim*eta*etam, xip*xim*etap*etam]
    # first derivatives of shape functions
    der = np.array([[0.25*x2m1*eta*etam, -0.5*x2m1*etap*etam, 0.25*x2m1*eta*etap, -xi*eta*etap, 0.25*x2p1*eta*etap, -0.5*x2p1*etap*etam, 0.25*x2p1*eta*etam, -xi*eta*etam, 2*xi*etap*etam], [0.25*xi*xim*e2m1, -xi*xim*eta, 0.25*xi*xim*e2p1, -0.5*xip*xim*e2p1, 0.25*xi*xip*e2p1, -xi*xip*eta, 0.25*xi*xip*e2m1, -0.5*xip*xim*e2m1, 2*xip*xim*eta]])
    #print(fun_s,np.ndim(der_s))    
    for i in range(nip):    
        fun_s[i, k] = fun [i]  # save shape functions
    for i in range(ndof):
        for j in range (nip):
            der_s[i,j,k] = der[i,j]  # save derivatives
#print(der_s.shape)
#-----------------------------------------
# initialisation
#-----------------------------------------
lhs = np.zeros((sdof, sdof))  # global stiffness matrix np.zeros((sdof, sdof)) # codigo antigo lhs = [[0 for i in range(sdof)] for j in range(sdof)]
b = np.zeros(sdof)  # global rhs vector np.zeros(sdof)# codigo antigo b = [0 for i in range(sdof)]
displ = np.zeros(sdof)  # solution vector (velocities) np.zeros(sdof) #displ = [0 for i in range(sdof)]
tensor = np.zeros((nels,nip,nst))  # stresses at integration points np.zeros((nels,nip,nst)) #tensor = [[[0 for i in range(nels)] for j in range(nip)] for k in range(nst)]
tensor0 = np.zeros((nels,nip,nst))  # old stresses np.zeros((nels,nip,nst))#tensor0 = [[[0 for i in range(nels)] for j in range(nip)] for k in range(nst)]
#----------------------------------------------------
# loading loop
#----------------------------------------------------
time = 0  # initialise time
for n in range(ntime):
    time = time + dt  # increment time
    #------------------------------------------
    # iterations
    #------------------------------------------
    error = eps*2  # initialise error
    iters = 0  # initialise iteration counter
    displ = [0 for i in range(sdof)]  # initialise solution vector
    while error > eps and iters < limit:
        iters = iters + 1  # increment iteration counter
        lhs = [[0 for i in range(sdof)] for j in range(sdof)]  # initialised global stiffness matrix
        b = [0 for i in range(sdof)]  # initialised global load vector
        #-----------------------------------------
        # element integration and assembly
        #-----------------------------------------
        for iel in range(nels):  # sum over elements
            num = g_num[iel]  # list of element nodes
            g = g_g[:, iel]  # element equation numbers
            coord = np.array([g_coord[num[i]-1] for i in range(nod)])  # nodal coordinates
            KM = [[0 for i in range(ntot)] for j in range(ntot)]  # initialise stiffness matrix
            R = [0 for i in range(ntot)]  # initialise stress load vector
            uv = [displ[g[i]-1] for i in range(ntot)]  # current nodal velocities
            # retieve material properties for the current element
            smod = smod_v[phase[iel]-1]  # shear modulus
            K = bmod_v[phase[iel]-1]  # bulk modulus
            mu = visc_v[phase[iel]-1]  # viscosity
            phi = phi_v[phase[iel]-1]  # friction coeff.
            psi = psi_v[phase[iel]-1]  # dilation angle
            coh = coh_v[phase[iel]-1]  # cohesion
            rhog = grav_v[phase[iel]-1]  # rho x gravity
            # viscoelastic material matrix
            di = dt*(3*mu*K + 3*dt*smod*K + 4*smod*mu)
            od = dt*(-2*smod*mu + 3*mu*K + 3*dt*smod*K)
            d = 3*(mu + dt*smod)
            ed = 2*mu*dt*smod/(2*mu + dt*smod)
            dee = [[di/d, od/d, 0], [od/d, di/d, 0], [0, 0, ed]]
            # stress matrix
            di = 3*mu + dt*smod
            od = dt*smod
            ed = 2 * mu / (2 * mu + dt * smod)
            dees = [[di/d, od/d, 0], [od/d, di/d, 0], [0, 0, ed]]
            for k in range(nip):  # integration loop
                fun = fun_s[:, k]  # shape functions
                fune = [0 for i in range(ntot)]  # extended shape function array
                fune[1::2] = fun  # shape function for z-dof only
                # shape functions in local coords
                der = der_s[:, :, k]
                print(der)
                print(coord)
                #jac = np.array([[sum([der[i][j]*coord[j][k] for j in range(nod)]) for k in range(2)] for i in range(ndof)])  # jacobian matrix
                #print(jac.shape)
                jac = der@coord
                #print(jac)
                detjac = jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]  # det. of the Jacobian
                dwt = detjac*wts[k]  # detjac x weight
                invjac = [[jac[1][1]/detjac, -jac[0][1]/detjac], [-jac[1][0]/detjac, jac[0][0]/detjac]]  # inverse of the Jacobian
                deriv = [[sum([invjac[i][j]*der[j][k] for j in range(nod)]) for k in range(ndof)] for i in range(ndof)]  # shape functions in physical coords
                bee = [[0 for i in range(ntot)] for j in range(nst)]  # kinematic matrix
                bee[0][0::2] = deriv[0]
                bee[1][1::2] = deriv[1]
                bee[2][0::2] = deriv[1]
                bee[2][1::2] = deriv[0]
                strain_rate = [sum([bee[i][j]*uv[j] for j in range(ntot)]) for i in range(nst)]  # strain rates
                stress = [sum([dee[i][j]*strain_rate[j] for j in range(nst)]) + sum([dees[i][j]*tensor0[j][k][iel] for j in range(nst)]) for i in range(nst)]  # stresses
                if plasticity:  # do only if plasticity included
                    tau = ((1/4*(stress[0]-stress[1])*2+stress[2])*(1/2))  # tau star
                    sigma = 1/2*(stress[0]+stress[1])  # sigma star
                    F = tau + sigma*math.sin(phi)-coh*math.cos(phi)  # plastic yield function
                    if F > 0:  # return stresses to yield surface
                        if sigma <= coh/math.tan(phi):  # 'normal' case
                            beta = abs(coh*math.cos(phi)-math.sin(phi)*sigma)/tau
                            sxx_new = sigma + beta*(stress[0]-stress[1])/2
                            szz_new = sigma - beta*(stress[0]-stress[1])/2
                            sxz_new = beta*stress[2]
                        else:  # special treatment for corners of yield surface
                            sxx_new = coh/math.tan(phi)
                            szz_new = coh/math.tan(phi)
                            sxz_new = 0
                        stress[0] = sxx_new
                        stress[1] = szz_new
                        stress[2] = sxz_new
                if iters == 1:  # normal rhs stress
                    stress_rhs = [dees[i][j]*tensor0[j][k][iel] for i in range(nst)]
                else:  # total stress
                    stress_rhs = stress
                KM = [[KM[i][j] + sum([bee[i][l]*dee[l][m]*bee[j][m]*dwt for l in range(nst)]) for j in range(ntot)] for i in range(ntot)]  # element stiffness matrix
                R = [R[i] + (sum([bee[i][j]*stress_rhs[j] for j in range(nst)]) + sum([fune[i]*rhog for i in range(ntot)]))*dwt for i in range(ntot)]  # load vector
                tensor[:, k, iel] = stress  # stresses at integration point
            # assemble global stiffness matrix and rhs vector
            for i in range(ntot):
                for j in range(ntot):
                    lhs[g[i]-1][g[j]-1] += KM[i][j]  # stiffness matrix
            for i in range(ntot):
                b[g[i]-1] -= R[i]  # load vector
        #-------------------------------------------------
        # implement boundary conditions and solve system
        #-------------------------------------------------
        # apply boundary conditions
        for i in bcdof:
            lhs[i-1] = [0 for j in range(sdof)]
            lhs[i-1][i-1] = 1
        if iters == 1:
            b = [bcval[i-1] if i in bcdof else b[i-1] for i in range(sdof)]
        else:
            b = [0 if i in bcdof else b[i-1] for i in range(sdof)]
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        displ_inc = spsolve(csc_matrix(lhs), b)  # solve for velocity increment
        displ = [displ[i] + displ_inc[i] for i in range(sdof)]  # update total velocity
        error = max(abs(displ_inc))/max(abs(displ))  # estimate error
    #----------------------------------------------
    # end of iterations
    #------------------------------------------------
    # update mesh coordinates
    g_coord = [[g_coord[i][0] + dt*displ[nf[0][i]-1], g_coord[i][1] + dt*displ[nf[1][i]-1]] for i in range(nn)]
    tensor0 = tensor  # save stresses
    shortening_percent = (1 - (max([g_coord[i][0] for i in range(nn)])-min([g_coord[i][0] for i in range(nn)]))/lx)*100
    #---------------------------------
    # plotting of results
    #---------------------------------
    xgrid = [[g_coord[i][0] for i in range(nn)][j::8] for j in range(8)]  # m
    zgrid = [[g_coord[i][1] for i in range(nn)][j::8] for j in range(8)]  # m
    c = length_scale/time_scale*seconds_per_year*1e3
    u_solution = [[displ[nf[0][i]-1]*c for i in range(nn)][j::8] for j in range(8)]  # (mm/yr)
    v_solution = [[displ[nf[1][i]-1]*c for i in range(nn)][j::8] for j in range(8)]  # (mm/yr)
    tensor_scaled = [[[tensor[i][j][k]*stress_scale/1e6 for i in range(nst)] for j in range(nip)] for k in range(nels)]  # MPa
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()  # mesh
    plt.plot(xgrid, zgrid, 'b')
    plt.plot(list(map(list, zip(*xgrid))), list(map(list, zip(*zgrid))), 'b')
    plt.plot([g_coord[i][0] for i in iit], [g_coord[i][1] for i in iit], 'r')
    plt.plot([g_coord[i][0] for i in iib], [g_coord[i][1] for i in iib], 'r')
    plt.axis('equal')
    plt.title('Deformed mesh after ' + str(shortening_percent) + ' % shortening')
    plt.figure(2)
    plt.clf()  # velocity field
    plt.quiver(xgrid, zgrid, u_solution, v_solution)
    plt.axis('equal')
    plt.title('Velocity vector field')
    plt.figure(3)
    plt.clf()  # x-velocity
    plt.pcolor(xgrid, zgrid, u_solution)
    #plt.figure.colormap('jet')
    plt.colorbar()
    #plt.shading('interp')
    plt.axis('equal')
    plt.title('x-velocity (mm/yr)')
    plt.figure(4)
    plt.clf()  # z-velocity
    plt.pcolor(xgrid, zgrid, v_solution)
    #plt.figure.colormap('jet')
    plt.colorbar()
    #plt.shading('interp')
    plt.axis('equal')
    plt.title('z-velocity (mm/yr)')
    plt.figure(5)
    plt.clf()  # plot negative mean stress
    for iel in range(nels):
        num = g_num[iel]
        coord = [g_coord[num[i]-1] for i in range(8)]
        means = [(tensor[0][j][iel]+tensor[1][j][iel])/2 for j in range(nip)]
        nodevalues = [fun_s[j].dot(means) for j in range(8)]
        h = plt.fill([coord[i][0] for i in range(8)], [coord[i][1] for i in range(8)], [-nodevalues[i] for i in range(8)])
        #plt.set(h, 'linestyle', 'none')
    plt.plot([xgrid[i][0] for i in range(8)], [zgrid[i][0] for i in range(8)], 'Color', [0.8, 0.8, 0.8])
    plt.plot([xgrid[i][j] for i in range(8) for j in range(8)], [zgrid[i][j] for i in range(8) for j in range(8)], 'Color', [0.8, 0.8, 0.8])
    plt.axis('equal')
    plt.colorbar()
    plt.title('Mean stress (MPa)')
    #------------------------------------------------------
# end of time integration
#----------------------------------------------------
