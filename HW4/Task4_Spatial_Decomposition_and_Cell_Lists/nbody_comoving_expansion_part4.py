"""
Create Your Own N-body Simulation (With Python)
Adapted from Philip Mocz (2020) Princeton University, @PMocz
New structure, Initial condition module, movie module, rotational analysis, comments: C.Welker

Simulate orbits of particles interacting only through gravitational interactions.
The code calculates pairwise forces according to Newton's Law of Gravity. 
Note that there is no expansion yet in this simulation. We are focusing on a patch decoupled from
expansion. Let's see how long it takes to virialize once a structure is collapsing.
Of course in a real simulation with billions of particles, we will need some better approximation 
to NOT calculate all pairwise interactions (see optional exercise of Homework 4) 
But for now, let's focus on a few hundreds particles
"""
############################
# MOD (John): header/docstring differs in 11.py (keeping original)
#############################
# --- 11.py version (commented out) ---
# #!/usr/bin/env python3
############################
# END MOD (John)
#############################
import numpy as np
############################
# MOD (John): replace block 2
#############################
# --- ORIGINAL (from orig_script.py) ---
# import matplotlib.pyplot as plt
# import os
# import moviepy.video.io.ImageSequenceClip
# from natsort import natsorted
#
# def InitialConditions(N,omega,mtot,v0):
#
#     # --------------This module generates Initial Conditions
#     #For N particles with total mass mtot, solid angular velocity omega
#
#     np.random.seed(17)            # set the random number generator seed
#
#     mass = mtot*np.ones((N,1))/N  # total mass of particles is mtot. all particles have the same mass here.
#     pos  = np.random.randn(N,3)   # randomly selected positions from a normal distribution. 
#    #Could be modified to take into account the initial density profile of the halo.
#
#     if(v0==1):
#         # for solid rotation: Vrot=radius*omega along e_theta. Let's calculate the radii of particles first
#         x = pos[:,0:1]
#         y = pos[:,1:2]
#         z = pos[:,2:3]
#         # in the frame of the centre of mass
#         x -= np.mean(mass * x) / np.mean(mass)
#         y -= np.mean(mass * y) / np.mean(mass)
#         z -= np.mean(mass * z) / np.mean(mass)
#         #polar coordinates (r, theta,z). We consider z the axis of rotation
#         norm_r=np.sqrt(x**2 + y**2)
#         theta=np.arctan(y/x)
#         #total rotational velocity, tangential to radius. Let assume the axis of rotation is z
#         vrot=norm_r*omega
#         vrot_x=-vrot*np.sin(theta)
#         vrot_y=vrot*np.cos(theta)
#         vrot_z=np.zeros(N)
#         vrot_z=vrot_z.reshape(N,1)
#         vrot=np.hstack((vrot_x,vrot_y,vrot_z))
#
#         vel  =  np.random.randn(N,3) 
#         # in the frame of the centre of mass
#         vel -= np.mean(mass * vel,0) / np.mean(mass)
#         #vrot and random variation
#         vel=vel+vrot
#         # --------------
#     else:
#         #zero initial velocities in the frame of reference of the halo
#         vel=np.zeros((N,3))    
#        # --------------
#
#
#     return mass, pos, vel
#
# def ScaleFactor(t):
#     # --------------This module computes a scale factor at time t in the LCDM, matter dominated era.
#     # useful if you want to include expansion
#     a=1.0 #modify this function
# --- MODIFIED (from 11.py) ---
import time

G = 1.0

def wrap_periodic(x, box_min, box_max):
    L = box_max - box_min
    return box_min + np.mod(x - box_min, L)

def assign_particles_to_cells(x, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng
    u = (x - box_min) / h
    ijk = np.floor(u).astype(int)
    ijk = np.clip(ijk, 0, Ng - 1)
    cid = ijk[:, 0] + Ng * ijk[:, 1] + (Ng * Ng) * ijk[:, 2]
    cell_lists = [[] for _ in range(Ng**3)]
    for p, c in enumerate(cid):
        cell_lists[c].append(p)
    return ijk, cid, cell_lists

def cid_to_ijk(cid, Ng):
    i = cid % Ng
    j = (cid // Ng) % Ng
    k = cid // (Ng * Ng)
    return i, j, k

def ijk_to_cid(i, j, k, Ng):
    return i + Ng*j + (Ng*Ng)*k

def grid_sanity(cell_lists):
    counts = np.array([len(lst) for lst in cell_lists])
    occupied = int(np.count_nonzero(counts))
    max_in_cell = int(counts.max()) if counts.size else 0
    mean_occ = float(counts[counts > 0].mean()) if occupied > 0 else 0.0
    return occupied, max_in_cell, mean_occ

def build_density_grid(cell_lists, m, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng
    rho = np.zeros((Ng, Ng, Ng), dtype=float)
    for cid, plist in enumerate(cell_lists):
        if not plist:
            continue
        i, j, k = cid_to_ijk(cid, Ng)
        rho[i, j, k] = np.sum(m[plist]) / (h**3)
    return rho

def mass_conservation_check(rho, m, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng
    M_grid = float(np.sum(rho) * (h**3))
    M_true = float(np.sum(m))
    return M_grid, M_true, abs(M_grid - M_true)

def k_grids(Ng, L):
    k1 = 2.0 * np.pi * np.fft.fftfreq(Ng, d=L/Ng)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    return kx, ky, kz

def mesh_accel_filtered(rho, box_min, box_max, r_split):
    """
    Long-range mesh force using Gaussian low-pass filter in k-space:
      W(k) = exp(-(k r_split)^2)
    """
    Ng = rho.shape[0]
    L = box_max - box_min

    rho_k = np.fft.fftn(rho)
    kx, ky, kz = k_grids(Ng, L)
    k2 = kx**2 + ky**2 + kz**2
    k = np.sqrt(k2)

    W = np.exp(-(k * r_split)**2)

    # Poisson in k-space for filtered density
    phi_k = np.zeros_like(rho_k, dtype=complex)
    mask = k2 > 0.0
    phi_k[mask] = -4.0 * np.pi * G * (rho_k[mask] * W[mask]) / k2[mask]
    phi_k[~mask] = 0.0 + 0.0j

    # spectral accel: a = -∇phi
    ax_k = -(1j * kx) * phi_k
    ay_k = -(1j * ky) * phi_k
    az_k = -(1j * kz) * phi_k

    ax = np.fft.ifftn(ax_k).real
    ay = np.fft.ifftn(ay_k).real
    az = np.fft.ifftn(az_k).real
    return ax, ay, az, phi_k

def sample_mesh_accel_NGP(x, ax, ay, az, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng
    u = (x - box_min) / h
    ijk = np.floor(u).astype(int)
    ijk = np.clip(ijk, 0, Ng - 1)

    a = np.zeros_like(x)
    a[:, 0] = ax[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    a[:, 1] = ay[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    a[:, 2] = az[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
############################
# END MOD (John)
#############################
    return a

############################
# MOD (John): replace block 3
#############################
# --- ORIGINAL (from orig_script.py) ---
# def getAcc( pos, mass, G, softening ):
#     """
#     Calculate the acceleration on each particle due to Newton's Law 
#     pos  is an N x 3 matrix of positions
#     mass is an N x 1 vector of masses
#     G is Newton's Gravitational constant
#     softening is the softening length
#     a is N x 3 matrix of accelerations
#     NOTE: You can see that everything is put in matrix form, allowing for matrix operations rather than looping over particles 
#     to get each update. This is not just because it looks cool to do all calculations in only one line rather than a FOR loop. 
#     It  also significantly improves the computational performance in python!!
#     """
#     # positions r = [x,y,z] for all particles
#     x = pos[:,0:1]
#     y = pos[:,1:2]
#     z = pos[:,2:3]
#
#     # matrix that stores all pairwise particle separations: r_j - r_i
#     dx = x.T - x
#     dy = y.T - y
#     dz = z.T - z
#
#     # matrix that stores 1/r^3 for all particle pairwise particle separations 
#     """
#     You can see that we included a "softening term". Its goal is to avoid getting an (near)infinite value when distance
#     between two particles is ~ zero. It can happen in simulations where resolution, 
#     number of particles and float precision is limited but would be unphysical. 
#     We've seen that close-encounter collisions are rather irrelevant in collisionless systems likes haloes.
#     So softening is essentially a user-defined resolution limit for numerical gravity. 
#     """
#     inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
#     inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)
#
#     # acceleration under gravity (Newton's second law) (notice we are calculating vec(r)/r^3 instead or 1/r^2 as we need the
#     #direction of the force for each pair of particles)
#     ax = G * (dx * inv_r3) @ mass
#     ay = G * (dy * inv_r3) @ mass
#     az = G * (dz * inv_r3) @ mass
#
#     # pack together the acceleration components (hstack performs a concatenation)
#     a = np.hstack((ax,ay,az))
# --- MODIFIED (from 11.py) ---
def neighbor_cids_for_cell(cid0, Ng):
    i0, j0, k0 = cid_to_ijk(cid0, Ng)
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                i = (i0 + di) % Ng
                j = (j0 + dj) % Ng
                k = (k0 + dk) % Ng
                out.append(ijk_to_cid(i, j, k, Ng))
    return out

def accel_local_PP(x, m, eps, box_min, box_max, Ng, ijk, cell_lists):
    """
    Short-range PP: interactions only inside the 27-cell neighborhood.
    Uses minimum-image convention for periodic displacement.
    """
    N = x.shape[0]
    Lbox = box_max - box_min
    a = np.zeros_like(x)
    cid = ijk[:, 0] + Ng * ijk[:, 1] + (Ng * Ng) * ijk[:, 2]

    for p in range(N):
        c0 = cid[p]
        neigh = neighbor_cids_for_cell(c0, Ng)
        xp = x[p]
        ap = np.zeros(3)

        for c in neigh:
            for q in cell_lists[c]:
                if q == p:
                    continue
                r = x[q] - xp
                r = r - Lbox * np.round(r / Lbox)  # minimum-image
                dist2 = np.dot(r, r) + eps**2
                ap += G * m[q] * r / dist2**1.5

        a[p] = ap
############################
# END MOD (John)
#############################

    return a

############################
# MOD (John): replace block 4
#############################
# --- ORIGINAL (from orig_script.py) ---
# def getEnergy( pos, vel, mass, G ):
#     """
#     Get kinetic energy (KE) and potential energy (PE) of simulation
#     pos is N x 3 matrix of positions
#     vel is N x 3 matrix of velocities
#     mass is an N x 1 vector of masses
#     G is Newton's Gravitational constant
#     KE is the kinetic energy of the system
#     PE is the potential energy of the system
#     """
#     # Kinetic Energy:
#     KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
#     # Potential Energy:
#
#     # positions r = [x,y,z] for all particles
#     x = pos[:,0:1]
#     y = pos[:,1:2]
#     z = pos[:,2:3]
#
#     # matrix that stores all pairwise particle separations: r_j - r_i. Note that each pair appears twice: dx(i,j)=-dx(j,i)
#     dx = x.T - x
#     dy = y.T - y
#     dz = z.T - z
#
#     # matrix that stores 1/r for all particle pairwise particle separations 
#     inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
#     inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
#
#     # sum over upper triangle, to count each interaction only once
#     PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
#
#     #Radial kinetic energy: first Convert to Center-of-Mass frame
#     x -= np.mean(mass * x) / np.mean(mass)
#     y -= np.mean(mass * y) / np.mean(mass)
#     z -= np.mean(mass * z) / np.mean(mass)
#     norm_r=np.sqrt(x**2 + y**2 + z**2)
#     x1=x/norm_r
#     y1=y/norm_r
#     z1=z/norm_r
#     r=np.hstack((x1,y1,z1))
#     vel_r=np.sum(vel*r,axis=1)
#     vr2=vel_r**2
#     N=vr2.shape[0]
#     vr2=np.reshape(vr2,(N,1))
#     KE_rad = 0.5 * np.sum(mass * vr2,axis=0 )
#     KE_rad=KE_rad.reshape(1)[0]
#
#     #Orbital kinetic energy
#     KE_orb=KE-KE_rad
#
#     return KE, PE, KE_rad, KE_orb
# --- MODIFIED (from 11.py) ---
def leapfrog_step_hybrid_split(x, v, m, dt, eps, box_min, box_max, Ng, r_split):
    """
    a_total = a_mesh_long + a_PP_local
    where a_mesh_long comes from filtered density (low-pass in k-space).
    """
    # cell lists + rho
    ijk, cid, cell_lists = assign_particles_to_cells(x, box_min, box_max, Ng)
    rho = build_density_grid(cell_lists, m, box_min, box_max, Ng)

    # mesh long-range accel
    ax, ay, az, phi_k = mesh_accel_filtered(rho, box_min, box_max, r_split)
    a_mesh = sample_mesh_accel_NGP(x, ax, ay, az, box_min, box_max, Ng)

    # local PP short-range accel
    a_pp = accel_local_PP(x, m, eps, box_min, box_max, Ng, ijk, cell_lists)

    a = a_mesh + a_pp

    # leapfrog
    v_half = v + 0.5 * dt * a
    x_new = wrap_periodic(x + dt * v_half, box_min, box_max)

    # accel at new positions
    ijk2, cid2, cell_lists2 = assign_particles_to_cells(x_new, box_min, box_max, Ng)
    rho2 = build_density_grid(cell_lists2, m, box_min, box_max, Ng)
    ax2, ay2, az2, phi_k2 = mesh_accel_filtered(rho2, box_min, box_max, r_split)
    a_mesh2 = sample_mesh_accel_NGP(x_new, ax2, ay2, az2, box_min, box_max, Ng)
    a_pp2 = accel_local_PP(x_new, m, eps, box_min, box_max, Ng, ijk2, cell_lists2)

    a2 = a_mesh2 + a_pp2
    v_new = v_half + 0.5 * dt * a2

    # diagnostics
    max_im_phi = float(np.max(np.abs(np.fft.ifftn(phi_k2).imag)))
    return x_new, v_new, rho2, max_im_phi
############################
# END MOD (John)
#############################

def main():
############################
# MOD (John): replace block 5
#############################
# --- ORIGINAL (from orig_script.py) ---
#     """ N-body simulation """
#
#     # Simulation parameters ----------------------------------------------------
#     N         = 20    # Number of particles
#     t         = 0      # current time of the simulation
#     tEnd      = 10.0   # time at which simulation ends
#     dt        = 0.01   # timestep
#     softening = 0.1    # softening length
#     G         = 1.0    # Newton's Gravitational Constant. Here set to 1 in code units for covenience.
#     mtot      = 20.0  # Total mass of the object
#     plotRealTime = False # switch on for plotting as the simulation goes along
#     omega        = 0.2  # initial angular velocity if solid rotation
#     v0           =1.0   #if not 1.0, set initial velocitites to 0
#     #-------------------------------------------------------------------------------
#
#
#     # Set where to store outputs-------------------------------------------------------------------------------    
#     # Create a directory for the Simulation
#     directory = "MyNbodyRun_Om"+str(omega)+"_N"+str(N)+"part"
#     # Parent Directory path: change to your own path
#     parent_dir = "/Users/johnmeftah/Desktop/sim_test"
#     isdir = os.path.isdir(parent_dir)
#     if isdir==False: # create parent directory if it does not exist.
#         os.mkdir(parent_dir)
#     # Make complete Path
#     path = os.path.join(parent_dir, directory)
#     #Create directory
#     isdir = os.path.isdir(path)
#     if isdir==False:
#         os.mkdir(path)
#     #-----------------------------------------------------------------------------
#
#     #Now let's run the simulation!
#     mass,pos,vel = InitialConditions(N,omega,mtot,v0) #load initial conditions
#
#     # calculate initial gravitational accelerations
#     acc = getAcc( pos, mass, G, softening )
#
#     # calculate initial energy of system
#     KE, PE, KE_rad, KE_orb  = getEnergy( pos, vel, mass, G )
#
#     # number of timesteps
#     Nt = int(np.ceil(tEnd/dt))
#
#     # save energies, particle orbits for plotting trails
#     pos_save = np.zeros((N,3,Nt+1))
#     pos_save[:,:,0] = pos
#     KE_save = np.zeros(Nt+1)
#     KE_save[0] = KE
#     PE_save = np.zeros(Nt+1)
#     PE_save[0] = PE
#     KE_Rsave = np.zeros(Nt+1)
#     KE_Rsave[0] = KE_rad
#     KE_Osave = np.zeros(Nt+1)
#     KE_Osave[0] = KE_orb
#     t_all = np.arange(Nt+1)*dt
#
#     # prep figure
#     fig = plt.figure(figsize=(4,5), dpi=80)
#     grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
#     ax1 = plt.subplot(grid[0:2,0])
#     ax2 = plt.subplot(grid[2,0])
#
#     # Simulation Main Loop
#     """
#     We are using a numerical (=approximate) scheme to solve the equation of motion.
#     The basic idea is that for each discrete timestep we calculate the acceleration at start time t and consider it constant
#     during the timestep duration Dt. So we can calculate the velocity and displacement of the particle easily. 
#     With the simplest version of this (1 coarse timestep= 1 acceleration update, then 1 velocity update, then 1 "drift"
#     (displacement) update across Dt), errors tend to pile up over timesteps and you end up far from the solution. 
#     A more stable scheme is used here, the kick-drift-kick version of the Leapfrog scheme. Can you explain how it is different?
#     """
#     for i in range(Nt):
#         # (1/2) kick
#         vel += acc * dt/2.0
#
#         # drift
#         pos += vel * dt
#
#         # update accelerations
#         acc = getAcc( pos, mass, G, softening )
#
#         # (1/2) kick
#         vel += acc * dt/2.0
#
#         # update time
#         t += dt
#
#         # get energy of system
#         KE, PE, KE_rad, KE_orb  = getEnergy( pos, vel, mass, G )
#
#         # save energies, positions for plotting trail
#         pos_save[:,:,i+1] = pos
#         KE_save[i+1] = KE
#         PE_save[i+1] = PE
#         KE_Rsave[i+1] = KE_rad
#         KE_Osave[i+1] = KE_orb
#
#         # plot in real time
#         if plotRealTime or (i == Nt-1):
#             plt.sca(ax1)
#             plt.cla()
#             xx = pos_save[:,0,max(i-50,0):i+1]
#             yy = pos_save[:,1,max(i-50,0):i+1]
#             plt.scatter(xx,yy,s=1,color=[.7,.7,1])
#             plt.scatter(pos[:,0],pos[:,1],s=10,color='red')
#             ax1.set(xlim=(-2, 2), ylim=(-2, 2))
#             ax1.set_aspect('equal', 'box')
#             ax1.set_xticks([-2,-1,0,1,2])
#             ax1.set_yticks([-2,-1,0,1,2])
#
#             plt.sca(ax2)
#             plt.cla()
#             ax2.scatter(t_all,KE_save,color='red',s=1,label='KE')
#             ax2.scatter(t_all,PE_save,color='blue',s=1,label='PE')
#             ax2.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot')
#             ax2.scatter(t_all,2*KE_save+PE_save,color='pink',s=1,label='2KE+PE' )   
#             ax2.scatter(t_all,KE_Rsave,color='green',s=1,label='KE_rad')
#             ax2.scatter(t_all,KE_Osave,color='orange',s=1,label='KE_orb')
#             ax2.legend(bbox_to_anchor=(1.35, 1.35), loc='upper right', borderaxespad=0)
#             ax2.set(xlim=(0, tEnd+3), ylim=(-300, 300))
#             ax2.set_aspect(0.005)
#
#             # add labels/legend
#             plt.sca(ax2)
#             plt.xlabel('time')
#             plt.ylabel('energy')
#
#             plt.sca(ax1)
#             plt.xlabel('x')
#             plt.ylabel('y')
#
#
#             #What can you say about the validity of the virial theorem vs that of the conservation of mechanical energy?
#             # Save figure
#             plt.savefig(str(path) +'/nbody_' + str(N) + 'part_omega' + str(omega) + '_step' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
#
#             plt.pause(0.001)
#
#     plt.show()
#
#     return 0
#
#
# if __name__== "__main__":
#   main()
#
#   #def make_movie(omega,N):
#
# directory = "MyNbodyRun_Om"+str(omega)+"_N"+str(N)+"part"
#     # Parent Directory path: change to your own path
# parent_dir = "/Users/johnmeftah/Desktop/sim_test/"
# image_folder=os.path.join(parent_dir, directory)
#
# fps=10 #number of frames per second
# image_files = [os.path.join(image_folder,img)
#                    for img in os.listdir(image_folder)
#                    if img.endswith(".png")]
# image_files_sorted = natsorted(image_files,reverse=False)
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_sorted, fps=fps)
# clip.write_videofile(str(parent_dir) +'/nbody-cluster_Om'+str(omega)+'_N'+str(N)+'part.mp4')
# return 0
#
# #def make_movie(omega,N):
#
# directory = "MyNbodyRun_Om"+str(omega)+"_N"+str(N)+"part"
#     # Parent Directory path: change to your own path
# parent_dir = "/Users/johnmeftah/Desktop/sim_test/"
# image_folder=os.path.join(parent_dir, directory)
#
# fps=10 #number of frames per second
# image_files = [os.path.join(image_folder,img)
#                    for img in os.listdir(image_folder)
#                    if img.endswith(".png")]
# image_files_sorted = natsorted(image_files,reverse=False)
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_sorted, fps=fps)
# clip.write_videofile(str(parent_dir) +'/nbody-cluster_Om'+str(omega)+'_N'+str(N)+'part.mp4')
# return 0
# --- MODIFIED (from 11.py) ---
    N = int(input("Enter N [default=300]: ") or "300")
    tEnd = float(input("Enter tEnd [default=1.0]: ") or "1.0")
    dt = float(input("Enter dt [default=0.01]: ") or "0.01")
    Ng = int(input("Enter Ng [default=10]: ") or "10")
    eps = float(input("Enter softening eps [default=0.1]: ") or "0.1")

    # Split scale r_split ~ a few cell widths
    # default: 2h is a reasonable starting point
    r_split_in_h = float(input("Enter r_split in units of h [default=2.0]: ") or "2.0")

    nsteps = int(np.round(tEnd / dt))

    np.random.seed(1)
    box_min, box_max = -1.0, 1.0
    L = box_max - box_min
    h = L / Ng
    r_split = r_split_in_h * h

    x = np.random.uniform(box_min, box_max, size=(N, 3))
    v = np.zeros((N, 3))
    m = np.ones(N) / N

    _, _, cells0 = assign_particles_to_cells(x, box_min, box_max, Ng)
    occ0, max0, mean0 = grid_sanity(cells0)
    print(f"\n=== t=0 sanity ===")
    print(f"cells={Ng**3}, occupied={occ0}, max_in_cell={max0}, mean_occ={mean0:.2f}")
    print(f"r_split = {r_split_in_h:.2f} h = {r_split:.4f}")

    t0 = time.time()
    rho = None
    mesh_time = 0.0
    pp_time = 0.0

    for step in range(nsteps):
        t_step0 = time.time()
        # We time mesh+pp internally by crude splits:
        # just re-run pieces for timing is expensive, so we time whole step and also time PP alone inside function? keep simple:
        x, v, rho, max_im_phi = leapfrog_step_hybrid_split(x, v, m, dt, eps, box_min, box_max, Ng, r_split)
        t_step1 = time.time()

        if step % 20 == 0 or step == nsteps - 1:
            M_grid, M_true, dM = mass_conservation_check(rho, m, box_min, box_max, Ng)
            print(f"[step {step:4d}/{nsteps}] |ΔM|={dM:.3e}, max(Im Phi)~{max_im_phi:.3e}", flush=True)

    t1 = time.time()
    runtime = t1 - t0

    print("\n================ RUN SUMMARY ================")
    print(f"Mode=hybrid split (filtered PM + local PP)")
    print(f"N={N}, steps={nsteps}, dt={dt}, tEnd={tEnd}, Ng={Ng}, eps={eps}")
    print(f"Runtime: {runtime:.3f} s  ({runtime/nsteps:.6f} s/step)")
    print("============================================\n")

if __name__ == "__main__":
    main()
############################
# END MOD (John)
#############################
