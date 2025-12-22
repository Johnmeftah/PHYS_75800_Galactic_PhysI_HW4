A"""
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

import numpy as np
import matplotlib.pyplot as plt
import os
import moviepy.video.io.ImageSequenceClip
from natsort import natsorted

############################################
# MOD (John): imports for the video method
############################################

# saving frames with a clean numbering then, we  convert ffmpeg -> mp4.
import time
import shutil
import subprocess


def InitialConditions(N,omega,mtot,v0):

    # --------------This module generates Initial Conditions
    #For N particles with total mass mtot, solid angular velocity omega

    np.random.seed(17)            # set the random number generator seed

    mass = mtot*np.ones((N,1))/N  # total mass of particles is mtot. all particles have the same mass here.
    pos  = np.random.randn(N,3)   # randomly selected positions from a normal distribution.
   #Could be modified to take into account the initial density profile of the halo.

    if(v0==1):
        # for solid rotation: Vrot=radius*omega along e_theta. Let's calculate the radii of particles first
        x = pos[:,0:1]
        y = pos[:,1:2]
        z = pos[:,2:3]
        # in the frame of the centre of mass
        x -= np.mean(mass * x) / np.mean(mass)
        y -= np.mean(mass * y) / np.mean(mass)
        z -= np.mean(mass * z) / np.mean(mass)
        #polar coordinates (r, theta,z). We consider z the axis of rotation
        norm_r=np.sqrt(x**2 + y**2)
        theta=np.arctan(y/x)
        #total rotational velocity, tangential to radius. Let assume the axis of rotation is z
        vrot=norm_r*omega
        vrot_x=-vrot*np.sin(theta)
        vrot_y=vrot*np.cos(theta)
        vrot_z=np.zeros(N)
        vrot_z=vrot_z.reshape(N,1)
        vrot=np.hstack((vrot_x,vrot_y,vrot_z))

        vel  =  np.random.randn(N,3)
        # in the frame of the centre of mass
        vel -= np.mean(mass * vel,0) / np.mean(mass)
        #vrot and random variation
        vel=vel+vrot
        # --------------
    else:
        #zero initial velocities in the frame of reference of the halo
        vel=np.zeros((N,3))
       # --------------

    return mass, pos, vel


def ScaleFactor(t):
    # --------------This module computes a scale factor at time t in the LCDM, matter dominated era.
    # useful if you want to include expansion
    a=1.0 #modify this function
    return a


def getAcc( pos, mass, G, softening ):
    """
    Calculate the acceleration on each particle due to Newton's Law
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    NOTE: You can see that everything is put in matrix form, allowing for matrix operations rather than looping over particles
    to get each update. This is not just because it looks cool to do all calculations in only one line rather than a FOR loop.
    It  also significantly improves the computational performance in python!!
    """
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    """
    You can see that we included a "softening term". Its goal is to avoid getting an (near)infinite value when distance
    between two particles is ~ zero. It can happen in simulations where resolution,
    number of particles and float precision is limited but would be unphysical.
    We've seen that close-encounter collisions are rather irrelevant in collisionless systems likes haloes.
    So softening is essentially a user-defined resolution limit for numerical gravity.
    """
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    # acceleration under gravity (Newton's second law)
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components (hstack performs a concatenation)
    a = np.hstack((ax,ay,az))

    return a


def getEnergy( pos, vel, mass, G ):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    # Potential Energy:

    # acknowledging positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i.
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))

    #Radial kinetic energy: first Convert to Center-of-Mass frame
    x -= np.mean(mass * x) / np.mean(mass)
    y -= np.mean(mass * y) / np.mean(mass)
    z -= np.mean(mass * z) / np.mean(mass)
    norm_r=np.sqrt(x**2 + y**2 + z**2)
    x1=x/norm_r
    y1=y/norm_r
    z1=z/norm_r
    r=np.hstack((x1,y1,z1))
    vel_r=np.sum(vel*r,axis=1)
    vr2=vel_r**2
    N=vr2.shape[0]
    vr2=np.reshape(vr2,(N,1))
    KE_rad = 0.5 * np.sum(mass * vr2,axis=0 )
    KE_rad=KE_rad.reshape(1)[0]

    #Orbital kinetic energy
    KE_orb=KE-KE_rad

    return KE, PE, KE_rad, KE_orb


############################################
# MOD (John): ffmpeg movie helpers
############################################

# naming the frames as nbody_00000.png, nbody_00001.png, and so no
# then ffmpeg converts frames to  mp4 
def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Run: which ffmpeg")


def run_cmd(cmd_list):
    subprocess.run(cmd_list, check=True)


def make_mp4_from_frames(frames_dir, out_mp4, fps=20):
    """
    H.264 requires even width/height.
    Force even dimensions with scale=trunc(iw/2)*2:trunc(ih/2)*2
    """
    require_ffmpeg()
    pattern = os.path.join(frames_dir, "nbody_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4
    ]
    run_cmd(cmd)


def main():
    """ N-body simulation """

    # Simulation parameters ----------------------------------------------------
    N         = 20    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 10.0   # time at which simulation ends
    dt        = 0.01   # timestep
    softening = 0.1    # softening length
    G         = 1.0    # Newton's Gravitational Constant. Here set to 1 in code units for covenience.
    mtot      = 20.0  # Total mass of the object
    plotRealTime = False # switch on for plotting as the simulation goes along
    omega        = 0.2  # initial angular velocity if solid rotation
    v0           =1.0   #if not 1.0, set initial velocitites to 0
    #-------------------------------------------------------------------------------

    ############################################
    # MOD (John): user inputs
    ############################################
    # keeping defaults above, but now we let the user override them.
    # pressing enter without entering a value, script will use the default.

    try:
        N_in = input("Enter number of particles N (default = 20): ")
        if N_in.strip() != "":
            N = int(N_in)
    except Exception:
        print("Bad N input. Using default N = 20")

    try:
        om_in = input("Enter initial solid rotation omega (default = 0.2): ")
        if om_in.strip() != "":
            omega = float(om_in)
    except Exception:
        print("Bad omega input. Using default omega = 0.2")

    try:
        m_in = input("Enter total mass mtot (default = 20.0): ")
        if m_in.strip() != "":
            mtot = float(m_in)
    except Exception:
        print("Bad mtot input. Using default mtot = 20.0")

    try:
        eps_in = input("Enter softening length epsilon (default = 0.1): ")
        if eps_in.strip() != "":
            softening = float(eps_in)
    except Exception:
        print("Bad epsilon input. Using default epsilon = 0.1")
    ######################################################################

    # Set where to store outputs-------------------------------------------------------------------------------
    # Create a directory for the Simulation

    ############################################
    # MOD (John): folder naming includes M and eps
    ############################################
    # adding M and eps to the folder name so runs don't overwrite each other.
    directory = "MyNbodyRun_Om"+str(omega)+"_N"+str(N)+"part"+"_M"+str(mtot)+"_eps"+str(softening)

    # Parent Directory path: change to your own path
    parent_dir = "/Users/johnmeftah/Desktop/sim_test"
    isdir = os.path.isdir(parent_dir)
    if isdir==False: # create parent directory if it does not exist.
        os.mkdir(parent_dir)
    # Make complete Path
    path = os.path.join(parent_dir, directory)
    #Create directory
    isdir = os.path.isdir(path)
    if isdir==False:
        os.mkdir(path)

    ############################################
    # MOD (John): frames folder for the video
    ############################################
    # saving the frames

    frames_dir = os.path.join(path, "frames")
    isdir = os.path.isdir(frames_dir)
    if isdir==False:
        os.mkdir(frames_dir)

    #-----------------------------------------------------------------------------

    #Now let's run the simulation!
    mass,pos,vel = InitialConditions(N,omega,mtot,v0) #load initial conditions

    # calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening )

    # calculate initial energy of system
    KE, PE, KE_rad, KE_orb  = getEnergy( pos, vel, mass, G )

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N,3,Nt+1))
    pos_save[:,:,0] = pos
    KE_save = np.zeros(Nt+1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt+1)
    PE_save[0] = PE
    KE_Rsave = np.zeros(Nt+1)
    KE_Rsave[0] = KE_rad
    KE_Osave = np.zeros(Nt+1)
    KE_Osave[0] = KE_orb
    t_all = np.arange(Nt+1)*dt

    # prep figure
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])

    ############################################
    # MOD (John): video settings
    ############################################
    # save_every to  skip frames if we want fewer images.

    fps = 20
    save_every = 1

    comp_start = time.time()

    # Simulation Main Loop
    """
    We are using a numerical (=approximate) scheme to solve the equation of motion.
    The basic idea is that for each discrete timestep we calculate the acceleration at start time t and consider it constant
    during the timestep duration Dt. So we can calculate the velocity and displacement of the particle easily.
    With the simplest version of this (1 coarse timestep= 1 acceleration update, then 1 velocity update, then 1 "drift"
    (displacement) update across Dt), errors tend to pile up over timesteps and you end up far from the solution.
    A more stable scheme is used here, the kick-drift-kick version of the Leapfrog scheme. Can you explain how it is different?
    """
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc( pos, mass, G, softening )

        # (1/2) kick
        vel += acc * dt/2.0

        # update time
        t += dt

        # get energy of system
        KE, PE, KE_rad, KE_orb  = getEnergy( pos, vel, mass, G )

        # save energies, positions for plotting trail
        pos_save[:,:,i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE
        KE_Rsave[i+1] = KE_rad
        KE_Osave[i+1] = KE_orb

        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:,0,max(i-50,0):i+1]
            yy = pos_save[:,1,max(i-50,0):i+1]
            plt.scatter(xx,yy,s=1,color=[.7,.7,1])
            plt.scatter(pos[:,0],pos[:,1],s=10,color='red')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-2,-1,0,1,2])
            ax1.set_yticks([-2,-1,0,1,2])

            plt.sca(ax2)
            plt.cla()
            ax2.scatter(t_all,KE_save,color='red',s=1,label='KE')
            ax2.scatter(t_all,PE_save,color='blue',s=1,label='PE')
            ax2.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot')
            ax2.scatter(t_all,2*KE_save+PE_save,color='pink',s=1,label='2KE+PE' )
            ax2.scatter(t_all,KE_Rsave,color='green',s=1,label='KE_rad')
            ax2.scatter(t_all,KE_Osave,color='orange',s=1,label='KE_orb')
            ax2.legend(bbox_to_anchor=(1.35, 1.35), loc='upper right', borderaxespad=0)
            ax2.set(xlim=(0, tEnd+3), ylim=(-300, 300))
            ax2.set_aspect(0.005)

            # add labels/legend
            plt.sca(ax2)
            plt.xlabel('time')
            plt.ylabel('energy')

            plt.sca(ax1)
            plt.xlabel('x')
            plt.ylabel('y')

            ############################################
            # MOD (John): run info box
            ############################################
            # printing the inputs on the figure, on the right side.
            # putting it under the legend area (the position is in axes coords).
            ax1.text(
                1.02, 0.45,
                "N = "+str(N)+"\n"+"ω = "+str(omega)+"\n"+"M = "+str(mtot)+"\n"+"ε = "+str(softening),
                transform=ax1.transAxes,
                ha='left', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
            )

            #What can you say about the validity of the virial theorem vs that of the conservation of mechanical energy?
            # Save figure (original save)
            plt.savefig(str(path) +'/nbody_' + str(N) + 'part_omega' + str(omega) + '_step' + str(i) + '.png',
                        dpi=240,  bbox_inches='tight', pad_inches = 0)

            ############################################
            # MOD (John): saving the frames for mp4
            ############################################
            # saving frames with fixed numbering so ffmpeg can read them easily

            if (i % save_every) == 0:
                frame_path = os.path.join(frames_dir, "nbody_%05d.png" % i)
                plt.savefig(frame_path, dpi=240)  # no tight cropping here

            plt.pause(0.001)

    plt.show()

    comp_elapsed = time.time() - comp_start

    ############################################
    # MOD (John): make the mp4 using ffmpeg
    ############################################
    # after the run, convert frames to  mp4
    # video is saved inside the run folder. This will run automatically
    out_mp4 = os.path.join(path, "VIDEO_run_Om"+str(omega)+"_N"+str(N)+"_M"+str(mtot)+"_eps"+str(softening)+".mp4")
    try:
        make_mp4_from_frames(frames_dir, out_mp4, fps=fps)
        print("Saved video to:", out_mp4)
    except Exception as e:
        print("Video export failed:", str(e))
        print("Frames are still saved in:", frames_dir)

    print("Simulation runtime (s):", comp_elapsed)

    return 0


if __name__== "__main__":
  main()
