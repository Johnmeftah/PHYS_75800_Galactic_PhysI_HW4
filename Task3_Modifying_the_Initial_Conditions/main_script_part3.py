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
import numpy as np
import matplotlib.pyplot as plt
import os
import moviepy.video.io.ImageSequenceClip
from natsort import natsorted

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
    
    # acceleration under gravity (Newton's second law) (notice we are calculating vec(r)/r^3 instead or 1/r^2 as we need the
    #direction of the force for each pair of particles)
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
    
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]
    
    # matrix that stores all pairwise particle separations: r_j - r_i. Note that each pair appears twice: dx(i,j)=-dx(j,i)
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

############################
# MOD (John): Add Task 3 (two-peaks ICs, merger detection, extra diagnostics, and driver) from 18.py
#############################
import time


# inputs helpers

def ask_float(prompt, default):
    s = input(f"{prompt} (default = {default}): ").strip()
    if s == "":
        return float(default)
    try:
        return float(s)
    except Exception:
        print(f"Invalid input. Using default {default}")
        return float(default)



# utilities

def total_angular_momentum(pos, vel, mass):
    return np.sum(mass * np.cross(pos, vel), axis=0)


def peak_COM(pos, vel, mass, grp):
    M1 = np.sum(mass[grp == 0])
    M2 = np.sum(mass[grp == 1])

    R1 = np.sum(mass[grp == 0] * pos[grp == 0], axis=0) / M1
    R2 = np.sum(mass[grp == 1] * pos[grp == 1], axis=0) / M2

    V1 = np.sum(mass[grp == 0] * vel[grp == 0], axis=0) / M1
    V2 = np.sum(mass[grp == 1] * vel[grp == 1], axis=0) / M2

    D = np.linalg.norm(R2 - R1)
    impact_yz = np.linalg.norm((R2 - R1)[1:])  # sqrt(dy^2+dz^2)
    return R1, R2, V1, V2, D, impact_yz



# setting up acceleration (matrix style)

def InitialConditions_two_peaks(
    N, mtot, d=4.0, v_rel=0.0,
    sigma_pos=1.0, omega1=0.0, omega2=0.0,
    seed=17,
    enforce_head_on=True
):
    rng = np.random.default_rng(seed)

    N1 = N // 2
    N2 = N - N1

    mass = (mtot / N) * np.ones((N, 1))

    c1 = np.array([-d / 2.0, 0.0, 0.0])
    c2 = np.array([+d / 2.0, 0.0, 0.0])

    pos1 = rng.normal(loc=0.0, scale=sigma_pos, size=(N1, 3)) + c1
    pos2 = rng.normal(loc=0.0, scale=sigma_pos, size=(N2, 3)) + c2

    # enforcing exact head-on geometry (remove accidental impact parameter)
    if enforce_head_on:
        pos1 = pos1 - np.mean(pos1, axis=0) + c1
        pos2 = pos2 - np.mean(pos2, axis=0) + c2

    # bulk velocities (relative motion along x)
    vbulk1 = np.array([+v_rel / 2.0, 0.0, 0.0])
    vbulk2 = np.array([-v_rel / 2.0, 0.0, 0.0])

    vel1 = np.tile(vbulk1, (N1, 1))
    vel2 = np.tile(vbulk2, (N2, 1))

    # optional internal rotation around each peak center
    if omega1 != 0.0:
        rrel1 = pos1 - c1
        vel1 += omega1 * np.column_stack([-rrel1[:, 1], rrel1[:, 0], np.zeros(N1)])

    if omega2 != 0.0:
        rrel2 = pos2 - c2
        vel2 += omega2 * np.column_stack([-rrel2[:, 1], rrel2[:, 0], np.zeros(N2)])

    pos = np.vstack([pos1, pos2])
    vel = np.vstack([vel1, vel2])

    # remove total COM position + velocity drift
    M = np.sum(mass)
    rcom = np.sum(mass * pos, axis=0) / M
    vcom = np.sum(mass * vel, axis=0) / M
    pos = pos - rcom
    vel = vel - vcom

    grp = np.zeros(N, dtype=int)
    grp[N1:] = 1

    return mass, pos, vel, grp



# main

def main():
    N = 100  # Task 3 requirement

    mtot = ask_float("Enter total mass M", 20.0)
    eps = ask_float("Enter softening epsilon", 0.1)
    tEnd = ask_float("Enter tEnd", 10.0)
    dt = ask_float("Enter dt", 0.01)

    d = ask_float("Enter initial peak separation d", 4.0)
    v_rel = ask_float("Enter initial relative velocity v_rel (along x)", -0.5)

    omega1 = ask_float("Enter omega1 (peak 1 rotation)", 0.0)
    omega2 = ask_float("Enter omega2 (peak 2 rotation)", 0.0)
    sigma_pos = ask_float("Enter peak size sigma_pos", 1.0)

    G = 1.0
    enforce_head_on = True

    print("\n================ RUN SUMMARY ================")
    print("ICs = two nearby peaks (NO expansion)")
    print("User-selected parameters:")
    print(f"  N        = {N}")
    print(f"  M        = {mtot}")
    print(f"  epsilon  = {eps}")
    print(f"  d        = {d}")
    print(f"  v_rel    = {v_rel}")
    print(f"  sigma    = {sigma_pos}")
    print(f"  omega1   = {omega1}")
    print(f"  omega2   = {omega2}")
    print(f"  tEnd     = {tEnd}")
    print(f"  dt       = {dt}")
    print(f"  head_on_enforced = {enforce_head_on}")
    print("Coords: PHYSICAL only")
    print("============================================\n")

    mass, pos, vel, grp = InitialConditions_two_peaks(
        N, mtot, d=d, v_rel=v_rel,
        sigma_pos=sigma_pos, omega1=omega1, omega2=omega2,
        seed=17,
        enforce_head_on=enforce_head_on
    )

    # IC sanity check
    R1, R2, V1, V2, D0, impact_yz = peak_COM(pos, vel, mass, grp)
    M = np.sum(mass)
    vcom = np.sum(mass * vel, axis=0) / M
    rcom = np.sum(mass * pos, axis=0) / M
    L = total_angular_momentum(pos, vel, mass)

    print("=== IC sanity checks (t=0) ===")
    print(f"Total COM position r_com = {rcom}")
    print(f"Total COM velocity v_com = {vcom}, |v_com|={np.linalg.norm(vcom):.3e}")
    print(f"Peak COM separation |R2-R1| = {D0:.6f}")
    print(f"Impact parameter sqrt(dy^2+dz^2) = {impact_yz:.6f}  (should be ~0 for head-on)")
    print(f"  R1 = {R1},  R2 = {R2}")
    print(f"  V1 = {V1},  V2 = {V2}")
    print(f"Total L = {L}, |L|={np.linalg.norm(L):.3e}")

    print("\n=== What the merger plot shows ===")
    print("We compute the center-of-mass (COM) of each peak (group 0 and group 1).")
    print("Then we plot D(t) = |R2(t) - R1(t)|, the separation between those two COMs.")
    print("If D(t) drops below the threshold (here: 1*sigma_pos), we flag MERGED=True.")
    print("To be stricter, we also report whether they are still below threshold at tEnd.\n")

    # integrating
    acc = getAcc(pos, mass, G, eps)

    Nt = int(np.ceil(tEnd / dt))
    t_all = np.arange(Nt + 1) * dt

    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos

    KE_save = np.zeros(Nt + 1)
    PE_save = np.zeros(Nt + 1)
    KE_Rsave = np.zeros(Nt + 1)
    KE_Osave = np.zeros(Nt + 1)
    D_save = np.zeros(Nt + 1)

    KE, PE, KE_rad, KE_orb = getEnergy(pos, vel, mass, G)
    KE_save[0], PE_save[0], KE_Rsave[0], KE_Osave[0] = KE, PE, KE_rad, KE_orb
    _, _, _, _, D_save[0], _ = peak_COM(pos, vel, mass, grp)

    comp_start = time.time()

    # leapfrog
    for i in range(Nt):
        vel += acc * (dt / 2.0)
        pos += vel * dt
        acc = getAcc(pos, mass, G, eps)
        vel += acc * (dt / 2.0)

        pos_save[:, :, i + 1] = pos

        KE, PE, KE_rad, KE_orb = getEnergy(pos, vel, mass, G)
        KE_save[i + 1], PE_save[i + 1], KE_Rsave[i + 1], KE_Osave[i + 1] = KE, PE, KE_rad, KE_orb
        _, _, _, _, D_save[i + 1], _ = peak_COM(pos, vel, mass, grp)

    comp_elapsed = time.time() - comp_start
    print(f"\nDone. computational time = {comp_elapsed:.3f} s")

    # merger diagnostic
    merge_thresh = 1.0 * sigma_pos
    Dmin = float(np.min(D_save))
    D_end = float(D_save[-1])
    merged = (Dmin < merge_thresh)
    stay_merged = (D_end < merge_thresh)

    print("\n=== Merger diagnostic ===")
    print(f"min separation Dmin = {Dmin:.6f}")
    print(f"merge threshold ~ {merge_thresh:.6f} (set = 1*sigma_pos)")
    print(f"MERGED? -> {merged}")
    print(f"final separation D_end = {D_end:.6f}")
    print(f"STAY_MERGED at tEnd? -> {stay_merged}")


    # figure 1: energies

    fig1 = plt.figure(figsize=(4, 5), dpi=150)
    grid = plt.GridSpec(3, 1, wspace=0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    # trails
    xx = pos_save[:, 0, :].flatten()
    yy = pos_save[:, 1, :].flatten()
    ax1.scatter(xx, yy, s=1, color=[0.7, 0.7, 1.0])
    ax1.scatter(pos[:, 0], pos[:, 1], s=10, color="red")

    lim = np.percentile(np.sqrt(xx**2 + yy**2), 99.5)
    lim = max(2.0, lim)
    ax1.set(xlim=(-lim, lim), ylim=(-lim, lim))
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Two peaks merger (physical)")

    # energies
    sc_ke  = ax2.scatter(t_all, KE_save, color="red",    s=1, label="KE")
    sc_pe  = ax2.scatter(t_all, PE_save, color="blue",   s=1, label="PE")
    sc_et  = ax2.scatter(t_all, KE_save + PE_save, color="black", s=1, label="Etot")
    sc_vr  = ax2.scatter(t_all, 2 * KE_save + PE_save, color="pink", s=1, label="2KE+PE")
    sc_ker = ax2.scatter(t_all, KE_Rsave, color="green", s=1, label="KE_rad")
    sc_keo = ax2.scatter(t_all, KE_Osave, color="orange", s=1, label="KE_orb")

    ax2.set(xlim=(0, tEnd + 3), ylim=(-300, 300))
    ax2.set_aspect(0.005)
    ax2.set_xlabel("time")
    ax2.set_ylabel("energy")

    handles = [sc_ke, sc_pe, sc_et, sc_vr, sc_ker, sc_keo]
    labels = [h.get_label() for h in handles]
    ax1.legend(
        handles, labels,
        bbox_to_anchor=(1.02, 1.02),
        loc="upper left",
        borderaxespad=0,
        frameon=True
    )

    # param box
    ax1.text(
        1.02, 0.45,
        f"N = {N}\n"
        f"M = {mtot}\n"
        f"ε = {eps}\n"
        f"d = {d}\n"
        f"v_rel = {v_rel}\n"
        f"σ = {sigma_pos}\n"
        f"ω₁ = {omega1}\n"
        f"ω₂ = {omega2}",
        transform=ax1.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7)
    )


    # figure 2: merger diagnostic ONLY

    fig2, axm = plt.subplots(figsize=(6, 2.5), dpi=150)

    axm.plot(t_all, D_save, linewidth=1.5, label="COM separation between peaks, D(t)")
    axm.axhline(merge_thresh, linestyle="--", linewidth=1.2,
                label=f"merge threshold = {merge_thresh:.2f} (=1×sigma_pos)")

    axm.set_xlim(0, tEnd)
    axm.set_xlabel("time")
    axm.set_ylabel("D(t)")
    axm.set_title("Merger diagnostic: peak COM separation")
    axm.legend(loc="upper right", frameon=True)

    axm.text(
        0.02, 0.95,
        f"Dmin = {Dmin:.3f}\n"
        f"Dend = {D_end:.3f}\n"
        f"Thresh = {merge_thresh:.3f}\n"
        f"MERGED = {merged}\n"
        f"STAY_MERGED = {stay_merged}",
        transform=axm.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7)
    )

    plt.show()
    return 0


if __name__ == "__main__":
    main()

