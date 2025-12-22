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

############################
# MOD (John): imports updated
#############################

import time

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value



# LambdaCDM scale factor a(t) and Hubble parameter H(t)

def ScaleFactor(t, t_start_gyr=0.5, t_unit_gyr=1.0):
    """a(t) from Planck18 LambdaCDM, using cosmic time T in Gyr.
    Map: T(t_sim) = t_start_gyr + t_sim * t_unit_gyr
    """
    T = (t_start_gyr + t * t_unit_gyr) * u.Gyr
    z = z_at_value(Planck18.age, T)
    a = 1.0 / (1.0 + z)
    return float(a)


def HubbleParameter(t, t_start_gyr=0.5, t_unit_gyr=1.0):
    """H(t)=adot/a from Planck18, returned in 1/Gyr."""
    T = (t_start_gyr + t * t_unit_gyr) * u.Gyr
    z = z_at_value(Planck18.age, T)
    H = Planck18.H(z)              # km/s/Mpc
    H_gyr = H.to(1/u.Gyr).value    # 1/Gyr
    return float(H_gyr)


def test_scale_factor(tEnd, t_start_gyr=0.5, t_unit_gyr=1.0):
    print("\n=== Scale factor sanity checks (Planck18 ΛCDM) ===")
    a0 = ScaleFactor(0.0, t_start_gyr, t_unit_gyr)
    a1 = ScaleFactor(tEnd, t_start_gyr, t_unit_gyr)
    print(f"Mapping: T(t) = {t_start_gyr} + t * {t_unit_gyr}  [Gyr]")
    print(f"  a(t=0)    = {a0:.6f}")
    print(f"  a(t=tEnd) = {a1:.6f}")
    if not (0.0 < a0 < a1 <= 1.0):
        raise RuntimeError("FAILED: need 0 < a(0) < a(tEnd) <= 1 for expansion forward in time.")
    print("=== Scale factor checks passed ===\n")



# initial conditions (option A: physical to comoving)

def InitialConditions(N, omega, mtot, v0, a0):
    """
    Option A ICs:
      1) Draw physical positions r_phys ~ N(0,1)
      2) Convert to comoving x = r_phys / a0
      3) Build physical velocities (random + solid-body rotation), then convert:
           xdot = v_phys / a0
    """
    np.random.seed(17)

    mass = mtot * np.ones((N, 1)) / N

    # physical initial positions
    r_phys = np.random.randn(N, 3)
    r_phys -= np.mean(mass * r_phys, axis=0) / np.mean(mass)

    # comoving positions
    pos = r_phys / a0

    if v0 == 1:
        # solid-body rotation in PHYSICAL coords about z
        x = r_phys[:, 0:1]
        y = r_phys[:, 1:2]
        z = r_phys[:, 2:3]

        vrot_x = -omega * y
        vrot_y =  omega * x
        vrot_z = 0 * z
        vrot_phys = np.hstack((vrot_x, vrot_y, vrot_z))

        # random PHYSICAL velocities
        v_phys = np.random.randn(N, 3)
        v_phys -= np.mean(mass * v_phys, axis=0) / np.mean(mass)

        # add rotation
        v_phys = v_phys + vrot_phys

        # comoving peculiar velocity xdot
        vel = v_phys / a0

    else:

######################################################
# MOD (John): code update near: vel = np.zeros((N, 3))
######################################################
        vel = np.zeros((N, 3))


    return mass, pos, vel


#####################################
# MOD (John): function getAcc updated
#####################################


# Newtonian acceleration helper (physical separations expected)

def getAcc(pos, mass, G, softening):
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]


    dx = x.T - x
    dy = y.T - y
    dz = z.T - z


#############################################################################
# MOD (John): code update near: inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
#############################################################################
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)


    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

##############################################################
# MOD (John): code update near: return np.hstack((ax, ay, az))
##############################################################

    return np.hstack((ax, ay, az))



# energies in physical coords (r, rdot)

def getEnergy_physical(r, rdot, mass, G):
    KE = 0.5 * np.sum(np.sum(mass * rdot**2))

    x = r[:, 0:1]
    y = r[:, 1:2]
    z = r[:, 2:3]


    dx = x.T - x
    dy = y.T - y
    dz = z.T - z


########################################
# MOD (John): code removed from original
#######################################
# matrix that stores 1/r for all particle pairwise particle separations 

    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)

################################
# MOD (John): plotting updated
#################################
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))
    return KE, PE



# inputs

def ask_for_N():
    default_N = 150
    s = input(f"Enter number of particles N (default = {default_N}): ").strip()
    if s == "":
        return default_N
    try:
        N = int(s)
        if N > 0:
            return N
    except Exception:
        pass
    print("Invalid input. Using default N=150.")
    return default_N


def ask_for_omega():
    default_omega = 0.2
    s = input(f"Enter initial solid rotation omega (default = {default_omega}): ").strip()
    if s == "":
        return default_omega
    try:
        return float(s)
    except Exception:
        print("Invalid input. Using default omega=0.2")
        return default_omega


def ask_for_mtot():
    default_mtot = 20.0
    s = input(f"Enter total mass mtot (default = {default_mtot}): ").strip()
    if s == "":
        return default_mtot
    try:
        return float(s)
    except Exception:
        print("Invalid input. Using default mtot=20.0")
        return default_mtot


def ask_for_softening():
    default_eps = 0.1
    s = input(f"Enter softening length epsilon (default = {default_eps}): ").strip()
    if s == "":
        return default_eps
    try:
        return float(s)
    except Exception:
        print("Invalid input. Using default epsilon=0.1")
        return default_eps


def ask_for_check_mode():
    print("\nChoose run mode:")
    print("  0) NORMAL (expansion ON, gravity ON)  -> real Part 2B run")
    print("  1) CHECK: Expansion OFF (a=1, H=0)    -> should match Part 1 behavior")
    print("  2) CHECK: Free particle (G=0)         -> should keep a^2*v ~ constant")

    while True:
        choice = input("Enter 0, 1, or 2: ").strip()
        if choice in ["0", "1", "2"]:
            break
        print("Invalid. Type 0, 1, or 2.")

    return {"0": "normal", "1": "expansion_off", "2": "free_particle"}[choice]


def ask_for_plot_coords():
    print("\nPlot coordinates:")
    print("  0) comoving   (plot x)")
    print("  1) physical   (plot r=a x)")
    print("  2) both       (make two figures: physical then comoving)")
    print("  3) overlay    (ONE figure: plot x and r on same axes)")
    while True:
        c = input("Enter 0, 1, 2, or 3: ").strip()
        if c in ["0", "1", "2", "3"]:
            break
        print("Invalid. Type 0, 1, 2, or 3.")
    return {"0": "comoving", "1": "physical", "2": "both", "3": "overlay"}[c]



# plot helper

def make_summary_plot(pos_end, a_end, mode, N, omega, mtot, eps_user,
                      t_all, KE, PE, Q,
                      plot_coords="comoving",
                      lock_limits=True,
                      lim_pad_frac=0.08):
    """
    lock_limits=True:
      - axis limits derived from COMOVING positions, applied to all plot modes
      - makes shrinking/growth visible instead of hidden by autoscaling
    """
    r_end = a_end * pos_end

    # axis limits derived from comoving and applied to all plots
    if lock_limits:
        xref = pos_end[:, 0]
        yref = pos_end[:, 1]
        xmin, xmax = np.min(xref), np.max(xref)
        ymin, ymax = np.min(yref), np.max(yref)
        dx = xmax - xmin
        dy = ymax - ymin
        pad_x = lim_pad_frac * (dx if dx > 0 else 1.0)
        pad_y = lim_pad_frac * (dy if dy > 0 else 1.0)
        xlim = (xmin - pad_x, xmax + pad_x)
        ylim = (ymin - pad_y, ymax + pad_y)
    else:
        xlim = None
        ylim = None

    # setting up figure
    fig = plt.figure(figsize=(4, 5), dpi=150)
    grid = plt.GridSpec(3, 1, wspace=0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    ax2b = ax2.twinx()
    fig.subplots_adjust(right=0.72)

    # scattering
    if plot_coords == "physical":
        ax1.scatter(r_end[:, 0], r_end[:, 1], s=40, color="red")
        ax1.set_xlabel("x (physical)")
        ax1.set_ylabel("y (physical)")
        ax1.set_title("Physical coordinates (r = a x)")

    elif plot_coords == "comoving":
        ax1.scatter(pos_end[:, 0], pos_end[:, 1], s=40, color="red")
        ax1.set_xlabel("x (comoving)")
        ax1.set_ylabel("y (comoving)")
        ax1.set_title("Comoving coordinates (x)")

    elif plot_coords == "overlay":
        ax1.scatter(pos_end[:, 0], pos_end[:, 1], s=40, color="red", alpha=0.65, label="x (comoving)")
        ax1.scatter(r_end[:, 0],  r_end[:, 1],  s=40, color="black", alpha=0.65, label="r=a x (physical)")
        ax1.set_xlabel("x or r (same axis limits)")
        ax1.set_ylabel("y")
        ax1.set_title("Overlay: comoving x (red) vs physical r=a x (black)")
        ax1.legend(loc="upper left")

    else:
        raise ValueError("Invalid plot_coords")

    ax1.set_aspect("equal", "box")

    if xlim is not None and ylim is not None:
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)

    # energies + Q (Q is viria eq, normlized, to make sure the physics are working here)
    ax2.plot(t_all, KE, color="red",   linewidth=1, label="KE")
    ax2.plot(t_all, PE, color="blue",  linewidth=1, label="PE")
    ax2.plot(t_all, KE + PE, color="black", linewidth=1, label="Etot")
    ax2.plot(t_all, 2*KE + PE, color="pink", linewidth=1, label="2KE+PE")

    ax2.set_xlabel("t (sim units)")
    ax2.set_ylabel("energy")

    ax2b.plot(t_all, Q, linewidth=0.8, label="Q=2KE/|PE|")
    ax2b.set_ylim(0, 3.0)
    ax2b.set_ylabel("Q")
    ax2b.yaxis.set_label_coords(1.02, 0.5)
    ax2b.tick_params(axis="y", pad=10)

    h2, l2 = ax2.get_legend_handles_labels()
    hq, lq = ax2b.get_legend_handles_labels()
    ax1.legend(
        h2 + hq, l2 + lq,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0,
        frameon=True
    )

    ax1.text(
        1.02, 0.45,
        f"mode={mode}\nN={N}\nω={omega}\nM={mtot}\nε={eps_user}\n"
        f"a(end)={a_end:.3f}",
        transform=ax1.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7)
    )

    plt.show()



# RMS radius diagnostics (THIS is the expansion story)

def plot_rms_radius(t_all, a_all, Rx, Rr, mode, N, omega, mtot, eps_user):
    # plotting Rx(t), Rr(t), and a(t)Rx(t)
    plt.figure(figsize=(5, 3.5), dpi=150)
    plt.plot(t_all, Rx, label="R_x(t)  (comoving)")
    plt.plot(t_all, Rr, label="R_r(t)  (physical)")
    plt.plot(t_all, a_all * Rx, "--", label="a(t) * R_x(t)  (should match R_r)")
    plt.xlabel("t (sim units)")
    plt.ylabel("RMS radius")
    plt.title("RMS radius vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plotting ratio Rr/Rx and compare to a(t)
    plt.figure(figsize=(5, 3.5), dpi=150)
    plt.plot(t_all, Rr / Rx, label="R_r / R_x")
    plt.plot(t_all, a_all, "--", label="a(t)")
    plt.xlabel("t (sim units)")
    plt.ylabel("ratio")
    plt.title("Check: R_r/R_x should track a(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()



# main


def main():

#################################
# MOD (John): code update near: N= ask_for_N()
##################################

    # user inputs
    
    N         = ask_for_N()
    omega     = ask_for_omega()
    mtot      = ask_for_mtot()
    eps_user  = ask_for_softening()

    mode = ask_for_check_mode()
    plot_coords = ask_for_plot_coords()

    TEST_EXPANSION_OFF = (mode == "expansion_off")
    TEST_FREE_PARTICLE = (mode == "free_particle")


    # sim params

    t         = 0.0
    tEnd      = 10.0
    dt        = 0.01
    G         = 1.0
    v0        = 1.0

    # mapping sim time to cosmic time (Gyr)
    t_start_gyr = 0.5
    t_unit_gyr  = 1.0
    dt_gyr      = dt * t_unit_gyr

    # softening: interpret as physical (consistent with r=a x usage)
    EPS_IS_PHYSICAL = True

    if TEST_FREE_PARTICLE:
        G = 0.0


    # printting run summary
    # --------------------
    print("\n================ RUN SUMMARY ================")
    print(f"Mode: {mode}")
    print(f"N={N}, omega={omega}, M={mtot}, eps={eps_user}")
    print(f"tEnd={tEnd}, dt={dt} (dt_gyr={dt_gyr})")
    print(f"Mapping: T = {t_start_gyr} + t * {t_unit_gyr}  [Gyr]")
    print("IC Mode: Option A (physical ICs, then x=r/a0)")
    print(f"Plot coords: {plot_coords}")
    print("============================================\n")


    # scale factor sanity check

    if not TEST_EXPANSION_OFF:
        test_scale_factor(tEnd, t_start_gyr, t_unit_gyr)
    else:
        print("[CHECK] Expansion OFF: forcing a=1, H=0\n")

    if TEST_FREE_PARTICLE:
        print("[CHECK] Free particle: forcing G=0; expect a^2*v ~ constant\n")


    # output folder 

    directory = f"Part2B_{mode}_Om{omega}_N{N}_M{mtot}_eps{eps_user}"

    parent_dir = "/Users/johnmeftah/Desktop/sim_test"


    path = os.path.join(parent_dir, directory)

#################################################
# MOD (John): cosmology/expansion utilities added
#################################################
    os.makedirs(path, exist_ok=True)


    # a,H helper

    def get_aH(t_sim):
        if TEST_EXPANSION_OFF:
            return 1.0, 0.0
        a = ScaleFactor(t_sim, t_start_gyr, t_unit_gyr)
        H = HubbleParameter(t_sim, t_start_gyr, t_unit_gyr)
        return a, H

   
    # setting initial a0 and ICs
   
    a0, H0 = get_aH(0.0)
    mass, pos, vel = InitialConditions(N, omega, mtot, v0, a0)  # pos=x, vel=xdot

    Nt = int(np.ceil(tEnd / dt))
    t_all = np.arange(Nt + 1) * dt

    a_save = np.zeros(Nt + 1)
    H_save = np.zeros(Nt + 1)

    KE_save = np.zeros(Nt + 1)
    PE_save = np.zeros(Nt + 1)
    Q_save  = np.zeros(Nt + 1)

    # RMS radius diagnostics
    Rx_save = np.zeros(Nt + 1)  # comoving RMS
    Rr_save = np.zeros(Nt + 1)  # physical RMS

    # free particle check: track mean(a^2 v)
    av2_mean = np.zeros((Nt + 1, 3))


    # initial acceleration (prof equation from white board)

    a, H = get_aH(t)
    a_save[0], H_save[0] = a, H

    eps_phys = eps_user if EPS_IS_PHYSICAL else a * eps_user

    # physical separations for gravity
    r = a * pos
    acc_phys = getAcc(r, mass, G, eps_phys)

    # professor comoving equation:
    # xddot = (1/a) * acc_phys  -  2H * xdot
    acc_com  = (acc_phys / a) - (2.0 * H * vel)

    # physical velocity for energy: rdot = a xdot + adot x = a vel + (aH) pos
    rdot = a * vel + (a * H) * pos
    KE0, PE0 = getEnergy_physical(r, rdot, mass, G)

    KE_save[0], PE_save[0] = KE0, PE0
    Q_save[0] = 2.0 * KE0 / np.abs(PE0) if PE0 != 0 else np.nan
    av2_mean[0, :] = (a**2) * np.mean(vel, axis=0)

    # RMS radius initial
    Rx_save[0] = np.sqrt(np.mean(np.sum(pos**2, axis=1)))
    Rr_save[0] = np.sqrt(np.mean(np.sum(r**2, axis=1)))

    comp_start = time.time()


    # main loop  (kick-drift-kick, cosmic time)
   

    for i in range(Nt):

###############################################################
# MOD (John): code update near: vel += acc_com * (dt_gyr / 2.0)
###############################################################
        # kick
        vel += acc_com * (dt_gyr / 2.0)


        # drift

###############
#MOD (John): code update near: pos += vel * dt_gyr
###############
        pos += vel * dt_gyr
        t   += dt

        a, H = get_aH(t)
        a_save[i + 1], H_save[i + 1] = a, H

        eps_phys = eps_user if EPS_IS_PHYSICAL else a * eps_user

        # physical positions for gravity
        r = a * pos
        acc_phys = getAcc(r, mass, G, eps_phys)
        acc_com  = (acc_phys / a) - (2.0 * H * vel)

        # kick
        vel += acc_com * (dt_gyr / 2.0)

        # energies in physical space
        rdot = a * vel + (a * H) * pos
        KE, PE = getEnergy_physical(r, rdot, mass, G)

        KE_save[i + 1] = KE
        PE_save[i + 1] = PE
        Q_save[i + 1]  = 2.0 * KE / np.abs(PE) if PE != 0 else np.nan
        av2_mean[i + 1, :] = (a**2) * np.mean(vel, axis=0)

        # RMS radius each step
        Rx_save[i + 1] = np.sqrt(np.mean(np.sum(pos**2, axis=1)))
        Rr_save[i + 1] = np.sqrt(np.mean(np.sum(r**2, axis=1)))

    comp_elapsed = time.time() - comp_start


    # checking  r = a x numerically

    a_end = a_save[-1]
    x_end = pos
    r_end = a_end * x_end

    std_x = np.std(np.linalg.norm(x_end[:, :2], axis=1))
    std_r = np.std(np.linalg.norm(r_end[:, :2], axis=1))
    ratio = (std_r / std_x) if std_x > 0 else np.nan

    max_err  = np.max(np.linalg.norm(r_end - a_end * x_end, axis=1))
    max_diff = np.max(np.linalg.norm(r_end - x_end, axis=1))

    print("\n--- QUICK CONSISTENCY CHECK ---")
    print(f"a_end = {a_end:.6f}")
    print(f"std(r_xy)/std(x_xy) = {ratio:.6f}   (should be ~ a_end when expansion ON)")
    print(f"max|r - a*x| = {max_err:.3e}         (should be ~ 0)")
    print(f"max|r - x|   = {max_diff:.3e}         (nonzero if a_end != 1)")
    if TEST_EXPANSION_OFF:
        print("Expansion OFF => expect a_end=1 and ratio~1.")
    print("--------------------------------\n")

    if mode == "free_particle":
        drift = np.linalg.norm(av2_mean[-1, :] - av2_mean[0, :])
        print(f"[FREE-PARTICLE CHECK] ||(a^2<v>)_end - (a^2<v>)_start|| = {drift:.3e}")
        print("Expected: small (numerical drift only). If huge, your -2H v term / dt_gyr is wrong.\n")

    print(f"Done. computational time = {comp_elapsed:.3f} s")


    # plotting

    if plot_coords == "both":
        make_summary_plot(pos, a_end, mode, N, omega, mtot, eps_user,
                          t_all, KE_save, PE_save, Q_save,
                          plot_coords="physical",
                          lock_limits=True)
        make_summary_plot(pos, a_end, mode, N, omega, mtot, eps_user,
                          t_all, KE_save, PE_save, Q_save,
                          plot_coords="comoving",
                          lock_limits=True)

    elif plot_coords == "overlay":
        make_summary_plot(pos, a_end, mode, N, omega, mtot, eps_user,
                          t_all, KE_save, PE_save, Q_save,
                          plot_coords="overlay",
                          lock_limits=True)

    else:
        make_summary_plot(pos, a_end, mode, N, omega, mtot, eps_user,
                          t_all, KE_save, PE_save, Q_save,
                          plot_coords=plot_coords,
                          lock_limits=True)


    # expansion diagnostics plots 

    plot_rms_radius(t_all, a_save, Rx_save, Rr_save, mode, N, omega, mtot, eps_user)


if __name__ == "__main__":
    main()

