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
############### MOD (John): imports updated (cosmology + animation utilities) ###############
import time
import shutil
import subprocess

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value
############### END MOD (John) ###############

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


############### MOD (John): animation/video settings (from animai.py) ###############
# Default output folder (can be overridden in interactive prompts)
OUT_ROOT_DEFAULT = os.path.join(os.getcwd(), "TASK2_VIDEOS")
FPS = 20
FRAME_DPI = 300
SAVE_EVERY = 1      # save every Nth step as a frame (2+ if it is heavy)
ORBIT_TAIL = 50     # tail length in frames for orbit trail plotting

# Sim defaults (Task 2)
N_DEFAULT = 150
OMEGA_DEFAULT = 0.2
MTOT_DEFAULT = 20.0
EPS_PHYS_DEFAULT = 0.1

TEND = 10.0
DT = 0.01
V0 = 1.0
############### END MOD (John) ###############

############### MOD (John): video maker utilities copied from animai.py ###############

def InitialConditions_Task2(N, omega, mtot, v0, a0):
    np.random.seed(17)
    mass = mtot * np.ones((N, 1)) / N

    # physical initial positions
    r_phys = np.random.randn(N, 3)
    r_phys -= np.mean(mass * r_phys, axis=0) / np.mean(mass)

    # comoving positions
    pos = r_phys / a0

    if v0 == 1:
        x = r_phys[:, 0:1]
        y = r_phys[:, 1:2]
        z = r_phys[:, 2:3]

        # solid-body rotation in PHYSICAL coords about z
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
        vel = np.zeros((N, 3))

    return mass, pos, vel

def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it (brew install ffmpeg) and try again.")

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def make_mp4_from_frames(frames_dir, out_mp4, fps=FPS):
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

def montage_xstack(inputs, out_mp4, cols=3, rows=1, fps=FPS):
    require_ffmpeg()
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for v in inputs:
        cmd += ["-i", v]

    # layout positions
    layout_positions = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(inputs):
                break
            x = "0" if c == 0 else "+".join([f"w{j}" for j in range(c)])
            y = "0" if r == 0 else "+".join([f"h{j*cols}" for j in range(r)])
            layout_positions.append(f"{x}_{y}")

    layout = "|".join(layout_positions)
    filter_complex = (
        f"xstack=inputs={len(inputs)}:layout={layout}:fill=white,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4
    ]
    run_cmd(cmd)

def simulate_task2(mode, N, omega, mtot, eps_phys, tEnd=TEND, dt=DT, v0=V0):
    """
    mode:
      - "normal"        : expansion ON, gravity ON
      - "expansion_off" : a=1, H=0, gravity ON
      - "free_particle" : expansion ON, gravity OFF (G=0)
    """
    T_UNIT_GYR = 1.0 
    Nt = int(np.ceil(tEnd / dt))
    t_all = np.arange(Nt + 1) * dt
    dt_gyr = dt * T_UNIT_GYR

    def get_aH(t_sim):
        if mode == "expansion_off":
            return 1.0, 0.0
        return ScaleFactor(t_sim), HubbleParameter(t_sim)

    G = 1.0
    if mode == "free_particle":
        G = 0.0

    a0, H0 = get_aH(0.0)
    mass, pos, vel = InitialConditions_Task2(N, omega, mtot, v0, a0)  # pos=x, vel=xdot

    pos_save = np.zeros((N, 3, Nt + 1))
    a_save   = np.zeros(Nt + 1)
    KE_save  = np.zeros(Nt + 1)
    PE_save  = np.zeros(Nt + 1)
    Q_save   = np.zeros(Nt + 1)

    # initial accel
    t = 0.0
    a, H = get_aH(t)
    a_save[0] = a
    r = a * pos
    acc_phys = getAcc(r, mass, G, eps_phys)
    acc_com  = (acc_phys / a) - (2.0 * H * vel)

    rdot = a * vel + (a * H) * pos
    KE0, PE0 = getEnergy_physical(r, rdot, mass, G)

    KE_save[0] = KE0
    PE_save[0] = PE0
    Q_save[0]  = 2.0 * KE0 / np.abs(PE0) if PE0 != 0 else np.nan
    pos_save[:, :, 0] = pos

    for i in range(Nt):
        # kick
        vel += acc_com * (dt_gyr / 2.0)
        # drift
        pos += vel * dt_gyr
        t   += dt

        a, H = get_aH(t)
        a_save[i + 1] = a

        r = a * pos
        acc_phys = getAcc(r, mass, G, eps_phys)
        acc_com  = (acc_phys / a) - (2.0 * H * vel)

        # kick
        vel += acc_com * (dt_gyr / 2.0)

        # energies in physical space
        rdot = a * vel + (a * H) * pos
        KE, PE = getEnergy_physical(r, rdot, mass, G)

        pos_save[:, :, i + 1] = pos
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE
        Q_save[i + 1]  = 2.0 * KE / np.abs(PE) if PE != 0 else np.nan

    return {
        "mode": mode,
        "t_all": t_all,
        "pos_save": pos_save,  # comoving
        "a_save": a_save,
        "KE": KE_save,
        "PE": PE_save,
        "Q": Q_save,
        "N": N,
        "omega": omega,
        "mtot": mtot,
        "eps": eps_phys,
        "a0": a_save[0],
        "a_end": a_save[-1],
    }

def locked_limits_from_comoving(pos_save, pad_frac=0.10):
    x = pos_save[:, 0, :].ravel()
    y = pos_save[:, 1, :].ravel()
    x0, x1 = np.percentile(x, [1, 99])
    y0, y1 = np.percentile(y, [1, 99])

    dx = x1 - x0
    dy = y1 - y0
    pad_x = pad_frac * (dx if dx > 0 else 1.0)
    pad_y = pad_frac * (dy if dy > 0 else 1.0)

    xmin, xmax = x0 - pad_x, x1 + pad_x
    ymin, ymax = y0 - pad_y, y1 + pad_y
    return (xmin, xmax), (ymin, ymax)

def make_video_phys_vs_como(out_root, hist, fps=FPS, dpi=FRAME_DPI, save_every=SAVE_EVERY, orbit_tail=ORBIT_TAIL):
    tag = f"VIDEO_A_phys_vs_como_N{hist['N']}_om{hist['omega']}_M{hist['mtot']}_eps{hist['eps']}"
    case_dir = os.path.join(out_root, tag)
    frames_dir = os.path.join(case_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    t_all = hist["t_all"]
    pos_save = hist["pos_save"]
    a_save = hist["a_save"]
    KE = hist["KE"]
    PE = hist["PE"]
    Q  = hist["Q"]

    N = hist["N"]
    omega = hist["omega"]
    mtot  = hist["mtot"]
    eps   = hist["eps"]

    xlim, ylim = locked_limits_from_comoving(pos_save, pad_frac=0.10)

    # IMPORTANT: bigger width + reserved right margin so legend isn't clipped
    fig = plt.figure(figsize=(14.0, 7.2), dpi=dpi, facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.10, wspace=0.28, hspace=0.40)

    grid = plt.GridSpec(3, 2, wspace=0.28, hspace=0.40)
    ax_com = plt.subplot(grid[0:2, 0], facecolor="white")
    ax_phy = plt.subplot(grid[0:2, 1], facecolor="white")
    axE = plt.subplot(grid[2, :], facecolor="white")
    axQ = axE.twinx()

    Nt = pos_save.shape[2] - 1

    for i in range(Nt + 1):
        if (i % save_every) != 0:
            continue

        x_now = pos_save[:, :, i]
        a_now = a_save[i]
        r_now = a_now * x_now

        # ---- COMOVING ----
        ax_com.cla()
        ax_com.set_facecolor("white")
        xs = pos_save[:, 0, max(i - orbit_tail, 0):i + 1]
        ys = pos_save[:, 1, max(i - orbit_tail, 0):i + 1]
        ax_com.scatter(xs, ys, s=2, color=(0.65, 0.65, 1.0), alpha=0.8)
        ax_com.scatter(x_now[:, 0], x_now[:, 1], s=18, color="red")
        ax_com.set_xlim(*xlim); ax_com.set_ylim(*ylim)
        ax_com.set_aspect("equal", "box")
        ax_com.set_title("Comoving positions (x)")
        ax_com.set_xlabel("x (comoving)")
        ax_com.set_ylabel("y (comoving)")
        ax_com.text(
            0.02, 0.98,
            f"mode=normal\nN={N}\nω={omega}\nM={mtot}\nε={eps}\na(t)={a_now:.3f}",
            transform=ax_com.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
        )

        # ---- PHYSICAL ----
        ax_phy.cla()
        ax_phy.set_facecolor("white")
        rr_x = (a_save[max(i - orbit_tail, 0):i + 1] * pos_save[:, 0, max(i - orbit_tail, 0):i + 1])
        rr_y = (a_save[max(i - orbit_tail, 0):i + 1] * pos_save[:, 1, max(i - orbit_tail, 0):i + 1])
        ax_phy.scatter(rr_x, rr_y, s=2, color=(0.65, 0.65, 1.0), alpha=0.8)
        ax_phy.scatter(r_now[:, 0], r_now[:, 1], s=18, color="red")

        # KEY: same axis limits to show shrink/scale
        ax_phy.set_xlim(*xlim); ax_phy.set_ylim(*ylim)

        ax_phy.set_aspect("equal", "box")
        ax_phy.set_title("Physical positions (r = a x)")
        ax_phy.set_xlabel("x (physical)")
        ax_phy.set_ylabel("y (physical)")
        ax_phy.text(
            0.02, 0.98,
            f"a(t)={a_now:.3f}",
            transform=ax_phy.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
        )

        # ---- ENERGY ----
        axE.cla(); axE.set_facecolor("white")
        axQ.cla()

        axE.plot(t_all, KE, color="red",   linewidth=1, label="KE (phys)")
        axE.plot(t_all, PE, color="blue",  linewidth=1, label="PE (phys)")
        axE.plot(t_all, KE + PE, color="black", linewidth=1, label="Etot (phys)")
        axE.plot(t_all, 2 * KE + PE, color="pink", linewidth=1, label="2KE+PE (phys)")

        axE.set_xlim(0, t_all[-1] + 0.5)
        axE.set_xlabel("t (sim units)")
        axE.set_ylabel("energy")

        axQ.plot(t_all, Q, color="green", linewidth=0.9, label="Q=2KE/|PE|")
        axQ.set_ylim(0, 3.0)
        axQ.set_ylabel("Q")
        axQ.yaxis.set_label_coords(1.01, 0.5)
        axQ.tick_params(axis="y", pad=8)

        # Legend placed in figure margin (no cropping)
        hE, lE = axE.get_legend_handles_labels()
        hQ, lQ = axQ.get_legend_handles_labels()
        fig.legend(hE + hQ, lE + lQ,
                   loc="upper left",
                   bbox_to_anchor=(0.80, 0.92),
                   borderaxespad=0.0)

        frame_path = os.path.join(frames_dir, f"nbody_{i:05d}.png")
        plt.savefig(frame_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    out_mp4 = os.path.join(case_dir, f"{tag}.mp4")
    make_mp4_from_frames(frames_dir, out_mp4, fps=fps)
    return out_mp4

def make_video_single_mode(out_root, hist, fps=FPS, dpi=FRAME_DPI, save_every=SAVE_EVERY, orbit_tail=ORBIT_TAIL):
    mode = hist["mode"]
    tag = f"MODE_{mode}_N{hist['N']}_om{hist['omega']}_M{hist['mtot']}_eps{hist['eps']}"
    case_dir = os.path.join(out_root, tag)
    frames_dir = os.path.join(case_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    t_all = hist["t_all"]
    pos_save = hist["pos_save"]
    a_save = hist["a_save"]
    KE = hist["KE"]
    PE = hist["PE"]
    Q  = hist["Q"]

    N = hist["N"]
    omega = hist["omega"]
    mtot  = hist["mtot"]
    eps   = hist["eps"]

    xlim, ylim = locked_limits_from_comoving(pos_save, pad_frac=0.10)

    fig = plt.figure(figsize=(7.2, 7.2), dpi=dpi, facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.72, top=0.92, bottom=0.10, hspace=0.35)

    grid = plt.GridSpec(3, 1, hspace=0.35)
    ax1 = plt.subplot(grid[0:2, 0], facecolor="white")
    ax2 = plt.subplot(grid[2, 0], facecolor="white")
    ax2b = ax2.twinx()

    Nt = pos_save.shape[2] - 1

    for i in range(Nt + 1):
        if (i % save_every) != 0:
            continue

        pos = pos_save[:, :, i]
        a_now = a_save[i]

        # positions
        ax1.cla(); ax1.set_facecolor("white")
        xs = pos_save[:, 0, max(i - orbit_tail, 0):i + 1]
        ys = pos_save[:, 1, max(i - orbit_tail, 0):i + 1]
        ax1.scatter(xs, ys, s=2, color=(0.65, 0.65, 1.0), alpha=0.8)
        ax1.scatter(pos[:, 0], pos[:, 1], s=18, color="red")
        ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
        ax1.set_aspect("equal", "box")
        ax1.set_xlabel("x (comoving)")
        ax1.set_ylabel("y (comoving)")
        ax1.set_title("Comoving positions (expansion included)")

        # energies
        ax2.cla(); ax2.set_facecolor("white")
        ax2b.cla()

        ax2.plot(t_all, KE, color="red",   linewidth=1, label="KE (phys)")
        ax2.plot(t_all, PE, color="blue",  linewidth=1, label="PE (phys)")
        ax2.plot(t_all, KE + PE, color="black", linewidth=1, label="Etot (phys)")
        ax2.plot(t_all, 2 * KE + PE, color="pink", linewidth=1, label="2KE+PE (phys)")
        ax2.set_xlim(0, t_all[-1] + 0.5)
        ax2.set_xlabel("t (sim units)")
        ax2.set_ylabel("energy")

        ax2b.plot(t_all, Q, color="green", linewidth=0.9, label="Q=2KE/|PE|")
        ax2b.set_ylim(0, 3.0)
        ax2b.set_ylabel("Q")
        ax2b.yaxis.set_label_coords(1.01, 0.5)
        ax2b.tick_params(axis="y", pad=8)

        # legend on the right margin (no clipping)
        h2, l2 = ax2.get_legend_handles_labels()
        hq, lq = ax2b.get_legend_handles_labels()
        ax1.legend(
            h2 + hq, l2 + lq,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00),
            borderaxespad=0.0
        )

        # info box
        ax1.text(
            1.02, 0.45,
            f"mode={mode}\nN={N}\nω={omega}\nM={mtot}\nε={eps}\n"
            f"a(0)={hist['a0']:.3f}\n"
            f"a(end)={hist['a_end']:.3f}",
            transform=ax1.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
        )

        frame_path = os.path.join(frames_dir, f"nbody_{i:05d}.png")
        plt.savefig(frame_path, dpi=dpi, facecolor="white")

    plt.close(fig)

    out_mp4 = os.path.join(case_dir, f"{tag}.mp4")
    make_mp4_from_frames(frames_dir, out_mp4, fps=fps)
    return out_mp4

############### END MOD (John) ###############

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

############### MOD (John): optional video/animation maker (pulled from animai.py) ###############
    make_vid = input("\nMake videos/animations now? [y/N]: ").strip().lower()
    if make_vid == "y" or make_vid == "yes":
        out_root = input(f"Output folder (default: {OUT_ROOT_DEFAULT}): ").strip()
        out_root = out_root if out_root else OUT_ROOT_DEFAULT
        os.makedirs(out_root, exist_ok=True)

        # ffmpeg is required for MP4 writing
        require_ffmpeg()

        print(f"\nVIDEO SETTINGS: out_root={out_root}, FPS={FPS}, DPI={FRAME_DPI}, SAVE_EVERY={SAVE_EVERY}, ORBIT_TAIL={ORBIT_TAIL}")

        # A) Physical vs comoving comparison (two-panel) for the same run
        hist_run = simulate_task2("normal", N, omega, mtot, eps_phys)
        mp4_A = make_video_phys_vs_como(out_root, hist_run, fps=FPS, dpi=FRAME_DPI, save_every=SAVE_EVERY, orbit_tail=ORBIT_TAIL)
        print(f"Saved: {mp4_A}")

        # B) Validation: normal vs expansion_off vs free_particle (stacked montage)
        hist_normal = hist_run
        hist_off    = simulate_task2("expansion_off", N, omega, mtot, eps_phys)
        hist_free   = simulate_task2("free_particle", N, omega, mtot, eps_phys)

        mp4_normal = make_video_single_mode(out_root, hist_normal, fps=FPS, dpi=FRAME_DPI, save_every=SAVE_EVERY, orbit_tail=ORBIT_TAIL)
        mp4_off    = make_video_single_mode(out_root, hist_off,    fps=FPS, dpi=FRAME_DPI, save_every=SAVE_EVERY, orbit_tail=ORBIT_TAIL)
        mp4_free   = make_video_single_mode(out_root, hist_free,   fps=FPS, dpi=FRAME_DPI, save_every=SAVE_EVERY, orbit_tail=ORBIT_TAIL)

        montage_path = os.path.join(out_root, "VIDEO_B_modes_validation_3x1.mp4")
        montage_xstack([mp4_normal, mp4_off, mp4_free], montage_path, cols=3, rows=1, fps=FPS)
        print(f"Saved: {montage_path}")

############### END MOD (John) ###############



if __name__ == "__main__":
    main()
