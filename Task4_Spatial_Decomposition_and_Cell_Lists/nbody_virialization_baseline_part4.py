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
############################
# MOD (John): extra import 
#############################
import time

############################
# MOD (John): periodic mesh/FFT gravity utilities + driver
#############################



G = 1.0


# periodic wrapping into [box_min, box_max)

def wrap_periodic(x, box_min, box_max):
    L = box_max - box_min
    return box_min + np.mod(x - box_min, L)


# part 1: assigning particles to grid cells (Ng x Ng x Ng)

def assign_particles_to_cells(x, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng

    u = (x - box_min) / h
    ijk = np.floor(u).astype(int)
    ijk = np.clip(ijk, 0, Ng - 1)

    cell_id = ijk[:, 0] + Ng * ijk[:, 1] + (Ng * Ng) * ijk[:, 2]

    cell_lists = [[] for _ in range(Ng**3)]
    for p, cid in enumerate(cell_id):
        cell_lists[cid].append(p)

    return ijk, cell_id, cell_lists

def grid_sanity(cell_lists):
    counts = np.array([len(lst) for lst in cell_lists])
    occupied = int(np.count_nonzero(counts))
    max_in_cell = int(counts.max()) if counts.size else 0
    mean_occ = float(counts[counts > 0].mean()) if occupied > 0 else 0.0
    return occupied, max_in_cell, mean_occ


# part 2: building density grid rho(i,j,k)

def build_density_grid(cell_lists, m, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng
    rho = np.zeros((Ng, Ng, Ng), dtype=float)

    for cid, plist in enumerate(cell_lists):
        if not plist:
            continue
        i = cid % Ng
        j = (cid // Ng) % Ng
        k = cid // (Ng * Ng)
        rho[i, j, k] = np.sum(m[plist]) / (h**3)

    return rho

def mass_conservation_check(rho, m, box_min, box_max, Ng):
    L = box_max - box_min
    h = L / Ng
    M_grid = float(np.sum(rho) * (h**3))
    M_true = float(np.sum(m))
    return M_grid, M_true, abs(M_grid - M_true)


# part 3: FFT Poisson solver and mesh acceleration (spectral)

def k_grids(Ng, L):
    """
    Returns kx, ky, kz with units of 1/length, shaped (Ng,Ng,Ng)
    """
    k1 = 2.0 * np.pi * np.fft.fftfreq(Ng, d=L/Ng)  # (Ng,)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    return kx, ky, kz

def poisson_solve_fft(rho, box_min, box_max):
    """
    Solve ∇^2 Φ = 4πG ρ on periodic grid using FFT.
    Returns real-space Phi (Ng,Ng,Ng) and k-space Phi_k.
    """
    Ng = rho.shape[0]
    L = box_max - box_min

    rho_k = np.fft.fftn(rho)
    kx, ky, kz = k_grids(Ng, L)
    k2 = kx**2 + ky**2 + kz**2

    phi_k = np.zeros_like(rho_k, dtype=complex)

    # avoiding  divide by zero: setting k=0 mode to 0 (removing mean potential)
    mask = k2 > 0.0
    phi_k[mask] = -4.0 * np.pi * G * rho_k[mask] / k2[mask]
    phi_k[~mask] = 0.0 + 0.0j

    phi = np.fft.ifftn(phi_k)
    return phi.real, phi_k

def mesh_acceleration_spectral(phi_k, box_min, box_max):
    """
    Compute a = -∇Φ using spectral derivatives:
      ax_k = -i kx Φ_k, etc., then inverse FFT.
    Returns ax, ay, az (each Ng,Ng,Ng), real.
    """
    Ng = phi_k.shape[0]
    L = box_max - box_min
    kx, ky, kz = k_grids(Ng, L)

    ax_k = -(1j * kx) * phi_k
    ay_k = -(1j * ky) * phi_k
    az_k = -(1j * kz) * phi_k

    ax = np.fft.ifftn(ax_k).real
    ay = np.fft.ifftn(ay_k).real
    az = np.fft.ifftn(az_k).real
    return ax, ay, az

def sample_mesh_acceleration(x, ax, ay, az, box_min, box_max, Ng):
    """
    Nearest-grid-point (NGP) sampling: particle feels accel of its cell.
    (Simple and enough for this assignment stage.)
    """
    L = box_max - box_min
    h = L / Ng
    u = (x - box_min) / h
    ijk = np.floor(u).astype(int)
    ijk = np.clip(ijk, 0, Ng - 1)

    a = np.zeros_like(x)
    a[:, 0] = ax[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    a[:, 1] = ay[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    a[:, 2] = az[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return a


# integrators

def leapfrog_step_mesh(x, v, m, dt, box_min, box_max, Ng):
    """
    One leapfrog step using mesh-only forces.
    """
    # build mesh
    _, _, cell_lists = assign_particles_to_cells(x, box_min, box_max, Ng)
    rho = build_density_grid(cell_lists, m, box_min, box_max, Ng)
    phi, phi_k = poisson_solve_fft(rho, box_min, box_max)
    ax, ay, az = mesh_acceleration_spectral(phi_k, box_min, box_max)
    a = sample_mesh_acceleration(x, ax, ay, az, box_min, box_max, Ng)

    v_half = v + 0.5 * dt * a
    x_new = wrap_periodic(x + dt * v_half, box_min, box_max)

    # recompute accel at new positions (leapfrog)
    _, _, cell_lists2 = assign_particles_to_cells(x_new, box_min, box_max, Ng)
    rho2 = build_density_grid(cell_lists2, m, box_min, box_max, Ng)
    _, phi_k2 = poisson_solve_fft(rho2, box_min, box_max)
    ax2, ay2, az2 = mesh_acceleration_spectral(phi_k2, box_min, box_max)
    a2 = sample_mesh_acceleration(x_new, ax2, ay2, az2, box_min, box_max, Ng)

    v_new = v_half + 0.5 * dt * a2
    return x_new, v_new, rho2, phi, phi_k2

# (optional) vectorized direct forces for comparison
def compute_acceleration_direct_fast(x, m, eps):
    r = x[None, :, :] - x[:, None, :]
    dist2 = np.sum(r*r, axis=2) + eps**2
    np.fill_diagonal(dist2, np.inf)
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    a = G * np.sum(r * (m[None, :, None] * inv_dist3[:, :, None]), axis=1)
    return a

def leapfrog_step_direct(x, v, m, dt, eps, box_min, box_max):
    a = compute_acceleration_direct_fast(x, m, eps)
    v_half = v + 0.5 * dt * a
    x_new = wrap_periodic(x + dt * v_half, box_min, box_max)
    a_new = compute_acceleration_direct_fast(x_new, m, eps)
    v_new = v_half + 0.5 * dt * a_new
    return x_new, v_new


# plotting

def save_phi_slice(phi, outname="phi_slice.png"):
    Ng = phi.shape[0]
    mid = Ng // 2
    plt.figure()
    plt.imshow(phi[:, :, mid], origin="lower")
    plt.colorbar(label=r"$\Phi(x,y,z_{\rm mid})$")
    plt.title("Mid-plane potential slice")
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()


# main

def main():
    N = int(input("Enter number of particles N [default=300]: ") or "300")
    tEnd = float(input("Enter tEnd [default=1.0]: ") or "1.0")
    dt = float(input("Enter dt [default=0.01]: ") or "0.01")
    Ng = int(input("Enter grid cells per axis Ng [default=10]: ") or "10")
    eps = float(input("Enter softening eps for direct compare [default=0.1]: ") or "0.1")

    mode = input("Mode: mesh or direct [default=mesh]: ").strip().lower() or "mesh"
    nsteps = int(np.round(tEnd / dt))

    np.random.seed(1)
    box_min, box_max = -1.0, 1.0
    x = np.random.uniform(box_min, box_max, size=(N, 3))
    v = np.zeros((N, 3))
    m = np.ones(N) / N

    # quick part 1 sanity
    _, _, cells0 = assign_particles_to_cells(x, box_min, box_max, Ng)
    occ0, max0, mean0 = grid_sanity(cells0)
    print("\n=== Grid sanity (t=0) ===")
    print(f"Ng^3 cells = {Ng**3}, occupied={occ0}, max_in_cell={max0}, mean_occ={mean0:.2f}")

    t0 = time.time()

    last_phi = None
    last_rho = None
    last_phi_k = None

    for step in range(nsteps):
        if mode == "mesh":
            x, v, rho, phi, phi_k = leapfrog_step_mesh(x, v, m, dt, box_min, box_max, Ng)
            last_phi, last_rho, last_phi_k = phi, rho, phi_k
            if step % 50 == 0 or step == nsteps - 1:
                M_grid, M_true, dM = mass_conservation_check(rho, m, box_min, box_max, Ng)
                imag_phi = float(np.max(np.abs(np.fft.ifftn(phi_k).imag)))
                print(f"[step {step:4d}/{nsteps}] |ΔM|={dM:.3e}, max(Im Phi)~{imag_phi:.3e}", flush=True)
        else:
            x, v = leapfrog_step_direct(x, v, m, dt, eps, box_min, box_max)
            if step % 50 == 0 or step == nsteps - 1:
                print(f"[step {step:4d}/{nsteps}] direct evolving...", flush=True)

    t1 = time.time()
    runtime = t1 - t0

    print("\n================ RUN SUMMARY ================")
    print(f"Mode={mode}")
    print(f"N={N}, steps={nsteps}, dt={dt}, tEnd={tEnd}")
    print(f"Grid Ng={Ng} (mesh mode only)")
    print(f"Runtime: {runtime:.3f} s  ({runtime/nsteps:.6f} s/step)")
    print("============================================\n")

    # saving a phi slice if we ran mesh mode
    if mode == "mesh" and last_phi is not None:
        save_phi_slice(last_phi, outname="phi_slice.png")
        print("Saved: phi_slice.png")

if __name__ == "__main__":
    main()

