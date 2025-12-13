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

############################
# MOD (John): imports 
#############################
import time

G = 1.0


# utilities

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
    return abs(M_grid - M_true)


# direct (vectorized) forces

def accel_direct_fast(x, m, eps, box_min, box_max):
    # periodic minimum-image for pairwise displacements
    Lbox = box_max - box_min
    r = x[None, :, :] - x[:, None, :]                # (N,N,3)
    r = r - Lbox * np.round(r / Lbox)                # minimum img

    dist2 = np.sum(r*r, axis=2) + eps**2
    np.fill_diagonal(dist2, np.inf)
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    a = G * np.sum(r * (m[None, :, None] * inv_dist3[:, :, None]), axis=1)
    return a

def step_direct(x, v, m, dt, eps, box_min, box_max):
    a = accel_direct_fast(x, m, eps, box_min, box_max)
    v_half = v + 0.5 * dt * a
    x_new = wrap_periodic(x + dt * v_half, box_min, box_max)
    a_new = accel_direct_fast(x_new, m, eps, box_min, box_max)
    v_new = v_half + 0.5 * dt * a_new
    return x_new, v_new


# FFT poisson + spectral accel + NGP sampling

def k_grids(Ng, L):
    k1 = 2.0 * np.pi * np.fft.fftfreq(Ng, d=L/Ng)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    return kx, ky, kz

def mesh_accel_filtered(rho, box_min, box_max, r_split=None):
    """
    If r_split is None -> unfiltered (mesh-only).
    If r_split is float -> Gaussian low-pass W(k)=exp(-(k r_split)^2).
    """
    Ng = rho.shape[0]
    L = box_max - box_min

    rho_k = np.fft.fftn(rho)
    kx, ky, kz = k_grids(Ng, L)
    k2 = kx**2 + ky**2 + kz**2

    if r_split is None:
        W = 1.0

    else:

############################
# MOD (John): set k 
#############################
        k = np.sqrt(k2)
        W = np.exp(-(k * r_split)**2)

    phi_k = np.zeros_like(rho_k, dtype=complex)
    mask = k2 > 0.0
    phi_k[mask] = -4.0 * np.pi * G * (rho_k[mask] * (W[mask] if hasattr(W, "__getitem__") else W)) / k2[mask]
    phi_k[~mask] = 0.0 + 0.0j

    ax_k = -(1j * kx) * phi_k
    ay_k = -(1j * ky) * phi_k
    az_k = -(1j * kz) * phi_k

    ax = np.fft.ifftn(ax_k).real
    ay = np.fft.ifftn(ay_k).real
    az = np.fft.ifftn(az_k).real

    max_im_phi = float(np.max(np.abs(np.fft.ifftn(phi_k).imag)))
    return ax, ay, az, max_im_phi

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

    return a


##############################################
# MOD (John): update function step_mesh_only 
#############################################
def step_mesh_only(x, v, m, dt, box_min, box_max, Ng):
    ijk, cid, cell_lists = assign_particles_to_cells(x, box_min, box_max, Ng)
    rho = build_density_grid(cell_lists, m, box_min, box_max, Ng)
    ax, ay, az, max_im_phi = mesh_accel_filtered(rho, box_min, box_max, r_split=None)
    a = sample_mesh_accel_NGP(x, ax, ay, az, box_min, box_max, Ng)

    v_half = v + 0.5 * dt * a
    x_new = wrap_periodic(x + dt * v_half, box_min, box_max)

    ijk2, cid2, cell_lists2 = assign_particles_to_cells(x_new, box_min, box_max, Ng)
    rho2 = build_density_grid(cell_lists2, m, box_min, box_max, Ng)
    ax2, ay2, az2, max_im_phi2 = mesh_accel_filtered(rho2, box_min, box_max, r_split=None)
    a2 = sample_mesh_accel_NGP(x_new, ax2, ay2, az2, box_min, box_max, Ng)

    v_new = v_half + 0.5 * dt * a2
    return x_new, v_new, rho2, max(max_im_phi, max_im_phi2)


# local PP (27 cell neighborhood)

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
                r = r - Lbox * np.round(r / Lbox)
                dist2 = np.dot(r, r) + eps**2
                ap += G * m[q] * r / dist2**1.5

        a[p] = ap

    return a


############################
# MOD (John): update function step_hybrid_split (block 4)
#############################
def step_hybrid_split(x, v, m, dt, eps, box_min, box_max, Ng, r_split):
    # full cell lists + rho
    ijk, cid, cell_lists = assign_particles_to_cells(x, box_min, box_max, Ng)
    rho = build_density_grid(cell_lists, m, box_min, box_max, Ng)

    # filtered (long range) mesh accel
    ax, ay, az, max_im_phi = mesh_accel_filtered(rho, box_min, box_max, r_split=r_split)
    a_mesh = sample_mesh_accel_NGP(x, ax, ay, az, box_min, box_max, Ng)

    # local PP accel (short range)
    a_pp = accel_local_PP(x, m, eps, box_min, box_max, Ng, ijk, cell_lists)

    a = a_mesh + a_pp

    # leapfrog
    v_half = v + 0.5 * dt * a
    x_new = wrap_periodic(x + dt * v_half, box_min, box_max)

    # accel at new positions
    ijk2, cid2, cell_lists2 = assign_particles_to_cells(x_new, box_min, box_max, Ng)
    rho2 = build_density_grid(cell_lists2, m, box_min, box_max, Ng)

    ax2, ay2, az2, max_im_phi2 = mesh_accel_filtered(rho2, box_min, box_max, r_split=r_split)
    a_mesh2 = sample_mesh_accel_NGP(x_new, ax2, ay2, az2, box_min, box_max, Ng)
    a_pp2 = accel_local_PP(x_new, m, eps, box_min, box_max, Ng, ijk2, cell_lists2)

    a2 = a_mesh2 + a_pp2
    v_new = v_half + 0.5 * dt * a2

    return x_new, v_new, rho2, max(max_im_phi, max_im_phi2)


# bench runner

def run_mode(mode, N, nsteps, dt, eps, Ng, r_split_h, seed=1):
    box_min, box_max = -1.0, 1.0
    L = box_max - box_min
    h = L / Ng
    r_split = r_split_h * h

    rng = np.random.default_rng(seed)
    x = rng.uniform(box_min, box_max, size=(N, 3))
    v = np.zeros((N, 3))
    m = np.ones(N) / N

    max_im_phi = 0.0
    max_dM = 0.0

    t0 = time.perf_counter()
    for _ in range(nsteps):
        if mode == "direct":
            x, v = step_direct(x, v, m, dt, eps, box_min, box_max)
        elif mode == "mesh":
            x, v, rho, imphi = step_mesh_only(x, v, m, dt, box_min, box_max, Ng)
            max_im_phi = max(max_im_phi, imphi)
            max_dM = max(max_dM, mass_conservation_check(rho, m, box_min, box_max, Ng))
        elif mode == "hybrid":
            x, v, rho, imphi = step_hybrid_split(x, v, m, dt, eps, box_min, box_max, Ng, r_split)
            max_im_phi = max(max_im_phi, imphi)
            max_dM = max(max_dM, mass_conservation_check(rho, m, box_min, box_max, Ng))
        else:
            raise ValueError("mode must be direct, mesh, or hybrid")
    t1 = time.perf_counter()

    runtime = t1 - t0
    return runtime, runtime / nsteps, max_dM, max_im_phi

def print_table(rows):
    # rows: list of dicts
    headers = ["N", "direct(s/step)", "mesh(s/step)", "hybrid(s/step)", "speedup(mesh)", "speedup(hybrid)"]
    print("\n" + "="*88)
    print("{:>6}  {:>14}  {:>12}  {:>13}  {:>13}  {:>15}".format(*headers))
    print("-"*88)
    for r in rows:
        print("{:>6}  {:>14.6e}  {:>12.6e}  {:>13.6e}  {:>13.3f}  {:>15.3f}".format(
            r["N"], r["direct_sps"], r["mesh_sps"], r["hybrid_sps"],
            r["direct_sps"]/r["mesh_sps"], r["direct_sps"]/r["hybrid_sps"]
        ))
    print("="*88 + "\n")



def main():

############################
# MOD (John): set Ns_str 
#############################
    # inputs
    Ns_str = input("Enter N list (comma-separated) [default=100,300,600,1000]: ").strip() or "100,300,600,1000"
    Ns = [int(s) for s in Ns_str.split(",")]

    tEnd = float(input("Enter tEnd [default=0.2]: ") or "0.2")
    dt = float(input("Enter dt [default=0.01]: ") or "0.01")
    eps = float(input("Enter eps [default=0.1]: ") or "0.1")
    Ng = int(input("Enter Ng [default=10]: ") or "10")
    r_split_h = float(input("Enter r_split/h [default=2.0]: ") or "2.0")

    nsteps = int(np.round(tEnd / dt))
    print(f"\nRunning benchmark: steps={nsteps}, dt={dt}, tEnd={tEnd}, eps={eps}, Ng={Ng}, r_split/h={r_split_h}\n")

    rows = []
    for N in Ns:
        # direct
        t_dir, sps_dir, dM_dir, im_dir = run_mode("direct", N, nsteps, dt, eps, Ng, r_split_h)
        # mesh
        t_mesh, sps_mesh, dM_mesh, im_mesh = run_mode("mesh", N, nsteps, dt, eps, Ng, r_split_h)
        # hybrid
        t_hyb, sps_hyb, dM_hyb, im_hyb = run_mode("hybrid", N, nsteps, dt, eps, Ng, r_split_h)

        print(f"N={N:4d}  direct={t_dir:.3f}s  mesh={t_mesh:.3f}s  hybrid={t_hyb:.3f}s"
              f"  | mesh checks: max|ΔM|={dM_mesh:.2e}, maxImPhi={im_mesh:.2e}"
              f"  | hybrid checks: max|ΔM|={dM_hyb:.2e}, maxImPhi={im_hyb:.2e}")

        rows.append({
            "N": N,
            "direct_sps": sps_dir,
            "mesh_sps": sps_mesh,
            "hybrid_sps": sps_hyb
        })

    print_table(rows)

    print("NOTE:")
    print("- mesh is fastest but smooth (low spatial resolution).")
    print("- hybrid adds local PP cost for better small-scale forces.")
    print("- speedups grow with N if PP loops don’t dominate.\n")

if __name__ == "__main__":
    main()

