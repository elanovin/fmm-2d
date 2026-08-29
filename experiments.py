"""
Numerical experiments for the 2D fast multipole method.

Reproduces the two claims of Greengard & Rokhlin 1997 that can be
measured directly:

  1. accuracy is geometric in p          (Eq. 2.6:  error <= A * 2^-p)
  2. cost is linear in N                 (Section 3, p. 287)

Run with:   python experiments.py
Writes:     figures/error_vs_p.png
            figures/runtime_vs_N.png
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmm import fmm, direct_all

BLUE, RED, GREY = "#1a5490", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def random_particles(N, seed):
    rng = np.random.default_rng(seed)
    z = rng.random(N) + 1j * rng.random(N)
    q = rng.normal(size=N)
    return z, q


def depth_for(N):
    """Tree depth n ~ log_4(N): keeps O(1) particles per finest box."""
    return max(2, int(round(np.log(N) / np.log(4))))


# ---------------------------------------------------------------
# Experiment 1 - accuracy versus p
# ---------------------------------------------------------------

def error_vs_p(N=500, seed=7, p_values=range(1, 27)):
    z, q = random_particles(N, seed)
    n = depth_for(N)
    exact = direct_all(z, q).real
    scale = np.max(np.abs(exact))

    ps, errs = [], []
    for p in p_values:
        approx = fmm(z, q, n, p).real
        e = np.max(np.abs(approx - exact)) / scale
        ps.append(p)
        errs.append(e)
        print(f"  p = {p:2d}   relative error = {e:.3e}")

    ps, errs = np.array(ps), np.array(errs)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.semilogy(ps, 2.0**(-ps), color=GREY, ls="--", lw=2,
                label=r"bound of Eq. (2.6):  $2^{-p}$")
    ax.semilogy(ps, errs, color=BLUE, lw=2, marker="o", ms=5,
                label=f"measured, N = {N}, depth {n}")

    fit = np.polyfit(ps[3:], np.log(errs[3:]), 1)
    rate = np.exp(-fit[0])
    ax.text(0.04, 0.06, f"measured decay: ${rate:.2f}^{{-p}}$\n"
                        f"faster than $2^{{-p}}$ because typical\n"
                        f"separations exceed the worst case $c = 2.12$",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY))

    ax.set_xlabel("truncation order  $p$")
    ax.set_ylabel("max relative error vs direct summation")
    ax.set_title("Accuracy is geometric in $p$", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/error_vs_p.png", dpi=160)
    print("  -> figures/error_vs_p.png")
    return ps, errs


# ---------------------------------------------------------------
# Experiment 2 - runtime versus N
# ---------------------------------------------------------------

def runtime_vs_N(p=10, seed=3,
                 N_both=(100, 200, 400, 800, 1600, 3200),
                 N_fmm_only=(6400, 12800)):
    n_list, t_fmm, t_dir = [], [], []

    for N in N_both:
        z, q = random_particles(N, seed)
        n = depth_for(N)
        t0 = time.perf_counter(); fmm(z, q, n, p);   t1 = time.perf_counter()
        t2 = time.perf_counter(); direct_all(z, q);  t3 = time.perf_counter()
        n_list.append(N); t_fmm.append(t1 - t0); t_dir.append(t3 - t2)
        print(f"  N = {N:6d}  depth {n}   fmm {t1-t0:7.2f}s   direct {t3-t2:7.2f}s")

    n_far, t_far = list(n_list), list(t_fmm)
    for N in N_fmm_only:
        z, q = random_particles(N, seed)
        n = depth_for(N)
        t0 = time.perf_counter(); fmm(z, q, n, p); t1 = time.perf_counter()
        n_far.append(N); t_far.append(t1 - t0)
        print(f"  N = {N:6d}  depth {n}   fmm {t1-t0:7.2f}s   direct  (not run)")

    N_both = np.array(n_list); t_fmm = np.array(t_fmm); t_dir = np.array(t_dir)
    n_far = np.array(n_far);   t_far = np.array(t_far)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ref = np.array([N_both[0], n_far[-1]], dtype=float)
    ax.loglog(ref, t_fmm[0] * (ref / N_both[0]),      color=GREY, ls=":", lw=1.6)
    ax.loglog(ref, t_dir[0] * (ref / N_both[0])**2,   color=GREY, ls=":", lw=1.6)
    ax.text(n_far[-1] * 0.95, t_fmm[0] * (n_far[-1] / N_both[0]) * 0.42,
            r"slope 1   $O(N)$", color=GREY, ha="right", va="top", fontsize=10)
    ax.text(N_both[-1] * 0.98, t_dir[0] * (N_both[-1] / N_both[0])**2 * 2.2,
            r"slope 2   $O(N^2)$", color=GREY, ha="right", va="bottom", fontsize=10)

    ax.loglog(n_far, t_far, color=BLUE, lw=2, marker="o", ms=6,
              label=f"FMM,  p = {p}")
    ax.loglog(N_both, t_dir, color=RED, lw=2, marker="s", ms=6,
              label="direct summation")

    lo = np.log(t_fmm) - np.log(t_dir)
    k = np.argmax(lo < 0)
    if 0 < k < len(N_both):
        x0, x1 = np.log(N_both[k-1]), np.log(N_both[k])
        y0, y1 = lo[k-1], lo[k]
        Nc = np.exp(x0 - y0 * (x1 - x0) / (y1 - y0))
        ax.axvline(Nc, color=GREY, lw=1, ls="-")
        ax.text(Nc * 1.10, t_far[0] * 0.35, f"crossover\nN $\\approx$ {Nc:.0f}",
                fontsize=10, color=GREY)

    ax.set_xlabel("number of particles  $N$")
    ax.set_ylabel("wall-clock time  [s]")
    ax.set_title("Cost is linear in $N$", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/runtime_vs_N.png", dpi=160)
    print("  -> figures/runtime_vs_N.png")


if __name__ == "__main__":
    print("Experiment 1: accuracy versus p")
    error_vs_p()
    print("\nExperiment 2: runtime versus N")
    runtime_vs_N()
    print("\nDone.")