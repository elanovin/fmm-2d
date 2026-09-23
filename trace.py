"""
Watch the FMM run.  python walkthrough2.py

Executes the seven steps of Section 3 on the 8-particle case and prints
the contents of phi and psi after each one, so you can see the algorithm
build its state instead of reading the loops.
"""
import numpy as np
from fmm import (direct_all, box_center, box_index, well_separated,
                 interaction_list, build_tree, p2m, m2m, m2l, l2l,
                 eval_local, direct_sum)

z = np.array([0.42+0.42j, 0.45+0.40j, 0.10+0.10j, 0.90+0.15j,
              0.20+0.85j, 0.80+0.80j, 0.55+0.60j, 0.30+0.35j])
q = np.array([1.0, -1.0, 2.0, -1.0, 1.0, -2.0, 1.0, 1.0])
NAME = {0:'p1', 1:'p2', 2:'p3', 3:'p4', 4:'p5', 5:'p6', 6:'p7', 7:'p8'}
n, p = 3, 10
TARGET = (3, 3)          # the box we follow
tree = build_tree(z, n)


def rule(t):
    print("\n" + "=" * 66); print(t); print("=" * 66)


rule("THE TREE  (build_tree)")
print(f"{len(tree)} occupied boxes out of {4**n}.  The other "
      f"{4**n - len(tree)} are empty and never created.\n")
for k in sorted(tree):
    who = ", ".join(NAME[i] for i in tree[k])
    print(f"  level {k[0]}  box ({k[1]},{k[2]})   holds {who}")

# ---------------------------------------------------------------- STEP 1
rule("STEP 1 - P2M     particles -> multipole,  finest level only")
phi = {}
for (lev, c, r), idx in tree.items():
    phi[(n, c, r)] = p2m(z[idx], q[idx], box_center(n, c, r), p)
print("  Theorem 2.1 applied once per occupied box.  This is where")
print("  particles become coefficients - M2M, M2L and L2L never see")
print("  a particle again.\n")
print(f"  {'box':>10}   {'Q':>7}   {'a1':>26}")
for k in sorted(phi):
    a = phi[k]
    print(f"  {str(k[1:]):>10}   {a[0].real:7.3f}   {a[1]:>26.5f}")
print("\n  Note box (3,3): Q = 0 because +1 and -1 cancel, but a1 is not 0.")
print("  A neutral box still radiates - that is the dipole.")

# ---------------------------------------------------------------- STEP 2
rule("STEP 2 - M2M     multipole -> multipole,  UP the tree")
for lev in range(n - 1, -1, -1):
    made = []
    for (l0, c, r) in [k for k in list(phi) if k[0] == lev + 1]:
        pc, pr = c // 2, r // 2
        if (lev, pc, pr) not in phi:
            phi[(lev, pc, pr)] = np.zeros(p + 1, dtype=complex)
            made.append((pc, pr))
        phi[(lev, pc, pr)] += m2m(phi[(lev + 1, c, r)],
                                  box_center(lev + 1, c, r) - box_center(lev, pc, pr), p)
    boxes = sorted(k for k in phi if k[0] == lev)
    print(f"\n  level {lev}:  {len(boxes)} boxes now have an expansion")
    for k in boxes:
        kids = [kk[1:] for kk in phi if kk[0] == lev + 1
                and kk[1] // 2 == k[1] and kk[2] // 2 == k[2]]
        print(f"    box {str(k[1:]):>8}   Q = {phi[k][0].real:6.2f}   "
              f"built from children {kids}")
print("\n  No particle was touched. Coefficients only - that is why it is O(p^2)")
print("  per box regardless of how many particles lie underneath.")

# ---------------------------------------------------------------- STEPS 3,4
rule(f"STEPS 3a, 3b, 4 - the downward pass,  following box {TARGET}")
psi = {}
for lev in range(2, n + 1):
    for (l0, c, r) in [k for k in phi if k[0] == lev]:
        zc = box_center(lev, c, r)
        if lev == 2:
            b = np.zeros(p + 1, dtype=complex)
        else:
            pc, pr = c // 2, r // 2
            b = l2l(psi[(lev - 1, pc, pr)], box_center(lev - 1, pc, pr) - zc, p)
        inherited = b[0]
        added = []
        for (cc, rr) in interaction_list(lev, (c, r)):
            if (lev, cc, rr) in phi:
                b += m2l(phi[(lev, cc, rr)], box_center(lev, cc, rr) - zc, p)
                added.append((cc, rr))
        psi[(lev, c, r)] = b
        if (c, r) == (TARGET[0] >> (n - lev), TARGET[1] >> (n - lev)):
            print(f"\n  LEVEL {lev},  box ({c},{r})"
                  f"      <- the ancestor of {TARGET}")
            if lev == 2:
                print("    inherited : nothing.  Psi at level 1 is zero in free space,")
                print("                so the downward pass starts here.")
            else:
                print(f"    inherited : b0 = {inherited:.5f}   (L2L from the parent)")
            ilist = interaction_list(lev, (c, r))
            print(f"    interaction list: {len(ilist)} boxes, of which "
                  f"{len(added)} are occupied -> {added}")
            who = []
            for (cc, rr) in added:
                for k2 in tree:
                    if k2[0] == n and (k2[1] >> (n - lev), k2[2] >> (n - lev)) == (cc, rr):
                        who += [NAME[i] for i in tree[k2]]
            print(f"    so this level delivers: {', '.join(who) if who else '-'}")
            print(f"    Psi complete: b0 = {b[0]:.5f}")

# ---------------------------------------------------------------- STEP 5
rule("STEP 5 - evaluate  Psi at the particles of the finest box")
zc = box_center(n, *TARGET)
far = {}
for i in tree[(n,) + TARGET]:
    far[i] = eval_local(psi[(n,) + TARGET], z[i], zc, p)
    print(f"  {NAME[i]}   FAR = {far[i].real:.8f}      "
          f"one polynomial evaluation, {p+1} terms")
print("\n  Same coefficients for both particles - only the input differs.")

# ---------------------------------------------------------------- STEP 6
rule("STEP 6 - direct summation over the 9-box neighbourhood")
near = direct_sum(z, q, tree, n)
nb = []
for cc in range(TARGET[0]-1, TARGET[0]+2):
    for rr in range(TARGET[1]-1, TARGET[1]+2):
        if (n, cc, rr) in tree:
            nb += [NAME[i] for i in tree[(n, cc, rr)]]
print(f"  neighbours of {TARGET} holding particles: {', '.join(nb)}")
print(f"  p1 NEAR = {near[0].real:.8f}   (brute force over 3 particles, not 7)")

# ---------------------------------------------------------------- STEP 7
rule("STEP 7 - add,  and check")
tot = far[0] + near[0]
exact = direct_all(z, q)[0]
print(f"  FAR  {far[0].real:>14.10f}")
print(f"  NEAR {near[0].real:>14.10f}")
print(f"  ---------------------------")
print(f"  FMM  {tot.real:>14.10f}")
print(f"  exact{exact.real:>14.10f}")
print(f"\n  absolute error {abs(tot.real-exact.real):.2e}")
A = np.sum(np.abs(q))
print(f"  bound A*2^-p = {A*2.0**-p:.2e}    (A = {A:.0f}, p = {p})")
print("\n  Seven orders of magnitude under the bound. The bound assumes every")
print("  charge sits on the circle and every target at the closest legal point.")