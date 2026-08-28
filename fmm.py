"""
implemention of Eq. (2.7)=sum over all sources , evaluated at each targe
Where we are in the paper: Section 2, page 282, Equation (2.7). 
It's the short paragraph just before Section 2.1, right after
 Theorem 2.1 — the one that says "In order to obtain the potential 
 (or force) at the points {yⱼ} due to the charges at the points {xᵢ}
   directly, we could compute..."
   What it is physically: the brute-force answer. Every particle feels 
   every other particle. No boxes, no tree, no expansions — that machinery 
   doesn't exist yet at this point in the paper. Each pair contributes
   qjlog(zi-zj)and you add them all up. It's just Coulomb's law in 2D
     plus superposition.
     Why we code it first: it's the thing FMM replaces, so it's also the thing
       that proves FMM right. It's slow but unarguable.
"""
import numpy as np
def direct_all (z,q):
    """Direct O(N^2) evaluation of the potential at every particle.

    Implements Eq. (2.7)- the naive pairwise
    sum that the FMM replaces. Used as ground truth for verifying fmm().
    """
    N = len(z)  #how many particles
    pot = np.zeros(N, dtype=complex)
    for i in range(N):              # target particle  (the star)
        for j in range(N):          # every source     (the red lines)
            if j != i:              # a particle does not act on itself
                pot[i] += q[j]*np.log(z[i]-z[j])       # <-- Eq. (2.7): one red line's contribution
    return pot

# ---- quick check ----                    
z = np.array([0.42+0.42j, 0.45+0.40j, 0.10+0.10j, 0.90+0.15j,
              0.20+0.85j, 0.80+0.80j, 0.55+0.60j, 0.30+0.35j])
qq = np.array([1.0, -1.0, 2.0, -1.0, 1.0, -2.0, 1.0, 1.0])

print(direct_all(z, qq)[0].real)


"""
box_center and box_index - Section 3, p. 285the paragraph describing the mesh hierarchy, and Fig. 4. *"Mesh level 0 is equivalent to the entire box, while mesh level 
 𝑙 + 1 is obtained from leve𝑙 by subdivision of each region into four equal parts."
"""
def box_center(level, col, row):
    """Centre of box (col, row) at the given tree level.

    The computational box [0,1]^2 is divided into 2^level boxes per side
    (Section 3, Fig. 4). Box (col,row) spans [col/m, (col+1)/m] in x
    and [row/m, (row+1)/m] in y, so its centre is at the midpoint of each.
    """
    m = 2 ** level
    return complex((col + 0.5) / m, (row + 0.5) / m) 


def box_index(z, level):
    """Which box at `level` contains the point z. Inverse of box_center."""
    m = 2 ** level
    col = min(int(z.real * m), m - 1)   # min() guards a particle sitting at x = 1.0
    row = min(int(z.imag*m), m-1)
    return (col, row)

print(box_center(3, 3, 3))         # want (0.4375+0.4375j)
print(box_center(2, 1, 1))         # want (0.375+0.375j)
print(box_index(0.42+0.42j, 3))    # want (3, 3)
print(box_index(0.90+0.15j, 3))    # want (7, 1)


#well_separated Section 2, p. 282 -It's the single function that decides, for every pair of boxes, whether a multipole expansion is legal
def well_separated(b1, b2):
    """True if two same-level boxes are well separated (Section 2, p. 282).

    On a uniform grid, well-separation reduces to index arithmetic:
    boxes are neighbours only when BOTH index differences are <= 1.
    Anything else has c >= 2.12, so Eq. (2.6) applies and M2L is legal.
    """
    return abs(b1[0] - b2[0]) >= 2 or abs(b1[1] - b2[1]) >= 2

print(well_separated((2, 2), (3, 2)))    # want False  (touching)
print(well_separated((2, 2), (4, 2)))    # want True   (one gap)
print(well_separated((2, 2), (5, 5)))    # want True   (far)


#interaction_list-Section 3, p. 285
def interaction_list(level, box):
    """
     Interaction list of `box` at `level` (Section 3, p. 285, Fig. 5).
    Returns the boxes whose multipole expansions must be converted to a
    local expansion about `box` at this level: the children of the nearest
    neighbours of `box`'s parent that are well separated from `box`.
    At most 27 entries; fewer for boxes near the edge of the domain.
    """

    parent = (box[0] // 2, box[1] // 2)       # integer division: which parent
    m_parent = 2 ** (level - 1)               # boxes per side one level up
    result = []

    for pc in range(parent[0] - 1, parent[0] + 2):        # parent's 3 columns
        for pr in range(parent[1] - 1, parent[1] + 2):    # parent's 3 rows
            if not (0 <= pc < m_parent and 0 <= pr < m_parent):
                continue                                  # off the grid, skip
            for cc in (2 * pc, 2 * pc + 1):               # that box's 2 child columns
                for cr in (2 * pr, 2 * pr + 1):
                    if well_separated((cc, cr), box):   # keep only well-separated ones
                        result.append((cc, cr))
    return result

print(len(interaction_list(3, (3, 3))))              # want 27
print((0, 0) in interaction_list(3, (3, 3)))         # want True   (p3's box)
print((2, 2) in interaction_list(3, (3, 3)))         # want False  (touches i)
print(len(interaction_list(2, (1, 1))))              # want 7      (near tree top)