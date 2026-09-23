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
from math import comb

def direct_all (z,q):
    """Direct O(N^2) evaluation of the potential at every particle.

    Eq. (2.7) gives the structure - the potential at one particle is the
    sum of the fields of all others. The log itself comes from the
    definition of phi on p. 281. Used as ground truth for verifying fmm().
    """
    N = len(z)  #how many particles
    pot = np.zeros(N, dtype=complex)   #accumulating with += from zeros.
    for i in range(N):              # target particle
        for j in range(N):          # every source 
            if j != i:              # a particle does not act on itself
                pot[i] += q[j]*np.log(z[i]-z[j])       # Eq. (2.7)  
    return pot


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



#well_separated Section 2, p. 282 -It's the single function that decides, for every pair of boxes, whether a multipole expansion is legal
def well_separated(b1, b2):
    """True if two same-level boxes are well separated (Section 2, p. 282).

    On a uniform grid, well-separation reduces to index arithmetic:
    boxes are neighbours only when BOTH index differences are <= 1.
    Anything else has c >= 2.12, so Eq. (2.6) applies and M2L is legal.
    """
    return abs(b1[0] - b2[0]) >= 2 or abs(b1[1] - b2[1]) >= 2



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

    for pc in range(parent[0] - 1, parent[0] + 2):        # columns 0, 1, 2
        for pr in range(parent[1] - 1, parent[1] + 2):    # rows 0, 1, 2
            if not (0 <= pc < m_parent and 0 <= pr < m_parent):   #if this parent isn't on the grid, skip it.
                continue                                 
            for cc in (2 * pc, 2 * pc + 1):               # that box's 2 child columns
                for cr in (2 * pr, 2 * pr + 1):
                    if well_separated((cc, cr), box):   # keep only well-separated ones
                        result.append((cc, cr))
    return result



#build_tree- Section 3, p. 285–286. The tree is described on p. 285
"""
But one structural point worth having: only the finest level needs a particle list.
 Levels above never touch particles at all; they work purely with the 
Φ coefficients handed up by M2M. That's a big part of why the method is fast.
"""

def build_tree(z, n):
    """Sort particles into their finest-level boxes (Section 3, p. 285-286).

    Only the finest level needs a particle list: coarser levels work
    entirely with multipole coefficients passed up by M2M.

    Returns
    -------
    dict : (level, col, row) -> list of particle indices in that box
    """
    boxes = {}
    for i in range(len(z)):
        b = box_index(z[i], n)
        key = (n, b[0], b[1])
        if key not in boxes:
            boxes[key] = []
        boxes[key].append(i)
    return boxes




#p2m

def p2m(z_src, q_src, zc, p):
    """Particles -> multipole expansion (Theorem 2.1, Eq. 2.3).

    Compresses the charges in one box into p+1 coefficients describing the
    field they create OUTSIDE the box. Valid only for |z - zc| > r, where r
    is the radius of the circle containing the charges.

    Parameters
    ----------
    z_src : complex array   positions of the charges in this box
    q_src : float array     their charges
    zc    : complex         box centre -- the expansion is written about this
    p     : int             number of terms

    Returns
    -------
    complex array of length p+1 : [Q, a_1, ..., a_p]
    """
    a = np.zeros(p + 1, dtype=complex)
    a[0] = np.sum(q_src)                                # Q, the monopole
    for k in range(1, p + 1):
        a[k] = -np.sum(q_src * (z_src - zc)**k) / k     # Eq. (2.3)
    return a



#m2m

def m2m(a, z0, p):
    """
    M2M: shift a multipole expansion to a new (parent) centre.

    Greengard & Rokhlin 1997, Lemma 2.3, p. 283.
    Used in Section 3, Step 2 (p. 286).

    Input  a  : coefficients about the CHILD centre, a[0] = Q
           z0 : child centre minus parent centre (the offset)
           p  : truncation order
    Output b  : coefficients about the PARENT centre.

    Exact - no approximation, no error bound. b[0] = a[0] because
    total charge does not depend on where you measure it from.
    """
    b = np.zeros(p + 1, dtype=complex)
    b[0] = a[0]
    for l in range(1, p + 1):
        s = -a[0] * z0**l / l            # Piece 1
        for k in range(1, l + 1):
            s += a[k] * z0**(l - k) * comb(l - 1, k - 1)   # Piece 2
        b[l] = s
    return b



#m2l
def m2l(a, z0, p):
    """
    M2L: convert a multipole (outgoing) expansion into a
    local (incoming) expansion about the target box centre.

    Greengard & Rokhlin 1997, Lemma 2.4, pp. 283-284.
    Error bound: Eq. (2.11). Used in Section 3, Steps 3a and 4.

    Input  a  : multipole coefficients of the SOURCE box, a[0] = Q
           z0 : source centre minus target centre
           p  : truncation order
    Output b  : local coefficients about the TARGET centre.

    The ONLY approximate operator in the FMM: the inner sum over k
    is infinite in the lemma and is truncated at p here. All of the
    method's error originates in this truncation.

    b[0] uses log(-z0); the branch choice shifts only the imaginary
    part, and the physical potential is the real part, so it is safe.
    """
    b = np.zeros(p + 1, dtype=complex)

    s = 0
    for k in range(1, p + 1):
        s += a[k] / z0**k * (-1)**k
    b[0] = a[0] * np.log(-z0) + s

    for l in range(1, p + 1):
        s = 0
        for k in range(1, p + 1):
            s += a[k] / z0**k * comb(l + k - 1, k - 1) * (-1)**k
        b[l] = -a[0] / (l * z0**l) + s / z0**l

    return b



#l2l

def l2l(a, z0, p):
    """
    L2L: shift a local expansion to a new (child) centre.

    Greengard & Rokhlin 1997, Lemma 2.5, p. 284.
    Used in Section 3, Step 3b (p. 286).

    Input  a  : local coefficients about the PARENT centre
           z0 : parent centre minus child centre
           p  : truncation order
    Output b  : local coefficients about the CHILD centre.

    Exact - it is the ordinary binomial theorem applied to a
    polynomial, so no tail is discarded. Lemma 2.5 carries no
    error bound, unlike Lemma 2.4.
    """
    b = np.zeros(p + 1, dtype=complex)
    for l in range(p + 1):
        s = 0
        for k in range(l, p + 1):
            s += a[k] * comb(k, l) * (-z0)**(k - l)
        b[l] = s
    return b




#eval_local
def eval_local(b, zt, zc, p):
    """
    Evaluate a local expansion at a target point.

    Greengard & Rokhlin 1997, Section 3, Step 5, p. 286.
    Computes  sum_{l=0}^{p} b[l] * (zt - zc)**l  by Horner's rule.

    Input  b  : local coefficients about zc
           zt : target particle position
           zc : box centre the expansion is anchored to
           p  : truncation order
    Output    : complex value; the physical potential is its real part.

    Cost O(p) per particle, independent of how many sources are far
    away - this is the step that makes the far field cost O(1) each.
    """
    dz = zt - zc
    v = b[p]
    for l in range(p - 1, -1, -1):
        v = v * dz + b[l]
    return v




#direct_sum
def direct_sum(z, q, tree, n):
    """
    Near-field: brute-force summation over each box and its 8 neighbours.

    Greengard & Rokhlin 1997, Section 3, Step 6, p. 286.

    These are the boxes for which no multipole expansion is legal
    (c < 1, the series diverges). Cost stays O(N) because the tree
    depth n ~ log_4(N) keeps O(1) particles in each finest box.
    """
    N = len(z)
    pot = np.zeros(N, dtype=complex)
    for (lev, c, r), idx in tree.items():
        neigh = []
        for cc in range(c - 1, c + 2):
            for rr in range(r - 1, r + 2):
                if (n, cc, rr) in tree:
                    neigh += tree[(n, cc, rr)]
        for i in idx:
            for j in neigh:
                if j != i:
                    pot[i] += q[j] * np.log(z[i] - z[j])
    return pot

#fmm
def fmm(z, q, n, p):
    """
    The 2D free-space fast multipole method.

    Greengard & Rokhlin 1997, Section 3, pp. 286-287, seven steps.

    Input  z : complex particle positions in [0,1)^2
           q : real charges
           n : tree depth (choose n ~ log_4(N))
           p : truncation order (choose p ~ -log_2(eps))
    Output   : complex potentials; physical potential is the real part.
    """
    N = len(z)
    tree = build_tree(z, n)

    # ---- Step 1: P2M at the finest level ----
    phi = {}
    for (lev, c, r), idx in tree.items():
        phi[(n, c, r)] = p2m(z[idx], q[idx], box_center(n, c, r), p)

    # ---- Step 2: M2M upward ----
    for lev in range(n - 1, -1, -1):
        for (l0, c, r) in [k for k in list(phi) if k[0] == lev + 1]:
            pc, pr = c // 2, r // 2
            if (lev, pc, pr) not in phi:
                phi[(lev, pc, pr)] = np.zeros(p + 1, dtype=complex)
            phi[(lev, pc, pr)] += m2m(phi[(lev + 1, c, r)],
                                      box_center(lev + 1, c, r) - box_center(lev, pc, pr), p)

    # ---- Steps 3a, 3b, 4: downward ----
    psi = {}
    for lev in range(2, n + 1):                       # free space: levels 0,1 give nothing
        for (l0, c, r) in [k for k in phi if k[0] == lev]:
            zc = box_center(lev, c, r)
            if lev == 2:
                b = np.zeros(p + 1, dtype=complex)    # Psi_1 = 0, so tilde-Psi_2 = 0
            else:
                pc, pr = c // 2, r // 2               # Step 3b: inherit from the parent
                b = l2l(psi[(lev - 1, pc, pr)],
                        box_center(lev - 1, pc, pr) - zc, p)
            for (cc, rr) in interaction_list(lev, (c, r)):    # Steps 3a / 4
                if (lev, cc, rr) in phi:
                    b += m2l(phi[(lev, cc, rr)], box_center(lev, cc, rr) - zc, p)
            psi[(lev, c, r)] = b

    # ---- Step 5: evaluate the far field ----
    pot = np.zeros(N, dtype=complex)
    for (lev, c, r), idx in tree.items():
        zc = box_center(n, c, r)
        for i in idx:
            pot[i] = eval_local(psi[(n, c, r)], z[i], zc, p)

    # ---- Steps 6 and 7: near field, added on ----
    pot += direct_sum(z, q, tree, n)
    return pot




