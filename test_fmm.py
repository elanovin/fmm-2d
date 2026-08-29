"""
Test suite for the 2D free-space fast multipole method.

Reference values come from an independent hand trace of an 8-particle
configuration (see FMM_Numeric_Trace). Every test names the result of
Greengard & Rokhlin 1997 that it verifies.

Run with:   pytest -q
"""

import numpy as np
import pytest

from fmm import (direct_all, box_center, box_index, well_separated,
                 interaction_list, build_tree, p2m, m2m, m2l, l2l,
                 eval_local, direct_sum, fmm)

P = 10
N_LEVELS = 3


@pytest.fixture
def particles():
    """The 8-particle configuration used throughout the hand trace."""
    z = np.array([0.42 + 0.42j, 0.45 + 0.40j, 0.10 + 0.10j, 0.90 + 0.15j,
                  0.20 + 0.85j, 0.80 + 0.80j, 0.55 + 0.60j, 0.30 + 0.35j])
    q = np.array([1.0, -1.0, 2.0, -1.0, 1.0, -2.0, 1.0, 1.0])
    return z, q


@pytest.fixture
def tree(particles):
    z, _ = particles
    return build_tree(z, N_LEVELS)


# --------------------------------------------------------------------
# Geometry: Section 3, p. 285
# --------------------------------------------------------------------

def test_box_center():
    assert box_center(3, 3, 3) == pytest.approx(0.4375 + 0.4375j)
    assert box_center(2, 1, 1) == pytest.approx(0.375 + 0.375j)


def test_box_index(particles):
    z, _ = particles
    assert box_index(z[0], 3) == (3, 3)
    assert box_index(z[3], 3) == (7, 1)


def test_box_index_clamps_upper_edge():
    """A particle exactly on the boundary must not fall outside the grid."""
    assert box_index(0.999999 + 0.999999j, 3) == (7, 7)


# --------------------------------------------------------------------
# Well-separation: Section 2, p. 282
# --------------------------------------------------------------------

def test_well_separated():
    assert not well_separated((3, 3), (4, 4))          # touching
    assert well_separated((3, 3), (5, 3))              # two columns apart
    assert well_separated((3, 3), (0, 0))


def test_interaction_list_has_27_entries():
    """9 parent-neighbour boxes x 4 children - 9 near boxes = 27."""
    assert len(interaction_list(3, (3, 3))) == 27


def test_interaction_list_members_are_well_separated():
    for b in interaction_list(3, (3, 3)):
        assert well_separated(b, (3, 3))


def test_interaction_list_is_clipped_at_the_boundary():
    """Boxes near the edge have fewer than 27 entries: some parents are off-grid."""
    assert len(interaction_list(3, (0, 0))) == 12    # corner box
    assert len(interaction_list(3, (1, 1))) == 7


# --------------------------------------------------------------------
# Tree: Section 3, pp. 285-286
# --------------------------------------------------------------------

def test_build_tree(tree, particles):
    z, _ = particles
    assert len(tree) == 7                   # 7 occupied boxes, 57 empty
    assert tree[(3, 3, 3)] == [0, 1]        # p1 and p2 share a box
    assert tree[(3, 0, 0)] == [2]
    assert sum(len(v) for v in tree.values()) == len(z)


# --------------------------------------------------------------------
# Theorem 2.1 / Eq. (2.3)
# --------------------------------------------------------------------

def test_p2m_monopole_and_dipole(tree, particles):
    z, q = particles
    idx = tree[(3, 0, 0)]
    a = p2m(z[idx], q[idx], box_center(3, 0, 0), P)
    assert a[0].real == pytest.approx(2.0)
    assert a[1] == pytest.approx(-0.075 - 0.075j)


def test_p2m_neutral_box_still_radiates(tree, particles):
    """Box (3,3) holds +1 and -1: Q = 0 but the dipole moment is not."""
    z, q = particles
    idx = tree[(3, 3, 3)]
    a = p2m(z[idx], q[idx], box_center(3, 3, 3), P)
    assert a[0].real == pytest.approx(0.0, abs=1e-14)
    assert a[1] == pytest.approx(0.03 - 0.02j)


# --------------------------------------------------------------------
# Lemma 2.3 (M2M)
# --------------------------------------------------------------------

def test_m2m_matches_the_trace(tree, particles):
    z, q = particles
    pc = box_center(2, 1, 1)
    total = np.zeros(P + 1, dtype=complex)
    for (c, r) in [(2, 2), (3, 3)]:
        idx = tree[(3, c, r)]
        cc = box_center(3, c, r)
        total += m2m(p2m(z[idx], q[idx], cc, P), cc - pc, P)
    assert total[0] == pytest.approx(1.0 + 0j)
    assert total[1] == pytest.approx(0.105 + 0.005j)


def test_m2m_is_exact(tree, particles):
    """Shifting up must equal expanding directly at the parent centre."""
    z, q = particles
    idx = tree[(3, 0, 0)]
    cc, pc = box_center(3, 0, 0), box_center(2, 0, 0)
    shifted = m2m(p2m(z[idx], q[idx], cc, P), cc - pc, P)
    direct = p2m(z[idx], q[idx], pc, P)
    assert np.max(np.abs(shifted - direct)) < 1e-15


# --------------------------------------------------------------------
# Lemma 2.4 (M2L) - the only approximate operator
# --------------------------------------------------------------------

def test_m2l_matches_the_trace(tree, particles):
    z, q = particles
    idx = tree[(3, 0, 0)]
    a = p2m(z[idx], q[idx], box_center(3, 0, 0), P)
    b = m2l(a, box_center(3, 0, 0) - box_center(3, 3, 3), P)
    assert b[0] == pytest.approx(-1.4792323568 + 1.5707963268j)
    assert b[1] == pytest.approx(2.9629629629 - 2.9629629629j)


def test_m2l_reproduces_the_true_field(tree, particles):
    """P2M then M2L then evaluate must reproduce q*log(z - z_src)."""
    z, q = particles
    idx = tree[(3, 0, 0)]
    a = p2m(z[idx], q[idx], box_center(3, 0, 0), P)
    zc = box_center(3, 3, 3)
    b = m2l(a, box_center(3, 0, 0) - zc, P)
    approx = eval_local(b, z[0], zc, P)
    exact = q[2] * np.log(z[0] - z[2])
    assert approx.real == pytest.approx(exact.real, abs=1e-10)


def test_m2l_error_respects_eq_2_11():
    """The measured error must never exceed the bound of Eq. (2.11)."""
    zs, qs = np.array([0.10 + 0.10j]), np.array([2.0])
    zc_s, zc_t = box_center(3, 0, 0), box_center(3, 3, 3)
    a = p2m(zs, qs, zc_s, P)
    b = m2l(a, zc_s - zc_t, P)
    zt = 0.42 + 0.42j
    err = abs(eval_local(b, zt, zc_t, P) - qs[0] * np.log(zt - zs[0]))

    A = np.sum(np.abs(qs))
    R = 0.125 / np.sqrt(2)
    c = abs(zc_s - zc_t) / R - 1.0
    bound = A * (4 * np.e * (P + c) * (c + 1) + c**2) / (c * (c - 1)) * (1 / c)**(P + 1)
    assert err < bound


# --------------------------------------------------------------------
# Lemma 2.5 (L2L)
# --------------------------------------------------------------------

def test_l2l_is_exact():
    """Re-anchoring a polynomial must not change the field it represents."""
    a = np.zeros(P + 1, dtype=complex)
    a[0], a[1], a[2] = 1 + 0j, 2 - 1j, 0.5 + 0.5j
    zp, zc = 0.375 + 0.375j, 0.4375 + 0.4375j
    b = l2l(a, zp - zc, P)
    zt = 0.42 + 0.42j
    old = sum(a[k] * (zt - zp)**k for k in range(P + 1))
    new = eval_local(b, zt, zc, P)
    assert abs(old - new) < 1e-14


# --------------------------------------------------------------------
# Section 3, Step 5 - evaluation
# --------------------------------------------------------------------

def test_eval_local_agrees_with_naive_sum():
    """Horner's rule must agree with the literal power sum."""
    rng = np.random.default_rng(1)
    b = rng.normal(size=P + 1) + 1j * rng.normal(size=P + 1)
    zt, zc = 0.42 + 0.42j, 0.4375 + 0.4375j
    naive = sum(b[l] * (zt - zc)**l for l in range(P + 1))
    assert eval_local(b, zt, zc, P) == pytest.approx(naive)


# --------------------------------------------------------------------
# Section 3, Step 6 - near field
# --------------------------------------------------------------------

def test_direct_sum_matches_the_trace(tree, particles):
    z, q = particles
    near = direct_sum(z, q, tree, N_LEVELS)
    assert near[0].real == pytest.approx(-0.15604518, abs=1e-8)


def test_direct_sum_uses_only_the_nine_box_neighbourhood(tree, particles):
    """p3 is well separated from p1's box, so it must NOT appear in NEAR."""
    z, q = particles
    near = direct_sum(z, q, tree, N_LEVELS)
    expected = sum(q[k] * np.log(z[0] - z[k]) for k in (1, 6, 7))
    assert near[0] == pytest.approx(expected)


# --------------------------------------------------------------------
# The full algorithm
# --------------------------------------------------------------------

def test_fmm_matches_the_hand_trace(particles):
    z, q = particles
    f = fmm(z, q, N_LEVELS, P)
    assert f[0].real == pytest.approx(-0.6309443154, abs=1e-9)


def test_fmm_matches_direct_summation(particles):
    z, q = particles
    f = fmm(z, q, N_LEVELS, P)
    d = direct_all(z, q)
    assert np.max(np.abs(f.real - d.real)) < 1e-8


def test_fmm_on_random_particles():
    """500 random particles, depth 4, p = 20."""
    rng = np.random.default_rng(0)
    z = rng.random(500) + 1j * rng.random(500)
    q = rng.normal(size=500)
    f = fmm(z, q, 4, 20)
    d = direct_all(z, q)
    rel = np.max(np.abs(f.real - d.real)) / np.max(np.abs(d.real))
    assert rel < 1e-8


@pytest.mark.parametrize("p, tol", [(5, 1e-1), (10, 1e-3), (20, 1e-7)])
def test_accuracy_improves_with_p(p, tol):
    """Eq. (2.6): the error must fall roughly like 2^-p."""
    rng = np.random.default_rng(2)
    z = rng.random(200) + 1j * rng.random(200)
    q = rng.normal(size=200)
    f = fmm(z, q, 3, p)
    d = direct_all(z, q)
    rel = np.max(np.abs(f.real - d.real)) / np.max(np.abs(d.real))
    assert rel < tol