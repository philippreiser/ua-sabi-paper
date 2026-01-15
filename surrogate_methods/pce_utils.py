import numpy as np
import pandas as pd
from math import comb, floor
import os
from scipy.special import eval_legendre, legendre
from scipy.io import loadmat
from numpy.polynomial.polynomial import Polynomial


def poly_idx(p, M):
    P = comb(p + M, M)
    
    out = np.zeros((P, M), dtype=int)
    tA = np.zeros(M, dtype=int)
    l = 1

    pmax = (p + 1) ** M
    
    for i in range(1, pmax + 1):
        ri = i
        for d in range(M):
            md = (p + 1) ** (M - d - 1)
            val = ri // md
            tA[d] = val
            ri = ri - val * md
        
        if np.sum(tA) <= p:
            out[l, :] = tA
            l += 1
    
    return out

def legendre_coefficients(p):
    coeffs = []
    for n in range(p + 1):
        poly_coeffs = np.polynomial.legendre.Legendre.basis(n).convert(kind=np.polynomial.Polynomial).coef

        # Normalize the Legendre polynomials
        norm_factor = np.sqrt((2 * n + 1) / 2)
        poly_coeffs = poly_coeffs * norm_factor

        coeffs.append(poly_coeffs)
    return coeffs

def get_l_poly_coeffs_mat(p):
    l_poly_coeffs = legendre_coefficients(p)

    # Create the coefficients matrix
    l_poly_coeffs_mat = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j, coeff in enumerate(l_poly_coeffs[i]):
            l_poly_coeffs_mat[i, j] = coeff
        if i + 1 <= p:
            l_poly_coeffs_mat[i, i + 1:] = 0
    
    return l_poly_coeffs_mat

def get_pce_vars(p, M, idx=None):
    # Generate Legendre polynomial coefficients
    l_poly_coeffs_mat = get_l_poly_coeffs_mat(p)

    # Combination and polynomial selection
    comb = poly_idx(p, M)

    # Remove the first row (constant polynomial)
    comb = comb[1:, :]

    # Select specific indices if provided
    if idx is not None:
        comb = comb[idx, :]

    return l_poly_coeffs_mat, comb



def legendre_polynomials(p, normalized=True):
    """
    Generate Legendre polynomials up to degree p.
    
    Parameters:
        p (int): Maximum polynomial degree.
        normalized (bool): Whether to return normalized Legendre polynomials.

    Returns:
        list: List of callables for each degree.
    """
    if normalized:
        return [
            lambda x, n=n: np.sqrt((2 * n + 1)/2) * eval_legendre(n, x) for n in range(p + 1)
        ]
    else:
        return [lambda x, n=n: eval_legendre(n, x) for n in range(p + 1)]

def get_co2_apc_poly_coeffs(p):
    """
    """
    poly1 = np.array(pd.read_csv("data/CO2_Response/npc_0_10.dat", sep=r"\s+", header=None))
    poly2 = np.array(pd.read_csv("data/CO2_Response/npc_2_10.dat", sep=r"\s+", header=None))
    poly3 = np.array(pd.read_csv("data/CO2_Response/npc_3_10.dat", sep=r"\s+", header=None))

    return np.array([poly1[0:p+1, 0:p+1], poly2[0:p+1, 0:p+1], poly3[0:p+1, 0:p+1]])

def get_co2_apc_vars(p, M, idx=None):
    # Generate Legendre polynomial coefficients
    co2_apc_poly_coeffs = get_co2_apc_poly_coeffs(p)

    # Combination and polynomial selection
    comb = poly_idx(p, M)

    # Remove the first row (constant polynomial)
    comb = comb[1:, :]

    # Select specific indices if provided
    if idx is not None:
        comb = comb[idx, :]

    return co2_apc_poly_coeffs, comb    

def co2_polynomials(p=10):
    """
    Generate Arbitrary Polynomial Chaos Expansion polynomials.
    """
    poly1_df = pd.read_csv("data/CO2_Response/npc_0_10.dat", sep=r"\s+", header=None)
    poly1 = [Polynomial(row) for _, row in poly1_df.iterrows()]
    poly2_df = pd.read_csv("data/CO2_Response/npc_2_10.dat", sep=r"\s+", header=None)
    poly2 = [Polynomial(row) for _, row in poly2_df.iterrows()]
    poly3_df = pd.read_csv("data/CO2_Response/npc_3_10.dat", sep=r"\s+", header=None)
    poly3 = [Polynomial(row) for _, row in poly3_df.iterrows()]

    return [poly1, poly2, poly3]

def get_micp_apc_poly_coeffs(p: int) -> np.ndarray:
    """
    aPC polynomials from:
    https://git.iws.uni-stuttgart.de/dumux-pub/scheurer2019a/-/tree/master/surrogate-based_bayesian_justifiability_analysis/full_complexity_model?ref_type=heads

    Args:
        p (int): Maximum polynomial degree (has to be <= 3).

    Returns:
        np.ndarray: Coefficients matrix of aPC polynomials.
    """

    poly1 = loadmat('data/MICP/full_complexity_model/PolynomialBasis_1.mat')['Polynomial']
    poly2 = loadmat('data/MICP/full_complexity_model/PolynomialBasis_2.mat')['Polynomial']
    poly3 = loadmat('data/MICP/full_complexity_model/PolynomialBasis_3.mat')['Polynomial']
    poly4 = loadmat('data/MICP/full_complexity_model/PolynomialBasis_4.mat')['Polynomial']

    return np.array([poly1[0:p+1, 0:p+1], poly2[0:p+1, 0:p+1], poly3[0:p+1, 0:p+1], poly4[0:p+1, 0:p+1]])


def get_micp_apc_vars(p: int, M: int, idx=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate polynomial coefficients and polynomial combinations for MICP aPC.

    Args:
        p (int): Maximum polynomial degree.
        M (int): Number of variables.
        idx (array-like): Indices to select specific polynomials.

    Returns:
        Tuple: Tuple containing the MICP polynomial coefficients matrix and polynomial combinations.
    """

    # Generate Legendre polynomial coefficients
    micp_apc_poly_coeffs = get_micp_apc_poly_coeffs(p)

    # Combination and polynomial selection
    comb = poly_idx(p, M)

    # Remove the first row (constant polynomial)
    comb = comb[1:, :]

    # Select specific indices if provided
    if idx is not None:
        comb = comb[idx, :]

    return micp_apc_poly_coeffs, comb


def micp_polynomials(p: int = 2, path: str = 'data/MICP/full_complexity_model') -> List:
    """
    Load MICP Arbitrary Polynomial Chaos Expansion polynomials.
    aPC polynomials from:
    https://git.iws.uni-stuttgart.de/dumux-pub/scheurer2019a/-/tree/master/surrogate-based_bayesian_justifiability_analysis/full_complexity_model?ref_type=heads

    Args:
        p (int): Maximum polynomial degree.

    Returns:
        List: List of callables for each degree.
    """
    base_path = os.path.join(path, 'PolynomialBasis_')
    polynomials = []
    for i in range(1, 5):
        mat_data = loadmat(f"{base_path}{i}.mat")
        poly_array = mat_data['Polynomial']
        poly_list = [Polynomial(row) for row in poly_array]
        polynomials.append(poly_list)

    return polynomials


def get_pce(*args, p=10, idx=None, comb=None, poly=legendre_polynomials):
    """
    Computes the Polynomial Chaos Expansion (PCE).

    Parameters:
        *args: Input arrays.
        p (int): Maximum polynomial degree.
        idx (array-like): Indices to select specific polynomials.
        poly (function or list): if function: a callable returning polynomials,
        if list: a list containing callables returning polynomials

    Returns:
        numpy.ndarray: PCE output matrix.
    """
    dots = args
    N = len(dots[0])
    M = len(dots)   

    # Generate polynomial combination indices
    if comb is None:
        comb = poly_idx(p, M)
    comb = comb[1:]  # Exclude the constant polynomial

    if idx is not None:
        comb = comb[idx]

    if callable(poly):
        poly = [poly(p) for _ in range(M)]  # Generate polynomials for each dimension

    assert isinstance(poly, list) and len(poly) == M, "Poly should be a list of length M."

    # Precompute the predictions for all polynomials
    poly_predictions = [
        np.array([poly[j][degree](dots[j]) for degree in range(p + 1)]) for j in range(M)
    ]


    # Compute the PCE matrix
    num_terms = len(comb)
    out = np.ones((N, num_terms))

    for i, combination in enumerate(comb):
        for j, degree in enumerate(combination):
            out[:, i] *= poly_predictions[j][degree]

    return out

