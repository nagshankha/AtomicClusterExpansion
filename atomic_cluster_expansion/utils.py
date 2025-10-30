import numpy as np
from scipy.special import chebyt

def radial_basis(r, rcut, lbd, c):

    g0 = 1
    g1 = 1 + np.cos(np.pi*r/rcut)
    def g_func(k, lbd, g1, r):
        if not (isinstance(k, int) and k>=2):
            raise ValueError("k must be an integer >= 2")
        x = 2*(np.exp(-lbd*((r/rcut)-1))-1)/(np.exp(-lbd)-1)
        x = 1-x
        return 0.25*(1-chebyt(k-1)(x))*g1

def get_radial_basis_functions_from_rdf_peaks(rdf_peaks):

    from scipy.stats import norm
    radial_basis_functions = []

    for i, peak in enumerate(rdf_peaks):
        std=1 ## NEEDS TO CHANGE
        dist = norm(loc=peak, scale=std)
        radial_basis_functions.append(
            lambda x: dist.pdf(x)
        )

    return radial_basis_functions

    

