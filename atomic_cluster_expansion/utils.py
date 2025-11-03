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

def get_radial_basis_functions_from_rdf_peaks(rdf_peaks, 
                                              overlap=0.01):

    if (isinstance(rdf_peaks, np.ndarray) and 
        np.issubdtype(rdf_peaks.dtype, np.floating) and
        (len(rdf_peaks.shape)==1) and (len(rdf_peaks)!=0)):
        pass
    elif (isinstance(rdf_peaks, (list, tuple)) and 
          (len(rdf_peaks)!=0) and
           np.all([isinstance(x, float) for x in rdf_peaks])):
        pass
    else:
        raise ValueError("rdf_peaks must be a non-empty 1d numpy array "+
                         "or list or tuple with only float entries")
    
    if np.any(rdf_peaks<0) or np.any(np.isclose(rdf_peaks, 0)):
        raise ValueError("All rdf peaks must be positive")
    else:
        rdf_peaks = np.sort(rdf_peaks)

    from scipy.stats import norm
    from scipy.special import lambertw

    rdf_peaks_diff = np.diff(rdf_peaks)
    if np.any(np.isclose(rdf_peaks_diff, 0)):
        raise ValueError("all rdf peaks needs to be unique")
    min_rdf_peaks_diff = np.min(np.c_[rdf_peaks_diff[:-1], 
                                      rdf_peaks_diff[1:]], axis=1)
    # Sanity check
    if np.any(min_rdf_peaks_diff<0) or np.any(np.isclose(min_rdf_peaks_diff, 0)):
        raise RuntimeError("min_rdf_peaks_diff must be all positives. Bug alert!")
    
    overlap_upper_bounds = (np.exp(-0.5)/np.sqrt(2*np.pi)/
                               min_rdf_peaks_diff)
    
    def create_Rn(dist):
        return lambda x: dist.pdf(x)
    
    radial_basis_functions = []
    for i in range(1, len(rdf_peaks)-1):
        peak = rdf_peaks[i]
        inner_prod = np.min([overlap, 
                             overlap_upper_bounds[i-1]])
        std=np.min([min_rdf_peaks_diff[i-1]/np.sqrt(-2*lambertw(
                         -2*np.pi*min_rdf_peaks_diff[i-1]**2*inner_prod**2, k=0).real),
                    min_rdf_peaks_diff[i-1]/np.sqrt(-2*lambertw(
                         -2*np.pi*min_rdf_peaks_diff[i-1]**2*inner_prod**2, k=-1).real)])
        dist = norm(loc=peak, scale=std)
        radial_basis_functions.append(create_Rn(dist))

    return radial_basis_functions

    

