import numpy as np
from scipy.special import chebyt

def radial_basis_chebyt(n_max, rcut, lbd):

    def create_Rn_func(n_max, rcut, lbd):
        def func(r):
            g0 = np.ones(len(r))
            g1 = 0.5*(1 + np.cos(np.pi*r/rcut))
            x = 2*(np.exp(-lbd*((r/rcut)-1))-1)/(np.exp(lbd)-1)
            x = 1-x
            g = np.c_[*[0.5*(1-chebyt(k-1)(x))*g1
                        for k in np.arange(2,n_max+1)]]
            g = np.c_[g0, g1, g]
            return g
        return func
    
    radial_basis_functions = create_Rn_func(n_max, rcut, lbd)
    return radial_basis_functions

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
    min_rdf_peaks_diff = np.r_[rdf_peaks_diff[0],
                               min_rdf_peaks_diff,
                               rdf_peaks_diff[-1]]
    # Sanity check
    if np.any(min_rdf_peaks_diff<0) or np.any(np.isclose(min_rdf_peaks_diff, 0)):
        raise RuntimeError("min_rdf_peaks_diff must be all positives. Bug alert!")
    
    overlap_upper_bounds = (np.exp(-0.5)/np.sqrt(2*np.pi)/
                               min_rdf_peaks_diff)
    inner_prod = np.min(np.c_[[overlap]*len(rdf_peaks), 
                              overlap_upper_bounds], axis=1)
    W1 = lambertw(-2*np.pi*min_rdf_peaks_diff**2*inner_prod**2, k=0)
    W2 = lambertw(-2*np.pi*min_rdf_peaks_diff**2*inner_prod**2, k=-1)
    # Sanity check
    if not np.allclose(W1.imag, 0) and np.allclose(W2.imag, 0):
        raise RuntimeError("W1 and W2 must be real. Bug alert!")

    std=np.min(np.c_[min_rdf_peaks_diff/np.sqrt(-2*W1.real),
                     min_rdf_peaks_diff/np.sqrt(-2*W2.real)], axis=1)
    
    dist = norm(loc=rdf_peaks, scale=std)

    def create_Rn_func(dist):
        def func(x):
            if isinstance(x, (int, float)):
                return dist.pdf(x)
            elif isinstance(x, np.ndarray):
                pass
            elif isinstance(x, (list, tuple)):
                x = np.array(x)
            else:
                raise TypeError("x must be integer, float, numpy array, list or tuple")
    
            if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating):
                pass
            else:
                raise TypeError("If x is numpy array, list or tuple, it must be either "+
                                "integer or floating type")
    
            if len(x.shape)==2 and x.shape[1]==1:
                return dist.pdf(x)
            elif len(x.shape)==1:
                return dist.pdf(x[:,None])
            else:
                raise TypeError("If x is numpy array, list or tuple, it's shape must be (N,) or (N,1)")
    
        return func
    
    radial_basis_functions = create_Rn_func(dist)
    return radial_basis_functions

def get_single_bond_basis(r, radial_basis_functions,
                                   l_max):
    
    r_norm = np.linalg.norm(r, axis=1)
    r_polar = np.arccos(r[:,-1]/r_norm)
    r_azimuth = np.arccos(r[:,0]/r_norm)
    r_azimuth[r[:,1]<0] = (2*np.pi)-r_azimuth[r[:,1]<0]

    Rn = radial_basis_functions(r_norm)
    
    from scipy.special import sph_harm_y
    l = np.arange(l_max+1)
    l = np.repeat(l, (2*l)+1)
    m = np.r_[*[np.arange(-x,x+1) for x in l]]
    Y_lm = sph_harm_y(l, m, r_polar[:,None], r_azimuth[:,None])

    return np.sqrt(4*np.pi)*Rn[:,:,None]*Y_lm[:,None,:]

def get_invariance_products_of_atomic_bases(single_bond_basis):

    import xarray as xr

    v_size = single_bond_basis.shape[1:]

    B1 = np.sum(single_bond_basis[:,:,0], axis=0) 
    B1 = xr.DataArray(
                B1,
                dims=("n",),
                coords={
                        "n": np.arange(v_size[0])+1,
                       }
                    )
    
    if len(v_size) != 2:
        raise ValueError("single_bond_basis must be 3d numpy array")
    elif np.sqrt(v_size[1]) != int(np.sqrt(v_size[1])):
        raise ValueError("The size of the 3rd dimension of single_bond_basis must be a square")
    else:
        l_max = int(np.sqrt(v_size[1])-1)
    unique_l = np.arange(l_max+1)
    m_span_lengths = (2*unique_l)+1
    l_start_inds = np.cumsum(m_span_lengths)-m_span_lengths
       
    B2 = []
    for i,l in enumerate(unique_l):   
        inds = l_start_inds[i] + (m_span_lengths[i]//2)
        B2.append( np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[0], None, 
                                            inds], axis=0) * 
                   np.sum(single_bond_basis[:, None, np.triu_indices(v_size[0])[1], 
                                            inds], axis=0) )
        for m in np.arange(1,l+1):
            inds = inds + np.array([m,-m])
            B2[i] += ( np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[0], None, 
                                            inds[0]], axis=0) * 
                       np.sum(single_bond_basis[:, None, np.triu_indices(v_size[0])[1], 
                                            inds[1]], axis=0) ) * (-1.)**m
            B2[i] += ( np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[0], None, 
                                            inds[1]], axis=0) * 
                       np.sum(single_bond_basis[:, None, np.triu_indices(v_size[0])[1], 
                                            inds[0]], axis=0) ) * (-1.)**m
    B2 = np.stack(B2, axis=2)
    B2 = xr.DataArray(
                B2,
                dims=("n1", "n2", "l"),
                coords={
                        "n1": np.triu_indices(v_size[0])[0]+1,
                        "n2": np.triu_indices(v_size[0])[1]+1,
                        "l": unique_l
                       }
                    )
    

    l1, l2, l3 = np.ogrid[:l_max+1, :l_max+1, :l_max+1]
    mask = (np.abs(l1 - l2) <= l3) & (l3 <= l1 + l2) & ((l1 + l2 + l3) % 2 == 0)
    l_tuples_B3 = np.column_stack(np.where(mask))
    B3 = []
    
    
           

    

    

