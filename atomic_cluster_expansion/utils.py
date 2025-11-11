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
    unique_l = np.arange(l_max+1)
    l = np.repeat(unique_l, (2*unique_l)+1)
    m = np.r_[*[np.arange(-x,x+1) for x in unique_l]]
    Y_lm = sph_harm_y(l, m, r_polar[:,None], r_azimuth[:,None])

    return np.sqrt(4*np.pi)*Rn[:,:,None]*Y_lm[:,None,:]

def get_single_component_invariance_products_of_atomic_bases(single_bond_basis, 
                                                             body_order=2):
    
    if body_order<2:
        raise ValueError("body_order must be >= 2")

    import pandas as pd

    v_size = single_bond_basis.shape[1:]

    B1 = np.sum(single_bond_basis[:,:,0], axis=0) 
    B1 = pd.DataFrame(
                np.c_[np.arange(v_size[0])+1, B1],
                columns=["n", "B1"]
                    )
    
    if body_order == 2:
        return B1

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
        B2.append( np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[0], 
                                            inds], axis=0) * 
                   np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[1], 
                                            inds], axis=0) )
        for m in np.arange(1,l+1):
            inds = l_start_inds[i] + (m_span_lengths[i]//2) + np.array([m,-m])
            B2[i] += ( np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[0],  
                                            inds[0]], axis=0) * 
                       np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[1], 
                                            inds[1]], axis=0) ) * (-1.)**m
            B2[i] += ( np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[0], 
                                            inds[1]], axis=0) * 
                       np.sum(single_bond_basis[:, np.triu_indices(v_size[0])[1], 
                                            inds[0]], axis=0) ) * (-1.)**m
    B2 = np.r_[B2]
    B2 = pd.DataFrame(
                np.c_[np.tile(np.triu_indices(v_size[0])[0]+1, len(unique_l)), 
                      np.tile(np.triu_indices(v_size[0])[1]+1, len(unique_l)),
                      np.repeat(unique_l, int(v_size[0]*(v_size[0]+1)/2)),
                      B2],
                columns=["n1", "n2", "l", "B2"]
                    )
    
    if body_order == 3:
        return B1, B2
    
    from sympy.physics.wigner import wigner_3j
    
    n1, n2, n3 = np.ogrid[:v_size[0], :v_size[0], :v_size[0]]
    mask = (n3 >= n2) & (n2 >= n1)    
    contracted_n_tuples_B3 = np.column_stack(np.where(mask))
    del n1, n2, n3, mask

    l1, l2, l3 = np.ogrid[:l_max+1, :l_max+1, :l_max+1]
    mask = (np.abs(l1 - l2) <= l3) & (l3 <= l1 + l2) & ((l1 + l2 + l3) % 2 == 0)
    l_tuples_B3 = np.column_stack(np.where(mask))
    del l1, l2, l3, mask

    B3 = [], n_tuples_B3 = []
    for i,l_tup in enumerate(l_tuples_B3):   
        if np.all(l_tup==l_tup[0]):
            n1, n2, n3 = contracted_n_tuples_B3.T
        else:
            n1, n2, n3 = np.meshgrid(*([np.arange(v_size[0])]*3))
            n1 = n1.ravel(); n2 = n2.ravel(); n3 = n3.ravel()
        inds = l_start_inds[l_tup] + (m_span_lengths[l_tup]//2)
        coeff = float(wigner_3j(*l_tup, 0, 0, 0))
        B3.append( coeff *
                   np.sum(single_bond_basis[:, n1, inds[0]], axis=0) * 
                   np.sum(single_bond_basis[:, n2, inds[1]], axis=0) *
                   np.sum(single_bond_basis[:, n3, inds[2]], axis=0) )
        
        m1 = np.arange(-l_tup[0], l_tup[0]+1)
        m2 = np.arange(-l_tup[1], l_tup[1]+1)
        m3 = np.arange(-l_tup[2], l_tup[2]+1)

        M1, M2, M3 = np.ogrid[-l_tup[0]:l_tup[0]+1, 
                              -l_tup[1]:l_tup[1]+1, 
                              -l_tup[2]:l_tup[2]+1]

        # Condition: m1 + m2 + m3 = 0
        mask = (M1 + M2 + M3 == 0)

        # Extract coordinates (m1,m2,m3)
        triples = np.column_stack(np.where(mask))

        # Convert index triples back to actual m values
        # Because np.where returns indices into m1,m2,m3 arrays
        m_tuples = np.column_stack((m1[triples[:,0]],
                                    m2[triples[:,1]],
                                    m3[triples[:,2]]))

        # Exclude (0,0,0)
        m_tuples = m_tuples[~np.all(m_tuples==0, axis=1)]
        del m1, m2, m3, M1, M2, M3, triples, mask

        for m_tup in m_tuples:
            inds = l_start_inds[l_tup] + (m_span_lengths[l_tup]//2) + m_tup
            coeff = float(wigner_3j(*l_tup, *m_tup))
            B3[i] += ( coeff *
                       np.sum(single_bond_basis[:, n1, inds[0]], axis=0) * 
                       np.sum(single_bond_basis[:, n2, inds[1]], axis=0) *
                       np.sum(single_bond_basis[:, n3, inds[2]], axis=0) )
        n_tuples_B3.append(np.c_[n1,n2,n3])
        del n1, n2, n3

    B3 = np.r_[B3]
    B3 = pd.DataFrame(
                np.c_[np.tile(n_tuples_B3, (len(l_tuples_B3),1)),
                      np.repeat(l_tuples_B3, len(n_tuples_B3), axis=0),
                      B2],
                columns=["n1", "n2", "n3", "l1", "l2", "l3"]
                    )
    del n_tuples_B3, l_tuples_B3, m_tuples
    if body_order == 4:
        return B1, B2, B3
    
    if body_order > 4:
        print("NOTE: This function is implemented upto 4 body-order terms")
        return B1, B2, B3
    
    from sympy.physics.wigner import clebsch_gordan

    n1, n2, n3, n4 = np.ogrid[:v_size[0], :v_size[0], :v_size[0], 
                          :v_size[0]]
    mask = (n4 >= n3) & (n3 >= n2) & (n2 >= n1)    
    contracted_n_tuples_B4 = np.column_stack(np.where(mask))
    del n1, n2, n3, n4, mask

    # Broadcast to shape (l_max+1, l_max+1, l_max+1, l_max+1)
    l1, l2, l3, l4 = np.ogrid[:l_max+1, :l_max+1, :l_max+1, :l_max+1]

    # Compute lower and upper valid bounds for L
    L_min = np.maximum(np.abs(l1 - l2), np.abs(l3 - l4))
    L_max = np.minimum(l1 + l2, l3 + l4)

    # Condition for non-empty L range
    mask = (L_min <= L_max)

    # Extract valid quadruples:
    quads = np.column_stack(np.where(mask))

    # Convert indices to actual l-values:
    l_tuples_B4 = np.column_stack((
        unique_l[quads[:,0]],
        unique_l[quads[:,1]],
        unique_l[quads[:,2]],
        unique_l[quads[:,3]]
    ))
    del l1, l2, l3, l4, L_min, L_max, mask, quads

    B4 = [], n_tuples_B4 = []
    for i,l_tup in enumerate(l_tuples_B4):   
        if np.all(l_tup==l_tup[0]):
            n1, n2, n3, n4 = contracted_n_tuples_B4.T
        else:
            n1, n2, n3, n4 = np.meshgrid(*([np.arange(v_size[0])]*4))
            n1 = n1.ravel(); n2 = n2.ravel(); n3 = n3.ravel(); n4 = n4.ravel()
        inds = l_start_inds[l_tup] + (m_span_lengths[l_tup]//2)
        coeff = clebsch_gordan(*l_tup, 0, 0, 0)
        B3.append( coeff *
                   np.sum(single_bond_basis[:, n1, inds[0]], axis=0) * 
                   np.sum(single_bond_basis[:, n2, inds[1]], axis=0) *
                   np.sum(single_bond_basis[:, n3, inds[2]], axis=0) )
        
        m1 = np.arange(-l_tup[0], l_tup[0]+1)
        m2 = np.arange(-l_tup[1], l_tup[1]+1)
        m3 = np.arange(-l_tup[2], l_tup[2]+1)

        M1, M2, M3 = np.ogrid[-l_tup[0]:l_tup[0]+1, 
                              -l_tup[1]:l_tup[1]+1, 
                              -l_tup[2]:l_tup[2]+1]

        # Condition: m1 + m2 + m3 = 0
        mask = (M1 + M2 + M3 == 0)

        # Extract coordinates (m1,m2,m3)
        triples = np.column_stack(np.where(mask))

        # Convert index triples back to actual m values
        # Because np.where returns indices into m1,m2,m3 arrays
        m_tuples = np.column_stack((m1[triples[:,0]],
                                    m2[triples[:,1]],
                                    m3[triples[:,2]]))

        # Exclude (0,0,0)
        m_tuples = m_tuples[~np.all(m_tuples==0, axis=1)]
        del m1, m2, m3, M1, M2, M3, triples, mask

        for m_tup in m_tuples:
            inds = l_start_inds[l_tup] + (m_span_lengths[l_tup]//2) + m_tup
            coeff = wigner_3j(*l_tup, *m_tup)
            B3[i] += ( coeff *
                       np.sum(single_bond_basis[:, n1, inds[0]], axis=0) * 
                       np.sum(single_bond_basis[:, n2, inds[1]], axis=0) *
                       np.sum(single_bond_basis[:, n3, inds[2]], axis=0) )
        n_tuples_B3.append(np.c_[n1,n2,n3])
        del n1, n2, n3

    B3 = np.r_[B3]
    B3 = pd.DataFrame(
                np.c_[np.tile(n_tuples_B3, (len(l_tuples_B3),1)),
                      np.repeat(l_tuples_B3, len(n_tuples_B3), axis=0),
                      B2],
                columns=["n1", "n2", "n3", "l1", "l2", "l3"]
                    )
    
    if body_order == 5:
        return B1, B2, B3, B4
    
           

    

    

