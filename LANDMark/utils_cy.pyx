import cython
from cython.parallel import prange

# cython: language_level=3

import numpy as np
cimport numpy as cnp

cnp.import_array()

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef wd_star_c(float [:, :] D, 
             signed int [:] y, 
             signed int [:] D_ind, 
             signed int [:] y_set, 
             signed int [:] n, 
             k, 
             n_samp):

    cdef signed int n_samp_c = n_samp
    cdef signed int k_c = k

    cdef float W = 0.0

    cdef cnp.ndarray [float, ndim = 1] w_t = np.zeros(shape = (k_c,), dtype = np.float32)
    cdef cnp.ndarray [signed int, ndim = 2] i_locs = np.zeros(shape = (k, n_samp_c), dtype = np.int32) - 1
    cdef cnp.ndarray [float, ndim = 2] SSDs = np.zeros(shape = (k_c, k_c), dtype = np.float32)
    cdef cnp.ndarray [float, ndim = 1] s_t = np.zeros(shape = (k_c,), dtype = np.float32)

    cdef signed int i, index, treatment_id, loc_a, loc_b, j
    cdef float outer, c_ij, c_i, c_j, eqn_16, eqn_17, inner, h, W_star_d

    #Determine the index of each sample
    for i in range(k_c):
        treatment_id = y_set[i]

        for index in range(n_samp_c):
            if y[index] == y_set[i]:
                i_locs[i, index] = D_ind[index]

    #Calculate the SSDs within and between groups
    for i in range(0, k_c - 1):
        #SSD within Class i
        for loc_a in range(n_samp_c):
            for loc_b in range(n_samp_c):
                if i_locs[i, loc_a] >= 0 and i_locs[i, loc_b] >= 0:
                    SSDs[i, i] = SSDs[i, i] + D[i_locs[i, loc_a], i_locs[i, loc_b]] #** 2

        #SSDs for j
        for j in range(i+1, k_c):
            #SSD within Class i
            for loc_a in range(n_samp_c):
                for loc_b in range(n_samp_c):
                    if i_locs[j, loc_a] >= 0 and i_locs[j, loc_b] >= 0:
                        SSDs[j, j] = SSDs[j, j] + D[i_locs[j, loc_a], i_locs[j, loc_b]] #** 2

            #SSD between Class i and j
            for loc_a in range(n_samp_c):
                for loc_b in range(n_samp_c):
                    
                    if i_locs[i, loc_a] >= 0 and i_locs[j, loc_b] >= 0:
                        SSDs[i, j] = SSDs[i, j] + D[i_locs[j, loc_b], i_locs[i, loc_a]] #** 2

            SSDs[i, j] = SSDs[i, j]
            SSDs[j, i] = SSDs[i, j]

    for i in range(k_c):
        for j in range(k_c):
            SSDs[i, j] = SSDs[i, j] / 2.0

    #Calculate W and each w_i
    for i in range(k_c):
        s_t[i] = SSDs[i, i] / (n[i] * (n[i] - 1))
        w_t[i] = n[i] / s_t[i]

        W = W + w_t[i]

    #Calculate Equation 16 and 17
    eqn_16 = 0.0
    for i in range(0, k_c-1):
        for j in range(i+1, k_c):

            outer = (n[i] + n[j]) / (s_t[i] * s_t[j])

            c_ij = (SSDs[i, i] + SSDs[j, j] + SSDs[i, j] + SSDs[j, i]) / (n[i] + n[j])
            c_i = SSDs[i, i] / n[i]
            c_j = SSDs[j, j] / n[j]

            inner = c_ij - (c_i + c_j)

            eqn_17 = outer * inner

            eqn_16 = eqn_16 + eqn_17

    #Calculate h
    h = 0.0
    for i in range(k_c):
        h = h + ((1 - n[i]/s_t[i]/W) ** 2) / (n[i] - 1)

    #Equation One (Calculate W*d)
    W_star_d = eqn_16 / W / (<float>k_c-1.0) / (1.0 + 2 * (<float>k_c - 2.0) / (<float>k_c**2 - 1.0) * h)

    return W_star_d