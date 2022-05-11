from cython import boundscheck, wraparound
cimport numpy as np
import numpy as np

cdef double G = 6.6743e-11

@boundscheck(False)
@wraparound(False)
cpdef np.ndarray[double, ndim=2] accelerations_cython(np.ndarray[double, ndim=1] m, np.ndarray[double, ndim=2] r):
    cdef int N = m.shape[0]
    cdef np.ndarray a = np.zeros((N, 2))
    cdef int i, j
    cdef double norm
    for i in range(N):
        for j in range(N):
            if i != j and i != 0:
                norm = np.linalg.norm(r[j] - r[i], 2)
                a[i] += G * m[j] * (r[j] - r[i]) / (norm * norm * norm)
    return a

@boundscheck(False)
@wraparound(False)
cpdef sol_verlet_cython(np.ndarray[double, ndim=1] m, np.ndarray[double, ndim=2] r, np.ndarray[double, ndim=2] v, double T, int k):
    cdef int N = m.shape[0]
    cdef np.ndarray res_r = np.zeros((N, 2, k))
    cdef np.ndarray res_v = np.zeros((N, 2, k))
    cdef double dt = T / (k - 1)
    cdef np.ndarray cur_a = accelerations_cython(m, r)
    cdef np.ndarray next_a
    res_r[:, : , 0] = r
    res_v[:, : , 0] = v
    cdef int i
    for i in range(1, k):
        r = r + v * dt + 0.5 * cur_a * dt * dt
        next_a = accelerations_cython(m, r)
        v = v + 0.5 * (cur_a + next_a) * dt
        cur_a = next_a
        res_r[:, : , i] = r
        res_v[:, : , i] = v
    return res_r, res_v