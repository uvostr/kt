import numpy as np
import functools
from multiprocessing import Pool, cpu_count


G = float(6.6743e-11)

def accelerations_mp(indexes, m, r):
    N = m.shape[0]
    n = indexes.shape[0]
    a = np.zeros((n, 2))
    for i in range(n):
        for j in range(N):
            if indexes[i] != j and indexes[i] != 0:
                a[i] += G * m[j] * (r[j] - r[indexes[i]]) / (np.linalg.norm(r[j] - r[indexes[i]], 2) ** 3)
    return a


def sol_verlet_mp(m, r, v, T, k):
    N = m.shape[0]
    res_r = np.zeros((N, 2, k))
    res_v = np.zeros((N, 2, k))
    dt = T / (k - 1)
    indexes = np.arange(0, N)
    num_workers = cpu_count()
    splitted_indexes = np.array_split(indexes, num_workers)
    func = functools.partial(accelerations_mp, m = m, r = r)
    pool_obj = Pool(num_workers)
    result = pool_obj.starmap(func, zip(splitted_indexes))
    cur_a = np.vstack(result)
    res_r[:, : , 0] = r
    res_v[:, : , 0] = v
    for i in range(1, k):
        r = r + v * dt + 0.5 * cur_a * (dt ** 2)
        func = functools.partial(accelerations_mp, m = m, r = r)
        result = pool_obj.starmap(func, zip(splitted_indexes))
        next_a = np.vstack(result)
        v = v + 0.5 * (cur_a + next_a) * dt
        cur_a = next_a
        res_r[:, : , i] = r
        res_v[:, : , i] = v
    return res_r, res_v