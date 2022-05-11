import numpy as np
import time
import random
import matplotlib.pyplot as plt

from python_verlet import sol_verlet
from python_verlet_mp import sol_verlet_mp
from cythonVerlet import sol_verlet_cython
from opencl_verlet import sol_verlet_opencl
from odeint import sol_odeint


def generate_random_data(N):
    m = np.random.rand(N) * random.randint(1, 10000)
    r0 = np.random.rand(N, 2) * random.randint(1, 10000)
    v0 = np.random.rand(N, 2) * random.randint(1, 10000)
    return m, r0, v0


if __name__=="__main__":
    T = 1
    k = T * 10
    
    
    
    N = np.array([50, 100, 200, 300])
    
    n_experiments = 3    
    
    time_res = np.zeros((N.shape[0], 4))

    for i in range(N.shape[0]):
        
        m, r0, v0 = generate_random_data(N[i])
        
        for j in range(n_experiments):
            start_time = time.time()
            sol_verlet(m, r0, v0, T, k)
            end_time = time.time()
            time_res[i, 0] += end_time - start_time
            
            start_time = time.time()
            sol_verlet_cython(m, r0, v0, T, k)
            end_time = time.time()
            time_res[i, 1] += end_time - start_time
            
            start_time = time.time()
            sol_verlet_mp(m, r0, v0, T, k)
            end_time = time.time()
            time_res[i, 2] += end_time - start_time
            
            start_time = time.time()
            sol_verlet_opencl(m, r0, v0, T, k)
            end_time = time.time()
            time_res[i, 3] += end_time - start_time
            
    time_res /= n_experiments
    
    plt.plot(N, time_res[:, 0], marker = 'o', color = "red", label = 'python')
    plt.plot(N, time_res[:, 1], marker = 'o', color = 'green', label = 'cython')
    plt.plot(N, time_res[:, 2], marker = 'o', color = 'blue', label = 'multiprocessing')
    plt.plot(N, time_res[:, 3], marker = 'o', color = 'magenta', label = 'opencl')
    plt.xlabel('number of bodies')
    plt.ylabel('time, sec')
    plt.title("Run Time")
    plt.legend()
    plt.savefig('graphs/run time.png')
    plt.show()
    
    plt.plot(N, time_res[:, 0] / time_res[:, 0], marker = 'o', color = 'red', label = 'python')
    plt.plot(N, time_res[:, 0] / time_res[:, 1], marker = 'o', color = 'green', label = 'cython')
    plt.plot(N, time_res[:, 0] / time_res[:, 2], marker = 'o', color = 'blue', label = 'multiprocessing')
    plt.plot(N, time_res[:, 0] / time_res[:, 3], marker = 'o', color = 'magenta', label = 'opencl')
    plt.xlabel('number of bodies')
    plt.ylabel('acceleration coefficient')
    plt.title("Acceleration")
    plt.legend()
    plt.savefig('graphs/acceleration.png')
    plt.show()
    
    plt.plot(N, time_res[:, 0] / time_res[:, 0], marker = 'o', color = 'red', label = 'python')
    plt.plot(N, time_res[:, 0] / time_res[:, 1], marker = 'o', color = 'green', label = 'cython')
    plt.plot(N, time_res[:, 0] / time_res[:, 2], marker = 'o', color = 'blue', label = 'multiprocessing')
    plt.xlabel('number of bodies')
    plt.ylabel('acceleration coefficient')
    plt.title("Acceleration")
    plt.legend()
    plt.savefig('graphs/acceleration whithout opencl.png')
    plt.show()
    
    
    
    
    m_planet = np.array([1989000.00000e24, 0.32868e24, 4.81068e24, 0.63345e24, 5.97600e24, 1876.64328e24, 561.80376e24, 86.05440e24, 101.59200e24], np.double)
    distance_from_sun = np.array([0, 58e9, 108e9, 150e9, 228e9, 778e9, 1429e9, 2875e9, 4497e9], np.double)
    planet_v = np.array([0, 47.36, 35.02, 29.78, 24.13, 13.07, 9.69, 6.81, 5.43], np.double) * 1e3

    N = m_planet.shape[0]

    r0 = np.zeros((N, 2), np.double)
    r0[:, 0] = distance_from_sun

    v0 = np.zeros((N, 2), np.double)
    v0[:, 1] = planet_v
    
    r, v = sol_odeint(m_planet, r0, v0, T, k)
    r1, v1 = sol_verlet(m_planet, r0, v0, T, k)
    r2, v2 = sol_verlet_cython(m_planet, r0, v0, T, k)
    r3, v3 = sol_verlet_mp(m_planet, r0, v0, T, k)
    r4, v4 = sol_verlet_opencl(m_planet, r0, v0, T, k)
    
    error1 = ((r - r1) ** 2).mean()
    error2 = ((r - r2) ** 2).mean()
    error3 = ((r - r3) ** 2).mean()
    error4 = ((r - r4) ** 2).mean()
    
    fig, ax = plt.subplots()
    ax.bar(['python', 'cython', 'multiprocessing', 'opencl'], [error1, error2, error3, error4])
    plt.title("Error relative to odeint")
    plt.savefig('graphs/error.png')
    plt.show()