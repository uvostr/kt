import numpy as np
import pyopencl as cl
import os

os.environ['PYOPENCL_CTX'] = '0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


text_prg = """
__kernel void accelerations(__global double *m,
                  __global double *r, 
                  __global double *a,
                  const double G,
                  const int N)
{
    int i = get_global_id(0);
    for (int j = 0; j < N; j++)
    {
         if ((i != j) && (i != 0))
         {   
             double norm = 0.0;
             norm = (r[2 * j] - r[2 * i]) * (r[2 * j] - r[2 * i]) + (r[2 * j + 1] - r[2 * i + 1]) * (r[2 * j + 1] - r[2 * i + 1]);
             norm = sqrt(norm);
             norm = norm * norm * norm;
             a[2 * i] += G * m[j] * (r[2 * j] - r[2 * i]) / norm;
             a[2 * i + 1] += G * m[j] * (r[2 * j + 1] - r[2 * i + 1]) / norm;
         }
    }
}
"""

def accelerations_opencl(m, r):
    
    N = m.shape[0]
    G = float(6.6743e-11)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    m_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = m)
    r_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = r)
    a = np.zeros(2 * N, np.double)
    a_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    
    prg = cl.Program(ctx, text_prg).build()
    
    knl = prg.accelerations
    knl(queue, (N, ), None, m_buf, r_buf, a_buf, np.float64(G), np.int32(N))
    
    a_np = np.empty_like(a)
    cl.enqueue_copy(queue, a_np, a_buf).wait()
    
    return np.reshape(a_np, (-1, 2))


def sol_verlet_opencl(m, r, v, T, k):
    N = m.shape[0]
    res_r = np.zeros((N, 2, k))
    res_v = np.zeros((N, 2, k))
    dt = T / (k - 1)
    cur_a = accelerations_opencl(m, r.flatten())
    res_r[:, : , 0] = r
    res_v[:, : , 0] = v
    for i in range(1, k):
        r = r + v * dt + 0.5 * cur_a * (dt ** 2)
        next_a = accelerations_opencl(m, r.flatten())
        v = v + 0.5 * (cur_a + next_a) * dt
        cur_a = next_a
        res_r[:, : , i] = r
        res_v[:, : , i] = v
    return res_r, res_v
    
