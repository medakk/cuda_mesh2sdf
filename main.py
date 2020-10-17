import nrrd
from tqdm import tqdm
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
import trimesh

# from inigo quilez
def dot2(v):
    return np.dot(v,v)

def triDist(v1, v2, v3, p):
    v21 = v2 - v1; p1 = p - v1
    v32 = v3 - v2; p2 = p - v2
    v13 = v1 - v3; p3 = p - v3
    nor = np.cross( v21, v13 )

    return np.sqrt(
                  min( min( 
                  dot2(v21*np.clip(np.dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
                  dot2(v32*np.clip(np.dot(v32,p2)/dot2(v32),0.0,1.0)-p2) ), 
                  dot2(v13*np.clip(np.dot(v13,p3)/dot2(v13),0.0,1.0)-p3) )
                  if
                 (np.sign(np.dot(np.cross(v21,nor),p1)) + 
                  np.sign(np.dot(np.cross(v32,nor),p2)) + 
                  np.sign(np.dot(np.cross(v13,nor),p3))<2.0) 
                  else
                  np.dot(nor,p1)*np.dot(nor,p1)/dot2(nor) );

def cpu_version(mesh, N):
    V = np.array(mesh.vertices)
    F = np.array(mesh.faces)

    extent = V.max(axis=0) - V.min(axis=0)
    scale = extent / N
    centroid = np.average(V, axis=0)
    V -= centroid # recenter

    print('centroid: ', centroid)
    print('scale: ', scale)

    sdf = np.zeros((64, 64, 64), dtype=np.float32)

    for f in tqdm(mesh.faces):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    pt = np.array([i,j,k])
                    pt = (pt - N/2) * scale
                    sdf[i,j,k] = triDist(V[f[0]], V[f[1]], V[f[2]], pt)
    nrrd.write('sdf.nrrd', sdf, header={'encoding': 'raw'})


# sizes: 172 311 2636

def gpu_version1(mesh, N):
    mod = SourceModule(open('mesh2sdf.cu').read())

    mesh2sdf = mod.get_function("mesh2sdf")
    sdf = np.ones((N, N, N), dtype=np.float32) * 1e7
    sdf_gpu = drv.mem_alloc(sdf.nbytes)
    drv.memcpy_htod(sdf_gpu, sdf)

    V = np.array(mesh.vertices).astype('float32')
    centroid = np.average(V, axis=0)
    V -= centroid # recenter

    F = np.array(mesh.faces).astype('int32')

    V_gpu = drv.In(V)
    F_gpu = drv.In(F)

    mesh2sdf(sdf_gpu, np.int32(N), np.int32(N), np.int32(N), V_gpu, F_gpu, np.int32(F.shape[0]),
            block=(1, 32, 32), grid=(1, 17, 17))

    drv.memcpy_dtoh(sdf, sdf_gpu)
    nrrd.write('sdf.nrrd', sdf, header={'encoding': 'raw'})

    print(f'min: {sdf.min()}\nmax: {sdf.max()}')

def main():
    mesh = trimesh.load('femur.ply')
    N = 544

    gpu_version1(mesh, N)
    # cpu_version(mesh, N)

if __name__ == '__main__':
    main()