import time

from mesh_to_sdf import mesh_to_voxels
import trimesh
import skimage
import skimage.measure

import nrrd

mesh = trimesh.load('femur.ply')

start = time.time()
voxels = mesh_to_voxels(mesh, 64, pad=True)
elapsed = time.time() - start
print(f'Time to voxelize: {elapsed:0.3}s') # about 18.5 s


# vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()

nrrd.write('existing_sdf.nrrd', voxels, header={'encoding': 'raw'})
