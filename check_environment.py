"""Check dependencies and GPU operations without running PIFu reconstruction/training."""
import importlib
import sys

import numpy as np
import torch
import trimesh
from skimage import measure

print('Python:', sys.version)
for module in ('torchvision', 'cv2', 'PIL', 'scipy', 'skimage', 'matplotlib',
               'pyexr', 'OpenGL.GL', 'OpenGL.GLU', 'OpenGL.GLUT', 'embreex',
               'rtree', 'shapely', 'xxhash', 'tqdm', 'gdown',
               'lib.model', 'lib.data', 'lib.train_util', 'apps.prt_util',
               'apps.render_data'):
    importlib.import_module(module)
    print('Import OK:', module)

assert torch.cuda.is_available(), 'CUDA is unavailable'
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('CUDA matrix multiplication and backward: OK')

from OpenGL import GLUT, GLU
assert bool(GLUT.glutInit) and bool(GLU.gluPerspective)
print('FreeGLUT / GLU runtime libraries: OK')

mesh = trimesh.creation.icosphere(subdivisions=1)
assert trimesh.ray.has_embree, 'Accelerated Embree backend unavailable'
assert mesh.ray.intersects_any([[0, 0, 3]], [[0, 0, -1]])[0]
print('Embree accelerated ray intersection: OK')

# The local compatibility fix retains the Lewiner algorithm under its current API.
volume = np.zeros((8, 8, 8))
volume[2:6, 2:6, 2:6] = 1
vertices, faces, _, _ = measure.marching_cubes(volume, 0.5, method='lewiner')
assert len(vertices) and len(faces)
print('Marching cubes: OK')
print('Environment checks passed; no PIFu reconstruction or training was run.')
