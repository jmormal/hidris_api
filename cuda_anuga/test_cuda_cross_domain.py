"""  Test of the CUDA implementation on the rectangular cross domain
"""

import unittest, os

import anuga

from anuga import Reflective_boundary
from anuga import rectangular_cross_domain

from anuga import Domain

import numpy as num
import warnings
import time
import math

from anuga.shallow_water.sw_domain_cuda import nvtxRangePush, nvtxRangePop


nx = 500
ny = 500

def create_domain(name='domain'):

    domain = anuga.rectangular_cross_domain(nx, ny, len1=1., len2=1.)

    domain.set_flow_algorithm('DE0')
    domain.set_low_froude(0)

    domain.set_name(name)  
    domain.set_datadir('.')

    scale_me=1.0

    def topography(x,y):
        return 0.0

    def stagefun(x,y):
        stage=1.0-0.5*x
        return stage

    domain.set_quantity('elevation',topography)
    domain.set_quantity('friction',0.03)
    domain.set_quantity('stage', stagefun)

    Br=anuga.Reflective_boundary(domain)
    Bd=anuga.Dirichlet_boundary([-0.1*scale_me,0.,0.])

    domain.set_boundary({'left': Br, 'right': Bd, 'top': Br, 'bottom':Br})

    return domain

print('')
print(70*'=')
print('Test Runup')
print(70*'=')


nvtxRangePush('create domain1')
domain1 = create_domain('domain_original')
domain1.set_multiprocessor_mode(1)
nvtxRangePop()

nvtxRangePush('create domain1')
domain2 = create_domain('domain_cuda')
domain2.set_multiprocessor_mode(1)
nvtxRangePop()

yieldstep = 0.0002
finaltime = 0.0002
nvtxRangePush('evolve domain1')
print('Evolve domain1')
print('domain1 number of triangles ',domain1.number_of_elements)
for t in domain1.evolve(yieldstep=yieldstep,finaltime=finaltime):
    domain1.print_timestepping_statistics()
nvtxRangePop()

nvtxRangePush('evolve domain2')
print('Evolve domain2')
print('domain2 number of triangles ',domain2.number_of_elements)
for t in domain2.evolve(yieldstep=yieldstep,finaltime=finaltime):
    domain2.print_timestepping_statistics()
nvtxRangePop()


timestep = 0.1

nvtxRangePush('distribute domain1')
domain1.distribute_to_vertices_and_edges()
nvtxRangePop()

nvtxRangePush('update boundary domain1')
domain1.update_boundary()
nvtxRangePop()

nvtxRangePush('compute fluxes domain1')
domain1.compute_fluxes()
timestep1 = domain1.flux_timestep
boundary_flux1 = domain1.boundary_flux_sum[0]
nvtxRangePop()


nvtxRangePush('distribute on cpu for domain2')
domain2.distribute_to_vertices_and_edges()
nvtxRangePop()

nvtxRangePush('update boundary domain2')
domain2.update_boundary()
nvtxRangePop()

from anuga.shallow_water.sw_domain_cuda import GPU_interface
gpu_interface2 = GPU_interface(domain2)

nvtxRangePush('allocate gpu arrays for domain2')
gpu_interface2.allocate_gpu_arrays()
nvtxRangePop()

nvtxRangePush('compile gpu kernels for domain2')
gpu_interface2.compile_gpu_kernels()
nvtxRangePop()

nvtxRangePush('compute fluxes on gpu for domain2')
timestep2 = domain2.evolve_max_timestep 
timestep2 = gpu_interface2.compute_fluxes_ext_central_kernel(timestep2)
nvtxRangePop()

boundary_flux2 = domain2.boundary_flux_sum[0]


print('domain1 timestep ', timestep1)
print('domain2 timestep ', timestep2)

print('domain1 boundary_flux ', boundary_flux1)
print('domain2 boundary_flux ', boundary_flux2)


quantities1 = domain1.quantities
stage1 = quantities1["stage"]
xmom1 = quantities1["xmomentum"]
ymom1 = quantities1["ymomentum"]
max_speed_1 = domain1.max_speed

quantities2 = domain2.quantities
stage2 = quantities2["stage"]
xmom2 = quantities2["xmomentum"]
ymom2 = quantities2["ymomentum"]
max_speed_2 = domain2.max_speed

N = domain1.number_of_elements
sqrtN = 1.0/math.sqrt(N)

print('timestep diff                ', abs(timestep1-timestep2))
print('boundary_flux diff           ', abs(boundary_flux1-boundary_flux2))
print('max_speed diff L2-norm       ', num.linalg.norm(max_speed_1-max_speed_2)*sqrtN)
print('stage update diff L2-norm    ', num.linalg.norm(stage1.explicit_update-stage2.explicit_update)*sqrtN)
print('xmom  update diff L2-norm    ', num.linalg.norm(xmom1.explicit_update-xmom2.explicit_update)*sqrtN)
print('ymom  update diff L2-norm    ', num.linalg.norm(ymom1.explicit_update-ymom2.explicit_update)*sqrtN)

print('stage update diff Linf-norm  ', num.linalg.norm(stage1.explicit_update-stage2.explicit_update,num.inf))
print('xmom  update diff Linf-norm  ', num.linalg.norm(xmom1.explicit_update-xmom2.explicit_update,num.inf))
print('ymom  update diff Linf-norm  ', num.linalg.norm(ymom1.explicit_update-ymom2.explicit_update,num.inf))
