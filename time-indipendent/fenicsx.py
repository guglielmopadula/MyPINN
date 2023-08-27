import numpy as np

from dolfinx.fem import (Constant,  Function, FunctionSpace, assemble_scalar, 
                         dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_cube, locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction, 
                 div, dot, dx, grad, inner, lhs, rhs)
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh

mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10,10)

with XDMFFile(mesh.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)






x = SpatialCoordinate(mesh)

###We are going to solve
## laplacian u = f where f =6
## u(x,y,z)=u_d on x=0 or x=1 where u_d=1/4+y**2-y+1+z**2-z 
##du/dn(x,y,z)=g on  on z=0 or z=1 where g=1
## du/dn(x,y,z)=(u-s) on y=0 or y=1 where s=(1/2-x)**2+(z**2-z) 
##which has solution
##u=(1/2-x)**2+(z**2-z)+y**2-y+1

u_true = lambda x: (1/2-x[0])**2+x[2]**2-x[2]+(x[1]**2-1*x[1]+1)
f=Constant(mesh, ScalarType(6))
u_d=lambda x: 1/4+0*x[0]+x[2]**2-x[2]+(x[1]**2-1*x[1]+1)
s=lambda x: (1/2-x[0])**2+(x[2]**2-x[2])
g=lambda x: 0*x[0]+1



boundaries = [(1, lambda x: np.logical_or(np.isclose(x[0], 0.0),np.isclose(x[0], 1.0))),
              (2, lambda x: np.logical_or(np.isclose(x[1], 0.0),np.isclose(x[1], 1.0))),
              (3, lambda x: np.logical_or(np.isclose(x[2], 0.0),np.isclose(x[2], 1.0)))
              ]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

V = FunctionSpace(mesh, ("CG", 2))
u, v = TrialFunction(V), TestFunction(V)
ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)


class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = Function(V)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(V, fdim, facets)
            self._bc = dirichletbc(u_D, dofs)
        elif type == "Neumann":
                u_N = Function(V)
                u_N.interpolate(values)
                self._bc = -inner(u_N, v) * ds(marker)
        elif type == "Robin":
            u_R = Function(V)
            u_R.interpolate(values)
            self._bc = -inner(u-u_R, v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type



# Define the Dirichlet condition
boundary_conditions = [BoundaryCondition("Dirichlet", 1, u_d),
                       BoundaryCondition("Robin", 2, s),
                       BoundaryCondition("Neumann", 3, g)
                       ]


F = inner(grad(u), grad(v)) * dx + inner(f, v) * dx
bcs = []
for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        F += condition.bc


# Solve linear variational problem
a = lhs(F)
L = rhs(F)
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


# Compute L2 error and error at nodes
V_ex = FunctionSpace(mesh, ("CG", 2))
u_exact = Function(V_ex)
u_exact.interpolate(u_true)
error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(form((uh - u_exact)**2 * dx)), op=MPI.SUM))

u_vertex_values = uh.x.array
uex_1 = Function(V)
uex_1.interpolate(u_true)
u_ex_vertex_values = uex_1.x.array
print(uex_1.x.array)
print(uh.x.array)


error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
error_max = mesh.comm.allreduce(error_max, op=MPI.MAX)
print(f"Error_L2 : {error_L2:.2e}")
print(f"Error_max : {error_max:.2e}")

