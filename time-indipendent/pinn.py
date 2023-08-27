import meshio
import numpy as np
from tqdm import trange
mesh=meshio.read("deformation.xdmf")
points=mesh.points
internal_points=points.copy()
facets=mesh.cells_dict["tetra"]
from torch import nn
from torch.func import hessian,vmap,jacrev

def list_faces(t):
  t.sort(axis=1)
  n_t, m_t= t.shape 
  f = np.empty((4*n_t, 3) , dtype=int)
  i = 0
  for j in range(4):
    f[i:i+n_t,0:j] = t[:,0:j]
    f[i:i+n_t,j:3] = t[:,j+1:4]
    i=i+n_t
  return f

def extract_unique_triangles(t):
  _, indxs, count  = np.unique(t, axis=0, return_index=True, return_counts=True)
  return t[indxs[count==1]]

def extract_surface(t):
  f=list_faces(t)
  f=extract_unique_triangles(f)
  return f

facets = extract_surface(facets)


facets_neumann_0=facets[np.isclose(np.mean(points[facets][:,:,2],axis=1),0)]
points_neumann_0=np.mean(points[facets_neumann_0],axis=1)
facets_neumann_1=facets[np.isclose(np.mean(points[facets][:,:,2],axis=1),1)]
points_neumann_1=np.mean(points[facets_neumann_1],axis=1)
facets_robin_0=facets[np.isclose(np.mean(points[facets][:,:,1],axis=1),0)]
points_robin_0=np.mean(points[facets_robin_0],axis=1)
facets_robin_1=facets[np.isclose(np.mean(points[facets][:,:,1],axis=1),1)]
points_robin_1=np.mean(points[facets_robin_1],axis=1)
points_dirichlet_0=points[np.isclose(points[:,0],0)]
points_dirichlet_1=points[np.isclose(points[:,0],1)]

points_neumann=np.concatenate((points_neumann_0,points_neumann_1))
points_robin=np.concatenate((points_robin_0,points_robin_1))
points_dirichlet=np.concatenate((points_dirichlet_0,points_dirichlet_1))

print(points_robin)

normals_robin_0=np.zeros_like(points_robin_0)
normals_robin_0[:,1]=-1
normals_robin_1=np.zeros_like(points_robin_1)
normals_robin_1[:,1]=1
normals_neumann_0=np.zeros_like(points_neumann_0)
normals_neumann_0[:,2]=-1
normals_neumann_1=np.zeros_like(points_neumann_1)
normals_neumann_1[:,2]=1

normals_neumann=np.concatenate((normals_neumann_0,normals_neumann_1))
normals_robin=np.concatenate((normals_robin_0,normals_robin_1))

import torch
def laplacian(f,x):
  return torch.diagonal(vmap(hessian(f),0,0)(x),dim1 = -2, dim2 = -1).sum(axis=1)


def pde_main_loss(f,x):
  return torch.linalg.norm(laplacian(f,x)-6)

def dirichlet_loss(f,g,x):
  return torch.linalg.norm(f(x).reshape(-1)-g(x).reshape(-1))

def neumann_loss(f,g,x,v):
  return torch.linalg.norm(normal_derivative(f,x,v).reshape(-1)-g(x).reshape(-1))

def robin_loss(f,a,b,g,x,v):
  return torch.linalg.norm(a*f(x).reshape(-1)+b*normal_derivative(f,x,v).reshape(-1)-g(x).reshape(-1))

def normal_derivative(f,x,v):
  return torch.vmap(torch.dot)(vmap(jacrev(f))(x).reshape(v.shape),v)

def divergence(f,x):
  return torch.sum(vmap(jacrev(f))(x),dim=1)


u_exact=lambda x: (1/2-x[0])**2+x[2]**2-x[2]+(x[1]**2-1*x[1]+1)


#u_exact=vmap(u_exact)
u_d= lambda x: 1/4+x[2]**2-x[2]+(x[1]**2-1*x[1]+1) 
u_d=torch.vmap(u_d)
s=lambda x: -((1/2-x[0])**2+(x[2]**2-x[2]))
s=torch.vmap(s)
g=lambda x: torch.tensor(1.)
g=torch.vmap(g)



internal_points=torch.tensor(internal_points).float()
points_neumann=torch.tensor(points_neumann).float()
points_dirichlet=torch.tensor(points_dirichlet).float()
points_robin=torch.tensor(points_robin).float()
normals_neumann=torch.tensor(normals_neumann).float()
normals_robin=torch.tensor(normals_robin).float()

model=nn.Sequential(nn.Linear(3,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),
                    nn.Linear(100,100),nn.ReLU(),                    nn.Linear(100,1))
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
points=torch.tensor(points).float()


for i in trange(10000):
  optimizer.zero_grad()
  loss=0
  loss=loss+pde_main_loss(model,internal_points)
  loss=loss+neumann_loss(model,g,points_neumann,normals_neumann)
  loss=loss+dirichlet_loss(model,u_d,points_dirichlet)
  loss=loss+robin_loss(model,-1,1,s,points_robin,normals_robin)
  loss.backward()
  print(torch.linalg.norm(model(internal_points).reshape(-1)-vmap(u_exact)(internal_points))/torch.linalg.norm(vmap(u_exact)(internal_points)))
  optimizer.step()




