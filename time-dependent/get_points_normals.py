import meshio
import numpy as np
from tqdm import trange
mesh=meshio.read("deformation.xdmf")
points=mesh.points
internal_points=points.copy()
facets=mesh.cells_dict["tetra"]
from torch import nn
from torch.func import hessian,vmap,jacrev


###We are going to solve
## laplacian u = u_t 
## u(0,x,y,z)=u_d on x=0 or x=1 where u_d=1/4+y**2-y+1+z**2-z 
##du/dn(t,x,y,z)=g on  on z=0 or z=1 where g=1
## du/dn(t,x,y,z)=(u-s) on y=0 or y=1 where s=(1/2-x)**2+(z**2-z)+6*t 
##which has solution
##u(t,x,t,z)=(1/2-x)**2+(z**2-z)+y**2-y+1+6*t


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


time=np.linspace(0,1,5)

time_neumann=np.tile(time,len(normals_neumann))
time_robin=np.tile(time,len(normals_robin))
time_dirichlet=np.tile(time,len(points_dirichlet))
time_internal_points=np.tile(time,len(internal_points))


normals_neumann=np.tile(normals_neumann,(len(time),1))
normals_robin=np.tile(normals_robin,(len(time),1))
points_neumann=np.tile(points_neumann,(len(time),1))
points_robin=np.tile(points_robin,(len(time),1))
internal_points=np.tile(internal_points,(len(time),1))
points_dirichlet=np.tile(points_dirichlet,(len(time),1))


import torch
def laplacian(f,t,x):
  return torch.diagonal(vmap(hessian(f,argnums=1),0,0)(t,x),dim1 = -2, dim2 = -1).sum(axis=1)


def pde_main_loss(f,t,x):
  return torch.linalg.norm(laplacian(f,t,x).reshape(-1)-6)

def dirichlet_loss(f,g,t,x):
  return torch.linalg.norm(f(t,x).reshape(-1)-g(t,x).reshape(-1))

def neumann_loss(f,g,t,x,v):
  return torch.linalg.norm(normal_derivative(f,t,x,v).reshape(-1)-g(t,x).reshape(-1))

def robin_loss(f,a,b,g,t,x,v):
  return torch.linalg.norm(a*f(t,x).reshape(-1)+b*normal_derivative(f,t,x,v).reshape(-1)-g(t,x).reshape(-1))

def ic_loss(f,g,x):
  return torch.linalg.norm(f(torch.zeros([x.shape[0],1]),x).reshape(-1)-g(x).reshape(-1))

def normal_derivative(f,t,x,v):
  return torch.vmap(torch.dot)(vmap(jacrev(f,argnums=1))(t,x).reshape(v.shape),v)

def divergence(f,t,x):
  return torch.sum(vmap(jacrev(f,argnums=1))(t,x),dim=1)


u_exact=lambda t,x: (1/2-x[0])**2+x[2]**2-x[2]+(x[1]**2-1*x[1]+1)+6*t


#u_exact=vmap(u_exact)
u_d= lambda t,x: 1/4+x[2]**2-x[2]+(x[1]**2-1*x[1]+1)+6*t
u_d=torch.vmap(u_d)
s=lambda t,x: -((1/2-x[0])**2+(x[2]**2-x[2])+6*t)
s=torch.vmap(s)
g=lambda t,x: torch.tensor(1.)
g=torch.vmap(g)
u0=lambda x: (1/2-x[0])**2+x[2]**2-x[2]+(x[1]**2-1*x[1]+1)
u0=torch.vmap(u0)

time_internal_points=torch.tensor(time_internal_points).float().reshape(-1,1)
time_dirichlet=torch.tensor(time_dirichlet).float().reshape(-1,1)
time_neumann=torch.tensor(time_neumann).float().reshape(-1,1)
time_robin=torch.tensor(time_robin).float().reshape(-1,1)
internal_points=torch.tensor(internal_points).float()
points_neumann=torch.tensor(points_neumann).float()
points_dirichlet=torch.tensor(points_dirichlet).float()
points_robin=torch.tensor(points_robin).float()
normals_neumann=torch.tensor(normals_neumann).float()
normals_robin=torch.tensor(normals_robin).float()


class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=nn.Sequential(nn.Linear(4,100),nn.ReLU(),
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
                    nn.Linear(100,100),nn.ReLU(),                    
                    nn.Linear(100,1))
  def forward(self,t,x):
        return self.model(torch.concatenate((t.reshape(-1,1),x.reshape(-1,3)),dim=1))

model=MyModel()
optimizer = torch.optim.AdamW(model.model.parameters(),lr=0.001)
points=torch.tensor(points).float()



for i in trange(10000):
  optimizer.zero_grad()
  loss=0
  loss=loss+pde_main_loss(model,time_internal_points,internal_points)
  loss=loss+neumann_loss(model,g,time_neumann,points_neumann,normals_neumann)
  loss=loss+dirichlet_loss(model,u_d,time_dirichlet,points_dirichlet)
  loss=loss+robin_loss(model,-1,1,s,time_robin,points_robin,normals_robin)
  loss=loss+ic_loss(model,u0,internal_points)
  loss.backward()
  print(torch.linalg.norm(model(time_internal_points,internal_points).reshape(-1)-vmap(u_exact)(time_internal_points,internal_points).reshape(-1))/torch.linalg.norm(vmap(u_exact)(time_internal_points,internal_points).reshape(-1)))
  optimizer.step()




