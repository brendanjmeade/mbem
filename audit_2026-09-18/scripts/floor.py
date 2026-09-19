import sys, numpy as np, pathlib, warnings
DD=pathlib.Path("/Users/meade/Desktop/moss-org/ddbem")
sys.path.insert(0,str(DD)); sys.path.insert(0,str(DD/"verify")); sys.path.insert(0,"/Users/meade/Desktop/moss-org/msd")
import ddbem
from _exact import kelvin_displacement, kelvin_stress, kelvin_traction
import mollified_bem as mb
from mbem.backends.dense import DenseBackend
from mbem.evaluate import evaluate_displacement, evaluate_stress
from mbem.model.core import BCType as MBC, Patch as MPatch, Region as MRegion, RegionModel
from mbem.model.equations import generate_system
MU,NU=1.0,0.25; X0=np.array([1.7,.9,-1.3]); F=np.array([.3,-.7,.5])
OBS=np.array([[0,0,0],[.25,.1,-.1],[-.15,.2,.1],[.05,-.28,.12],[-.2,-.1,-.22],[.12,.22,.18],[-.26,.06,-.08]],float)
lam=2*MU*NU/(1-2*NU)
for level in (2,3):
  v,t=ddbem.icosphere(level,radius=1.0); tv=ddbem.tri_verts(v,t); h=float(ddbem.element_h(tv).mean())
  mesh=mb.TriMesh(vertices=np.ascontiguousarray(v,float),triangles=np.ascontiguousarray(t))
  c=mesh.centroids(); n,_=mesh.normals_and_areas()
  print(f"\n### icosphere {tv.shape[0]} tri, h={h:.4f}")
  for bc in ("dirichlet","neumann"):
    print(f"  -- {bc}");  print(f"     {'eps/h':>6} {'u_int':>10} {'sigma':>10} {'u_surf':>10} {'cond':>9}")
    for eoh in (0.02,0.04,0.075,0.10,0.15,0.225,0.30,0.45,0.60,0.80,1.0,1.25):
      eps=eoh*h
      if bc=="dirichlet":
        p=MPatch("bdy",mesh,MBC.PRESCRIBED_DISPLACEMENT,value=kelvin_displacement(c,X0,F,MU,NU)); jump="half"; defl=False
      else:
        p=MPatch("bdy",mesh,MBC.FREE_TRACTION,value=kelvin_traction(c,n,X0,F,MU,NU)); jump="calibrated"; defl=True
      reg=MRegion("body",mb.ElasticMaterial(mu=MU,lam=lam),[p],np.zeros(3))
      model=RegionModel([reg]); model.validate(); em={"bdy":eps}
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back=DenseBackend(mode="direct",jump=jump,deflate=defl).assemble(generate_system(model),em)
        sol=back.solve()
        u=evaluate_displacement(model,reg,sol,OBS,em,warn_near=False)
        sg=evaluate_stress(model,reg,sol,OBS,em,warn_near=False)
      ue=kelvin_displacement(OBS,X0,F,MU,NU); se=kelvin_stress(OBS,X0,F,MU,NU); du=u-ue
      us=np.nan
      if bc=="neumann":
        A=np.zeros((3*len(OBS),6)); b=du.ravel()
        for i,q in enumerate(OBS):
          A[3*i:3*i+3,:3]=np.eye(3); A[3*i:3*i+3,3:]=np.array([[0,q[2],-q[1]],[-q[2],0,q[0]],[q[1],-q[0],0]])
        du=(b-A@np.linalg.lstsq(A,b,rcond=None)[0]).reshape(-1,3)
        tr=sol["u:bdy"]; ur=kelvin_displacement(c,X0,F,MU,NU); d2=tr-ur
        A=np.zeros((3*len(c),6)); b=d2.ravel()
        for i,q in enumerate(c):
          A[3*i:3*i+3,:3]=np.eye(3); A[3*i:3*i+3,3:]=np.array([[0,q[2],-q[1]],[-q[2],0,q[0]],[q[1],-q[0],0]])
        d2=(b-A@np.linalg.lstsq(A,b,rcond=None)[0]).reshape(-1,3)
        us=np.max(np.abs(d2))/np.max(np.abs(ur))
      print(f"     {eoh:6.3f} {np.max(np.abs(du))/np.max(np.abs(ue)):10.3e} "
            f"{np.max(np.abs(sg-se))/np.max(np.abs(se)):10.3e} {us:10.3e} {getattr(back,'cond_estimate',np.nan):9.2e}", flush=True)
