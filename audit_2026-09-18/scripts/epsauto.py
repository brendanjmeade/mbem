"""Spot check: msd's eps='auto' (EPS_OVER_H=1.25) vs the tree's working eps/h."""
import sys, numpy as np, pathlib
DD=pathlib.Path("/Users/meade/Desktop/moss-org/ddbem")
sys.path.insert(0,str(DD)); sys.path.insert(0,str(DD/"verify"))
sys.path.insert(0,"/Users/meade/Desktop/moss-org/msd")
import ddbem
from _exact import kelvin_displacement, kelvin_stress, kelvin_traction
import mollified_bem as mb
from mbem.backends.dense import DenseBackend
from mbem.evaluate import evaluate_displacement, evaluate_stress
from mbem.model.core import BCType as MBC, Patch as MPatch, Region as MRegion, RegionModel
from mbem.model.equations import generate_system
from mbem import defaults
print("msd defaults.EPS_OVER_H =", defaults.EPS_OVER_H)
MU,NU=1.0,0.25
X0=np.array([1.7,0.9,-1.3]); F=np.array([0.3,-0.7,0.5])
OBS=np.array([[0,0,0],[.25,.1,-.1],[-.15,.2,.1],[.05,-.28,.12],[-.2,-.1,-.22],[.12,.22,.18],[-.26,.06,-.08]],float)
v,t=ddbem.icosphere(3,radius=1.0); tv=ddbem.tri_verts(v,t); h=float(ddbem.element_h(tv).mean())
mesh=mb.TriMesh(vertices=np.ascontiguousarray(v,float),triangles=np.ascontiguousarray(t))
c=mesh.centroids(); n,_=mesh.normals_and_areas()
lam=2*MU*NU/(1-2*NU)
for bc in ("dirichlet","neumann"):
  print(f"\n-- {bc}, icosphere 1280 tri, h(mean edge over ddbem def)={h:.4f}")
  for label,eps in (("eps/h=0.15",0.15*h),("eps/h=0.30",0.30*h),("eps/h=0.50",0.50*h),
                    ("eps='auto' (1.25 h_j)","auto")):
    if bc=="dirichlet":
        p=MPatch("bdy",mesh,MBC.PRESCRIBED_DISPLACEMENT,value=kelvin_displacement(c,X0,F,MU,NU)); jump="half"; defl=False
    else:
        p=MPatch("bdy",mesh,MBC.FREE_TRACTION,value=kelvin_traction(c,n,X0,F,MU,NU)); jump="calibrated"; defl=True
    reg=MRegion("body",mb.ElasticMaterial(mu=MU,lam=lam),[p],np.zeros(3))
    model=RegionModel([reg]); model.validate(); em={"bdy":eps}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back=DenseBackend(mode="direct",jump=jump,deflate=defl).assemble(generate_system(model),em)
        sol=back.solve()
        u=evaluate_displacement(model,reg,sol,OBS,em,warn_near=False)
        sg=evaluate_stress(model,reg,sol,OBS,em,warn_near=False)
    ue=kelvin_displacement(OBS,X0,F,MU,NU); se=kelvin_stress(OBS,X0,F,MU,NU)
    du=u-ue
    if bc=="neumann":
        A=np.zeros((3*len(OBS),6)); b=du.ravel()
        for i,q in enumerate(OBS):
            A[3*i:3*i+3,:3]=np.eye(3); A[3*i:3*i+3,3:]=np.array([[0,q[2],-q[1]],[-q[2],0,q[0]],[q[1],-q[0],0]])
        du=(b-A@np.linalg.lstsq(A,b,rcond=None)[0]).reshape(-1,3)
    eu=np.max(np.abs(du))/np.max(np.abs(ue)); es=np.max(np.abs(sg-se))/np.max(np.abs(se))
    print(f"   {label:24s} u_int={eu:9.3e}  sigma={es:9.3e}  cond={getattr(back,'cond_estimate',np.nan):9.2e}")
