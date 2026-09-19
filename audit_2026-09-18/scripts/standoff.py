"""Interior-field standoff: where does the mollified BEM interior stress
become trustworthy?  msd P0 on an icosphere, exact = Kelvin point force outside.
"""
import json, sys, time, pathlib
import numpy as np
DD = pathlib.Path("/Users/meade/Desktop/moss-org/ddbem")
sys.path.insert(0, str(DD)); sys.path.insert(0, str(DD/"verify"))
sys.path.insert(0, "/Users/meade/Desktop/moss-org/msd")
import ddbem
from _exact import kelvin_displacement, kelvin_stress, kelvin_traction
import mollified_bem as mb
from mbem.backends.dense import DenseBackend
from mbem.evaluate import evaluate_displacement, evaluate_stress
from mbem.model.core import BCType as MBC, Patch as MPatch, Region as MRegion, RegionModel
from mbem.model.equations import generate_system

MU, NU = 1.0, 0.25
X0 = np.array([1.7, 0.9, -1.3]); FORCE = np.array([0.3, -0.7, 0.5])
SC = pathlib.Path(sys.argv[0]).parent

def shell(r, n=60, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n,3)); v /= np.linalg.norm(v,axis=1)[:,None]
    return r*v

def run(level, bc, eps_over_h, out):
    v,t = ddbem.icosphere(level, radius=1.0)
    tv = ddbem.tri_verts(v,t); h = float(ddbem.element_h(tv).mean())
    eps = eps_over_h*h
    mesh = mb.TriMesh(vertices=np.ascontiguousarray(v,float), triangles=np.ascontiguousarray(t))
    c = mesh.centroids(); nrm,_ = mesh.normals_and_areas()
    if bc=="dirichlet":
        patch = MPatch("bdy",mesh,MBC.PRESCRIBED_DISPLACEMENT,
                       value=kelvin_displacement(c,X0,FORCE,MU,NU)); jump="half"; defl=False
    else:
        patch = MPatch("bdy",mesh,MBC.FREE_TRACTION,
                       value=kelvin_traction(c,nrm,X0,FORCE,MU,NU)); jump="calibrated"; defl=True
    lam = 2*MU*NU/(1-2*NU)
    region = MRegion("body", mb.ElasticMaterial(mu=MU,lam=lam),[patch],np.zeros(3))
    model = RegionModel([region]); model.validate()
    em = {"bdy": eps}
    back = DenseBackend(mode="direct",jump=jump,deflate=defl).assemble(generate_system(model),em)
    sol = back.solve()
    for r in (0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98):
        P = shell(r)
        u = evaluate_displacement(model,region,sol,P,em,warn_near=False)
        sg = evaluate_stress(model,region,sol,P,em,warn_near=False)
        ue = kelvin_displacement(P,X0,FORCE,MU,NU)
        se = kelvin_stress(P,X0,FORCE,MU,NU)
        du = u-ue
        if bc=="neumann":   # remove rigid motion fitted on this shell
            A = np.zeros((3*len(P),6)); b=du.ravel()
            for i,p in enumerate(P):
                A[3*i:3*i+3,:3]=np.eye(3)
                A[3*i:3*i+3,3:]=np.array([[0,p[2],-p[1]],[-p[2],0,p[0]],[p[1],-p[0],0]])
            du = (b - A@np.linalg.lstsq(A,b,rcond=None)[0]).reshape(-1,3)
        eu = float(np.max(np.abs(du)))/float(np.max(np.abs(ue)))
        es = float(np.max(np.abs(sg-se)))/float(np.max(np.abs(se)))
        standoff = 1.0-r
        rec = dict(level=level,bc=bc,n_tri=int(tv.shape[0]),h=h,eps=eps,
                   eps_over_h=eps_over_h,r=r,standoff=standoff,
                   d_over_h=standoff/h,d_over_eps=standoff/eps,u_err=eu,sig_err=es)
        out.write(json.dumps(rec)+"\n"); out.flush()
        print(f"L{level} {bc:9s} ntri={tv.shape[0]:5d} eps/h={eps_over_h} r={r:.2f} "
              f"d/h={standoff/h:6.2f} d/eps={standoff/eps:7.2f} u={eu:.3e} sig={es:.3e}", flush=True)

if __name__=="__main__":
    with open(SC/"standoff.jsonl","w") as f:
        for level in (1,2,3):
            for bc in ("dirichlet","neumann"):
                for eoh in (0.3,):
                    run(level,bc,eoh,f)
        # eps dependence at one mesh
        for eoh in (0.15,0.5):
            for bc in ("dirichlet","neumann"):
                run(3,bc,eoh,f)
