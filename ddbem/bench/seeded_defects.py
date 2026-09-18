"""Seed known defects into COPIES of ddbem/clq and report which gates catch them.

    python bench/seeded_defects.py                    # all defects, 4 at a time
    python bench/seeded_defects.py D1_clq_U_lam_mu_swap
    python bench/seeded_defects.py --list
    python bench/seeded_defects.py --workdir /tmp/x --jobs 2

A gate that has never failed is an untested gate.  This is the other half of
verification: instead of asking "do the gates pass on correct code", it asks
"would they fail on incorrect code", by breaking the code on purpose and
running the real gates against it.

HOW IT IS SAFE.  Nothing here writes to the real tree.  Each defect gets its
own sandbox --  ``<workdir>/<defect>/{ddbem, clq, msd -> symlink}`` -- built by
copying ``ddbem`` and ``clq``, applying a unique-match text substitution, and
running the gates from the copy with ``DDBEM_CLQ_ROOT`` pointed at the copied
clq.  ``msd`` is symlinked because it is only ever read.  If a substitution is
not a unique match the run aborts rather than guessing, so a defect silently
becoming a no-op after a refactor is a hard error, not a false "caught".

WHAT IT MEASURED, 2026-09-18 (README, "Adversarial verification"):
14 of 17 defects were caught; the three that were not are the finding --
the Voigt row order of the public stress API, the ``FAR_FIELD`` default, and
anything about ``orientation = -1``.  Two control entries (``CTRL_``, ``HARD_``)
collapse the gates' ``nu`` sweep to 1/4 and demonstrate that the lam/mu pairing
swap really is invisible there: with every ``nu`` literal in
``verify_kernels.py`` at 0.25 that gate PASSES with the bug in.

Cost: one full gate suite per defect, ~4 min each, ~20 min for all 17 at
``--jobs 4``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]      # .../ddbem
MOSS_ORG = ROOT.parent
GATES = ["verify_kernels.py", "verify_layout.py", "verify_solver.py",
         "verify_msd_parity.py", "verify_convergence.py"]


# --------------------------------------------------------------------------
# the defects.  Each is a list of (path relative to the sandbox, old, new);
# ``old`` must occur EXACTLY once in the file.
# --------------------------------------------------------------------------

DEFECTS = {
    # ---- the lam/mu pairing swap (invisible at nu = 1/4) ------------------
    "D1_clq_U_lam_mu_swap": [(
        "clq/clq/kernels.py",
        '        term1 = mu * np.einsum("m,nkijm->nkij", n, G1)\n'
        '        term2 = lam * np.einsum("j,nkimm->nkij", n, G1)',
        '        term1 = lam * np.einsum("m,nkijm->nkij", n, G1)\n'
        '        term2 = mu * np.einsum("j,nkimm->nkij", n, G1)')],
    "D1b_clq_H_lam_mu_swap": [(
        "clq/clq/kernels.py",
        '        B = (lam * np.einsum("j,nkrs->nkrsj", n, trD)\n'
        '             + mu * np.einsum("q,nkrjsq->nkrsj", n, D2)',
        '        B = (mu * np.einsum("j,nkrs->nkrsj", n, trD)\n'
        '             + lam * np.einsum("q,nkrjsq->nkrsj", n, D2)')],

    # ---- wrong eigenstress subtraction -----------------------------------
    "D2a_eigen_lam_mu_swap": [(
        "ddbem/ddbem/assemble.py",
        '    return (lam * np.einsum("ml,j->mlj", eye, n)\n'
        '            + mu * (np.einsum("mj,l->mlj", eye, n) + np.einsum("m,lj->mlj", n, eye)))',
        '    return (mu * np.einsum("ml,j->mlj", eye, n)\n'
        '            + lam * (np.einsum("mj,l->mlj", eye, n) + np.einsum("m,lj->mlj", n, eye)))')],
    "D2b_eigen_not_subtracted": [(
        "ddbem/ddbem/assemble.py",
        "    H = inf.H                                        # (N, K, 3, 3, 3) = [n,k,m,l,j]\n"
        "    if not subtract:\n        return H",
        "    H = inf.H                                        # (N, K, 3, 3, 3) = [n,k,m,l,j]\n"
        "    if True:\n        return H")],
    "D2c_eigen_2pct_low": [(
        "ddbem/ddbem/assemble.py",
        "    return H - inf.E[:, :, None, None, None] * sig0[None, None]",
        "    return H - 0.98 * inf.E[:, :, None, None, None] * sig0[None, None]")],

    # ---- permuted P2 node order ------------------------------------------
    "D3a_p2_node_perm_mesh": [(
        "ddbem/ddbem/mesh.py",
        "        lam = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],\n"
        "                        [.5, .5, 0.], [0., .5, .5], [.5, 0., .5]])",
        "        lam = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],\n"
        "                        [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])")],
    "D3b_p2_shape_perm": [(
        "ddbem/ddbem/mesh.py",
        "        return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),\n"
        "                         4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=1)",
        "        return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),\n"
        "                         4 * l2 * l3, 4 * l3 * l1, 4 * l1 * l2], axis=1)")],
    # a CONSISTENT relabelling of clq's own P2 edge nodes: nodes and shape
    # functions move together, so clq stays self-consistent and only its
    # agreement with ddbem's node order and the textbook basis breaks
    "D3c_clq_p2_node_perm": [(
        "clq/clq/shape.py",
        "    for (i, j) in ((0, 1), (1, 2), (2, 0)):",
        "    for (i, j) in ((1, 2), (2, 0), (0, 1)):")],

    # ---- free term: I, or the analytic 1/2, instead of N_k(x_c) ----------
    "D4a_freeterm_identity": [(
        "ddbem/ddbem/mesh.py",
        '    return shape_at(order, barycentric_nodes(order, shrink))',
        '    return np.eye(n_nodes(order))')],
    "D4b_freeterm_analytic_half": [(
        "ddbem/ddbem/model.py",
        '            R = (rowsum[c_lo:c_lo + n_coll] if self.jump == "calibrated"\n'
        '                 else np.broadcast_to(np.eye(3) * _ANALYTIC_R[rt],\n'
        '                                      (n_coll, 3, 3)))',
        '            R = np.broadcast_to(np.eye(3) * _ANALYTIC_R[rt],\n'
        '                                (n_coll, 3, 3))')],

    # ---- sign flip on slip -----------------------------------------------
    "D5a_U_sign_flip": [(
        "ddbem/ddbem/assemble.py",
        "def _u_block(inf, nhat, mu, nu):\n    return inf.U",
        "def _u_block(inf, nhat, mu, nu):\n    return -inf.U")],
    "D5b_clq_UH_sign_flip": [
        ("clq/clq/kernels.py",
         '        out["U"] = -(term1 + term2 + term3)',
         '        out["U"] = (term1 + term2 + term3)'),
        ("clq/clq/kernels.py",
         '        H = -(lam * np.einsum("ml,nkj->nkmlj", eye, trB)\n'
         '              + mu * (B + np.swapaxes(B, 2, 3)))',
         '        H = (lam * np.einsum("ml,nkj->nkmlj", eye, trB)\n'
         '             + mu * (B + np.swapaxes(B, 2, 3)))'),
        ("clq/clq/kernels.py",
         '        out["E"] = (15.0 * eps ** 4 / (8.0 * np.pi)) * W[7][:, :, 0, 0]',
         '        out["E"] = -(15.0 * eps ** 4 / (8.0 * np.pi)) * W[7][:, :, 0, 0]')],

    # ---- conventions and defaults ----------------------------------------
    "X1_eps_not_per_element": [(
        "ddbem/ddbem/assemble.py",
        "        inf = clq.influence(x_field, tri, mu, nu, eps[s], order=order,",
        "        inf = clq.influence(x_field, tri, mu, nu, eps[0], order=order,")],
    "X2_voigt_yz_xz_swapped": [(
        "ddbem/ddbem/defaults.py",
        "VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))",
        "VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 2), (1, 2), (0, 1))")],
    "X3_far_field_analytic": [(
        "ddbem/ddbem/defaults.py",
        'FAR_FIELD = "hybrid"',
        'FAR_FIELD = "analytic"')],
    "X4_rowsum_sign": [(
        "ddbem/ddbem/model.py",
        "            sgn[col_slice[p.name]] = -float(p.orientation)",
        "            sgn[col_slice[p.name]] = float(p.orientation)")],
    "X5_orientation_ignored": [(
        "ddbem/ddbem/model.py",
        "            Fq = float(q.orientation) * (R - np.eye(3) * _TARGET[rt])",
        "            Fq = 1.0 * (R - np.eye(3) * _TARGET[rt])")],
    "X6_shrink_default_0p3": [(
        "ddbem/ddbem/defaults.py",
        "COLLOCATION_SHRINK_BY_ORDER = {0: 0.0, 1: 0.5, 2: 0.5}",
        "COLLOCATION_SHRINK_BY_ORDER = {0: 0.0, 1: 0.3, 2: 0.3}")],
}

#: ``CTRL_<name>`` = that defect with the gates' nu sweep collapsed to 1/4.
NU_QUARTER = [
    ("ddbem/verify/_common.py",
     "NU_SWEEP = (0.25, 0.30, 0.45)      # the lam/mu pairing bug is invisible at 1/4",
     "NU_SWEEP = (0.25,)                 # CONTROL: collapsed to lam = mu"),
]

#: ``HARD_<name>`` = the same, plus the two hard-coded ``nu = 0.30`` inside
#: verify_kernels.py, so that gate genuinely never leaves lam = mu.
NU_QUARTER_HARD = NU_QUARTER + [
    ("ddbem/verify/verify_kernels.py",
     "    ak = msd_analytical()\n    nu = 0.30\n    for p in (0, 1, 2):",
     "    ak = msd_analytical()\n    nu = 0.25\n    for p in (0, 1, 2):"),
    ("ddbem/verify/verify_kernels.py",
     "    t1 = tri[0]\n    nu = 0.30",
     "    t1 = tri[0]\n    nu = 0.25"),
]


def edits_for(name: str):
    if name.startswith("HARD_"):
        return list(DEFECTS[name[5:]]) + NU_QUARTER_HARD
    if name.startswith("CTRL_"):
        return list(DEFECTS[name[5:]]) + NU_QUARTER
    return list(DEFECTS[name])


# --------------------------------------------------------------------------

def sandbox(work: pathlib.Path, name: str) -> pathlib.Path:
    d = work / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ig = shutil.ignore_patterns("__pycache__", "convergence_cache", "*.jsonl")
    shutil.copytree(ROOT, d / "ddbem", ignore=ig)
    shutil.copytree(MOSS_ORG / "clq", d / "clq", ignore=ig)
    os.symlink(MOSS_ORG / "msd", d / "msd")
    return d


def apply_patch(root: pathlib.Path, edits) -> None:
    for rel, old, new in edits:
        f = root / rel
        s = f.read_text()
        if s.count(old) != 1:
            raise SystemExit(
                f"{rel}: the defect's anchor text occurs {s.count(old)} times, "
                f"need exactly 1 -- the file moved under the harness.  Fix the "
                f"anchor; do NOT relax it, or the defect becomes a silent no-op "
                f"and this script will report a false 'caught'.\n  {old[:120]!r}")
        f.write_text(s.replace(old, new))


def run_gate(root: pathlib.Path, gate: str, timeout: float = 3600.0) -> dict:
    env = dict(os.environ)
    env["DDBEM_CLQ_ROOT"] = str(root / "clq")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, f"verify/{gate}"],
                           cwd=str(root / "ddbem"), capture_output=True,
                           text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"gate": gate, "verdict": "TIMEOUT", "failed": [],
                "n_checks_failed": 0, "dt": timeout, "tail": ""}
    verdict = "ERROR"
    for ln in reversed([l for l in p.stdout.splitlines() if l.strip()]):
        if ln.startswith("PASS") or ln.startswith("FAIL"):
            verdict = ln.split(":")[0]
            break
    if p.returncode != 0 and verdict == "PASS":
        verdict = "FAIL"
    failed = [l.split("]", 1)[1].strip() for l in p.stdout.splitlines() if "[XX]" in l]
    tail = (p.stderr.strip().splitlines()[-1]
            if verdict == "ERROR" and p.stderr.strip() else "")
    return {"gate": gate, "verdict": verdict, "failed": failed,
            "n_checks_failed": len(failed),
            "dt": time.perf_counter() - t0, "tail": tail}


def run_defect(work: pathlib.Path, name: str, gates) -> dict:
    root = sandbox(work, name)
    apply_patch(root, edits_for(name))
    res = [run_gate(root, g) for g in gates]
    rec = {"defect": name, "gates": res}
    (work / f"{name}.json").write_text(json.dumps(rec, indent=1))
    caught = [r["gate"] for r in res if r["verdict"] != "PASS"]
    short = [g.replace("verify_", "").replace(".py", "") for g in caught]
    print(f"{name:28s} caught by {len(caught)}/{len(res)}: "
          f"{', '.join(short) or 'NONE -- UNCAUGHT, this is a finding'}", flush=True)
    for r in res:
        for c in r["failed"][:2]:
            print(f"      {r['gate'].replace('verify_', ''):18s} {c[:88]}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("defects", nargs="*", help="defect names (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workdir", default=None,
                    help="sandbox root (default: a fresh temp directory)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gates", nargs="+", default=GATES)
    a = ap.parse_args()
    if a.list:
        for k in DEFECTS:
            print(f"  {k}")
        print("  (prefix CTRL_ or HARD_ for the nu = 1/4 controls)")
        return 0
    import tempfile
    work = pathlib.Path(a.workdir) if a.workdir else pathlib.Path(
        tempfile.mkdtemp(prefix="ddbem_defects_"))
    work.mkdir(parents=True, exist_ok=True)
    names = a.defects or list(DEFECTS)
    print(f"sandboxes in {work}\n")
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        recs = list(ex.map(lambda n: run_defect(work, n, a.gates), names))
    uncaught = [r["defect"] for r in recs
                if all(g["verdict"] == "PASS" for g in r["gates"])]
    print("-" * 72)
    print(f"{len(recs) - len(uncaught)} / {len(recs)} defects caught")
    if uncaught:
        print("UNCAUGHT (each one is a gate that needs writing): "
              + ", ".join(uncaught))
    return 0


if __name__ == "__main__":
    sys.exit(main())
