# Closed-form mollified dislocation kernels for constant, linear and quadratic slip on a flat triangle

This note states the closed-form results implemented in `clq` in the notation
of the manuscript appendix *Closed-form mollified DD-triangle integration*
(`moss/manuscript/appendix_kernels.tex`).  Everything reduces to the same
three ingredients as the constant-slip case — the Van Oosterom solid angle,
elementary edge antiderivatives, and the in-plane moment recurrence — carried
to higher order.  Section numbers in brackets refer to `clq` modules.

## 1. Setting

Mollified Kelvin solution (Galerkin/Cortez form), $\mathbf d = \mathbf x - \mathbf y$,
$R_\varepsilon = \sqrt{|\mathbf d|^2 + \varepsilon^2}$, $C_1 = 1/(16\pi\mu(1-\nu))$:

$$
G^\varepsilon_{ij}(\mathbf d) = C_1\Big[(3-4\nu)\frac{\delta_{ij}}{R_\varepsilon}
 + \frac{d_i d_j}{R_\varepsilon^{3}} + 2(1-\nu)\varepsilon^2\frac{\delta_{ij}}{R_\varepsilon^{3}}\Big],
\qquad
\mathcal L_{ik} G^\varepsilon_{kj} = -\delta_{ij}\,\phi_\varepsilon,\quad
\phi_\varepsilon(r) = \frac{15\varepsilon^4}{8\pi R_\varepsilon^{7}} .
$$

The last identity (regularised Cauchy–Navier equation with the blob
$\phi_\varepsilon$ as source; `verify_pointwise.py`) is what makes the
eigenstress of section 6 exact.

Displacement of a dislocation with slip $\Delta\mathbf u(\mathbf y)$ on
$\mathcal T$ with unit normal $\hat{\mathbf n}$ (Volterra representation;
slip and normal share the **first** index pair of $C$ — the traction operator):

$$
u_i(\mathbf x) = \int_{\mathcal T} \Delta u_j(\mathbf y)\, C_{jmpq}\, n_m\,
\frac{\partial G^\varepsilon_{ip}}{\partial y_q}(\mathbf x-\mathbf y)\, dS
= -\int_{\mathcal T} \Delta u_j\, C_{jmpq}\, n_m\, \partial_{x_q} G^\varepsilon_{ip}\, dS .
$$

With $C_{jmpq}=\lambda\delta_{jm}\delta_{pq}+\mu(\delta_{jp}\delta_{mq}+\delta_{jq}\delta_{mp})$
and $G^1_{ipq} := \partial_{x_q}G^\varepsilon_{ip}$ this is, per unit slip in
direction $j$,

$$
U_{ij} = -\Big[\mu\, n_m G^1_{ijm} + \lambda\, n_j\, G^1_{imm} + \mu\, n_m G^1_{imj}\Big]. \tag{1.1}
$$

The stress read from Hooke's law on $\nabla\mathbf u$ is the **total** stress
of the mollified dislocation (section 6); its kernel is
$K_{mn,k} = -C_{mnrs}C_{kjpq}n_j\,\partial_s\partial_q G^\varepsilon_{rp}$
(appendix eq. Kchain, unchanged).  Slip sign:
$\Delta\mathbf u = \mathbf u^{+} - \mathbf u^{-}$ with $+$ the side
$\hat{\mathbf n}$ points to; $\hat{\mathbf n} = (\mathbf v_2-\mathbf v_1)\times(\mathbf v_3-\mathbf v_1)/|\cdot|$.

*Remark (index pairing).* `msd/mollified_kernel/analytical_kernels.py` (pre-fix)
and its batch/numba twins carried $\lambda$ on the first term of (1.1) and
$\mu$ on the second; the two forms coincide iff $\lambda=\mu$ ($\nu = 1/4$).
Convention-free tests that distinguish them: the closed-surface identity
$\sum_{\mathcal T}\mathbf U = -\mathbf I$ inside a closed mesh, and the
displacement jump across an element (section 7).  `moss` commit `f721a6a`
carries (1.1); `msd` is fixed in the same session as this note.

## 2. Nodal slip [`clq/shape.py`]

Lagrange interpolation of order $p$ on $\mathcal T$: $K=(p+1)(p+2)/2$ nodes at
barycentric positions $\boldsymbol\alpha/p$, $|\boldsymbol\alpha|=p$, with

$$
\Delta\mathbf u(\mathbf y) = \sum_{k=1}^{K} N_k(\mathbf y)\,\mathbf s_k,\qquad
N_{\boldsymbol\alpha}(\boldsymbol\lambda)=\prod_{i=1}^{3}\prod_{j=0}^{\alpha_i-1}\frac{p\lambda_i-j}{j+1}.
$$

| $p$ | nodes (fixed order) | shape functions |
|---|---|---|
| 0 | centroid | $N=1$ |
| 1 | $\mathbf v_1,\mathbf v_2,\mathbf v_3$ | $N_k=\lambda_k$ |
| 2 | $\mathbf v_1,\mathbf v_2,\mathbf v_3,\ \mathbf m_{12},\mathbf m_{23},\mathbf m_{31}$ | $\lambda_k(2\lambda_k-1)$, $4\lambda_i\lambda_j$ |

In the in-plane frame $(\hat{\mathbf e}_1,\hat{\mathbf e}_2)$ of the appendix
the barycentric coordinates are affine.  `clq` writes them about the centroid,
$\lambda_k(\boldsymbol\eta)=A_k^c+B_k\eta_1+C_k\eta_2$ (one $3\times3$ inverse per
triangle), and about the projection $\mathbf x_\parallel$ of the observer,
$\boldsymbol\xi=\boldsymbol\eta-\mathbf X$:

$$
\lambda_k(\boldsymbol\xi)=A_k(\mathbf x)+B_k\xi_1+C_k\xi_2,\qquad A_k(\mathbf x)=A_k^c+B_kX_1+C_kX_2 .
$$

Every $N_k$ is then a polynomial $N_k(\boldsymbol\xi)=\sum_{a+b\le p}c^{(k)}_{ab}(\mathbf x)\,\xi_1^a\xi_2^b$
with, for $p=2$ (vertex node $2\lambda^2-\lambda$, edge node $4\lambda_i\lambda_j$),

$$
\begin{aligned}
&c_{00}=2A^2-A,\ c_{10}=4AB-B,\ c_{01}=4AC-C,\ c_{20}=2B^2,\ c_{11}=4BC,\ c_{02}=2C^2;\\
&c_{00}=4A_iA_j,\ c_{10}=4(A_iB_j+A_jB_i),\ c_{01}=4(A_iC_j+A_jC_i),\ c_{20}=4B_iB_j,\ c_{11}=4(B_iC_j+B_jC_i),\ c_{02}=4C_iC_j .
\end{aligned}
$$

Only the $A_k$ depend on the observer.

## 3. Nodal influence integrals

Per node $k$ (all three are what `clq.influence` returns):

$$
U^{(k)}_{ij}(\mathbf x)=\int_{\mathcal T}N_k\,U_{ij}\,dS,\qquad
H^{(k)}_{mn,j}(\mathbf x)=\int_{\mathcal T}N_k\,K_{mn,j}\,dS,\qquad
E^{(k)}(\mathbf x)=\int_{\mathcal T}N_k\,\phi_\varepsilon\,dS ,
$$

so that $u_i=\sum_k U^{(k)}_{ij}s_{k,j}$, $\sigma^{\rm tot}_{mn}=\sum_k H^{(k)}_{mn,j}s_{k,j}$ and
(section 6) $C\!:\!\boldsymbol\varepsilon^*=\sum_k E^{(k)}\big[\lambda(\mathbf s_k\!\cdot\!\hat{\mathbf n})\mathbf I+\mu(\mathbf s_k\hat{\mathbf n}^{\sf T}+\hat{\mathbf n}\mathbf s_k^{\sf T})\big]$.

## 4. In-plane moments to arbitrary order [`clq/moments.py`, `clq/primitives.py`]

Definition (appendix eq. Mab_def), $R_\varepsilon^2=\xi_1^2+\xi_2^2+h_\varepsilon^2$,
$h_\varepsilon^2=z^2+\varepsilon^2$:

$$
M_n^{(a,b)}=\int_{\mathcal T}\frac{\xi_1^a\xi_2^b}{R_\varepsilon^{\,n}}\,dS .
$$

**Seeds.**  $I_3=-\Omega(\mathbf x_{\rm eff})/h_\varepsilon$ (solid angle) and the
vertical identity $(2-n)I_n+n h_\varepsilon^2 I_{n+2}=E_n$ with
$E_n=\sum_{\rm edges}d_\perp\,[J_n]_{u_a}^{u_b}$:

$$
I_1=E_1-h_\varepsilon^2I_3,\quad I_5=\frac{E_3+I_3}{3h_\varepsilon^2},\quad I_7=\frac{E_5+3I_5}{5h_\varepsilon^2},\quad
I_{-1}=\frac{E_{-1}+h_\varepsilon^2I_1}{3}.
$$

**Master recurrence** (divergence of $\xi_1^a\xi_2^b\hat{\mathbf e}_\alpha/R_\varepsilon^{n-2}$; valid for every
$a,b\ge0$ and every odd $n\neq2$, including $n=1$ where $R_\varepsilon^{2-n}=R_\varepsilon$):

$$
M_n^{(a+1,b)}=\frac{a\,M_{n-2}^{(a-1,b)}-\mathcal B^{1}_{n-2}(a,b)}{n-2},\qquad
M_n^{(0,b+1)}=\frac{b\,M_{n-2}^{(0,b-1)}-\mathcal B^{2}_{n-2}(0,b)}{n-2},\qquad
\mathcal B^{\alpha}_{m}(a,b)=\oint_{\partial\mathcal T}\frac{\xi_1^a\xi_2^b\,(\hat{\mathbf n}_{\rm out}\!\cdot\!\hat{\mathbf e}_\alpha)}{R_\varepsilon^{\,m}}\,d\ell .
$$

The two routes to $M_n^{(a,b)}$ with $a,b\ge1$ must agree; `verify_moments.py`
gates the $(1,1)$ entry (the only one without an interior term) at $10^{-13}$ and
observes $10^{-16}$.

**Required degrees.**  For slip order $p$ the kernels consume (constant-slip
degrees raised by $p$):

| kernel | $n=1$ | $n=3$ | $n=5$ | $n=7$ |
|---|---|---|---|---|
| $U$ ($V_3$, $V_5$, $T^{[3]}_5$) | $p-1$ | $1+p$ | $3+p$ | – |
| $H$ ($I_3$, $I_5$, $T^{[2]}_5$, $T^{[2]}_7$, $T^{[4]}_7$) | $p-2$ | $p$ | $2+p$ | $4+p$ |
| $E$ | – | – | – | $p$ |

so quadratic slip closes at $a+b=6$, $n=7$ (constant: $a+b=4$).  The $n=1$
column is what the recursion consumes, not what the kernels read; the closure
of every row also needs the lower $n$ at degree reduced by two per step and the
seeds $I_3, I_5, I_7$ (and $I_1, I_{-1}, I_{-3},\dots$ for high $p$), which
`clq.moments._close_degrees` adds automatically for any $p$.

**Edge integrals.**  With the edge parameterisation of the appendix
($\xi_1=c_1d_\perp+s_1u$, $\xi_2=c_2d_\perp+s_2u$, $\rho_\varepsilon^2=d_\perp^2+h_\varepsilon^2$)
each $\mathcal B$ is a binomial sum of
$P^m_k=\int_{u_a}^{u_b}u^k\,du/(u^2+\rho_\varepsilon^2)^{m/2}$, with

$$
P^m_k=P^{m-2}_{k-2}-\rho_\varepsilon^2P^m_{k-2},\qquad P^m_0=[J_m],\quad P^m_1=[K_m],
$$

$$
J_1=\log(u+R),\quad J_3=\frac{u}{\rho_\varepsilon^2R},\quad
J_{m+2}=\frac{(m-1)J_m+uR^{-m}}{m\rho_\varepsilon^2},\quad
K_m=-\frac{R^{2-m}}{m-2},\quad
\boxed{J_{-1}=\tfrac12\big(uR+\rho_\varepsilon^2\log(u+R)\big)},\quad K_{-1}=\tfrac13R^3 .
$$

$J_{-1}$ (i.e. $\int R\,du$) is the only primitive beyond the constant-slip
set; it enters through $M_1^{(1,0)},M_1^{(0,1)}$ (needed by the $n=3$ moments
of degree 3 for quadratic slip in $U$) and through $\int u^2/R\,du$.  No new
transcendental function appears: the closed form is still `atan2` (solid
angle), `log`, and algebraic terms.  For $p=2$ the edge table is
$\{m=5:k\le5,\ m=3:k\le4,\ m=1:k\le2,\ m=-1:k\le0\}$.

**Cancellation-free differences.**  The edge terms are differences
$F(u_b)-F(u_a)$; for observers many mollification widths along an edge
($|u|\gg\rho_\varepsilon$) or with $\rho_\varepsilon\gg|u|$ the naive forms lose
$(u/\rho_\varepsilon)^{m-1}$, resp. $(\rho_\varepsilon/u)^k$, digits.  `clq` uses
$\Delta R=(u_b-u_a)(u_b+u_a)/(R_a+R_b)$, a `log1p` form of $\Delta J_1$
(reflected for $u<0$, product form for mixed signs),
$\Delta J_3=(u_b^2-u_a^2)/\big(R_aR_b(u_bR_a+u_aR_b)\big)$ for same-sign $u$,
factored $\Delta R^q$ for $\Delta K_m$, the recurrence in difference form for
$J_{m\ge5}$, and the binomial series in $(\rho_\varepsilon/u)^2$ (same sign,
$\min|u|\ge2\rho_\varepsilon$) or $(u/\rho_\varepsilon)^2$ ($\max|u|\le\rho_\varepsilon/2$).
Every regime agrees with a 40-digit reference to rounding
(`verify_primitives.py`).

## 5. Weighted moments, lift and the closed-form kernels [`clq/kernels.py`]

Per node, the shape polynomial shifts the table:

$$
W^{(k)}_n{}^{(a,b)}=\sum_{a'+b'\le p}c^{(k)}_{a'b'}(\mathbf x)\,M_n^{(a+a',\,b+b')} .
$$

With $\mathbf d=-\xi_1\hat{\mathbf e}_1-\xi_2\hat{\mathbf e}_2+z\hat{\mathbf n}$ the lift is the
appendix expansion (eq. T2_lift and its rank-1, 3, 4 analogues) with
$M\to W^{(k)}$:

$$
T^{[r](k)}_{i_1\cdots i_r,n}=\sum_{\rm assignments}(-1)^{a+b}\,z^{\,r-a-b}\,W^{(k)}_n{}^{(a,b)}\,
\hat e_{\cdot,i_1}\cdots\hat e_{\cdot,i_r} .
$$

Then, exactly as for constant slip (appendix eqs. G1-integrated, D2G_integrated, H_assembled) but per node,

$$
\begin{aligned}
G^{1(k)}_{ijm}&=C_1\big[-(3-4\nu)\delta_{ij}V^{(k)}_{3,m}+\delta_{im}V^{(k)}_{3,j}+\delta_{jm}V^{(k)}_{3,i}-3T^{[3](k)}_{ijm,5}-6(1-\nu)\varepsilon^2\delta_{ij}V^{(k)}_{5,m}\big],\\
U^{(k)}_{ij}&=-\big[\mu n_mG^{1(k)}_{ijm}+\lambda n_jG^{1(k)}_{imm}+\mu n_mG^{1(k)}_{imj}\big],\\
\overline{\mathcal D}^{(k)}_{rpsq}&=C_1\Big\{-(3-4\nu)\delta_{rp}[\delta_{sq}I^{(k)}_3-3T^{[2](k)}_{sq,5}]+\delta_{rs}[\delta_{pq}I^{(k)}_3-3T^{[2](k)}_{pq,5}]+\delta_{ps}[\delta_{rq}I^{(k)}_3-3T^{[2](k)}_{rq,5}]\\
&\qquad-3[\delta_{rq}T^{[2](k)}_{ps,5}+\delta_{pq}T^{[2](k)}_{rs,5}+\delta_{sq}T^{[2](k)}_{rp,5}]+15T^{[4](k)}_{rpsq,7}+2(1-\nu)\varepsilon^2\delta_{rp}[-3\delta_{sq}I^{(k)}_5+15T^{[2](k)}_{sq,7}]\Big\},\\
B^{(k)}_{rsj}&=\lambda n_j\overline{\mathcal D}^{(k)}_{rpsp}+\mu n_q\overline{\mathcal D}^{(k)}_{rjsq}+\mu n_p\overline{\mathcal D}^{(k)}_{rpsj},\qquad
H^{(k)}_{mn,j}=-\big[\lambda\delta_{mn}B^{(k)}_{rrj}+\mu(B^{(k)}_{mnj}+B^{(k)}_{nmj})\big],
\end{aligned}
$$

where $I^{(k)}_n=W^{(k)}_n{}^{(0,0)}$.

**Linear slip, stated explicitly.**  With $N_k=A_k+B_k\xi_1+C_k\xi_2$,

$$
W^{(k)}_n{}^{(a,b)}=A_k\,M_n^{(a,b)}+B_k\,M_n^{(a+1,b)}+C_k\,M_n^{(a,b+1)} :
$$

the linear-slip kernel is the constant-slip formula applied to the first-moment
shift of the table (one extra order at every $n$).  **Quadratic slip** adds the
second-moment shift with the six coefficients of section 2 (two extra orders).
Constant slip is the case $c_{00}=1$; the three orders share one code path.

## 6. Exact eigenstress of the smeared slip [`clq/api.py`]

Because $G^\varepsilon=G^0*\phi_\varepsilon$, the mollified dislocation field is the
exact elastic response to the eigenstrain
$\boldsymbol\varepsilon^*(\mathbf x)=\int_{\mathcal T}\phi_\varepsilon(\mathbf x-\mathbf y)\,{\rm sym}(\Delta\mathbf u\otimes\hat{\mathbf n})\,dS$,
and the kernel stress is $C\!:\!({\rm sym}\nabla\mathbf u)=\boldsymbol\sigma_{\rm el}+C\!:\!\boldsymbol\varepsilon^*$.  Hence

$$
\boldsymbol\sigma_{\rm el}=\boldsymbol\sigma_{\rm tot}-C\!:\!\boldsymbol\varepsilon^*,\qquad
E^{(k)}(\mathbf x)=\frac{15\varepsilon^4}{8\pi}\,W^{(k)}_7{}^{(0,0)}(\mathbf x),
$$

exact for the finite triangle and any slip order, at no extra cost (the
$n=7$ table is already built for $H$).  Numerically the weight inherits the
$(D/h_\varepsilon)^4$ loss of the $I_7$ seed for observers on the plane far
outside the footprint, so the eigenstress tail there is accurate relative to
the elastic-stress scale (the quantity it is subtracted from), not relative to
itself.  Infinite-plane limit
($\mathcal T\to$ plane, interior points): $E^{(k)}\to\rho_\varepsilon(z)N_k(\mathbf x_\parallel)+\varepsilon^4\nabla^2N_k/(8h_\varepsilon^3)$
with $\rho_\varepsilon(z)=\tfrac34\varepsilon^4/(z^2+\varepsilon^2)^{5/2}$ — the marginal used by
`msd/anelastic.py` (`verify_eigenstress.py`).  On the fault the raw kernel
peaks at $\tfrac34\mu\,s(\mathbf x)/\varepsilon$; the elastic stress is bounded and
converges as $\varepsilon\to0$ (`fig_eps_finiteness`).

## 7. Identities and limits (all gated in `verify/`)

* $\varepsilon\to0$ off the plane: $h_\varepsilon\to|z|$ reproduces the singular closed form
  (allowed in `clq`; $\varepsilon=0$ on the plane raises on every code path).
  Elastic on-fault stress converges as $\varepsilon\to0$ at interior points; on
  edges and vertices where the nodal slip is non-zero the classical
  singularity re-emerges.
* Partition of unity: $\sum_kU^{(k)}=U^{(p=0)}$ (same for $H$, $E$); a linear
  slip expressed on the quadratic nodes gives the linear result; cyclic vertex
  relabelling permutes nodes; reversing the orientation and negating the slip
  leaves everything unchanged.
* Rigid covariance and scaling $(\mathcal T,\mathbf x,\varepsilon)\to\alpha(\cdot)$: $U$ invariant,
  $H,E\propto\alpha^{-1}$.
* Hooke consistency: the finite-difference gradient of the closed-form
  displacement, through Hooke's law, equals the closed-form total stress at
  $\nu=0.3$.  Since the stress kernel is built from the traction-operator
  pairing, this rejects a swapped displacement contraction (the `msd` pre-fix
  state); the absolute pairing itself is fixed by the Volterra derivation and
  by the convention-free closed-surface closure and jump tests
  (`msd/verify/verify_dd_pairing.py`, `verify_jump.py`).
* Displacement jump: for interior points of a large element,
  $\mathbf u(\mathbf x+z_0\hat{\mathbf n})-\mathbf u(\mathbf x-z_0\hat{\mathbf n})\to f(z_0/\varepsilon)\,\Delta\mathbf u(\mathbf x)$
  with $f(t)=t(2t^2+3)/(2(1+t^2)^{3/2})$ (the mass of the mollified profile $\rho_\varepsilon$ within $\pm t\varepsilon$, i.e. $2F(t)-1$ with $F$ its CDF), at every $\nu$.
* Far field: the closed form loses digits with the effective distance
  $R=\sqrt{D^2+\varepsilon^2}$ (closure sum $\sim D/L$ and the observer-dependent
  $A_k\sim D/L$, squared for quadratic weights; intrinsic to the divergence
  theorem, a centroid frame only moves it).  The loss is largest for quadratic
  weights and in-plane observer directions, grows roughly like $(R/L)^{4\text{–}5}$,
  and is larger for smaller $\varepsilon$.  Measured worst relative error of the
  $U$, $H$ tensors (P0–P2, four directions, $\varepsilon=0.05L$, against Gauss
  quadrature) for the equilateral and the generic test triangle:
  $10^{-12}$ at $R/L=2$, $2\times10^{-9}$ at 10, $4\times10^{-8}$ at 20,
  $7\times10^{-6}$ at 50, $2\times10^{-4}$ at 100, $2\times10^{-3}$ at 200,
  $0.6$ at 500 and $\approx9$ at 1000.  At $\varepsilon=0.01L$ these are 2–20
  times larger.  Thin triangles are much worse, because the relevant ratio
  involves the triangle height: for $h/L=0.02$ the error is $3\times10^{-8}$
  at $R/L=2$, $4\times10^{-5}$ at 10 and $2\times10^{-3}$ at 20.  The
  closed-form producer (`far_field="analytic"`) is therefore not usable
  beyond $R\approx20L$.  Beyond $R>D_\star L$ ($D_\star=10$) the default
  `far_field="hybrid"` fills the same $W^{(k)}$ tables by Gauss quadrature of
  the smooth integrand, which is exact to rounding ($\le2\times10^{-14}$ for
  every shape above).  Within $D_\star L$ the hybrid is the closed form:
  about $10^{-9}$ just inside the switch for well-shaped triangles, but
  $3\times10^{-7}$ ($h/L=0.1$) and $10^{-5}$ ($h/L=0.02$) for thin ones,
  since $D_\star$ is relative to the longest edge.  The full table is in
  `README.md`.

## 8. Implementation and verification maps

| equation / object | module | gate |
|---|---|---|
| $J_m,K_m,P^m_k$, stable differences, solid angle | `clq/primitives.py` | `verify_primitives.py` |
| seeds, master recurrence, degrees, $W^{(k)}$, far-field producer | `clq/moments.py` | `verify_moments.py`, `verify_far_field.py` |
| nodes, $N_{\boldsymbol\alpha}$, $c^{(k)}_{ab}$, interpolation, grids | `clq/shape.py` | `verify_identities.py`, `verify_api.py` |
| lift, $G^1$, $U$, $\overline{\mathcal D}$, $H$, $E$ | `clq/kernels.py` | `verify_nodal_vs_quadrature.py`, `verify_subdivision.py`, `verify_hooke_consistency.py`, `verify_jump.py` |
| public API, $\boldsymbol\sigma_{\rm el}$, $C\!:\!\boldsymbol\varepsilon^*$ | `clq/api.py` | `verify_api.py`, `verify_eigenstress.py` |
| point kernels, blob, oracle quadrature | `clq/pointwise.py`, `clq/quadrature.py` | `verify_pointwise.py` |
| constant-slip parity with the frozen oracles | — | `verify_order0_parity.py` |
