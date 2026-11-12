# Flat-spectrum connections with convex training at multiple depths

This is a theory extension requested on 2026-11-12 after the convex stress
experiments. It supports both **plain residual connections** and a new
**phase-Fourier connection**. In the interacting construction, each nonlinear
branch reads the actual state produced by earlier blocks. The learned
coordinates are joint weights over complete paths through a fixed finite
block library. Mixing final path logits gives convex ordinary cross-entropy;
the paths themselves retain their nonlinear feature interactions.

A two-block residual witness represents an input product through correlated
routing. Independent routing with identical layer marginals erases that
product. Thus the extension fixes the additive baseline's missing interaction
capacity, at the cost of changing the training coordinates: a dense tensor
has `|J|^depth` entries. It does **not** make arbitrary continuous layer weights
jointly convex. No stress results, optimizer convergence or native runtime
are claimed. CD5–CD7 give the interaction extension; CD1–CD4 record the
baseline, the transport construction, and the obstruction that motivates it.

The reference for the user's “new mHC” is the local repository
`/home/misha/python_projects/diff-gpt`, inspected at `3555299`. Its installed
phase-mHC implementation and associated proof were introduced at `fe1d528`:

- `src/native_transformer/src/phase_mhc.rs`: the phase table and `write` kernel;
- `src/native_transformer/src/mhc.rs`: dynamic router and finite Sinkhorn;
- `src/native_transformer/src/engine_mhc.rs`: attention/FFN branch integration;
- `research/main_model/README.txt`: the selected architecture;
- `research/main_model/proofs/ComplexPhaseRouting.lean`: exact phase algebra.

That model uses `D P D*`, with `r_0=1` and
`r_s=(1+i b_s)/sqrt(1+b_s^2)` for the other three streams. Its phases, dynamic
routers, attention, complex-product FFN and tied embedding/readout train.
RMS normalization also depends on the current hidden state. The existing
phase proof concerns frozen coefficients and does not assert joint convexity.

The original mHC operator and ideal double stochasticity were also checked
against the local manuscript `papers/arXiv-2512.24880v2/main.tex`, §1,
Eqs. (1)–(4), and §§4.1–4.2. The manuscript motivates stable transport; it
does not claim that its complete training objective is convex.

## CD1. Additive baseline and its exact training coordinates

For a visible input or causal prefix `x`, use

\[
h_0(x)=e(x),\qquad
h_{\ell+1}(x)=U_\ell h_\ell(x)
  +\alpha_\ell W_\ell\phi_\ell(x),\qquad
z_W(x)=R h_d(x).
\]

The trained coordinates are **all entries of every `W_l`**, jointly. Each
`W_l` is a real-linear map; its input and output spaces can include both real
and imaginary coordinates. The initial representation `e`, feature maps
`phi_l`, transport `U_l`, scales `alpha_l`, and final linear readout `R` are
fixed before optimization. Features can be nonlinear functions of the visible
input, including a frozen causal encoder. They may not depend on `W` or on
the evolving learned hidden state. No labels or arithmetic answer rules are
supplied by these definitions.

The objective is ordinary full-vocabulary teacher-forced cross-entropy:

\[
L(W)=\sum_s\left[\log\sum_c\exp(z_W(x_s)_c)-z_W(x_s)_{y_s}\right].
\]

`ConvexDepth/Basic.lean` proves by induction on the executed recurrence that

\[
h_{aW+bV}(x)=a h_W(x)+b h_V(x)\quad(a+b=1).
\]

Thus the same CE is jointly convex in every layer's injection parameters.
The feasible set `norm(W_l) <= rho_l` for all `l<d` is convex as well, and
nonnegative radii admit the zero initialization. These are upper bounds on
the largest singular value, not rank or equal-singular-value constraints on
trained matrices. A concrete two-block example has nonzero contributions
from both separately learned injections.

`run_difference_linear` also proves

\[
h_W(x)-h_V(x)=h^{e=0}_{W-V}(x).
\]

This is an expressivity limitation, not just a proof device: with the fixed
features and transports, the final predictor is affine in a concatenated
feature representation. It learns no interaction between representations
produced by different trained blocks. More blocks may add distinct prescribed
feature banks and parameter budgets, but this theory proves no intrinsic
advantage over the corresponding affine predictor.

Replacing fixed features by `phi_l(h_l)`, training the phase/router together
with the injections, training the readout jointly, tying it to a learned
embedding, or adding a hidden-state RMS normalization before the readout
falls outside this convexity theorem. The original native phase-mHC formula
is therefore not a direct instance of the convex model.

## CD2. Additive bounds through actual depth

For nonexpansive `U_l` and feasible `W`, the proved bound is

\[
\|h_d(x)\|\le\|e(x)\|
  +\sum_{\ell<d}|\alpha_\ell|\rho_\ell\|\phi_\ell(x)\|.
\]

There is no product of learned block norms. A bound on the entire sum gives
a bound independent of the number of blocks; flat transport alone does not
bound the accumulated injection. The four-block witness uses fixed scales
`1/4` and nonzero injections with a total budget of one.

For two parameter choices, the same executed recurrence obeys

\[
\|h_W(x)-h_V(x)\|
 \le\sum_{\ell<d}|\alpha_\ell|\|W_\ell-V_\ell\|\|\phi_\ell(x)\|.
\]

A fixed linear readout multiplies this bound by its operator norm.
When injections are zero and the transports are isometries, the initial
state norm is preserved at every depth. Nonzero injections can still add
or cancel signal; no isometry claim is made for the complete block.

## CD3. A concrete flat four-stream skip

Replace the stochastic skip matrix by the fixed normalized Fourier matrix

\[
F_4=\tfrac12\begin{pmatrix}
1&1&1&1\\
1&i&-1&-i\\
1&-1&1&-1\\
1&-i&-1&i
\end{pmatrix},\qquad U_\ell=D_\ell F_4D_\ell^*.
\]

`D_l` uses three fixed phase coordinates with DiffGPT's exact phase formula
and its stream-zero gauge. The formalization proves both Gram identities
for `F_4`, unit magnitude of the phases, and unitarity after conjugation.
It connects these properties to the existing `FlatSpectrum.IsFlat` and
`singularValues` definitions: every transport singular value is exactly one.
Each Fourier entry has magnitude `1/2`, so this is actual all-stream mixing.
The corresponding continuous real-linear operator preserves the L2 norm
of the complex four-stream state, supplying CD2's transport hypothesis.

Unlike ideal mHC's positive stochastic mixer, this connection permits
complex cancellation and preserves every skip direction's norm. It need not
preserve the ordinary stream mean or constant-stream direction. The theorem
is for exact arithmetic, not the generated FP32 phase coefficients.
The concrete transport is proved for four complex stream coordinates;
replicating it across feature coordinates is an implementation step not
benchmarked here.

The spectrum is flat **by construction of the fixed transport**. Imposing
unitarity as an equality on freely trained transport matrices would not make
their feasible set convex. Replacing that equality by an operator-norm ball
would make each individual feasible set convex, but would not eliminate the
products of parameters across trained layers. CD4 addresses that issue.

This proposal intentionally differs from both native phase-mHC and the
earlier convex probability-table models. Its hidden state and logits are
general real/complex linear features. The one-hot probability-coordinate CE
minimum certificate of CM8 is not transferred to it.

## CD4. Why residual or mHC mixing alone is insufficient

The scalar residual block `r_t(x)=x+(-2t)x` gives a binary classifier with
logits `[r_b(r_a(1)),0]`. For `(a,b)=(1/4,1/4)` or `(3/4,3/4)`, its target
margin is `1/4`. At the parameter midpoint `(1/2,1/2)`, its margin is zero:

\[
L(\tfrac14,\tfrac14)=L(\tfrac34,\tfrac34)
=\log(1+e^{-1/4})<\log 2=L(\tfrac12,\tfrac12).
\]

Lean proves the strict midpoint violation for ordinary CE on the full
two-parameter gate square. The two blocks are evaluated, rather than
assuming their desired outputs.

The same obstruction occurs in a four-stream mHC-style skip. Define
`P(t)[i,j]=(1-t)/2` when `i,j` belong to the same pair of streams, and `t/2`
otherwise. Every `P(t)` for `0<=t<=1` is doubly stochastic and has operator
norm at most one. The endpoint and midpoint matrices are strictly positive.
On the disagreement `[1,1,-1,-1]`, two such mixers realize the residual
classifier's first logit. The gate-to-matrix map is affine, so the failure
does not result from a nonlinear parameter chart. `fourMixer_loss_not_convex`
proves the failure for this executed four-stream objective.

This witness has zero phase and a fixed stream-sensitive readout. It refutes
the general inference from ideal mHC stability to convex training; it is not
a theorem about every detail of DiffGPT's mean/RMS/tied output head. Neither
the mHC paper nor the existing DiffGPT phase proof claims that inference.

## CD5. Preserve nonlinear depth; learn its joint path tensor

Let `J` be a finite library of choices at each depth. For a complete path
`p=(j_0,...,j_{d-1})`, execute

\[
h_0^p(x)=x,\qquad h_{\ell+1}^p(x)=B_{\ell,j_\ell}(h_\ell^p(x)).
\]

The branch functions may be nonlinear, with attention, products or hidden
normalization inside them. They are fixed during this optimization. The two
connection families are

| Connection | Executed block |
| --- | --- |
| Plain residual | `B_l,j(h) = h + F_l,j(h)` |
| Phase-Fourier | `B_l,j(h) = D_l F_4 D_l* h + F_l,j(h)` |

Identity transport recovers the first formula exactly. `pathRun_residual`
identifies its execution with the existing `Residual.residualRun`, for
arbitrary branches, inputs and depth. A concrete nonlinear example has two
squaring residual branches and executes `1 -> 2 -> 6`.

Let `g` be a fixed terminal decoder, possibly nonlinear. Train one joint
tensor `w_p >= 0`, `sum_p w_p = 1`, and predict

\[
z_w(x)=\sum_p w_p\,g(h_d^p(x)).
\]

`pathLoss_convex` proves ordinary CE of these mixed logits is convex in all
joint coordinates. This is neither a mixture of path losses nor a mixture
of probabilities. Each path executes its entire nonlinear recurrence and
decoder **before** logits are averaged. Applying a nonlinear block or decoder
to the mean hidden state would define a different model and generally break
the argument. A delta tensor reproduces any complete path and decoder exactly.

Independent layer distributions embed through
`w_p = product_l theta_l,j_l`; `productWeights_mem` proves feasibility.
The convex domain also contains correlated distributions, which independent
factors cannot express. The product constraint is deliberately removed, not
proved convex. Continuous branch/router weights can only be included here
by fixing a finite selection of candidate blocks before joint optimization;
this theorem does not optimize those raw weights. In particular, it does not
cover jointly training the native DiffGPT library or its embedding/decoder.

`path_card` proves the exact dense coordinate count `|J|^d`. Path enumeration
and storing states can therefore be expensive. No polynomial compression or
equivalence to one inexpensive residual forward pass has been established.
The model remains affine in a bank of **complete nonlinear path features**;
its improvement over CD1 is how those features preserve composition, and
its extra freedom over independent routing is cross-depth correlation.

## CD6. Bounds with interacting branches

Suppose `U_l` is nonexpansive and the actual nonlinear branches satisfy
`norm(F_l,j(h)) <= beta_l` for every candidate and state. Then

\[
\|h_d^p(x)\|\le\|x\|+\sum_{\ell<d}\beta_\ell,
\qquad
\Big\|\sum_p w_p h_d^p(x)\Big\|
\le\|x\|+\sum_{\ell<d}\beta_\ell.
\]

`pathRun_norm_le` proves the bound through actual path execution;
`pathPredict_norm_le` proves its preservation under feasible joint weights.
`fourier_path_bound` instantiates the connection with CD3's exact phase-Fourier
transport; `residual_path_bound` gives the same result for plain residuals.
Nonzero bounded branches witness the premises. These are conditional bounds:
an arbitrary polynomial branch need not be bounded, including the global
interaction example below. No bound on an arbitrary decoder is inferred.

Only the skip transport is an isometry. A branch can add or cancel signal,
and averaging different paths can cancel it too. This theory does not claim
a flat Jacobian spectrum for the entire nonlinear network or ensemble.

## CD7. A concrete cross-block interaction, using just residuals

For input `(x,y)`, choose signs `a,b` in `{-1,+1}`. The first residual branch
adds `((a-1)x,0)`, yielding `(a x,y)`. The second branch at state `(u,v)` adds
`(b u v-u,0)`, yielding `(b a x y,y)`. It multiplies the feature actually
produced by the first block. `interaction_path` checks both executed blocks.

Put weight `1/2` on each equal-sign path and zero on the others. Both layer
marginals are uniform, but this joint tensor is not a product of any two
layer factors (`correlatedWeights_not_product`). Its output is `(xy,y)`.
Independent uniform routing, with the same library and marginals, outputs
`(0,y)`. Deterministic paths can also compute the product; the witness shows
that layer marginals alone do not specify the joint predictor, not that
independent choices are incapable of parity in every parameter setting.

With fixed logits `[-u,u]`, the correlated predictor separates the four
signed-parity inputs with exact correct-class margin two.
`residual_interaction_margin` proves this, while `no_affine_interaction`
proves that an affine score in the original two features cannot strictly
separate those labels. An affine baseline supplied with `xy` in advance
would of course remove that separation. The labels and branches here are
an expressivity witness, not answer rules supplied to a stress model.

The tensor is a feasible learned coordinate point and can be optimized with
the convex objective of CD5. Its existence and margin do not prove that a
particular optimizer finds it, that it is the unique optimum, or that parity
generalization follows from an incomplete training split.

## Validation and next empirical question

The new namespace contains no unfinished proofs. Each theorem with numerical
or feasibility premises has concrete examples satisfying them. Source
correspondence includes the complete fixed/trainable distinction above.
Generated build and axiom evidence stays local in `.kanban/artifacts/167/`.

```sh
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexDepthAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
git diff --check
rg -n 'native_decide|^axiom |linter\..* false' src
```

The last command must produce no matches. No Python is needed. `INDEX.md`
is generated by the native tool and committed with these Lean sources.

The next empirical comparison is joint routing against independent routing,
with plain residual and phase-Fourier connections under the same finite
library. Include CD1's additive baseline; report library construction cost,
path count, parameter count, total compute and the original held-out stress
metrics. The current completed convex cohorts and the ongoing common-hull
suite are separate evidence; this theory adds no measured pass to them.
