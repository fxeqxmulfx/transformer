# Mean-field approximation of Argand AdamW training in width

Board task #245. Lean: `src/Transformer/Grokking/FiniteRecovery/MeanFieldWidth/`,
namespace `Transformer.Grokking.FiniteRecovery.MeanFieldWidth`, umbrella
`MeanFieldWidth.lean`; axiom script `scripts/MeanFieldWidthAxioms.lean`. Paths
without `src/` are relative to `src/Transformer/Grokking/FiniteRecovery/`.

## Outcome

1. **Transfer theorem (proved).** For every mean-field AdamW training (§1) the
   expectation of the maximum over updates $B\le t\le H$ of the supremum over
   $0<u<1$ of $(R^W_t(u)-\bar R_t(u))_+$ is at most an explicit
   $\varepsilon(W,B,H)$: `expected_residual_gap_le`
   (`MeanFieldWidth/Transfer.lean:28`). The maximum is inside the expectation.
2. **Handoff (proved).** The parent adds $\varepsilon$ (updates $B..H$) and
   $E[\mathrm{tail}]$ (updates after $H$) to its bound $\bar r$ on the limiting
   residual. Permanent recovery with probability above $q$ follows once
   $\max(\bar r,0)+\varepsilon+E[\mathrm{tail}]<(1-q)(1-\rho)(1/2-r)$:
   `permanent_of_width_transfer` (`MeanFieldWidth/Recovery.lean:83`).
3. **Width limit.** $\varepsilon\to0$ as $W\to\infty$ at a fixed horizon and a
   fixed floor (`widthError_vanishes`), at rate
   $O(G^{H+1}H\,W^{-1/2}+H^{1/2}W^{-1/4})$ after a hand optimization of the
   grid, with a per-update factor $G>1$.
4. **At $W=64$ the bound is vacuous.** At the division peak rate $G\ge56251$ per
   update, and $\varepsilon\ge L_\Gamma L_\chi L_G\,k\,C_\varphi(H+1)\,56251^{H+2}/4$,
   about $10^{194579}$ times the constants at $H=40960$ (`widthError_width64`).
   The factor $1/\epsilon$ in $G$ is intrinsic to AdamW
   (`first_direction_lipschitz`).
5. **Argand is not in the class.** Its gradient is not a function of a width
   unit's own fixed-size coordinates and finitely many width averages. Its
   certificate `marginalBound` is not of the form $F\circ\mathrm{mean}\,\psi$. For
   Argand the proposed limit is therefore not established; §4 isolates the
   missing estimates.
6. **Tail.** The Gronwall factor cannot reach the infinite tail: $G\ge19/16$ even
   at the floor rate. The needed stability estimate is a contraction of the
   coupled distance, with a law of large numbers uniform in time. Under it the
   tail allowance is explicit (`residual_le_tail`).
7. **Sorry audit.** `weak_error_clean`, `mean_field_limit` and
   `meanField_exponential_rate` are unproved, so unusable. None of their
   hypotheses describes AdamW training, and nothing in the chain uses them.
8. **Axioms.** All 116 declarations of the package depend on `propext`,
   `Classical.choice` and `Quot.sound` only. The global audit reads
   `87 sorry · 0 resting on a sorry · 0 extra axioms`; the package adds no
   `sorry`.

## 1. Training dynamics and the limit

### Particles, interaction, update

- A **particle** is one width unit $i\in\{1,\dots,N\}$, $N=W$, with a finite
  coordinate set $P$. Its state `AdamW.State P` holds the weights $w_i$, the
  first moment $m_i$ and the second moment $v_i$. AdamW acts coordinate by
  coordinate, so both buffers travel with the particle (`MeanFieldWidth/Defs.lean`).
- The particles **interact only through the mean field**
  $\Phi=\frac1N\sum_i\varphi(w_i)\in\mathbb R^K$, $K$ finite (`meanField`,
  `Defs.lean:52`; `Training.field`). The gradient of particle $i$ at update $t$
  is `A.grad t w_i Φ` (`Training`, `Defs.lean:45`).
- One update (`step`, `Defs.lean:38`) at rate $\eta_t$ = `A.rate t` and floor
  $\epsilon$ = `A.eps`:
  $m\leftarrow\frac9{10}m+\frac1{10}g$, $v\leftarrow\frac{95}{100}v+\frac5{100}g^2$,
  $w\leftarrow(1-\eta_t/10)\,w-\eta_t\,m\big/\big((1-0.9^{t+1})(\sqrt v/\sqrt{1-0.95^{t+1}}+\epsilon)\big)$.
  These are the equations of the existing `AdamW.run`
  (`AdamWTrajectory.lean:33`): both bias corrections, decoupled weight decay
  $0.1$ and the given schedule. `direction_literal` (`Defs.lean:34`) identifies
  the direction at the literal floor $10^{-8}$ with `AdamW.direction` by `rfl`.
- The finite system `system` (`Defs.lean:56`) starts from `init i` with zero
  moments. The limiting particle `flow` (`Defs.lean:65`) from $\theta$ uses the
  deterministic field $\bar\Phi_t=E_{x\sim\mu}\,\varphi(\bar w_t(x))$
  (`limitField`), where $\bar w_t(x)$ = `(flow A μ t x).weights`.

### Scaling in width

- In the mean-field parametrization particle $i$ receives its per-particle
  gradient divided by $N$ (`literalGrad`, `Literal.lean:62`). AdamW is invariant
  under rescaling the gradient together with the floor (`direction_scale`,
  `Literal.lean:25`). Hence `run_literal` (`Literal.lean:100`): the literal
  `AdamW.run` with floor $10^{-8}$, gradient clipping and decoupled decay on the
  flat parameters `Fin N × P` **is** the $N$-particle `system` with floor
  $\epsilon_N=N\cdot10^{-8}$, with moments divided by $N$ and $N^2$, while
  clipping does not act.
- Clipping does not act at squared gradient norm at most 1
  (`clip_of_sum_sq_le`). That holds for per-particle gradients bounded by $C$
  once $|P|C^2\le N$ (`sum_sq_literalGrad_le`). The squared norm of the literal
  gradient is $1/N$ times a width average, so clipping switches off as $N$ grows.
- The literal family thus has floor $N\cdot10^{-8}$ in mean-field units, and
  $|\mathrm{direction}|\le10|m|/\epsilon$ (`abs_direction_le_floor`,
  `Literal.lean:37`). If the literal floor $10^{-8}$ is kept while $N\to\infty$,
  the direction vanishes and the limit is pure weight decay. The nondegenerate
  limit holds the floor at its value for one width; for the division run,
  $W=64$ gives $\epsilon=64\cdot10^{-8}$.

### Shared randomness: the coupling

- **Initialization.** The $N$ initial draws are i.i.d. with law $\mu$ on
  $\mathbb R^P$ (`Measure.pi fun _ => μ`). Limiting particle $i$ starts from the
  same draw `init i` (synchronous coupling). The coupled distance `coupling`
  (`Defs.lean:82`) is the mean over $i$ of the $\ell^1$ distance between the two
  states in weights, first moments and square roots of second moments (`gap`).
- **Minibatches.** The minibatch of update $t$ enters only through `A.grad t`,
  which the finite system and the limit share. Every theorem holds for each
  fixed minibatch sequence. A random sequence independent of the
  initialization is covered by applying them to each sequence; $\bar R_t$ is
  then the quenched limit driven by the same sequence. Integration over the
  minibatch law is not stated in Lean.
- **Division protocol** (`experiments/argand/division-stock-persistence-protocol.json`):
  "stock Gaussian initialization from the u32 seed, production minibatches
  shuffled by the seed plus 10000". The Lean statements condition on the
  shuffle; the initialization law is the per-unit law $\mu$.

### Learning-rate schedule

The theorems take any schedule with $0\le\eta_t\le\eta\le10$ (`Regular.rate`,
`Regular.rate_le`). `Argand.wsdRate` (`ArgandNative.lean:20`) satisfies this
with $\eta=1/1000$ (`wsdRate_bounds`) and equals the floor $10^{-8}$ after its
budget (`wsdRate_eq_floor`). The division protocol uses the same WSD shape
with peak $0.003$, budget 20000 and horizon 40960. The $W=64$ evaluation below
uses that peak and the floor $10^{-8}$. The sum of its rates over the 40960
updates is $\sum_t\eta_t=53.40$ (arithmetic).

### What transfers from the empirical attention identity

`vectorField_empiricalProb_eq_saField`
(`src/Transformer/Perspective/Section3_EmpiricalMeasures.lean:117`) and
`meanFieldIPS_of_SA` (`:139`) state that for $n$ tokens on the sphere the
mean-field velocity at the empirical measure equals the particle attention
velocity exactly. There the particles are token positions.

- **Transfers:** the form "the finite system is the limiting dynamics evaluated
  at the empirical measure". In width this is `system_succ` (`Defs.lean:94`):
  particle $i$ takes the limiting step with the empirical field
  $\frac1N\sum_j\varphi(w_j)$ in place of $\bar\Phi_t$. It is what makes the
  synchronous coupling a coupling of one and the same dynamics.
- **Does not transfer:**
  - (a) In width there is no exact identity between the finite system and the
    limit. Their difference is the sampling error of the field;
    `integral_fluctuation_le` bounds it by $kC_\varphi/\sqrt N$ and `coupling_le`
    propagates it.
  - (b) The particles there are positions. The division task has five fixed
    positions, so there is no limit in positions to take.
  - (c) The identity concerns one attention layer with fixed parameters acting
    on tokens. In width the particles are parameters, and Argand's attention
    couples pairs of width indices (§4).

## 2. The error bound the recovery theorem consumes

### Residual and its mean-field counterpart

`ObservedConstraints.permanent_of_observed_drift` (`ObservedDrift.lean:58`)
asks, for every start $z$ and every update $t\ge B$, that
$\frac{d}{du}U(w_t+u\Delta_t)\le-(1-\rho)(U(w_t)-c)+\mathrm{excess}(z)$ for $0<u<1$,
with its `r` as the floor $c$. Equivalently $R^W_t(u)\le\mathrm{excess}(z)$ for
$R^W_t(u)=DU(w_t+u\Delta_t)[\Delta_t]+(1-\rho)(U(w_t)-c)$.

- **General form.** `residual A Γ χ z t u` is
  $\Gamma\big(\frac1N\sum_i\chi_u(w_{t,i},w_{t+1,i})\big)$ (`Residual.lean:44`).
  Its mean-field counterpart `limitResidual` is
  $\bar R_t(u)=\Gamma\big(E_{x\sim\mu}\,\chi_u(\bar w_t(x),\bar w_{t+1}(x))\big)$
  (`Residual.lean:55`).
- **Certificates $U=F\circ\mathrm{mean}\,\psi$.** `certificate_residual_of_init`
  (`Certificate.lean:107`) proves by the chain rule that the recovery residual
  equals `residual` for $\Gamma$ = `certificateMap F ρ c` and $\chi$ =
  `certificateFeatures ψ D`. It needs $|\mathrm{init}_{i,p}|\le C_0$ for every
  $i,p$, and takes $D=\eta(\max(C_0,120)/10+12)$. The clamp of the displacement
  at $D$ is then inactive, since the displacement is at most $D$ in every
  coordinate (`abs_displacement_le`, `Bounds.lean:84`).
- **The counterpart $\bar R_t(u)$.** With $\bar\Delta_t=\bar w_{t+1}-\bar w_t$,
  clamped at $D$,
  $\bar R_t(u)=DF\big(E\psi(\bar w_t+u\bar\Delta_t)\big)\big[E\,D\psi(\bar w_t+u\bar\Delta_t)\,\bar\Delta_t\big]+(1-\rho)\big(F(E\psi(\bar w_t))-c\big)$.

### Hypotheses: `Regular` (`Residual.lean:25`)

The structure collects the following conditions.

- **Optimizer:** $\epsilon>0$ and $0\le\eta_t\le\eta\le10$.
- **Gradient:** $\ell^1$-Lipschitz with constant $L_G$ in (own weights, field).
- **Field $\varphi$:** $\ell^1$-Lipschitz with constant $L_\varphi$, and bounded by $C_\varphi$.
- **Map $\Gamma$:** one-sided Lipschitz with constant $L_\Gamma$ on the box $|a_j|\le C_\chi$.
- **Features $\chi_u$:** Lipschitz with constant $L_\chi$ in both arguments,
  Lipschitz with constant $M_\chi$ in $u$, and bounded by $C_\chi$.

For a certificate these become conditions on $F$ and $\psi$: $\psi$ and
$D\psi$ bounded and Lipschitz, $DF$ Lipschitz on the box. The Lean chain takes
`Regular` as a hypothesis. It does not derive `Regular` from conditions on $F$
and $\psi$.

### Theorem `expected_residual_gap_le` (`Transfer.lean:28`)

```lean
theorem expected_residual_gap_le [Fintype P] [Fintype K] [Fintype J] (A : Training P K)
    (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ}
    {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) (B H : ℕ) {n : ℕ} (hn : 0 < n) :
    ∫⁻ z, ⨆ t ∈ Icc B H, ⨆ u ∈ Set.Ioo (0:ℝ) 1,
        ENNReal.ofReal (max (residual A Γ χ z t u - limitResidual A μ Γ χ t u) 0)
        ∂(Measure.pi fun _ : Fin N => μ) ≤
      ENNReal.ofReal (widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n)
```

The lower integral needs no measurability of the supremum. The proof bounds the
supremum pathwise by a measurable majorant (`residual_sub_le_majorant`,
`ErrorBound.lean:64`) and integrates that majorant (`integral_majorant_le`,
`ErrorBound.lean:93`).

### The explicit error: `widthError` (`ErrorBound.lean:36`)

$$\varepsilon(W,B,H)=L_\Gamma\Big(2L_\chi\,G^{H+1}F\,(H+1)\,\frac{kC_\varphi}{\sqrt W}+(H+1-B)(n+1)\,\frac{jC_\chi}{\sqrt W}+\frac{2M_\chi}{n}\Big),$$

where:

- $G=(1+12\eta/\epsilon)(1+L_G(1+L_\varphi))$ (`growth`),
- $F=(1+12\eta/\epsilon)L_G$ (`forcing`),
- $k=|K|$ and $j=|J|$,
- $n\ge1$ is free.

The three terms come from three steps of the proof:

- **Propagation of chaos.** `coupling_le` (`Coupling.lean:144`) bounds the
  coupled distance pathwise by $G^tF\sum_{s<t}\mathrm{fluctuation}_s$.
  `integral_fluctuation_le` (`Sampling.lean:163`) bounds each expected
  fluctuation by $kC_\varphi/\sqrt W$. The factor $1+12\eta/\epsilon$ is the
  Lipschitz constant of one AdamW update (`direction_lipschitz` and
  `coordinate_step`, `Basic.lean:78,134`). It holds in the moment cone
  $14m^2\le19v$, which AdamW preserves (`cone_system`).
- **Sampling the residual features** on a grid of $n+1$ fractions at each
  update $B..H$.
- **Interpolation** between grid fractions (`segmentError_le_grid`, `Grid.lean:66`).

**Dependence on width and horizon.**

- In width the bound falls like $W^{-1/2}$ at fixed $n$.
- Choosing $n\approx\big(2M_\chi\sqrt W/((H+1-B)\,jC_\chi)\big)^{1/2}$ gives
  $O\big(G^{H+1}H\,W^{-1/2}+(H\,jC_\chi M_\chi)^{1/2}\,W^{-1/4}\big)$. This
  optimization over $n$ is by hand, not in Lean.
- In the horizon the bound grows like $G^{H+1}$, and $G\ge1+12\eta/\epsilon>1$
  whenever $\eta>0$.

### $W\to\infty$: `widthError_vanishes` (`Transfer.lean:51`)

Fix $\epsilon$, $\eta$, the constants, $B$ and $H$. For every $\delta>0$ there
are $n>0$ and $N_0$ such that $\varepsilon<\delta$ for all $N\ge N_0$; the proof
chooses $n$ first, then $N$.

For nonzero constants the bound is small only when $\sqrt W\gg G^{H+1}H$, that is
$\log W\gtrsim2(H+1)\log G$. So it does not vanish uniformly in $H$, and it
needs the floor held fixed (for the literal floor see §1).

### $W=64$ (`MeanFieldWidth/Obstruction.lean`)

By `run_literal` the division run of width 64 has floor $\epsilon=64\cdot10^{-8}$.

| Lean | Statement | Meaning |
|---|---|---|
| `growth_width64_peak` (`:23`) | $G\ge56251$ at $\eta=3/1000$ | one update at the peak multiplies the bound by at least 56251 |
| `growth_width64_floor` (`:31`) | $G\ge19/16$ at $\eta=10^{-8}$ | even the floor rate expands |
| `forcing_width64_peak` (`:38`) | $F=56251\,L_G$ at $\eta=3/1000$ | |
| `widthError_width64_ge` (`:43`) | $\varepsilon\ge2L_\Gamma L_\chi L_G\,kC_\varphi(H+1)\,56251^{H+2}/\sqrt N$ | every width, at this floor and peak |
| `widthError_width64` (`:65`) | $\varepsilon\ge L_\Gamma L_\chi L_G\,kC_\varphi(H+1)\,56251^{H+2}/4$ at $N=64$ | |
| `first_direction` (`:78`) | from zero moments the first direction is $g/(\lvert g\rvert+\epsilon)$ | a sign function smoothed at scale $\epsilon$ |
| `first_direction_lipschitz` (`:89`) | every Lipschitz constant of the first direction in $g$ is at least $1/(2\epsilon)$ | 781250 at $\epsilon=64\cdot10^{-8}$ |

The following numbers are arithmetic, not Lean theorems:

- $\log_{10}56251=4.75013$, so at $H=40960$,
  $(H+1)\,56251^{H+2}/4\approx10^{194578.85}$.
- Already at $H=0$ the bound is below 1 only if
  $L_\Gamma L_\chi L_G\,kC_\varphi<4/56251^2\approx1.26\cdot10^{-9}$.
- The 20960 floor-rate updates of the division run alone carry a Gronwall
  factor of at least $(19/16)^{20960}\approx10^{1564}$.
- With the ARGAND.md peak $10^{-3}$ instead, $G\ge18751$. The verdict is the same.

**Vacuity is not only an artifact of the proof.**

- The first AdamW update moves a coordinate by $\eta\,g/(|g|+\epsilon)$.
- Its Lipschitz constant in the gradient is at least $\eta/(2\epsilon)=2343.75$
  at the peak rate: `first_direction_lipschitz` times $\eta$.
- The width-64 fluctuation of a field bounded by $C_\varphi$ is of order
  $C_\varphi/8$, far above $\epsilon$.
- Any argument that propagates the width error through per-update Lipschitz
  constants therefore pays this factor already at the first update.

### The training tail (`MeanFieldWidth/Recovery.lean`)

- `le_of_contraction` (`:30`): if $D_{t+1}\le aD_t+b\zeta_t$ with $0\le a<1$ and
  $\zeta_t\le Z$ for $t\ge H$, then $D_t\le D_H+bZ/(1-a)$ for all $t\ge H$.
- `residual_le_tail` (`:53`): suppose that for one run $z$, at every update
  after $H$:
  1. the coupled distance contracts:
     $\mathrm{coupling}_{t+1}\le a\cdot\mathrm{coupling}_t+b\cdot\mathrm{fluctuation}_t$
     with $a<1$ (`hcontr`);
  2. the fluctuations are at most $Z$ (`hZ`) and the segment errors at most $S$
     (`hS`), a law of large numbers uniform in time;
  3. the limiting residual is at most $r$ (`hlim`).

  Then $R^W_t(u)\le r+L_\Gamma\big(2L_\chi(\mathrm{coupling}_H+bZ/(1-a))+S\big)$
  for all $t>H$ and $0<u<1$.
- **This is the additional stability estimate.** The Gronwall recursion
  `coupling_succ_le` has coefficient $G>1$ at every positive rate, and $G\ge19/16$
  at the division floor, so it never yields condition 1.

### Handoff to the parent

- **Excess.** `drift_excess` (`Transfer.lean:122`) takes a bound
  $\bar r\ge\bar R_t(u)$ for $B\le t\le H$, and a measurable, integrable,
  nonnegative `tail` that dominates the residual after $H$. The excess
  $\max(\bar r,0)+\mathrm{majorant}(z)+\mathrm{tail}(z)$ is then measurable,
  integrable and nonnegative. It dominates $R^W_t(u)$ for every $t\ge B$, and
  its mean is at most $\max(\bar r,0)+\varepsilon(W,B,H)+E[\mathrm{tail}]$.
- **Recovery.** `permanent_of_width_transfer` (`Recovery.lean:83`) feeds this
  excess into `permanent_of_observed_drift`:
  $$\max(\bar r,0)+\varepsilon(W,B,H)+E[\mathrm{tail}]<(1-q)(1-\rho)(1/2-r)\ \Longrightarrow\ \exists G,\ \mu^{\otimes W}(G)>q\ \text{and every}\ z\in G\ \text{is eventually strict}.$$
- **Its `slope` hypothesis** asks that the finite derivative along the update,
  plus $(1-\rho)(U(w_t)-r)$, be at most `residual`. For
  $U=F\circ\mathrm{mean}\,\psi$ on bounded initializations this holds with
  equality: `certificate_residual_of_init`.

**The error term the parent adds to its bound on the limiting drift is
$\varepsilon(W,B,H)$ = `widthError` for the updates $B..H$, plus
$E[\mathrm{tail}]$ after $H$.**

## 3. Audit of the sorry declarations

All three are `sorry` in `INDEX.md`, so by the repository's rule nothing may rest
on them, and nothing does. Independently of that, none of their hypotheses holds
for the training process.

### `Homogenized.weak_error_clean`

`src/Transformer/Homogenized/WeakError.lean:60`; arXiv:2604.01978v1 §2.3.2.

- **Would supply:** a weak error in the step size $\eta$ between a random layer
  chain and a homogenized SDE, for $C^4$ observables of $n$ normalized tokens:
  $\sup_{t\le\eta L}\lvert E\varphi(X(t))-E\varphi(X^\eta(t))\rvert\le Ce^{C\eta L}\eta(\eta L+1)\max(1,\alpha)$.
- **Against training:**
  - `IsRandomChain` (`src/Transformer/Homogenized/RandomChain.lean:144`) draws
    the head parameters independently, with law $\rho^*$, at every layer. These
    are random weights, not trained ones.
  - The small parameter is the step $\eta$ per layer; the width never enters.
  - The state is the token configuration, not the weights.
  - A weak error compares one expectation at a time. It cannot put the maximum
    inside the expectation.

### `Kinetic.mean_field_limit`

`src/Transformer/Kinetic/MeanField.lean:147`; arXiv:2605.09213v1 th:mf-gpt(i).

- **Would supply:** existence and uniqueness of a weak solution of the layered
  McKean–Vlasov equation, and weak-* convergence to it of the empirical measure
  of $N$ token angles under the ALiBi flow, as the number of positions
  $N\to\infty$.
- **Against training:**
  - `IsGPTFlow` is a fixed-parameter ODE on token angles; nothing is trained.
  - The limit is in positions, and the division task has five.
  - The result is qualitative. Without a rate it gives no
    $\varepsilon(W,B,H)$, and nothing at $W=64$.

### `MeanField.meanField_exponential_rate`

`src/Transformer/MeanField/GlobalRate.lean:56`; arXiv:2512.01868v4 §4 thm:mfclust.

- **Would supply:** exponential clustering of token representations. For
  $d\ge2$, an $L^2$ density and $R_0>0$, and small $\lvert\beta\rvert$, the
  unnormalized self-attention continuity equation satisfies
  $W_2(\mu_t,\delta_{x_\infty})\le C_0e^{-t/100}$ for $t\ge T_0$.
- **Against training:**
  - It is the only one of the three shaped like a tail estimate, but its time is
    the forward attention dynamics of tokens, not optimizer steps.
  - `usaContinuityEquation` does not hold for AdamW weights.
  - Clustering of tokens does not give `hcontr`, the contraction of the coupled
    training distance.

**What the chain uses instead** (all proved): Mathlib for product measures,
Bochner and lower integrals and $L^2$; from the repository,
`permanent_of_observed_drift`, `AdamW.run`, `AdamW.moment_cone`,
`firstBias_lower` and the Kernite `clip`.

## 4. Where the proposed limit fails, and the missing estimates

1. **Argand is not a mean-field system in width.**
   - Its parameters (`ArgandParameters.lean:24`) are an embedding $A\times c$,
     shared LeRoPE adjustments, and per layer: $W_{Q,K,V}$ ($c\times Hd$),
     $W_O$ ($Hd\times c$), $W_{g,u}$ ($c\times4c$), $W_F$ ($4c\times c$), a
     score multiplier and residual gates. ARGAND.md's example has $c=Hd=64$ and
     $H=4$; the division run has width $c=64$.
   - A stream index carries $16c+4$ complex entries per layer. The coordinate
     set of a "width unit" therefore grows with the width.
   - The logits are sums over pairs of width indices, not averages of per-unit
     features, and the heads and LeRoPE frequencies are not exchangeable.
   - **Missing:** a parametrization in which Argand's gradient is
     `A.grad t w_i Φ`, with $P$ independent of $W$. That is a mean-field or
     μP-type parametrization of the matrices: a limit of matrices, not of
     particles. Without it the transfer theorem covers mean-field models, not
     Argand.
2. **The existing certificate is outside the certificate class.**
   - `marginalBound` (`ArgandObservedJacobian.lean:23`) is
     $\sum_{x\in S}(1-p_w(y_x\mid x))+|J|\sum_x(1-p_w(g_x\mid x))$, a function of
     the logits.
   - It has the form $F\circ\mathrm{mean}\,\psi$, with $F$ the sums of
     $1-\mathrm{softmax}$ and $\psi$ the per-unit logit contributions, exactly
     when the logits are width averages. By item 1 this fails for Argand.
3. **Unbounded initialization.**
   - `certificate_residual_of_init` needs $|\mathrm{init}_{i,p}|\le C_0$ for every
     $i,p$. The division run draws each part from $\mathcal N(0,0.02^2)$, which
     is unbounded.
   - `permanent_of_observed_drift` asks for its slope bound at every start.
   - **Missing:** either an almost-everywhere version of the recovery theorem,
     or its application to the initialization law conditioned on the box. The
     conditioning moves $\mu$ by $P(\max|\mathrm{init}|>C_0)$ in total variation.
4. **No Lipschitz stability estimate at $W=64$.**
   - The upper bound $12\eta/\epsilon$ per update is within a factor 24 of the
     first update's Lipschitz constant, which is at least
     $\eta/(2\epsilon)=2343.75$ at the peak rate (`first_direction_lipschitz`).
     Later updates can have smaller constants only where $\sqrt v$ stays away
     from 0, which is the next point.
   - A lower bound $\sqrt v\ge\sigma$ on the second moments along both
     trajectories would replace $1/\epsilon$ by roughly $1/\sigma$. This is a
     hand estimate, not proved here.
   - Even then the factor stays huge. Over the division schedule
     $\prod_t(1+12\eta_t/\sigma)$ is $10^{273.6}$ at $\sigma=1$, $10^{27.8}$ at
     $\sigma=10$ and $10^{2388.6}$ at $\sigma=0.1$ (arithmetic), before the factor
     $1+L_G(1+L_\varphi)$ per update.
5. **The precise missing estimate.**
   - **(a) Contraction along the whole run:**
     $\mathrm{coupling}_{t+1}\le a\cdot\mathrm{coupling}_t+b\cdot\mathrm{fluctuation}_t$
     with $a<1$, from the start, or from an update $B_0$ at which $G^{B_0}$ is
     still moderate.
   - **(b) Uniform-in-time concentration** of the fluctuations and the segment
     errors: $E\sup_t$, not $\sup_tE$.
   - Given (a) and (b), `residual_le_tail` with $B_0$ in place of $H$ bounds the
     residual gap at every later update by
     $L_\Gamma\big(2L_\chi(\mathrm{coupling}_{B_0}+bZ/(1-a))+S\big)$, with no factor
     $G^H$.
   - Such a contraction must come from dissipativity of the trained dynamics:
     the decay factor $1-\eta_t/10$, and monotonicity of the preconditioned
     gradient near the solution. Lipschitz bounds cannot give it, and nothing in
     the repository proves it for AdamW.

## 5. Every new declaration and its example

Each example applies its theorem with every hypothesis discharged. The test
objects are:

- `identityTraining` (`Coupling.lean:170`): one coordinate, gradient equal to
  the weight, zero field, floor 1, rate 1;
- `identityRegular` (`Residual.lean:127`), its `Regular` witness;
- `quietTraining` (`Literal.lean:141`): zero gradient, the literal floor of
  width 2, rate $10^{-3}$;
- zero states.

"none" in the example column means the theorem has no hypotheses.

| Module | Theorem (line) | Statement | Example |
|---|---|---|---|
| Defs | `direction_literal` (34), `system_succ` (94), `flow_succ` (98) | the literal direction; one update of the system and of the limit (`rfl`) | none |
| Basic | `cone_step` (19), `second_nonneg_of_cone` (26) | the step preserves the cone $14m^2\le19v$; the cone gives $v\ge0$ | `Basic.lean:188-189` |
| Basic | `sqrt_second_lipschitz` (31) | $\lvert\sqrt{0.95v+0.05g^2}-\sqrt{0.95v'+0.05g'^2}\rvert\le\lvert\sqrt v-\sqrt{v'}\rvert+\lvert g-g'\rvert/4$ | `Basic.lean:190` |
| Basic | `direction_lipschitz` (78) | in the cone, $\lvert d(m,v)-d(m',v')\rvert\le(10\lvert m-m'\rvert+12\lvert\sqrt v-\sqrt{v'}\rvert)/\epsilon$ | `Basic.lean:191` |
| Basic | `coordinate_step` (134), `gap_step` (179) | one update is $(1+12\eta/\epsilon)$-Lipschitz in (state, gradient) | `Basic.lean:192-194` |
| Coupling | `cone_system` (19), `cone_flow` (25), `l1_nonneg` (31), `l1_sub_le` (34), `gap_nonneg` (39), `l1_weights_le_gap` (42), `coupling_zero` (69) | both dynamics stay in the cone; norm facts; zero coupling at the start | none |
| Coupling | `meanField_sub_le` (50) | an $L$-Lipschitz field gives an $L$-Lipschitz mean field | Coupling Examples |
| Coupling | `coupling_succ_le` (75), `gronwall` (123), `coupling_le` (144) | one-step recursion, discrete Gronwall, pathwise propagation of chaos | Coupling Examples |
| Coupling | `identityTraining_grad` (172), `identityTraining_field` (178) | Lipschitz facts of the test model | used by the examples |
| Sampling | `integral_abs_le_sqrt` (21), `integral_abs_empirical_sub_le` (36), `integrable_abs_empirical_sub` (99) | $E\lvert Y\rvert\le\sqrt{EY^2}$; $E\lvert\text{empirical mean}-\text{mean}\rvert\le C/\sqrt N$ and its integrability | `Sampling.lean:179-183` |
| Sampling | `continuous_of_l1` (118), `continuous_direction` (132), `continuous_flow` (143) | continuity of Lipschitz maps, the direction, the limiting particle | `Sampling.lean:184-186` |
| Sampling | `integral_fluctuation_le` (163) | $E\,\mathrm{fluctuation}_t\le kC_\varphi/\sqrt N$ | `Sampling.lean:187` |
| Residual | `l1_meanField_sub_le` (66) | mean of differences | none |
| Residual | `abs_meanField_le` (79), `abs_limitFeatures_le` (86), `residual_sub_le` (96) | bounded features; $R-\bar R\le L_\Gamma(L_\chi(\mathrm{coupling}_t+\mathrm{coupling}_{t+1})+\mathrm{segmentError})$ | Residual Examples |
| Residual | `identityRegular` (127) | `Regular` holds for the test model | used by the examples |
| Grid | `continuous_of_l1₂` (18), `Regular.continuous_grad` (38), `Regular.Mχ_nonneg` (45), `integrable_features` (51), `segmentError_le_grid` (66) | continuity and integrability of the features; reduction of $\sup_u$ to $n+1$ grid points plus $2M_\chi/n$ | Grid Examples |
| ErrorBound | `integral_l1_meanField_sub_le` (20), `Regular.gronwall_nonneg` (54), `residual_sub_le_majorant` (64), `majorant_nonneg` (85), `integral_majorant_le` (93) | the vector law of large numbers; the pathwise majorant; $E\,\mathrm{majorant}\le\varepsilon$ | Transfer Examples |
| Transfer | `expected_residual_gap_le` (28), `widthError_vanishes` (51) | the transfer theorem; vanishing in width | Transfer Examples |
| Transfer | `continuous_l1_meanField_sub` (83), `continuous_fluctuation` (99), `continuous_segmentError` (106), `measurable_majorant` (110), `drift_excess` (122) | measurability; the excess for the recovery theorem | Transfer Examples |
| Bounds | `abs_direction_le` (20), `abs_step_weights_le` (45), `abs_step_sub_le` (62), `abs_weights_le` (74), `abs_displacement_le` (84) | $\lvert d\rvert\le12$ in the cone; the box $\max(\lvert\mathrm{init}\rvert,120)$ is invariant; displacement $\le\eta_t(\max(\lvert\mathrm{init}\rvert,120)/10+12)$ | Bounds Examples |
| Bounds | `step_weights` (40) | the weight update (`rfl`) | none |
| Certificate | `meanField_eq_smul_sum` (38), `hasFDerivAt_meanField` (43), `fderiv_certificate` (54), `clampDisp_of_abs_le` (64) | chain rule for $F\circ\mathrm{mean}\,\psi$; an inactive clamp | Certificate Examples |
| Certificate | `certificate_residual` (71), `certificate_residual_system` (89), `certificate_residual_of_init` (107) | the recovery residual of $F\circ\mathrm{mean}\,\psi$ is `residual` | Certificate Examples |
| Recovery | `le_of_contraction` (30), `coupling_nonneg` (47), `residual_le_tail` (53), `permanent_of_width_transfer` (83) | the tail estimate; recovery from the width transfer | Recovery Examples |
| Recovery | `identity_system_zero` (121), `identity_flow_zero` (130), `identity_coupling_zero` (139) | the test model stays at 0 | used by the examples |
| Literal | `direction_scale` (25), `abs_direction_le_floor` (37), `clip_of_sum_sq_le` (49), `sum_sq_literalGrad_le` (66), `run_literal` (100) | scale invariance; the degenerate literal limit; inactive clipping; literal AdamW = the system | Literal Examples |
| Literal | `run_succ_first` (81), `run_succ_second` (86), `run_succ_weights` (91) | one literal update (`rfl`) | none |
| Obstruction | `growth_width64_peak` (23), `growth_width64_floor` (31), `widthError_width64_ge` (43), `widthError_width64` (65), `first_direction_lipschitz` (89) | the $W=64$ evaluation; the $1/(2\epsilon)$ lower bound | Obstruction Examples |
| Obstruction | `forcing_width64_peak` (38), `first_direction` (78) | closed forms | none |

The definitions are `l1`, `denominator`, `direction`, `step`, `Training`,
`meanField`, `system`, `flow`, `limitField`, `gap`, `coupling`, `fluctuation`,
`Cone`, `growth`, `forcing`, `Regular`, `residual`, `limitFeatures`,
`limitResidual`, `segmentError`, `widthError`, `majorant`, `clampDisp`,
`certificateFeatures`, `certificateMap`, `weightsAt`, `flatten`, `literalGrad`,
`identityTraining` and `quietTraining`. None of them ignores an argument.

## 6. Checks and how to reproduce them

Run from the repository root with `LAKE_ARTIFACT_CACHE=false`:

| Command | Result |
|---|---|
| `lake build` | `Build completed successfully (30570 jobs).`, no warning in the package |
| `lake env lean scripts/MeanFieldWidthAxioms.lean` | 116 lines, each `depends on axioms: [propext, Classical.choice, Quot.sound]` |
| `lake env lean scripts/Axioms.lean` | `87 sorry · 0 resting on a sorry · 0 extra axioms` |
| `cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml -- --check` | `INDEX.md: 6103 modules, 32916 theorems, 87 sorry, 0 vacuous, 0 placeholder` |
| `grep -rn 'native_decide\|^axiom \|linter\..* false' src` | empty |
