# Landscape scheme: almost every gradient trajectory ends at a global optimum

Started 2026-10-18 at the user's request. Status: active. Work solo and follow
AGENTS.md. Only Lean results without `sorry` count as proved: check
`dt show` or the `INDEX.md` status before you use one. Literature, hand
arguments and numerics are hypotheses until they are proved in Lean.

## Objective and constraints

The goal is one reusable proof pattern for gradient descent and for gradient
flows. It has three inputs and a bridge:

- **(A) Avoidance:** from almost every start, the trajectory converges to no
  strict saddle.
- **(B) Convergence:** every trajectory converges. This follows from the
  Łojasiewicz inequality at the critical points.
- **(C) Landscape:** every critical point is a global optimum or a strict
  saddle.
- **(D) Bridge:** from a manuscript's model to the energy that (A)–(C) speak
  about.

From these the pattern concludes that almost every trajectory converges to a
global optimum. It is then applied to the phase model of grokking
(`Transformer.PhaseModel`) and to the particle models of arXiv:2312.10794v5
(`Transformer.Perspective`).

Constraints:

- **No conditional results on open problems.** Do not add a theorem whose
  hypothesis is an open problem: not `o:strictsaddle`, and not an unproved
  landscape claim.
- **Łojasiewicz enters only as a hypothesis.** Łojasiewicz's theorem
  (lojasiewicz1963propriete) is proved in the literature, not in Lean. It may
  enter only as an explicit hypothesis, as `hloj` does in
  `SaddleAvoidance.ae_exists_tendsto_isLocalMin`.
- **Never weaken a paper's statement under its name.** Partial regimes get
  their own names, and the paper's theorem stays `sorry`.
- **Claim a sorry before working on it.** The board `.kanban/` was absent at
  the root on 2026-10-18. Ask the user before running `agent-kanban init`.
  New theorems that close no `sorry` need no claim.

## Proved on 2026-10-18

| Layer | Lean | Status |
| --- | --- | --- |
| A, gradient descent | `SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle`; global form `measure_tendsto_symmetricUnstable_eq_zero` | proved |
| B, gradient descent | `SaddleAvoidance.exists_tendsto_of_isLojasiewiczAt`, which converges from a Łojasiewicz limit point taken modulo translations `Λ`; `isLojasiewiczAt_of_injective` | proved |
| A + B, gradient descent | `SaddleAvoidance.ae_exists_tendsto_isLocalMin`, with hypotheses `hcrit`, `hloj` and bounded orbits | proved |
| B, flows | `Normalization.lojasiewicz_modulated_of_local_gradient_inequality`, which needs a compact `K`, an analytic `E` and the local inequality at every critical point; `analytic_gradient_inequality_of_nondegenerate` | proved |
| B, degenerate analytic case | `Normalization.lojasiewicz_modulated` | sorry |
| Flows | `GlobalFlow.IsFlowField.contDiff_flow`, `hasFDerivAt_flow` (the variational equation), `flow_reverse` | proved |
| C, phase model | `PhaseModel.exists_sub_characterPhase_mem_periods_of_isLocalMin` and `isStrictSaddle_of_divisionEnergy_lt` for `V < n²/8` | proved |
| C, `𝖤_0` on spheres | `Perspective.yury_lemma` | proved |
| p:beta0 | `Perspective.almost_sure_consensus_beta0`, via an explicit inequality for `𝖤_0` and a cone argument at the saddles | proved |

## Milestones

### M0 — Sources and board

| Item | Deliverable | Status |
| --- | --- | --- |
| M0.1 | Fetch with `./make.py papers` once this file is committed: arXiv:2507.22857 (Polyanskiy, Rigollet, Yao, *Synchronization of mean-field models on the circle*) and arXiv:2405.18273 (Criscitiello, Rebjock, McRae, Boumal, *Synchronization on circles and spheres with nonlinear interactions*). For each paper, record which `β` and `d` it shows to satisfy "every non-maximal critical point of `𝖤_β` is a strict saddle", and how. | todo; unread, so a hypothesis |
| M0.2 | Ask the user about the board before claiming `strict_saddle` or any other Perspective `sorry`. | todo |

arXiv:2512.01868v4 (lines 254–256) reports that arXiv:2507.22857 proves
clustering on the circle for every `β > -0.16`. arXiv:2312.10794v5 (§7,
after `o:strictsaddle`) answers the question only for `β ≤ 1` and for
`β ≥ n²/π²`. Both are hypotheses until M0.1 reads the source.

### M1 — Layer A for flows (general, `Transformer.SaddleAvoidance`)

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M1.1 | **Linear flow.** For a constant symmetric `A`, `J τ = exp (τ A)` solves `J' = A J` with `J 0 = id`; `exp A` is symmetric; an eigenvalue `μ` of `A` gives the eigenvalue `e^μ` of `exp A`. Mathlib: `hasDerivAt_exp_smul_const`, `IsSelfAdjoint.exp`. | — |
| M1.2 | **Time-one map.** `Φ = flow (-∇f) 0 1` for `f` of class `C²` with `∇f` bounded and Lipschitz, which is `IsFlowField`. `Φ` is `C¹` (`contDiff_flow`) with an invertible derivative (backward flow, `flow_reverse`). At a critical point, `Φ z = z` and `DΦ z = exp (-∇²f z)`, obtained by feeding `J τ = exp (-τ ∇²f z)` to `hasFDerivAt_flow`. A strict saddle is therefore `IsSymmetricUnstable Φ z`. | M1.1 |
| M1.3 | **Avoidance for flows.** `ae_not_tendsto_flow_of_isStrictSaddle`: for `ν`-almost every `x`, `t ↦ flow (-∇f) 0 t x` converges to no strict saddle, because integer times are iterates of `Φ`. Ascent is the same statement for `-f`. | M1.2 |
| M1.4 | **Periodic energies.** `lipschitzWith_gradient_of_periodic`: if `f` is `C²` and invariant under a lattice spanning `E`, then `∇f` is bounded and globally Lipschitz. This supplies the hypotheses of M1.2 and of the descent theorems for `𝖤_β` on angles and for `V_full`. | — |
| M1.5 | **Symmetrizable derivatives.** `IsSymmetrizableUnstable`: the derivative is conjugate to a symmetric map by a linear isomorphism. Extend `measure_tendsto_symmetricUnstable_eq_zero` to it, and M1.2 to `ẋ = -G(x) ∇f(x)` with `G` symmetric positive definite and `C¹`. SA and the sphere bridge M5.3 need it. | M1.3 |
| M1.6 | **Optional, for networks.** Gradient descent with a compact forward-invariant sublevel set in place of a global Lipschitz constant. | — |

### M2 — Layer B, convergence

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M2.1 | **Łojasiewicz under a symmetry.** `isLojasiewiczAt_of_invariant`: if `f` is invariant under translations by a subspace `W` and its Hessian at the critical point `z` is injective on `Wᗮ`, then `IsLojasiewiczAt f z (1/2)`. Proof: restrict to `z + Wᗮ` and apply `isLojasiewiczAt_of_injective`. It covers the synchronized states of `𝖤_β`, where `W = ℝ𝟙`. | — |
| M2.2 | **Convergence for flows.** The flow analogue of `exists_tendsto_of_isLojasiewiczAt`, with translations `Λ`: a limit point taken modulo `Λ` at which the inequality holds is the limit. First check what the proof of `lojasiewicz_modulated_of_local_gradient_inequality` already gives; reuse it rather than duplicate it. | — |
| M2.3 | **Glue to Normalization.** Convert the form `\|E y - E z\| ^ α ≤ k ‖∇E y‖` used by the `hlocal` hypothesis of `lojasiewicz_modulated_of_local_gradient_inequality` into `IsLojasiewiczAt`, so the hard core plugs in once proved. | — |
| M2.4 | **Hard core, not this plan's work.** Łojasiewicz at degenerate critical points of analytic functions: the remaining input of `Normalization.lojasiewicz_modulated`. The survey proves convergence of every trajectory by Łojasiewicz's theorem. For `β = 0` the repository replaced it by an explicit inequality (`AppendixA_Lojasiewicz`); for `β > 0` no replacement is known. So `beta_tiny`, `beta_tiny_circle`, `beta_interval` and `boumal_clustering` cannot close before M2.4. | Normalization |

### M3 — Layer C, landscape

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M3.1 | **`Perspective.strict_saddle`.** Formalize the landscape lemma of the paper from M0.1 that answers `o:strictsaddle`. It is pure landscape and needs neither A nor B: the only Perspective `sorry` that can close without M2.4. If the paper covers only some `β`, prove those regimes under new names. | M0.1 |
| M3.2 | **The regimes `β ≤ 1` and `β ≥ n²/π²` on the torus.** For `β ≤ 1` use arXiv:2405.18273. For `β ≥ n²/π²` use Appendix B, where `taylor2_inequality`, `taylor3_inequality` and `τ_β_star_spec` are in place. These are new theorems under their own names. | M0.1 |
| M3.3 | **`d ≥ 3`.** The non-synchronized critical points of `𝖤_β` on `(𝕊^{d-1})^n` are strict saddles: move every particle in one common direction (arXiv:2405.18273; `dr1_skew_inequality`, `claim_yury`). | M0.1 |
| M3.4a | **Characters are nondegenerate.** The Hessian of `V_full` at a character is injective, since the rows `e_x - e_y - e_{x-y}` span `ℝⁿ`. This gives Łojasiewicz with exponent `1/2` at the characters. | — |
| M3.4b | **Research: `V_full` above `n²/8`.** First a lab search for local minimizers with `V ≥ n²/8`, `n ≤ 30`; a hypothesis until proved. | lab |
| M3.4c | **Research: partial tables `V_S`.** The analogy is Kuramoto on Erdős–Rényi graphs just above connectivity (abdalla2022expander, cited in arXiv:2312.10794v5 §6). A comment in that survey's TeX source (line 908) says the reference proves only that a convergent trajectory ends synchronized. So this is an analogy and a hypothesis. | — |
| M3.4d | **Finite abelian groups.** Generalize `Fourier` and `Landscape` from `ℤ/n`. | — |

### M4 — The scheme theorems (general)

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M4.1 | **Gradient descent.** `ae_exists_tendsto_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, then almost every orbit converges to a global minimizer. Periodic corollary: when `Λ` is a lattice spanning `E`, every orbit has a limit point modulo `Λ`, so that hypothesis is free. | M3.4a for the example |
| M4.2 | **Gradient flows.** The same statement for gradient flows. | M1.3, M2.2 |
| M4.3 | **`𝖤_β` on angles (USA).** In the regimes of M3.2, almost every start converges to a synchronized state, given `hloj` at the non-synchronized critical points. M2.1 covers the synchronized ones. | M3.2, M4.2, M1.4, M2.1 |

### M5 — Bridges (D)

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M5.1 | **Angles and the circle.** `θ ↦ (cos θ_i, sin θ_i)` carries `angularUSA` to USA on `SphereTuple 2 n`. `UniformTuple 2 n P` is the image of normalized Lebesgue measure on `[0, 2π)ⁿ`; the uniform law is unique (`57993faf`). A `2π`-periodic Lebesgue-null set therefore maps to a `P`-null set. | — |
| M5.2 | **SA in angles.** SA in angles is a gradient flow in the metric `diag(Z_i)`. | M1.5 |
| M5.3 | **Spheres for `d ≥ 3`, without manifolds.** On `(ℝ^d∖0)^n`, the field `F_i(x) = ‖x_i‖² ∇_i(𝖤_β ∘ normalize)(x)` has the sphere flow on every slice. Radially invariant Lebesgue-null sets correspond to uniform-null sets (`Measure.measurePreserving_homeomorphUnitSphereProd`). | M1.5 |
| M5.4 | **Research: grokking network to phase model.** No Lean until a statement is fixed on paper. | — |

## Order of work

1. M0.1, which decides M3.1–M3.3.
2. M1.1 → M1.2 → M1.3, then M1.4.
3. M2.1, M2.3, M3.4a, then M4.1.
4. M2.2 → M4.2.
5. M3.1, then M3.2, according to M0.1.
6. M5.1 → M4.3.
7. M1.5 → M5.2, M5.3, then M3.3.

The research items M2.4, M3.4b–d and M5.4 wait for an argument on paper.

## Target `sorry`s

| `sorry` | Closed by | Blocked on |
| --- | --- | --- |
| `Perspective.strict_saddle` | M3.1 | M0.1; board |
| `Perspective.beta_tiny_circle` | M3.2, M4.3, M5.1, M5.2 | M2.4 |
| `Perspective.beta_tiny` | Appendix C synthesis, M5.3 | M2.4 |
| `Perspective.beta_interval` | M3.2 for `d = 2`, Appendix B, M5.3 | M2.4 |
| `Perspective.boumal_clustering` | M3.3, M5.3 | M2.4 |
| `Normalization.lojasiewicz_modulated` | M2.4 | Normalization line |

`Perspective.beta0_consensus` is outside this scheme. It already follows from
`almost_sure_consensus_beta0`, but `Section3_SmallBeta` comes before
`AppendixA_Saddle` in the import order, so the statement has to move first.
Leave it to whoever claims it.
