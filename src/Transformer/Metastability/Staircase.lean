/-
# Metastability — Beyond metastability: the staircase setting (§6 of 2410.06833v1)

Equations and definitions covered:

* `Definition d: init_s1`         — well-prepared configurations on `𝕊^1`,
* `eq: usa.angles`                — `USA` on the circle, `Θ̇ = ∇𝖤_β(Θ)`,
* `Definition def: modified USA`  — modified `USA` enforcing collisions,
* `compt: reparam`                — the time-reparametrization `τ_β`.

`Theorem thm: staircase` is `Metastability.StaircaseProfile`, and
`lem: exact time scale of clustering` is `Metastability.StaircaseTimeScale`.

**What the source says and what is changed here.**

* `eq: usa.angles` is `Θ̇ = ∇𝖤_β(Θ)` with the printed
  `𝖤_β = (1/(2β e^β n²)) Σ_i Σ_j e^{β cos(θ_i - θ_j)}`, whose gradient is
  `θ̇_i = (1/n²) Σ_j e^{β(cos(θ_j - θ_i) - 1)} sin(θ_j - θ_i)`; the display
  "in other words" below it drops the `1/n²`, and `compt: derivative` in the
  proof of `thm: staircase` keeps it.  `usaVel` is the gradient, with `1/n²`.
* `def: modified USA` counts indices `i` with "`∃ j`" close to them; `j = i`
  is always close, and `j ≠ i` is meant.
* `compt: reparam` takes a `max` of `e^{β(1 - cos(θ_i - θ_j))}` over the pairs
  farther apart than `1/√(β log β)`.  The proof then plugs it into
  `compt: derivative` and obtains `e^{β(cos(θ̃_j - θ̃_i) - cos(θ̃_1 - θ̃_2))}`
  with `(1, 2)` the *closest* such pair (the display before
  `compt: derivative` is an `argmax` of `cos`); that identity holds for the
  `min`, not the `max`, and at the reparametrized configuration
  `θ̃ = θ ∘ τ_β`, not at `θ(t)`.  `staircaseReparam` is stated with the `min`
  along `θ̃`.  Where no pair is farther apart than the radius the `min` is
  over nothing; there the rate is `0` and the clock stops.
* `τ_β` is only asked to be continuous, and its rate jumps whenever a pair
  crosses the radius; a derivative has no jumps (Darboux), so the
  differential form is read in integral form, `τ_β(t) = ∫_0^t τ̇_β`.
-/

import Transformer.Basic
import Transformer.Perspective.Section6_Circle
import Transformer.Metastability.Basic
import Transformer.Metastability.MainTheorem
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (n : ℕ)

open Perspective

/-- **Definition (d: init_s1).** *Well-prepared configuration on `𝕊^1`.*

`(θ_1,…,θ_n) ∈ 𝕋^n` is *well-prepared* if `0 ≤ θ_1 < ⋯ < θ_n ≤ π` and
there exists a numerical constant `c > 1` such that for all `i ∈ {2,…,n-1}`
and `k > i`,

  `cos(θ_i - θ_1) > cos(θ_k - θ_i) + (c log β / β)`.

The configuration is indexed by `Idx (n + 2)`: the survey assumes `n ≥ 2`
throughout §6, and carrying the two extremal indices `0` and `n + 1` in the
type is what lets `θ_1` and `θ_n` be written without a side condition.

Source: arXiv:2410.06833v1, §6, `d: init_s1`. -/
def isWellPrepared (β : ℝ) (θ : Angles (n + 2)) : Prop :=
  (∀ i j : Idx (n + 2), (i : ℕ) < j → θ i < θ j) ∧
  0 ≤ θ 0 ∧ θ (Fin.last (n + 1)) ≤ Real.pi ∧
  ∃ c : ℝ, 1 < c ∧
    ∀ i : Idx (n + 2), 1 ≤ (i : ℕ) → ((i : ℕ) + 1) < n + 2 →
      ∀ k : Idx (n + 2), (i : ℕ) < k →
        Real.cos (θ k - θ i) + c * Real.log β / β < Real.cos (θ i - θ 0)

/-- **The velocity of `eq: usa.angles`**, for the particles of `S`:

  `θ̇_i = (1/|S|²) Σ_{j ∈ S} e^{β(cos(θ_j - θ_i) - 1)} sin(θ_j - θ_i)`,

the gradient of the printed `𝖤_β` of the `|S|` particles of `S`.

Source: arXiv:2410.06833v1, §3, `eq: usa.angles`. -/
noncomputable def usaVel {N : ℕ} (S : Finset (Idx N)) (β : ℝ) (θ : Angles N) (i : Idx N) : ℝ :=
  (1 / (S.card : ℝ) ^ 2) *
    ∑ j ∈ S, Real.exp (β * (Real.cos (θ j - θ i) - 1)) * Real.sin (θ j - θ i)

/-- The collision radius `1/√(β log β)` of §6. -/
noncomputable def collisionRadius (β : ℝ) : ℝ := 1 / Real.sqrt (β * Real.log β)

/-- The times at which two distinct particles are within the collision radius.

Source: arXiv:2410.06833v1, §6, `def: modified USA` (the set whose infimum
is `T_*`). -/
def collisionTimes {N : ℕ} (β : ℝ) (θ : ℝ → Angles N) : Set ℝ :=
  { t | 0 ≤ t ∧ ∃ i j : Idx N, i ≠ j ∧ |θ t i - θ t j| ≤ collisionRadius β }

/-- **Definition (def: modified USA).** *Modified `USA` enforcing collisions.*

`θ` solves `eq: usa.angles`.  If no two particles ever come within
`1/√(β log β)`, the modified dynamics `θ̄` is `θ`.  Otherwise `T_*` is the
infimum of the times at which two do; exactly the indices `0` and `1` (the
source's `1` and `2`, "without loss of generality") are then within the
radius of another particle; `θ̄ = θ` before `T_*`; from `T_*` on the first
particle is glued to the second, and the others — the indices `i ≥ 1` —
solve `eq: usa.angles` among themselves, started from `θ(T_*)`.

At `T_*` the velocity of a surviving particle jumps, since one particle has
left the system; the equation after `T_*` is therefore one-sided there.

Source: arXiv:2410.06833v1, §6, `def: modified USA`. -/
def modifiedUSA
    (β : ℝ) (θ θbar : ℝ → Angles (n + 2)) (T_star : ℝ) : Prop :=
  (∀ t : ℝ, ∀ i : Idx (n + 2),
    HasDerivAt (fun s => θ s i) (usaVel Finset.univ β (θ t) i) t) ∧
  ((collisionTimes β θ = ∅ ∧ θbar = θ) ∨
   (IsGLB (collisionTimes β θ) T_star ∧
    { i : Idx (n + 2) | ∃ j : Idx (n + 2), j ≠ i ∧
        |θ T_star i - θ T_star j| ≤ collisionRadius β } = {0, 1} ∧
    (∀ t : ℝ, t < T_star → θbar t = θ t) ∧
    (∀ i : Idx (n + 2), 1 ≤ (i : ℕ) → θbar T_star i = θ T_star i) ∧
    (∀ t : ℝ, T_star ≤ t → θbar t 0 = θbar t 1) ∧
    ∀ t : ℝ, T_star ≤ t → ∀ i : Idx (n + 2), 1 ≤ (i : ℕ) →
      HasDerivWithinAt (fun s => θbar s i)
        (usaVel (Finset.univ.filter fun j : Idx (n + 2) => 1 ≤ (j : ℕ)) β (θbar t) i)
        (Set.Ici T_star) t))

/-- The rates `e^{β(1 - cos(θ_i - θ_j))}` of the pairs of `θ` farther apart
than the collision radius.

Source: arXiv:2410.06833v1, §6, `compt: reparam`. -/
def farRates {N : ℕ} (β : ℝ) (θ : Angles N) : Set ℝ :=
  { r | ∃ p : Idx N × Idx N, collisionRadius β < |θ p.1 - θ p.2| ∧
      r = Real.exp (β * (1 - Real.cos (θ p.1 - θ p.2))) }

/-- **Time reparametrization (compt: reparam).**

  `τ̇_β(t) = log β · min_{(i,j): |θ̃_i - θ̃_j| > 1/√(β log β)}
                e^{β (1 - cos(θ̃_i(t) - θ̃_j(t)))}`,   `τ_β(0) = 0`,

with `θ̃ = θ̄ ∘ τ_β`, in integral form; the rate `m` is the least element of
`farRates` at `θ̃(t)`, and `0` when there is none (see the module header for
the `min`, the `θ̃` and the integral form).

Source: arXiv:2410.06833v1, §6, `compt: reparam`, `compt: derivative`. -/
def staircaseReparam
    (β : ℝ) (θbar : ℝ → Angles (n + 2)) (τ m : ℝ → ℝ) : Prop :=
  (∀ t : ℝ, 0 ≤ t → (farRates β (θbar (τ t))).Nonempty →
    IsLeast (farRates β (θbar (τ t))) (m t)) ∧
  (∀ t : ℝ, 0 ≤ t → farRates β (θbar (τ t)) = ∅ → m t = 0) ∧
  ∀ t : ℝ, 0 ≤ t → τ t = ∫ s in (0 : ℝ)..t, Real.log β * m s

/-- **The normalized energy on the circle**, `2β 𝖤_β`:

  `(1/n²) Σ_i Σ_j e^{β(cos(θ_i - θ_j) - 1)} ∈ (0, 1]`.

The printed `𝖤_β` is at most `1/(2β)` and makes `thm: staircase` empty
(`Metastability.EnergyScale`); this is the normalization whose maximum is
the `1` of the staircase figure.

Source: arXiv:2410.06833v1, §3 (`𝖤_β` below `eq: usa.angles`) and §6. -/
noncomputable def circleEnergy {N : ℕ} (β : ℝ) (θ : Angles N) : ℝ :=
  (1 / (N : ℝ) ^ 2) *
    ∑ i : Idx N, ∑ j : Idx N, Real.exp (β * (Real.cos (θ i - θ j) - 1))

end Metastability
end Transformer
