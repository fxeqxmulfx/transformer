/-
# The emergence of clusters in self-attention dynamics — the log-sum-exp
  potential

§7 of arXiv:2305.05465v6, `sec: self-att`: the dynamics `e:Idnonresca` the
proof of `t:boolean` runs on, the convexity of its potential `l:logsumexp`,
and the non-collision lemma `l:distnondec` that follows from it.

**What the source says and what is carried here.**

* `e:Idnonresca` is `eq:trans_dyn` at `Q = K = V = I_d`, and
  `transformerDynamics_one_iff` proves that literally.  The source writes it
  for `d = 1`, where the inner products are products of scalars, but
  `l:logsumexp` and `l:distnondec` are stated there "for any `d ≥ 1`", so they
  are carried in `ℝ^d`.

* `l:logsumexp` is proved, though not by the source's computation.  The
  source deduces midpoint convexity from `(a+b) ≥ 2√(ab)`; the same inequality
  in its weighted form, `p₁^{w₁} p₂^{w₂} ≤ w₁p₁ + w₂p₂`, gives convexity
  outright, after normalising by the two sums.  That is `sum_rpow_mul_rpow_le`
  below, Hölder's inequality for a finite sum.

* `e:infdist` is the sentence "particles never collide" following
  `l:distnondec`.  It is stated here with the conclusion of `l:distnondec` as
  an explicit hypothesis, so that it is genuinely proved.

Source: arXiv:2305.05465v6, `e:Idnonresca`, `e:logsumexpfct`, `l:logsumexp`,
`l:distnondec`, `e:infdist`.
-/

import Transformer.Clusters.Section1_Dynamics
import Mathlib.Analysis.MeanInequalities

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### Hölder's inequality for a finite sum -/

/-- **Hölder's inequality for a finite sum**, in the form the convexity of
`e:logsumexpfct` needs:

  `Σ_i f_i^a g_i^b ≤ (Σ_i f_i)^a (Σ_i g_i)^b`  for `a, b ≥ 0`, `a + b = 1`.

It is the weighted arithmetic-geometric mean inequality applied to the
normalised terms `f_i / Σf` and `g_i / Σg`, whose sums are both `1`. -/
theorem sum_rpow_mul_rpow_le {ι : Type*} (s : Finset ι) (f g : ι → ℝ)
    (hf : ∀ i ∈ s, 0 ≤ f i) (hg : ∀ i ∈ s, 0 ≤ g i)
    (hA : 0 < ∑ i ∈ s, f i) (hB : 0 < ∑ i ∈ s, g i)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    ∑ i ∈ s, f i ^ a * g i ^ b ≤ (∑ i ∈ s, f i) ^ a * (∑ i ∈ s, g i) ^ b := by
  set A := ∑ i ∈ s, f i with hAdef
  set B := ∑ i ∈ s, g i with hBdef
  have hApow : (0 : ℝ) < A ^ a * B ^ b := by positivity
  have key : ∀ i ∈ s, f i ^ a * g i ^ b ≤ A ^ a * B ^ b * (a * (f i / A) + b * (g i / B)) := by
    intro i hi
    have hfi := hf i hi
    have hgi := hg i hi
    have h1 : (f i / A) ^ a * (g i / B) ^ b ≤ a * (f i / A) + b * (g i / B) :=
      Real.geom_mean_le_arith_mean2_weighted ha hb (by positivity) (by positivity) hab
    have h2 : f i ^ a * g i ^ b = A ^ a * B ^ b * ((f i / A) ^ a * (g i / B) ^ b) := by
      rw [Real.div_rpow hfi hA.le, Real.div_rpow hgi hB.le]
      field_simp
    rw [h2]
    exact mul_le_mul_of_nonneg_left h1 hApow.le
  refine (Finset.sum_le_sum key).trans_eq ?_
  rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    ← Finset.sum_div, ← Finset.sum_div, ← hAdef, ← hBdef,
    div_self hA.ne', div_self hB.ne', mul_one, mul_one, hab, mul_one]

/-! ### `e:logsumexpfct` and `l:logsumexp` -/

/-- **Equation (e:logsumexpfct).**  The log-sum-exp potential of a
configuration:

  `f(x) = log( Σ_j e^{⟨x, x_j⟩} )`.

Source: arXiv:2305.05465v6, `e:logsumexpfct`. -/
noncomputable def logSumExp (X : Idx n → EucSpace d) (x : EucSpace d) : ℝ :=
  Real.log (∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) x (X j)))

/-- The sum inside `e:logsumexpfct` is positive as soon as there is a token. -/
theorem sum_exp_inner_pos (hn : 0 < n) (X : Idx n → EucSpace d) (x : EucSpace d) :
    0 < ∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) x (X j)) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) (Finset.univ_nonempty_iff.mpr
    (Fin.pos_iff_nonempty.mp hn))

/-- **Lemma (l:logsumexp).**  For any `x_1, …, x_n ∈ ℝ^d` the function

  `f : x ↦ log( Σ_j e^{⟨x, x_j⟩} )`

is convex.

Source: arXiv:2305.05465v6, `l:logsumexp`. -/
theorem convexOn_logSumExp (X : Idx n → EucSpace d) :
    ConvexOn ℝ (Set.univ : Set (EucSpace d)) (logSumExp X) := by
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp [logSumExp]
  have hx := sum_exp_inner_pos hn X x
  have hy := sum_exp_inner_pos hn X y
  have hsplit : ∀ j : Idx n,
      Real.exp (inner (𝕜 := ℝ) (a • x + b • y) (X j))
        = Real.exp (inner (𝕜 := ℝ) x (X j)) ^ a * Real.exp (inner (𝕜 := ℝ) y (X j)) ^ b := by
    intro j
    rw [inner_add_left, real_inner_smul_left, real_inner_smul_left, Real.exp_add,
      mul_comm a _, mul_comm b _, Real.exp_mul, Real.exp_mul]
  have hle : ∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) (a • x + b • y) (X j))
      ≤ (∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) x (X j))) ^ a *
        (∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) y (X j))) ^ b := by
    rw [Finset.sum_congr rfl fun j _ => hsplit j]
    exact sum_rpow_mul_rpow_le _ _ _ (fun _ _ => (Real.exp_pos _).le)
      (fun _ _ => (Real.exp_pos _).le) hx hy ha hb hab
  have hpos : 0 < ∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) (a • x + b • y) (X j)) :=
    sum_exp_inner_pos hn X _
  calc logSumExp X (a • x + b • y) ≤ Real.log
        ((∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) x (X j))) ^ a *
          (∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) y (X j))) ^ b) :=
        Real.log_le_log hpos hle
    _ = a * logSumExp X x + b * logSumExp X y := by
        rw [Real.log_mul (by positivity) (by positivity), Real.log_rpow hx, Real.log_rpow hy]
        rfl

/-! ### `e:Idnonresca` and `l:distnondec` -/

/-- **Equation (e:Idnonresca).**  The transformer dynamics at
`Q = K = V = I_d`:

  `ẋ_i(t) = Σ_j ( e^{⟨x_i,x_j⟩} / Σ_k e^{⟨x_i,x_k⟩} ) x_j(t)`.

Source: arXiv:2305.05465v6, `e:Idnonresca`. -/
def IdNonrescaledDynamics (X : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n),
    HasDerivAt (fun s => X s i)
      (∑ j : Idx n, attentionMatrix (1 : ParamMatrix d) 1 (X t) i j • X t j) t

/-- **`e:Idnonresca` is `eq:trans_dyn` at `Q = K = V = I_d`.** -/
theorem transformerDynamics_one_iff (X : ℝ → Idx n → EucSpace d) :
    TransformerDynamics (1 : ParamMatrix d) 1 1 X ↔ IdNonrescaledDynamics X := by
  constructor <;> intro H t i <;>
    simpa [one_apply_eq_self] using H t i

/-- **The configuration at the origin solves `e:Idnonresca`.**  Every drift
vanishes there; this is the solution in closed form that witnesses the
hypotheses below. -/
theorem idNonrescaledDynamics_zero (d n : ℕ) :
    IdNonrescaledDynamics (n := n) (fun _ _ => (0 : EucSpace d)) := by
  intro t i
  simpa using hasDerivAt_const t (0 : EucSpace d)

/-- **Lemma (l:distnondec).**  Along a solution of `e:Idnonresca`, the map
`t ↦ ‖x_i(t) - x_j(t)‖` is non-decreasing for every pair `i, j ∈ [n]`.

Not proved here.  The source's argument: `e:Idnonresca` is `ẋ_i = ∇f(x_i)` for
the potential `e:logsumexpfct`, and the gradient of a convex function is
monotone, so `⟨ẋ_i - ẋ_j, x_i - x_j⟩ ≥ 0`.

Source: arXiv:2305.05465v6, `l:distnondec`. -/
theorem norm_sub_monotone (X : ℝ → Idx n → EucSpace d) (hX : IdNonrescaledDynamics X)
    (i j : Idx n) : Monotone fun t => ‖X t i - X t j‖ := by
  sorry

/-- The hypothesis of `norm_sub_monotone` is satisfiable. -/
example : IdNonrescaledDynamics (n := n) (fun _ _ => (0 : EucSpace d)) :=
  idNonrescaledDynamics_zero d n

/-- **Equation (e:infdist).**  Particles never collide: two tokens that start
apart stay apart, by at least their initial separation.

The conclusion of `l:distnondec` is taken as an explicit hypothesis.

Source: arXiv:2305.05465v6, `e:infdist`. -/
theorem ne_of_norm_sub_monotone (X : ℝ → Idx n → EucSpace d)
    (hmono : ∀ i j : Idx n, Monotone fun t => ‖X t i - X t j‖)
    (i j : Idx n) (hij : X 0 i ≠ X 0 j) {t : ℝ} (ht : 0 ≤ t) : X t i ≠ X t j := by
  intro hcontra
  have h0 : ‖X 0 i - X 0 j‖ ≤ ‖X t i - X t j‖ := hmono i j ht
  rw [hcontra, sub_self, norm_zero] at h0
  exact hij (sub_eq_zero.mp (norm_le_zero_iff.mp h0))

/-- The hypotheses of `ne_of_norm_sub_monotone` are satisfiable: a
configuration whose tokens do not move has constant distances, and two of its
tokens may start apart. -/
example (z : EucSpace 1) (hz : z ≠ 0) :
    (∀ i j : Idx 2, Monotone fun t : ℝ => ‖(fun _ (k : Idx 2) => if k = 0 then z else 0) t i
        - (fun _ (k : Idx 2) => if k = 0 then z else 0) t j‖) ∧
      (if (0 : Idx 2) = 0 then z else 0) ≠ (if (1 : Idx 2) = 0 then z else 0) := by
  refine ⟨fun i j => monotone_const, ?_⟩
  simpa using hz

end Clusters
end Transformer
