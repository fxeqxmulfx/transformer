module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Basic

/-!
# One joint mixture over complete causal rules

Source: `review/convex_causal.md`, CCa2. A library member receives the whole
visible prefix and returns a distribution. It may execute nonlinear stateful
computation internally, but is fixed during this convex optimization. Learned
weights multiply fixed outputs, never separately learned routers or decoders.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators

variable {H X C : Type*} [Fintype H] [Fintype C]

/-- All joint mixture weights, trained on the simplex (CCa2). -/
def Simplex (w : H → ℝ) : Prop := (∀ h, 0 ≤ w h) ∧ ∑ h, w h = 1

/-- Mix complete predictions after each rule has executed (CCa2). -/
noncomputable def mix (E : H → X → C → ℝ) (w : H → ℝ) (x : X) (c : C) : ℝ :=
  ∑ h, w h * E h x c

/-- The trainable domain is convex in all weights (CCa2). -/
theorem simplex_convex : Convex ℝ {w : H → ℝ | Simplex w} := by
  intro w hw v hv a b ha hb hab
  constructor
  · intro h
    exact add_nonneg (mul_nonneg ha (hw.1 h)) (mul_nonneg hb (hv.1 h))
  · change (∑ h, (a * w h + b * v h)) = 1
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hw.2, hv.2]
    simpa using hab

/-- Mixing preserves the probability rows (CCa2). -/
theorem mix_rows {E : H → X → C → ℝ} (hE : ∀ h, Rows (E h))
    {w : H → ℝ} (hw : Simplex w) : Rows (mix E w) := by
  constructor
  · intro x c
    exact Finset.sum_nonneg fun h _ => mul_nonneg (hw.1 h) ((hE h).1 x c)
  · intro x
    simp only [mix]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, (hE _).2, mul_one]
    exact hw.2

omit [Fintype C] in
/-- The forward map is linear in the complete joint weights (CCa2). -/
theorem mix_linear (E : H → X → C → ℝ) (w v : H → ℝ) (a b : ℝ) :
    mix E (a • w + b • v) = a • mix E w + b • mix E v := by
  funext x c
  simp only [mix, Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_mul,
    Finset.sum_add_distrib, mul_assoc, ← Finset.mul_sum]

/-- The calibrated ordinary CE is jointly convex in the mixture (CCa2). -/
theorem mixture_loss_convex [Nonempty C] {E : H → X → C → ℝ}
    (hE : ∀ h, Rows (E h)) {S : Type*} [Fintype S] {ε : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (input : S → X) (label : S → C) :
    ConvexOn ℝ {w | Simplex w} (fun w => loss ε input label (mix E w)) := by
  refine ⟨simplex_convex, ?_⟩
  intro w hw v hv a b ha hb hab
  change loss ε input label (mix E (a • w + b • v)) ≤ _
  rw [mix_linear]
  exact (loss_convex hε hε1 input label).2 (mix_rows hE hw) (mix_rows hE hv) ha hb hab

/-- A single candidate is a legitimate learned point (CCa2). -/
def vertex [DecidableEq H] (h : H) : H → ℝ := fun k => if k = h then 1 else 0

/-- Every library member remains available without dilution (CCa2). -/
theorem simplex_vertex [DecidableEq H] (h : H) : Simplex (vertex h) := by
  constructor
  · intro k
    simp only [vertex]
    split_ifs <;> norm_num
  · simp [vertex]

omit [Fintype C] in
/-- A vertex evaluates the candidate on the actual visible input (CCa2). -/
theorem mix_vertex [DecidableEq H] (E : H → X → C → ℝ) (h : H) :
    mix E (vertex h) = E h := by
  funext x c
  simp [mix, vertex]

/-- Uniform weights on a nonempty surviving finite set (CCa2). -/
noncomputable def uniformOn [DecidableEq H] (s : Finset H) : H → ℝ :=
  fun h => if h ∈ s then (s.card : ℝ)⁻¹ else 0

/-- Finite elimination returns feasible weights whenever a candidate survives (CCa2). -/
theorem simplex_uniformOn [DecidableEq H] {s : Finset H} (hs : s.Nonempty) :
    Simplex (uniformOn s) := by
  constructor
  · intro h
    unfold uniformOn
    split_ifs <;> positivity
  · have hn : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_pos.mpr hs).ne'
    simp only [uniformOn]
    rw [← Finset.sum_filter]
    simp [hn]

example := mix_rows (fun h : Bool => rows_point (fun x : Bool => xor h x))
  (simplex_vertex true)

example := mixture_loss_convex
  (fun h : Bool => rows_point (fun x : Bool => xor h x)) (ε := 1 / 4)
  (by norm_num) (by norm_num) (id : Bool → Bool) Bool.not

example := simplex_uniformOn (s := ({false, true} : Finset Bool)) (by simp)

end Transformer.Grokking.FlatSpectrum.ConvexCausal
