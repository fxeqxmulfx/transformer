module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning.Work

/-!
# Actual CE work through both shared complex passes

Source: `review/convex/moments.md`, CM82. Each pass uses the same full learned
array. The residual of the averaged logits is pulled through both passes,
with the inverse object permutation aligning the second residual. This is
the derivative of the actual averaged prediction, not two separate CE losses.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning

open scoped BigOperators Topology
open Renaming

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- The full aligned direction of the two shared complex forwards, CM82. -/
noncomputable def pairedDirection (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (tokens : Prefix O T) (v : Parameters J O (ObjectWeights O T)) (c : O) : ℝ :=
  (rawDirection tokens (fun j k => mask j (tokens, k)) v c +
    rawDirection (renamed σ tokens) (fun j k => mask j (renamed σ tokens, k)) v (σ c)) / 2

/-- Both actual learned passes obey the derived difference on the complete
original cone, CM82. No symmetry of the target task is needed for this algebra. -/
theorem pairedLogits_difference (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    {p q : Parameters J O (ObjectWeights O T)}
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hq : q ∈ complexDomain rawInput Set.univ mask) (tokens : Prefix O T) :
    pairedLogits σ q tokens - pairedLogits σ p tokens = pairedDirection σ mask tokens (q - p) := by
  have hd (x : Prefix O T) : sequenceLogits q x - sequenceLogits p x =
      rawDirection x (fun j k => mask j (x, k)) (q - p) := by
    rw [rawDirection_eq]
    exact logits_difference (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
      (rawInput (O := O) (T := T)) Set.univ mask (p := p) (q := q) hp hq (Set.mem_univ x)
  funext c
  have h₁ := congrFun (hd tokens) c
  have h₂ := congrFun (hd (renamed σ tokens)) (σ c)
  simp only [Pi.sub_apply] at h₁ h₂ ⊢
  simp only [pairedLogits_average, pairedDirection]
  linarith

/-- Feasible perturbations change the actual paired logits by this full
shared direction, CM82. All original classes enter both passes. -/
theorem pairedLogits_shift (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    {p v : Parameters J O (ObjectWeights O T)} (amount : ℝ)
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hv : p + amount • v ∈ complexDomain rawInput Set.univ mask) (tokens : Prefix O T) :
    pairedLogits σ (p + amount • v) tokens =
      pairedLogits σ p tokens + amount • pairedDirection σ mask tokens v := by
  have hd (x : Prefix O T) : sequenceLogits (p + amount • v) x =
      sequenceLogits p x + amount • rawDirection x (fun j k => mask j (x, k)) v := by
    rw [rawDirection_eq]
    exact logits_shift (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
      (rawInput (O := O) (T := T)) Set.univ mask (p := p) (v := v) amount hp hv (Set.mem_univ x)
  funext c
  simp only [pairedLogits_average, hd, pairedDirection, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

variable [DecidableEq O]

/-- Pull the single paired CE residual through both shared parameter paths,
CM82. The second pass needs the inverse output permutation even when the
architecture is used with a permutation that is not an involution. -/
noncomputable def pairedGradient (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (tokens : Prefix O T) (target : O) (p : Parameters J O (ObjectWeights O T)) :
    GradientArray J O T :=
  let residual := fun c => ConvexDepth.rowCEGradient (pairedLogits σ p tokens) target c
  (1 / 2 : ℝ) • rawPullback tokens (fun j k => mask j (tokens, k)) residual +
  (1 / 2 : ℝ) • rawPullback (renamed σ tokens)
    (fun j k => mask j (renamed σ tokens, k)) (fun c => residual (σ.symm c))

/-- The entire paired gradient has exactly the actual CE slope, CM82.
The two class sums are explicitly aligned, retaining every output object. -/
theorem pairedGradient_work (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (tokens : Prefix O T) (target : O) (p v : Parameters J O (ObjectWeights O T)) :
    rawWork (pairedGradient σ mask tokens target p) v =
      ceSlope (pairedLogits σ p tokens) (pairedDirection σ mask tokens v) target := by
  let g := fun c => ConvexDepth.rowCEGradient (pairedLogits σ p tokens) target c
  have he := Equiv.sum_comp σ (fun c => g (σ.symm c) *
    rawDirection (renamed σ tokens) (fun j k => mask j (renamed σ tokens, k)) v c)
  simp only [Equiv.symm_apply_apply] at he
  unfold pairedGradient
  rw [rawWork_combination, rawPullback_work, rawPullback_work, ceSlope_eq]
  change (1 / 2 : ℝ) * (∑ c, g c * rawDirection tokens (fun j k => mask j (tokens, k)) v c) +
    (1 / 2 : ℝ) * (∑ c, g (σ.symm c) *
      rawDirection (renamed σ tokens) (fun j k => mask j (renamed σ tokens, k)) v c) = _
  rw [← he]
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro c hc
  change _ = g c * pairedDirection σ mask tokens v c
  unfold pairedDirection
  ring

/-- The complete shared update supports the actual paired ordinary CE
at every other feasible model, including boundary points, CM82. -/
theorem pairedLoss_first_order (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    {p q : Parameters J O (ObjectWeights O T)}
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hq : q ∈ complexDomain rawInput Set.univ mask) (tokens : Prefix O T) (target : O) :
    pairedLoss σ tokens target p + rawWork (pairedGradient σ mask tokens target p) (q - p) ≤
      pairedLoss σ tokens target q := by
  rw [pairedGradient_work, ← pairedLogits_difference σ mask hp hq tokens]
  exact Training.crossEntropy_first_order _ _ target

/-- Feasible two-sided queries differentiate the actual shared complex
prediction with its one ordinary CE, CM82. This is not a finite-step identity. -/
theorem pairedLoss_hasDerivAt (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    {p v : Parameters J O (ObjectWeights O T)} {radius : ℝ} (hr : 0 < radius)
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hplus : p + radius • v ∈ complexDomain rawInput Set.univ mask)
    (hminus : p + (-radius) • v ∈ complexDomain rawInput Set.univ mask)
    (tokens : Prefix O T) (target : O) :
    HasDerivAt (fun amount : ℝ => pairedLoss σ tokens target (p + amount • v))
      (rawWork (pairedGradient σ mask tokens target p) v) 0 := by
  rw [pairedGradient_work]
  have hd := ConvexDepth.rowCE_hasDerivAt (pairedLogits σ p tokens)
    (pairedDirection σ mask tokens v) target
  rw [← ceSlope_eq] at hd
  apply hd.congr_of_eventuallyEq
  filter_upwards [eventually_abs_sub_lt (0 : ℝ) hr] with amount ha
  have hb : |amount| ≤ radius := by simpa using ha.le
  have hm := query_interval (complexDomain_convex (E := ObjectWeights O T) (X := Prefix O T)
    (J := J) (C := O) (rawInput (O := O) (T := T)) Set.univ mask) hr hp hplus hminus hb
  unfold pairedLoss
  rw [pairedLogits_shift σ mask amount hp hm tokens]
  rfl

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning
