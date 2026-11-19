module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.DistanceDecay

/-!
# Signed geometric relative-position profiles

Source: `review/convex/moments.md`, CM119. Each direction has a trainable
adjacent level and a trainable nonnegative decay. Odds observed in one
interior positional row determine the complete relative law at any length.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open Perspective
open scoped BigOperators
variable {O : Type*} {d n : ℕ}

/-- CM119. The left and right adjacent levels are measured relative to
remaining at the source position. Each decay is shared along its direction. -/
structure SignedProfile where
  leftLevel : ℝ
  rightLevel : ℝ
  leftDecay : ℝ
  rightDecay : ℝ

/-- CM119. Signed positional logits with the central logit fixed at zero. -/
def SignedProfile.potential (s : SignedProfile) (δ : ℤ) : ℝ :=
  if δ = 0 then 0 else if δ < 0 then s.leftLevel - s.leftDecay * (-(δ : ℝ) - 1)
  else s.rightLevel - s.rightDecay * ((δ : ℝ) - 1)

/-- CM119. Insert the same trained positional law into both normalized reads. -/
def profileParameters (p : Parameters O d) (s : SignedProfile) : Parameters O d :=
  {p with relation := s.potential}

/-- CM119. One extra step into the positive tail subtracts the same learned
coefficient at every distance, including distances absent from training. -/
theorem profile_right_step (s : SignedProfile) (δ : ℤ) (h : 0 < δ) :
    s.potential (δ + 1) = s.potential δ - s.rightDecay := by
  have h' : 0 < δ + 1 := by omega
  simp [SignedProfile.potential, h.ne', h'.ne', h.not_gt, h'.not_gt]
  ring

/-- CM119. The negative side has its own learned decay. -/
theorem profile_left_step (s : SignedProfile) (δ : ℤ) (h : δ < 0) :
    s.potential (δ - 1) = s.potential δ - s.leftDecay := by
  have h' : δ - 1 < 0 := by omega
  simp [SignedProfile.potential, h.ne, h'.ne, h, h']
  ring

/-- CM119. The exponential relative law is exact in the complete row
for any memory length and any boundary truncation of the two tails. -/
theorem profile_link_odds (p : Parameters O d) (s : SignedProfile) (j k : Fin n) :
    softmaxWeight (linkScore (profileParameters p s) j) k =
      Real.exp (s.potential ((k : ℤ) - (j : ℤ))) *
        softmaxWeight (linkScore (profileParameters p s) j) j := by
  have unchanged : distanceParameters (profileParameters p s) 0 = profileParameters p s := by
    simp [distanceParameters]
  have zero : (profileParameters p s).relation 0 = 0 := by simp [profileParameters, SignedProfile.potential]
  have h := distance_link_odds (profileParameters p s) 0 j k
  rw [unchanged, zero, show linkScore (profileParameters p s) j k = s.potential ((k : ℤ) - (j : ℤ)) by rfl] at h
  simpa only [zero_mul, sub_zero] using h

/-- CM119. Two consecutive positive offsets in the actual normalized row
have the same geometric ratio regardless of every other visible position. -/
theorem profile_right_geometric (p : Parameters O d) (s : SignedProfile) (j k l : Fin n)
    (hk : j < k) (hl : l.val = k.val + 1) :
    softmaxWeight (linkScore (profileParameters p s) j) l =
      Real.exp (-s.rightDecay) * softmaxWeight (linkScore (profileParameters p s) j) k := by
  have hk' : (0 : ℤ) < (k : ℤ) - (j : ℤ) := by omega
  have hl' : (l : ℤ) - (j : ℤ) = (k : ℤ) - (j : ℤ) + 1 := by omega
  rw [profile_link_odds p s j l, profile_link_odds p s j k, hl', profile_right_step s _ hk',
    sub_eq_add_neg, Real.exp_add]
  ring

/-- CM119. Consecutive negative offsets in an actual row follow the
independently learned left decay even when the positive tail is truncated. -/
theorem profile_left_geometric (p : Parameters O d) (s : SignedProfile) (j k l : Fin n)
    (hk : k < j) (hl : l.val + 1 = k.val) :
    softmaxWeight (linkScore (profileParameters p s) j) l =
      Real.exp (-s.leftDecay) * softmaxWeight (linkScore (profileParameters p s) j) k := by
  have hk' : (k : ℤ) - (j : ℤ) < 0 := by omega
  have hl' : (l : ℤ) - (j : ℤ) = (k : ℤ) - (j : ℤ) - 1 := by omega
  rw [profile_link_odds p s j l, profile_link_odds p s j k, hl', profile_left_step s _ hk',
    sub_eq_add_neg, Real.exp_add]
  ring

/-- CM119. Numerical positional probabilities at distances 0, ±1 and ±2
identify both levels and both decays. The premise observes routing
probabilities, which can contain more information than next-object labels. -/
theorem profile_identified_by_interior_row (p : Parameters O d) (s t : SignedProfile)
    (same : softmaxWeight (linkScore (profileParameters p s) (2 : Fin 5)) =
      softmaxWeight (linkScore (profileParameters p t) (2 : Fin 5))) : s = t := by
  let u := linkScore (profileParameters p t) (2 : Fin 5)
  have center : 0 < softmaxWeight u 2 :=
    div_pos (Real.exp_pos _) (Landscape.Neural.sum_exp_pos u 2)
  have recover (k : Fin 5) : s.potential ((k : ℤ) - 2) = t.potential ((k : ℤ) - 2) := by
    have h₁ := profile_link_odds p s (2 : Fin 5) k
    have h₂ := profile_link_odds p t (2 : Fin 5) k
    rw [same] at h₁
    apply Real.exp_injective
    exact mul_right_cancel₀ center.ne' (h₁.symm.trans h₂)
  have left := recover 1
  have right := recover 3
  have left₂ := recover 0
  have right₂ := recover 4
  cases s
  cases t
  norm_num [SignedProfile.potential] at left right left₂ right₂
  congr 1 <;> linarith

/-- CM119. The identified profile determines the positional reads of
the full successor on every serialized prefix with the other weights fixed. -/
theorem profile_extrapolates_observed_row (p : Parameters O d) (s t : SignedProfile)
    (same : softmaxWeight (linkScore (profileParameters p s) (2 : Fin 5)) =
      softmaxWeight (linkScore (profileParameters p t) (2 : Fin 5)))
    (B : ℝ) (b r : O → ℝ) (xs : List O) :
    radialOffsetScore (profileParameters p s) B b r xs = radialOffsetScore (profileParameters p t) B b r xs := by
  rw [profile_identified_by_interior_row p s t same]

/-- CM119. Every full successor logit remains bounded on arbitrary input
prefixes when using a signed profile in both positional reads. -/
theorem profile_score_bound (p : Parameters O d) (hp : Admissible p) (s : SignedProfile)
    (B D : ℝ) (hB : 0 ≤ B) (b r : O → ℝ) (hb : ∀ o, |b o| ≤ D) (hr : ∀ o, 0 ≤ r o ∧ r o ≤ 1)
    (xs : List O) (c : O) : |radialOffsetScore (profileParameters p s) B b r xs c| ≤ B + D :=
  radialOffsetScore_bound (profileParameters p s) hp B D hB b r hb hr xs c

example : (SignedProfile.mk 1 2 (1/2) 1).potential (7 : ℤ) =
    (SignedProfile.mk 1 2 (1/2) 1).potential (6 : ℤ) - 1 :=
  profile_right_step _ 6 (by decide)

example : (SignedProfile.mk 1 2 (1/2) 1).potential (-7 : ℤ) =
    (SignedProfile.mk 1 2 (1/2) 1).potential (-6 : ℤ) - 1/2 :=
  profile_left_step _ (-6) (by decide)

example (p : Parameters Bool 0) : (SignedProfile.mk 1 2 (1/2) 1) = (SignedProfile.mk 1 (1+1) (1/2) 1) :=
  profile_identified_by_interior_row p _ _ (by norm_num)

example (p : Parameters Bool 0) (s : SignedProfile) :
    softmaxWeight (linkScore (profileParameters p s) (1 : Fin 4)) 3 =
      Real.exp (-s.rightDecay) * softmaxWeight (linkScore (profileParameters p s) (1 : Fin 4)) 2 :=
  profile_right_geometric p s 1 2 3 (by decide) (by decide)

example (p : Parameters Bool 0) (s : SignedProfile) :
    softmaxWeight (linkScore (profileParameters p s) (2 : Fin 4)) 0 =
      Real.exp (-s.leftDecay) * softmaxWeight (linkScore (profileParameters p s) (2 : Fin 4)) 1 :=
  profile_left_geometric p s 2 1 0 (by decide) (by decide)

example (p : Parameters Bool 0) : radialOffsetScore (profileParameters p ⟨1,2,1/2,1⟩)
    1 0 (fun _ => 1) [false,true,true] =
    radialOffsetScore (profileParameters p ⟨1,1+1,1/2,1⟩) 1 0 (fun _ => 1) [false,true,true] :=
  profile_extrapolates_observed_row p _ _ (by norm_num) ..

example : |radialOffsetScore (profileParameters (productParameters bitAngle bitAngle) ⟨1,2,1/2,1⟩)
    16 0 (fun _ => 1) [true,false,true] true| ≤ 16 + 0 :=
  profile_score_bound (productParameters bitAngle bitAngle) (product_admissible ..) _ 16 0
    (by norm_num) 0 (fun _ => 1) (by simp) (by simp) [true,false,true] true

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
