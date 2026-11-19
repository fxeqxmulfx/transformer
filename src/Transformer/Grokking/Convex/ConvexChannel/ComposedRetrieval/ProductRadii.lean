module

public import Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval.OutputOffsets

/-!
# Learned radii of the two operands in the product channel

Source: `review/convex/moments.md`, CM117. Each opaque object indexes one
bounded radius shared by the retrieved value and query rotation. Training
observations of numerical class contrasts can identify products of radii.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval

open scoped BigOperators
variable {O : Type*} {d n : ℕ}

/-- The query magnitude multiplies the normalized read of scaled values,
CM117. Each occurrence contributes with its own routing probability. -/
noncomputable def radialRead (p : Parameters O d) (r : O → ℝ) (x : Fin n → O) (q c : O) : ℝ :=
  r q * ∑ j, jump p x q j * (r (x j) * decode p (p.value (x j)) (p.phase q) c)

/-- CM117, complete normalized operator on the same causal prefixes as CM113. -/
noncomputable def radialScore (p : Parameters O d) (r : O → ℝ) (xs : List O) (c : O) : ℝ :=
  match xs.reverse with
  | [] => decode p p.start 0 c
  | [q] => decode p p.start (p.phase q) c
  | q :: a :: rest =>
    let past := (a :: rest).reverse
    let x := fun j : Fin past.length => past[j]
    blend p.product (blend p.depth (readOne p x q 0 c) (readTwo p x q c)) (radialRead p r x q c)

/-- CM117 applies the physical scale and complete learned offset table. -/
noncomputable def radialOffsetScore (p : Parameters O d) (B : ℝ) (b r : O → ℝ)
    (xs : List O) (c : O) : ℝ := B * radialScore p r xs c + b c

/-- CM117. Attenuation contracts a bounded component. -/
private theorem attenuate_bound {r s : ℝ} (hr : 0 ≤ r ∧ r ≤ 1) (hs : |s| ≤ 1) :
    |r * s| ≤ 1 := by
  rw [abs_mul, abs_of_nonneg hr.1]
  exact (mul_le_mul_of_nonneg_left hs hr.1).trans (by simpa using hr.2)

/-- CM117. The scaled physical operator remains bounded independently of
sequence length, including its boundary cases. -/
theorem radialOffsetScore_bound (p : Parameters O d) (hp : Admissible p) (B D : ℝ)
    (hB : 0 ≤ B) (b r : O → ℝ) (hb : ∀ o, |b o| ≤ D) (hr : ∀ o, 0 ≤ r o ∧ r o ≤ 1)
    (xs : List O) (c : O) : |radialOffsetScore p B b r xs c| ≤ B + D := by
  have h : |radialScore p r xs c| ≤ 1 := by
    unfold radialScore
    split
    · exact decode_bound ..
    · exact decode_bound ..
    · rename_i a rest he
      have hn : 0 < (a :: rest).reverse.length := by simp
      apply blend_bound hp.2
      · apply blend_bound hp.1
        · exact weighted_bound _ _ (fun _ => jump_nonneg ..) (sum_jump hn ..) (fun _ => decode_bound ..)
        · exact weighted_bound _ _ (fun _ => trail_nonneg ..) (sum_trail hn ..) (fun _ => decode_bound ..)
      · apply attenuate_bound (hr _)
        exact weighted_bound _ _ (fun _ => jump_nonneg ..) (sum_jump hn ..)
          (fun _ => attenuate_bound (hr _) (decode_bound ..))
  refine (abs_add_le _ _).trans (add_le_add ?_ (hb c))
  rw [abs_mul, abs_of_nonneg hB]
  simpa using mul_le_mul_of_nonneg_left h hB

/-- Every unit-radius point executes the original offset model, CM117. -/
theorem unit_radii_score (p : Parameters O d) (B : ℝ) (b : O → ℝ) (xs : List O) (c : O) :
    radialOffsetScore p B b (fun _ => 1) xs c = offsetScore p B b xs c := by
  simp only [radialOffsetScore, offsetScore, radialScore, score]
  generalize xs.reverse = backwards
  cases backwards with
  | nil => rfl
  | cons q rest =>
    cases rest with
    | nil => rfl
    | cons a rest => simp [logit, radialRead, readOne]

/-- The actual singleton-memory reduction has the product of both input
magnitudes for either distinct or repeated operands, CM117. -/
theorem radial_pair (p : Parameters O d) (r : O → ℝ) (a q c : O) :
    radialScore p r [a,q] c = blend p.product (decode p (p.value a) 0 c)
      (r a * r q * decode p (p.value a) (p.phase q) c) := by
  let x := fun j : Fin 1 => [a][j]
  change blend p.product (blend p.depth (readOne p x q 0 c) (readTwo p x q c)) (radialRead p r x q c) = _
  simp only [readOne, readTwo, radialRead, trail, Fintype.sum_prod_type, Fin.sum_univ_one, jump_singleton, one_mul]
  change blend _ (blend _ (decode p (p.value a) 0 c) (decode p (p.value a) 0 c))
    (r q * (r a * decode p (p.value a) (p.phase q) c)) = _
  unfold blend
  ring

/-- CM117. Freeze the other weights and use the product of the two
observed radii as the scalar coordinate of this pair's complete logits. -/
noncomputable def pairFactorLogit (p : Parameters O d) (B : ℝ) (b : O → ℝ)
    (a q : O) (u : ℝ) (c : O) : ℝ :=
  B * blend p.product (decode p (p.value a) 0 c) (u * decode p (p.value a) (p.phase q) c) + b c

/-- The conditional coordinate evaluates the physical logits at the
actual factor for each input pair, CM117. -/
theorem pairFactorLogit_actual (p : Parameters O d) (B : ℝ) (b r : O → ℝ) (a q c : O) :
    pairFactorLogit p B b a q (r a * r q) c = radialOffsetScore p B b r [a,q] c := by
  rw [radialOffsetScore, radial_pair, pairFactorLogit]

/-- Ordinary full-class CE is convex in this single effective factor
at fixed remaining coordinates, CM117. -/
theorem pair_factor_ce_convex [Fintype O] (p : Parameters O d) (B : ℝ) (b : O → ℝ) (a q y : O) :
    ConvexOn ℝ Set.univ (fun u => Landscape.Neural.crossEntropy (pairFactorLogit p B b a q u) y) := by
  refine ⟨convex_univ, ?_⟩
  intro u _ v _ l k hl hk hlk
  have affine : pairFactorLogit p B b a q (l • u + k • v) =
      l • pairFactorLogit p B b a q u + k • pairFactorLogit p B b a q v := by
    funext c
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, pairFactorLogit, blend]
    linear_combination -(B * (1 - p.product) * decode p (p.value a) 0 c + b c) * hlk
  change Landscape.Neural.crossEntropy (pairFactorLogit p B b a q (l • u + k • v)) y ≤ _
  rw [affine]
  exact (ConvexRecall.convex_crossEntropy y).2 (Set.mem_univ _) (Set.mem_univ _) hl hk hlk

/-- At fixed other parameters the numerical contrast identifies precisely
the radius product whenever its coefficient is nonzero, CM117. -/
theorem pair_factor_identified (p : Parameters O d) (B : ℝ) (b r s : O → ℝ) (a q c k : O)
    (hn : B * p.product *
      (decode p (p.value a) (p.phase q) c - decode p (p.value a) (p.phase q) k) ≠ 0) :
    (radialOffsetScore p B b r [a,q] c - radialOffsetScore p B b r [a,q] k =
     radialOffsetScore p B b s [a,q] c - radialOffsetScore p B b s [a,q] k) ↔
    r a * r q = s a * s q := by
  have h : (radialOffsetScore p B b r [a,q] c - radialOffsetScore p B b r [a,q] k) -
      (radialOffsetScore p B b s [a,q] c - radialOffsetScore p B b s [a,q] k) =
      (B * p.product * (decode p (p.value a) (p.phase q) c - decode p (p.value a) (p.phase q) k)) *
      (r a * r q - s a * s q) := by simp only [radialOffsetScore, radial_pair, blend]; ring
  rw [← sub_eq_zero, h, mul_eq_zero]
  simp only [hn, false_or, sub_eq_zero]

/-- CM117. At a repeated input, nonnegativity fixes the radius itself from
one nondegenerate numerical contrast. -/
theorem repeated_radius_identified (p : Parameters O d) (B : ℝ) (b r s : O → ℝ) (a c k : O)
    (hr : 0 ≤ r a) (hs : 0 ≤ s a)
    (hn : B * p.product *
      (decode p (p.value a) (p.phase a) c - decode p (p.value a) (p.phase a) k) ≠ 0) :
    (radialOffsetScore p B b r [a,a] c - radialOffsetScore p B b r [a,a] k =
     radialOffsetScore p B b s [a,a] c - radialOffsetScore p B b s [a,a] k) ↔ r a = s a :=
  (pair_factor_identified p B b r s a a c k hn).trans (mul_self_inj hr hs)

/-- CM117. Complete training logits can agree although a single unseen
object magnitude reverses the strict prediction on the unseen pair. -/
theorem unobserved_radius_ambiguous :
    let p := productParameters bitAngle bitAngle
    let b := fun o : Bool => if o then (0 : ℝ) else 1 / 2
    let u := radialOffsetScore p 1 b (fun _ => 1)
    let z := radialOffsetScore p 1 b (fun o => if o then 0 else 1)
    u [false,false] = z [false,false] ∧
    u [true,false] false < u [true,false] true ∧ z [true,false] true < z [true,false] false := by
  dsimp only
  constructor
  · funext c
    simp [radialOffsetScore, radial_pair]
  · norm_num [radialOffsetScore, radial_pair, productParameters, bitAngle, blend, decode,
      average, Fin.sum_univ_one]

example : |radialOffsetScore (productParameters bitAngle bitAngle) 16 (fun _ => 1)
    (fun o => if o then 0 else 1) [true,false] true| ≤ 16 + 1 :=
  radialOffsetScore_bound _ (product_admissible ..) _ _ (by norm_num) _ _ (by intro o; norm_num)
    (by intro o; cases o <;> norm_num) _ _

example : |(1 / 2 : ℝ) * (-3 / 4)| ≤ 1 := attenuate_bound (by norm_num) (by norm_num)

example (r s : Bool → ℝ) : let p := productParameters bitAngle bitAngle
    (radialOffsetScore p 1 0 r [true,false] true - radialOffsetScore p 1 0 r [true,false] false =
     radialOffsetScore p 1 0 s [true,false] true - radialOffsetScore p 1 0 s [true,false] false) ↔
    r true * r false = s true * s false :=
  pair_factor_identified _ _ _ _ _ _ _ _ _ (by
    norm_num [productParameters, bitAngle, decode, average, Fin.sum_univ_one])

example : let p := productParameters bitAngle bitAngle
    let r := fun _ : Bool => (1 : ℝ)
    let s := fun _ : Bool => (1 / 2 : ℝ)
    (radialOffsetScore p 1 0 r [false,false] false - radialOffsetScore p 1 0 r [false,false] true =
     radialOffsetScore p 1 0 s [false,false] false - radialOffsetScore p 1 0 s [false,false] true) ↔
    r false = s false :=
  repeated_radius_identified _ _ _ _ _ _ _ _ (by norm_num) (by norm_num)
    (by norm_num [productParameters, bitAngle, decode, average, Fin.sum_univ_one])

end Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval
