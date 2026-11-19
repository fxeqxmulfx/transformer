module

public import Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Domain

/-!
# Complete-score geometry and two-object interaction

Source: `review/convex/moments.md`, CM113. The physical score family has a
convex functional hull. An admissible point exactly separates every original
modular-addition answer. A different pair of points gives the same training
scores and opposite held-out predictions when one query is unobserved.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval

open scoped BigOperators
variable {O : Type*} {d : ℕ}

/-- The convex hull of complete bounded physical scores, CM113. -/
noncomputable def domain (O : Type*) (d : ℕ) (B : ℝ) : Set (List O → O → ℝ) :=
  convexHull ℝ {r | ∃ p : Parameters O d, Admissible p ∧ r = B • score p}

/-- Each admissible trained parameter point is included at the same scale,
CM113. -/
theorem generator_mem (p : Parameters O d) (hp : Admissible p) (B : ℝ) :
    B • score p ∈ domain O d B := subset_convexHull ℝ _ ⟨p, hp, rfl⟩

/-- The actual score bound persists over every finite mixture, CM113. -/
theorem domain_bound (B : ℝ) (hB : 0 ≤ B) (r : List O → O → ℝ) (hr : r ∈ domain O d B)
    (xs : List O) (c : O) : |r xs c| ≤ B := by
  obtain ⟨I, hI, w, z, hw, hs, hz, rfl⟩ := mem_convexHull_iff_exists_fintype.mp hr
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  calc
    _ ≤ ∑ j, |w j * z j xs c| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, w j * B := by
      apply Finset.sum_le_sum
      intro j _
      obtain ⟨p, hp, he⟩ := hz j
      rw [he, abs_mul, abs_of_nonneg (hw j)]
      apply mul_le_mul_of_nonneg_left _ (hw j)
      change |B * score p xs c| ≤ B
      rw [abs_mul, abs_of_nonneg hB]
      simpa using mul_le_mul_of_nonneg_left (score_bound p hp xs c) hB
    _ = _ := by rw [← Finset.sum_mul, hs, one_mul]

/-- Ordinary full-vocabulary CE is convex in the complete function mixture,
CM113. The independent phase, gate and routing arrays are physical coordinates. -/
theorem loss_convex [Fintype O] (B : ℝ) (xs : List O) (c : O) :
    ConvexOn ℝ (domain O d B) (fun r => Landscape.Neural.crossEntropy (r xs) c) := by
  have h := (ConvexRecall.convex_crossEntropy c).comp_linearMap (LearnedRecurrent.evaluation xs)
  have hu : ConvexOn ℝ Set.univ (fun r : List O → O → ℝ => Landscape.Neural.crossEntropy (r xs) c) := by
    simpa only [Set.preimage_univ, Function.comp_def, LearnedRecurrent.evaluation,
      LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (convex_convexHull ℝ _)

/-- CM113. One memory position has jump mass one for every query, as in CM112. -/
theorem jump_singleton (p : Parameters O d) (x : Fin 1 → O) (q : O) : jump p x q 0 = 1 := by
  simpa only [Fin.sum_univ_one] using sum_jump (by decide : 0 < 1) p x q

/-- The serialized pair now contains an actual learned two-operand rotation.
The shared one-hop and two-hop values coincide on a singleton memory, CM113. -/
theorem score_pair (p : Parameters O d) (a q c : O) :
    score p [a,q] c = blend p.product (decode p (p.value a) 0 c) (decode p (p.value a) (p.phase q) c) := by
  change logit p (fun j : Fin 1 => [a][j]) q c = _
  simp only [logit, readOne, readTwo, trail, Fintype.sum_prod_type, Fin.sum_univ_one,
    jump_singleton, one_mul]
  change blend _ (blend _ (decode p (p.value a) 0 c) (decode p (p.value a) 0 c))
    (decode p (p.value a) (p.phase q) c) = _
  unfold blend
  ring

/-- A feasible point placing its learned output gate on phase multiplication.
The two object tables are free coordinates, CM113. -/
def productParameters (v u : O → Fin (d+1) → ℝ) : Parameters O d :=
  ⟨v, v, u, 0, 0, 1, 0, 0, 0, 1⟩

/-- Any such learned phase pair belongs to the physical domain, CM113. -/
theorem product_admissible (v u : O → Fin (d+1) → ℝ) : Admissible (productParameters v u) := by
  norm_num [Admissible, productParameters]

/-- On a two-object input the product vertex executes the real part of
the complex phase product in each coordinate, CM113. -/
theorem product_pair (v u : O → Fin (d+1) → ℝ) (a q c : O) :
    score (productParameters v u) [a,q] c = average (fun f => Real.cos (v a f + u q f - v c f)) := by
  rw [score_pair]
  simp [blend, productParameters, decode]

/-- A width-one existence witness for the 97 original arithmetic objects.
These angles specify a proof point; the prospective native fit starts from
independent random angles, CM113. -/
noncomputable def residue (a : Fin 97) : Fin 1 → ℝ := ![2 * Real.pi * a.val / 97]

private theorem cos_residue_eq (a b c : Fin 97) :
    Real.cos (2 * Real.pi * ((a.val : ℝ) + b.val - c.val) / 97) = 1 ↔ c = a+b := by
  rw [Real.cos_eq_one_iff]
  constructor
  · rintro ⟨k, hk⟩
    have h : (k : ℝ) * 97 = (a.val : ℝ) + b.val - c.val := by
      apply mul_left_cancel₀ (a := 2 * Real.pi) (by positivity)
      linear_combination 97 * hk
    have hi : k * 97 = (a.val : ℤ) + b.val - c.val := by exact_mod_cast h
    apply Fin.ext
    rw [Fin.val_add]
    omega
  · rintro rfl
    let k : ℤ := ((a.val+b.val)/97 : ℕ)
    have hi : (a.val : ℤ) + b.val - (a+b).val = 97 * k := by
      rw [Fin.val_add]
      dsimp [k]
      omega
    have h : (a.val : ℝ) + b.val - (a+b).val = 97 * (k : ℝ) := by exact_mod_cast hi
    exact ⟨k, by rw [h]; ring⟩

/-- Every one of the 97² original two-object inputs has its true modular
sum as the strict complete-class winner at admissible learned weights, CM113. -/
theorem addition_correct (a b c : Fin 97) (hc : c ≠ a+b) :
    score (productParameters residue residue) [a,b] c <
      score (productParameters residue residue) [a,b] (a+b) := by
  have score_at (o : Fin 97) : score (productParameters residue residue) [a,b] o =
      Real.cos (2 * Real.pi * ((a.val : ℝ) + b.val - o.val) / 97) := by
    rw [product_pair]
    simp only [average, residue]
    norm_num
    congr 1
    ring
  rw [score_at c, score_at (a+b), (cos_residue_eq a b (a+b)).2 rfl]
  exact lt_of_le_of_ne (Real.cos_le_one _) (fun h => hc ((cos_residue_eq a b c).1 h))

/-- A two-object character used to compare admissible learned states, CM113. -/
noncomputable def bitAngle (b : Bool) : Fin 1 → ℝ := ![if b then Real.pi else 0]

/-- The two physical states agree on the complete scores at both observed
inputs ending in false, hence give identical ordinary CE for any observed
labels. They predict strictly different objects after the held-out true
query at first operand false, CM113. -/
theorem unobserved_query_ambiguous :
    let p := productParameters bitAngle bitAngle
    let r := productParameters bitAngle 0
    Admissible p ∧ Admissible r ∧
    (∀ a : Bool, score p [a,false] = score r [a,false]) ∧
    score p [false,true] false < score p [false,true] true ∧
    score r [false,true] true < score r [false,true] false := by
  refine ⟨product_admissible .., product_admissible .., ?_, ?_, ?_⟩
  · intro a
    funext c
    simp [product_pair, bitAngle]
  · norm_num [product_pair, average, Fin.sum_univ_one, bitAngle]
  · norm_num [product_pair, average, Fin.sum_univ_one, bitAngle]

example : score (productParameters residue residue) [17,92] 0 <
    score (productParameters residue residue) [17,92] 12 :=
  addition_correct 17 92 0 (by decide)
example : (1 : ℝ) • score (productParameters residue residue) ∈ domain (Fin 97) 0 1 :=
  generator_mem _ (product_admissible ..) _
example : |((1 : ℝ) • score (productParameters residue residue)) [17,92] 12| ≤ 1 :=
  domain_bound 1 (by norm_num) _ (generator_mem _ (product_admissible ..) 1) _ _

end Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval
