module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.ProductRadii

/-!
# Distance decay and observable offsets in the complete successor

Source: `review/convex/moments.md`, CM118. Both normalized reads use the
same signed residual and learned distance coefficient. Short training
prefixes observe only the effective offsets occurring between their positions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open scoped BigOperators
open Perspective
variable {O : Type*} {d n : ℕ}

/-- CM118. The residual at a signed offset is shared by all pairs of
positions at that offset. The coefficient scales their absolute distance. -/
noncomputable def distanceParameters (p : Parameters O d) (a : ℝ) : Parameters O d :=
  {p with relation := fun δ => p.relation δ - a * |(δ : ℝ)|}

/-- CM118. Exact odds for one normalized destination relative to remaining
at the source position, for arbitrary values of the residual and slope. -/
theorem distance_link_odds (p : Parameters O d) (a : ℝ) (j k : Fin n) :
    softmaxWeight (linkScore (distanceParameters p a) j) k =
      Real.exp (linkScore p j k - p.relation 0 - a * |(((k : ℤ) - (j : ℤ) : ℤ) : ℝ)|) *
        softmaxWeight (linkScore (distanceParameters p a) j) j := by
  let u := linkScore (distanceParameters p a) j
  have h : u k - u j =
      linkScore p j k - p.relation 0 - a * |(((k : ℤ) - (j : ℤ) : ℤ) : ℝ)| := by
    simp [u, linkScore, distanceParameters]
    ring
  rw [← h]
  change Real.exp (u k) / _ = Real.exp (u k - u j) * (Real.exp (u j) / _)
  rw [Real.exp_sub]
  field_simp

/-- CM118. For a finite set of destinations at distance at least L, an
upper bound on their residual gaps gives a bound on their actual mass. -/
theorem distance_tail_mass (p : Parameters O d) (a L M : ℝ) (ha : 0 ≤ a)
    (j : Fin n) (far : Finset (Fin n))
    (hf : ∀ k ∈ far, L ≤ |(((k : ℤ) - (j : ℤ) : ℤ) : ℝ)| ∧ linkScore p j k - p.relation 0 ≤ M) :
    ∑ k ∈ far, softmaxWeight (linkScore (distanceParameters p a) j) k ≤
      far.card * (Real.exp (M - a * L) * softmaxWeight (linkScore (distanceParameters p a) j) j) := by
  calc
    _ ≤ ∑ _k ∈ far, Real.exp (M - a * L) * softmaxWeight (linkScore (distanceParameters p a) j) j := by
      apply Finset.sum_le_sum
      intro k hk
      rw [distance_link_odds p a j k]
      apply mul_le_mul_of_nonneg_right _ (softmaxWeight_nonneg _ _)
      apply Real.exp_le_exp.mpr
      exact sub_le_sub (hf k hk).2 (mul_le_mul_of_nonneg_left (hf k hk).1 ha)
    _ = _ := by simp

/-- CM118. Compensating the distance coefficient in the unsigned linear
part of the signed residual preserves the complete physical parameters. -/
theorem distance_compensation (p : Parameters O d) (a t : ℝ) :
    distanceParameters {p with relation := fun δ => p.relation δ + t * |(δ : ℝ)|} (a + t)
      = distanceParameters p a := by
  unfold distanceParameters
  congr 1
  funext δ
  ring

/-- CM118. A training prefix of length at most H+1 observes offsets only
in the open interval (-H,H). This holds for every class, both reads,
the learned radii and offsets, and both empty-memory boundary rules. -/
theorem observed_offsets_determine_score (p : Parameters O d) (s : ℤ → ℝ) (H : ℕ)
    (same : ∀ δ : ℤ, -(H : ℤ) < δ → δ < H → s δ = p.relation δ)
    (B : ℝ) (b r : O → ℝ) (xs : List O) (hlen : xs.length ≤ H + 1) (c : O) :
    radialOffsetScore {p with relation := s} B b r xs c = radialOffsetScore p B b r xs c := by
  have hl : xs.reverse.length ≤ H + 1 := by simpa using hlen
  unfold radialOffsetScore radialScore
  generalize hrev : xs.reverse = ys at hl ⊢
  cases ys with
  | nil => rfl
  | cons q ys =>
    cases ys with
    | nil => rfl
    | cons a rest =>
      let m := (a :: rest).reverse.length
      have hm : m ≤ H := by simpa [m] using hl
      have links (j : Fin m) : linkScore {p with relation := s} j = linkScore p j := by
        funext k
        have hj := j.isLt
        have hk := k.isLt
        exact same _ (by omega) (by omega)
      have jumps (x : Fin m → O) (o : O) (k : Fin m) :
          jump {p with relation := s} x o k = jump p x o k := by
        simp only [jump, links]
        rfl
      simp only [readOne, readTwo, radialRead, trail, jumps]
      rfl

/-- CM118. The scale and offset bound holds on every generated prefix
for every setting of the learned distance coefficient. -/
theorem distance_score_bound (p : Parameters O d) (hp : Admissible p) (a B D : ℝ)
    (hB : 0 ≤ B) (b r : O → ℝ) (hb : ∀ o, |b o| ≤ D) (hr : ∀ o, 0 ≤ r o ∧ r o ≤ 1)
    (xs : List O) (c : O) : |radialOffsetScore (distanceParameters p a) B b r xs c| ≤ B + D :=
  radialOffsetScore_bound (distanceParameters p a) hp B D hB b r hb hr xs c

/-- CM118. Directional relative potentials for the unobserved-offset witness. -/
noncomputable def directionalWitness (offset : ℤ) : Parameters Bool 0 :=
  {productParameters bitAngle 0 with
    scale := 0
    product := 0
    relation := fun δ => if δ = offset then Real.log 3 else 0}

/-- CM118. Two predictors agree on every length-at-most-two training
prefix, yet have opposite strict winners on the same length-three prefix. -/
theorem unobserved_distance_ambiguous :
    let z := fun (dir : ℤ) => radialOffsetScore (directionalWitness dir) 1 0 (fun _ => 1)
    (∀ xs : List Bool, xs.length ≤ 2 → z 1 xs = z (-1) xs) ∧
    z 1 [false,true,false] false < z 1 [false,true,false] true ∧
    z (-1) [false,true,false] true < z (-1) [false,true,false] false := by
  dsimp only
  constructor
  · intro xs hx
    funext c
    apply observed_offsets_determine_score (directionalWitness (-1)) (directionalWitness 1).relation 1 _ 1 0 _ xs hx c
    intro δ h₁ h₂
    have : δ = 0 := by omega
    simp [this, directionalWitness]
  · have short (dir : ℤ) (c : Bool) : radialOffsetScore (directionalWitness dir) 1 0
          (fun _ => 1) [false,true,false] c =
        readOne (directionalWitness dir) (fun j : Fin 2 => [false,true][j]) false 0 c := by
      change 1 * blend 0 (blend 0 _ _) _ + 0 = _
      simp [blend]
      rfl
    simp only [short]
    norm_num [directionalWitness, productParameters, readOne, jump, keyScore, linkScore,
      softmaxWeight, decode, average, bitAngle, Fin.sum_univ_one, Fin.sum_univ_two,
      Real.exp_log (by norm_num : (0 : ℝ) < 3)]

example : radialOffsetScore (directionalWitness 1) 1 0 (fun _ => 1) [true,false] true =
    radialOffsetScore (directionalWitness (-1)) 1 0 (fun _ => 1) [true,false] true :=
  congrFun (unobserved_distance_ambiguous.1 [true,false] (by decide)) true

example : |radialOffsetScore (distanceParameters (productParameters bitAngle bitAngle) (1/4))
    16 0 (fun _ => 1) [true,false,true] true| ≤ 16 + 0 :=
  distance_score_bound (productParameters bitAngle bitAngle) (product_admissible ..) (1/4)
    16 0 (by norm_num) 0 (fun _ => 1) (by simp) (by simp) [true,false,true] true

example : softmaxWeight (linkScore (distanceParameters (productParameters bitAngle bitAngle) 1)
    (0 : Fin 2)) 1 ≤ Real.exp (-1) * softmaxWeight
      (linkScore (distanceParameters (productParameters bitAngle bitAngle) 1) (0 : Fin 2)) 0 := by
  simpa [linkScore, productParameters] using
    distance_tail_mass (productParameters bitAngle bitAngle) 1 1 0 (by norm_num) (0 : Fin 2) {1}
      (by intro k hk; simp at hk; subst k; norm_num [linkScore, productParameters])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
