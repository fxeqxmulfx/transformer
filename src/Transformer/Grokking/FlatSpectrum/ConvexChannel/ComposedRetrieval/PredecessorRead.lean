module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.PredecessorRouting

/-!
# Full causal prediction with predecessor-dependent content rows

Source: `review/convex/moments.md`, CM120. The two reads, rotated product
and every class use the actual normalized contextual routing. Short
observations can leave a strict winning class at the next length free.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open Perspective
open scoped BigOperators
variable {O : Type*} {d n : ℕ}

/-- CM120. One contextual content row followed by the original positional row. -/
noncomputable def predecessorJump (p : Parameters O d) (t : ℝ) (x : Fin n → O)
    (q : O) (a : Option O) (k : Fin n) : ℝ :=
  ∑ j, softmaxWeight (predecessorScore p t x q a) j * softmaxWeight (linkScore p j) k

/-- CM120. Each contextual jump has nonnegative physical mass. -/
theorem predecessorJump_nonneg (p : Parameters O d) (t : ℝ) (x : Fin n → O)
    (q : O) (a : Option O) (k : Fin n) : 0 ≤ predecessorJump p t x q a k :=
  Finset.sum_nonneg (fun j _ => mul_nonneg (softmaxWeight_nonneg _ j) (softmaxWeight_nonneg _ k))

/-- CM120. The actual jump preserves unit mass on every nonempty memory. -/
theorem sum_predecessorJump (hn : 0 < n) (p : Parameters O d) (t : ℝ) (x : Fin n → O)
    (q : O) (a : Option O) : ∑ k, predecessorJump p t x q a k = 1 := by
  unfold predecessorJump
  rw [Finset.sum_comm]
  simp only [← Finset.mul_sum, sum_softmaxWeight hn, mul_one]

/-- CM120. All inner content queries use the predecessor of their own
source position. The rotated path still uses both original object radii. -/
noncomputable def predecessorBody (p : Parameters O d) (t : ℝ) (r : O → ℝ)
    (x : Fin n → O) (q : O) (a : Option O) (c : O) : ℝ :=
  let w := predecessorJump p t x q a
  let v := fun k => decode p (p.value (x k)) 0 c
  blend p.product (blend p.depth (∑ k, w k * v k)
    (∑ k, w k * ∑ h, predecessorJump p t x (x k) (previous x k) h * v h))
    (r q * ∑ k, w k * (r (x k) * decode p (p.value (x k)) (p.phase q) c))

/-- CM120. Empty and singleton prefixes retain their learned boundary
rules. On longer prefixes the last memory object precedes the query. -/
noncomputable def predecessorReadout (p : Parameters O d) (t : ℝ) (r : O → ℝ)
    (xs : List O) (c : O) : ℝ :=
  match xs.reverse with
  | [] => decode p p.start 0 c
  | [q] => decode p p.start (p.phase q) c
  | q :: a :: rest =>
    let past := (a :: rest).reverse
    predecessorBody p t r (fun j : Fin past.length => past[j]) q (some a) c

/-- CM120. Each original class gets its own physical full-vocabulary score. -/
noncomputable def predecessorOutput (p : Parameters O d) (t B : ℝ) (b r : O → ℝ)
    (xs : List O) (c : O) : ℝ := B * predecessorReadout p t r xs c + b c

private theorem predecessorJump_zero (p : Parameters O d) (x : Fin n → O)
    (q : O) (a : Option O) (k : Fin n) : predecessorJump p 0 x q a k = jump p x q k := by
  have scores : predecessorScore p 0 x q a = keyScore p x q := by
    funext j
    simp [predecessorScore]
  simp only [predecessorJump, jump, scores]

/-- CM120. The complete initialized extension exactly executes CM117. -/
theorem predecessor_zero_output (p : Parameters O d) (B : ℝ) (b r : O → ℝ)
    (xs : List O) (c : O) :
    predecessorOutput p 0 B b r xs c = radialOffsetScore p B b r xs c := by
  simp only [predecessorOutput, radialOffsetScore, predecessorReadout, radialScore]
  generalize xs.reverse = back
  cases back with
  | nil => rfl
  | cons q rest =>
    cases rest with
    | nil => rfl
    | cons a rest =>
      simp only [predecessorBody, predecessorJump_zero, readOne, readTwo, radialRead,
        trail, Fintype.sum_prod_type, Finset.mul_sum, mul_assoc]

/-- CM120. Changing a content comparison inside a one-position softmax
leaves the physical jump equal to one. -/
theorem predecessor_singleton (p : Parameters O d) (t : ℝ) (x : Fin 1 → O)
    (q : O) (a : Option O) : predecessorJump p t x q a 0 = 1 := by
  simpa only [Fin.sum_univ_one] using sum_predecessorJump (by decide : 0 < 1) p t x q a

/-- CM120. Every coefficient is indistinguishable from zero on all
next-object training prefixes containing at most two objects. -/
theorem predecessor_short_output (p : Parameters O d) (t B : ℝ) (b r : O → ℝ)
    (xs : List O) (short : xs.length ≤ 2) (c : O) :
    predecessorOutput p t B b r xs c = radialOffsetScore p B b r xs c := by
  rw [← predecessor_zero_output]
  cases xs with
  | nil => rfl
  | cons a rest =>
    cases rest with
    | nil => rfl
    | cons q tail =>
      cases tail with
      | nil =>
        change B * predecessorBody p t r (fun j : Fin 1 => [a][j]) q (some a) c + b c =
          B * predecessorBody p 0 r (fun j : Fin 1 => [a][j]) q (some a) c + b c
        simp only [predecessorBody, Fin.sum_univ_one, predecessor_singleton]
      | cons z tail => simp at short

/-- CM120. Normalized content and position rows contract the bounded
learned values, uniformly over the coefficient and every prefix length. -/
theorem predecessor_output_bound (p : Parameters O d) (hp : Admissible p)
    (t B D : ℝ) (hB : 0 ≤ B) (b r : O → ℝ) (hb : ∀ o, |b o| ≤ D)
    (hr : ∀ o, 0 ≤ r o ∧ r o ≤ 1) (xs : List O) (c : O) :
    |predecessorOutput p t B b r xs c| ≤ B+D := by
  have attenuate (o : O) (s : ℝ) (hs : |s| ≤ 1) : |r o * s| ≤ 1 := by
    rw [abs_mul, abs_of_nonneg (hr o).1]
    exact (mul_le_mul_of_nonneg_left hs (hr o).1).trans (by simpa using (hr o).2)
  have bound : |predecessorReadout p t r xs c| ≤ 1 := by
    unfold predecessorReadout
    split
    · exact decode_bound ..
    · exact decode_bound ..
    · rename_i a rest he
      have hn : 0 < (a :: rest).reverse.length := by simp
      unfold predecessorBody
      apply blend_bound hp.2
      · apply blend_bound hp.1
        · exact weighted_bound _ _ (fun _ => predecessorJump_nonneg ..)
            (sum_predecessorJump hn ..) (fun _ => decode_bound ..)
        · apply weighted_bound _ _ (fun _ => predecessorJump_nonneg ..) (sum_predecessorJump hn ..)
          exact fun _ => weighted_bound _ _ (fun _ => predecessorJump_nonneg ..)
            (sum_predecessorJump hn ..) (fun _ => decode_bound ..)
      · apply attenuate
        exact weighted_bound _ _ (fun _ => predecessorJump_nonneg ..) (sum_predecessorJump hn ..)
          (fun k => attenuate _ _ (decode_bound ..))
  refine (abs_add_le _ _).trans (add_le_add ?_ (hb c))
  rw [abs_mul, abs_of_nonneg hB]
  simpa using mul_le_mul_of_nonneg_left bound hB

/-- CM120. All class offsets enter affinely with the other weights fixed.
Ordinary full-class next-object CE is jointly convex in these offsets. -/
theorem predecessor_offset_ce_convex [Fintype O] (p : Parameters O d) (t B : ℝ)
    (r : O → ℝ) (xs : List O) (y : O) : ConvexOn ℝ Set.univ
      (fun b : O → ℝ => Landscape.Neural.crossEntropy (predecessorOutput p t B b r xs) y) := by
  refine ⟨convex_univ, ?_⟩
  intro u _ v _ a h ha hh hs
  have affine : predecessorOutput p t B (a • u + h • v) r xs =
      a • predecessorOutput p t B u r xs + h • predecessorOutput p t B v r xs := by
    funext c
    simp only [predecessorOutput, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    linear_combination -(B * predecessorReadout p t r xs c) * hs
  change Landscape.Neural.crossEntropy (predecessorOutput p t B (a • u + h • v) r xs) y ≤ _
  rw [affine]
  exact (ConvexRecall.convex_crossEntropy y).2 (Set.mem_univ _) (Set.mem_univ _) ha hh hs

/-- CM120. Opposite context coefficients fit identical numerical scores
on every short prefix and reverse the strict winner on a length-three prefix. -/
theorem unobserved_predecessor_ambiguous :
    let p := {productParameters bitAngle bitAngle with
      scale := 0
      product := 0
      relation := fun δ => if δ = 0 then Real.log 3 else 0}
    let z := fun t => predecessorOutput p t 1 0 (fun _ => 1)
    (∀ xs : List Bool, xs.length ≤ 2 → z (Real.log 3) xs = z (-Real.log 3) xs) ∧
    z (Real.log 3) [false,true,false] true < z (Real.log 3) [false,true,false] false ∧
    z (-Real.log 3) [false,true,false] false < z (-Real.log 3) [false,true,false] true := by
  dsimp only
  let p : Parameters Bool 0 := {productParameters bitAngle bitAngle with
    scale := 0
    product := 0
    relation := fun δ => if δ = 0 then Real.log 3 else 0}
  constructor
  · intro xs hx
    funext c
    exact (predecessor_short_output p _ 1 0 (fun _ => 1) xs hx c).trans
      (predecessor_short_output p _ 1 0 (fun _ => 1) xs hx c).symm
  · let x := fun j : Fin 2 => [false,true][j]
    have reduced (t : ℝ) (c : Bool) : predecessorOutput p t 1 0 (fun _ => 1) [false,true,false] c =
        ∑ j, predecessorJump p t x false (some true) j * decode p (p.value (x j)) 0 c := by
      change 1 * predecessorBody p t (fun _ => 1) x false (some true) c + 0 = _
      have hp : p.product = 0 := rfl
      have hd : p.depth = 0 := rfl
      simp only [predecessorBody, hp, hd, blend, sub_zero, one_mul, zero_mul, add_zero]
    change predecessorOutput p (Real.log 3) 1 0 (fun _ => 1) [false,true,false] true <
      predecessorOutput p (Real.log 3) 1 0 (fun _ => 1) [false,true,false] false ∧
      predecessorOutput p (-Real.log 3) 1 0 (fun _ => 1) [false,true,false] false <
      predecessorOutput p (-Real.log 3) 1 0 (fun _ => 1) [false,true,false] true
    simp only [reduced]
    norm_num [predecessorJump, predecessorScore, predecessorKernel, keyScore, linkScore, softmaxWeight,
      decode, average, previous, p, productParameters, x, bitAngle, Fin.sum_univ_one, Fin.sum_univ_two,
      Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 3)]

example (p : Parameters Bool 0) : (∑ k, predecessorJump p 2 (![true,false] : Fin 2 → Bool)
    false (some true) k) = 1 := sum_predecessorJump (by decide : 0 < 2) p 2 _ false (some true)
example (p : Parameters Bool 0) : predecessorOutput p 2 1 0 (fun _ => 1) [true,false] true =
    radialOffsetScore p 1 0 (fun _ => 1) [true,false] true :=
  predecessor_short_output p 2 1 0 (fun _ => 1) [true,false] (by decide) true
example : |predecessorOutput (productParameters bitAngle bitAngle) 2 16 0 (fun _ => 1)
    [true,false,true] true| ≤ 16 + 0 := predecessor_output_bound _ (product_admissible ..) _ _ _
      (by norm_num) _ _ (by simp) (by simp) _ _

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
