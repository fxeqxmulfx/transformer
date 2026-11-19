module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RawRegions.Basic

/-!
# Excluding raw labels by observable convex hull conflicts

Source: `review/convex/moments.md`, CM49. A candidate assignment is impossible
when its augmented hull intersects a differently labelled observed hull.
An explicit candidate-pool or decoder-correctness hypothesis is required to
turn these exclusions into a unique missing answer. Flatness does not supply it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion

open RawRegions

variable {E I C : Type*} [AddCommGroup E] [Module ℝ E]

/-- An observable conflict between a hypothetical query class and another
observed class, CM49. No unknown target occurs in this predicate. -/
def Excluded (point : I → E) (labels : I → C) (x : E) (c : C) : Prop :=
  ∃ d z, d ≠ c ∧ z ∈ convexHull ℝ (observedClass point labels d) ∧
    z ∈ convexHull ℝ (insert x (observedClass point labels c))

/-- Convex target regions turn the geometric conflict into exclusion of a
candidate raw answer, CM49. The missing label is not an input assumption. -/
theorem target_ne_of_excluded (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    {x : E} {c : C} (hex : Excluded point labels x c) : target x ≠ c := by
  rcases hex with ⟨d, z, hdc, hd, hc⟩
  intro hx
  have hsubset : insert x (observedClass point labels c) ⊆ {u | target u = c} := by
    intro u hu
    rcases hu with rfl | hu
    · exact hx
    · rcases hu with ⟨i, hi, rfl⟩
      exact (hobs i).trans hi
  have hzc : target z = c := convexHull_min hsubset (hconv c) hc
  have hzd := label_of_hull point labels target hconv hobs hd
  exact hdc (hzd.symm.trans hzc)

/-- An observed point of a different class is a sufficient conflict witness,
CM49. A native barycentric certificate can establish the augmented membership. -/
theorem excluded_of_observed_point (point : I → E) (labels : I → C)
    {x : E} {c : C} (j : I) (hne : labels j ≠ c)
    (hj : point j ∈ convexHull ℝ (insert x (observedClass point labels c))) :
    Excluded point labels x c := by
  refine ⟨labels j, point j, hne, ?_, hj⟩
  apply subset_convexHull
  exact ⟨j, rfl, rfl⟩

/-- Exclusion of every alternative identifies the query when its true label
belongs to the declared pool, CM49. Pool closure is a separate premise. -/
theorem target_eq_of_exclusions (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (allowed : Set C) {x : E} {a : C} (hpool : target x ∈ allowed)
    (hex : ∀ c ∈ allowed, c ≠ a → Excluded point labels x c) : target x = a := by
  by_contra hne
  have hbad := hex (target x) hpool hne
  exact target_ne_of_excluded point labels target hconv hobs hbad rfl

/-- A correct many-to-one decoder fiber can provide the candidate pool,
CM49. Correctness at the query remains explicit and is not inferred from fit. -/
theorem target_eq_of_decoder_exclusions {L : Type*}
    (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (decode : C → L) (latent : L) {x : E} {a : C}
    (hcode : decode (target x) = latent)
    (hex : ∀ c, decode c = latent → c ≠ a → Excluded point labels x c) :
    target x = a := by
  exact target_eq_of_exclusions point labels target hconv hobs
    {c | decode c = latent} hcode hex

/-- All compatible convex-region targets in the same candidate pool agree
when the same observable certificates exclude every alternative, CM49. -/
theorem compatible_targets_agree (point : I → E) (labels : I → C) (f g : E → C)
    (hf : ConvexClasses f) (hg : ConvexClasses g)
    (hfobs : ∀ i, f (point i) = labels i) (hgobs : ∀ i, g (point i) = labels i)
    (allowed : Set C) {x : E} {a : C} (hfp : f x ∈ allowed) (hgp : g x ∈ allowed)
    (hex : ∀ c ∈ allowed, c ≠ a → Excluded point labels x c) : f x = g x := by
  have hfa := target_eq_of_exclusions point labels f hf hfobs allowed hfp hex
  have hga := target_eq_of_exclusions point labels g hg hgobs allowed hgp hex
  exact hfa.trans hga.symm

/-- The left exterior query has an augmented hull through an observed point
of the opposite raw class, CM49. All coordinates are original inputs. -/
theorem left_conflict : regionPoints 0 ∈
    convexHull ℝ (insert (-4) (observedClass regionPoints regionLabels 2)) := by
  have hx : (-4 : ℝ) ∈ convexHull ℝ
      (insert (-4) (observedClass regionPoints regionLabels 2)) := by
    apply subset_convexHull
    exact Or.inl rfl
  have hy : (1 : ℝ) ∈ convexHull ℝ
      (insert (-4) (observedClass regionPoints regionLabels 2)) := by
    apply subset_convexHull
    exact Or.inr ⟨(2 : Fin 4), rfl, rfl⟩
  have h := (convex_convexHull ℝ
    (insert (-4) (observedClass regionPoints regionLabels 2))) hx hy
    (by norm_num : (0 : ℝ) ≤ 4 / 5) (by norm_num : (0 : ℝ) ≤ 1 / 5)
    (by norm_num : (4 / 5 : ℝ) + 1 / 5 = 1)
  convert h using 1
  norm_num [regionPoints]

/-- The symmetric exterior query has the corresponding opposite-class
conflict, CM49, with the same positive rational coefficients. -/
theorem right_conflict : regionPoints 3 ∈
    convexHull ℝ (insert 4 (observedClass regionPoints regionLabels 0)) := by
  have hx : (4 : ℝ) ∈ convexHull ℝ
      (insert 4 (observedClass regionPoints regionLabels 0)) := by
    apply subset_convexHull
    exact Or.inl rfl
  have hy : (-1 : ℝ) ∈ convexHull ℝ
      (insert 4 (observedClass regionPoints regionLabels 0)) := by
    apply subset_convexHull
    exact Or.inr ⟨(1 : Fin 4), rfl, rfl⟩
  have h := (convex_convexHull ℝ
    (insert 4 (observedClass regionPoints regionLabels 0))) hx hy
    (by norm_num : (0 : ℝ) ≤ 4 / 5) (by norm_num : (0 : ℝ) ≤ 1 / 5)
    (by norm_num : (4 / 5 : ℝ) + 1 / 5 = 1)
  convert h using 1
  norm_num [regionPoints]

def rawPool : Set (Fin 4) := {0, 2}

/-- Both class regions of the threshold family lie in the declared pool. -/
theorem cut_in_pool (t x : ℝ) : cutTarget t x ∈ rawPool := by
  by_cases hx : x ≤ t <;> simp [cutTarget, rawPool, hx]

/-- Every alternative in the declared pool is excluded at the left query. -/
theorem left_exclusions (c : Fin 4) (hc : c ∈ rawPool) (hne : c ≠ 0) :
    Excluded regionPoints regionLabels (-4) c := by
  simp only [rawPool, Set.mem_insert_iff, Set.mem_singleton_iff] at hc
  rcases hc with rfl | rfl
  · exact (hne rfl).elim
  · exact excluded_of_observed_point regionPoints regionLabels 0 (by decide) left_conflict

/-- Every alternative in the declared pool is excluded at the right query. -/
theorem right_exclusions (c : Fin 4) (hc : c ∈ rawPool) (hne : c ≠ 2) :
    Excluded regionPoints regionLabels 4 c := by
  simp only [rawPool, Set.mem_insert_iff, Set.mem_singleton_iff] at hc
  rcases hc with rfl | rfl
  · exact excluded_of_observed_point regionPoints regionLabels 3 (by decide) right_conflict
  · exact (hne rfl).elim

-- Every conditional core theorem has an inhabited nonconstant target instance.
example := target_ne_of_excluded regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1)
  (left_exclusions 2 (by simp [rawPool]) (by decide))
example := excluded_of_observed_point regionPoints regionLabels 0 (by decide) left_conflict
example := target_eq_of_exclusions regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) rawPool
  (cut_in_pool 0 (-4)) left_exclusions
example := compatible_targets_agree regionPoints regionLabels (cutTarget 0) (cutTarget (1 / 2))
  (cutTarget_convex 0) (cutTarget_convex (1 / 2))
  (fun i => (cutTargets_observed i).1) (fun i => (cutTargets_observed i).2) rawPool
  (cut_in_pool 0 (-4)) (cut_in_pool (1 / 2) (-4)) left_exclusions
example := left_exclusions 2 (by simp [rawPool]) (by decide)
example := right_exclusions 0 (by simp [rawPool]) (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion
