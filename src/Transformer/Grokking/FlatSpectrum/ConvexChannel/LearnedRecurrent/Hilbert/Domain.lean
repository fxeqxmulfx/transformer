module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.Mixing
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Queries

/-!
# Preserve the complete old family while adding Hilbert-state recurrences

Source: `review/convex/moments.md`, CM107. Each finite component retains its
actual learned branch. Convexity is functional; all original objects and
complex coordinates remain, with no assumed factor-discovery oracle.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert

open scoped BigOperators

/-- CM107: retain old learned rotary parameters or complete Hilbert parameters. -/
abbrev Component (O : Type*) := Rotary.Parameters O ⊕ Parameters O State

/-- CM107: the actual geometry of each tagged component determines its budget. -/
def ComponentUnit {O : Type*} : Component O → Prop
  | .inl f => Rotary.Unit f
  | .inr f => Unit f

/-- CM107: both branches execute on the same original constant-object sequence. -/
noncomputable def componentScore {O : Type*} : Component O → List O → O → ℝ
  | .inl f => Rotary.score f
  | .inr f => score f

/-- CM107: all admissible learned generators of both complete branches. -/
def generators (O : Type*) (B : ℝ) : Set (List O → O → ℝ) :=
  {r | ∃ f : Component O, ComponentUnit f ∧ r = B • componentScore f}

/-- CM107: joint function capacity, with every old family point retained. -/
def domain (O : Type*) (B : ℝ) : Set (List O → O → ℝ) :=
  convexHull ℝ (generators O B)

variable {O : Type*}

/-- CM107: the enlarged joint function domain is convex. -/
theorem domain_convex (B : ℝ) : Convex ℝ (domain O B) := convex_convexHull ℝ _

/-- CM107: each actual tagged learned component has the same depth-free output bound. -/
theorem component_bound (f : Component O) (hf : ComponentUnit f) (xs : List O) (c : O) :
    |componentScore f xs c| ≤ 1 := by
  cases f with
  | inl f => exact Rotary.score_bound f hf xs c
  | inr f => exact score_bound f hf xs c

/-- CM107: an actual admissible component is feasible in the enlarged domain. -/
theorem generator_mem (B : ℝ) (f : Component O) (hf : ComponentUnit f) :
    B • componentScore f ∈ domain O B := subset_convexHull ℝ _ ⟨f, hf, rfl⟩

/-- CM107: every old learned-rotary function remains feasible, without refitting. -/
theorem old_domain_subset (B : ℝ) : Rotary.domain O B ⊆ domain O B := by
  apply convexHull_mono
  rintro r ⟨f, hf, hr⟩
  exact ⟨Sum.inl f, hf, hr⟩

/-- CM107: the entire bounded Hilbert family is added, not only the unitary witness. -/
theorem hilbert_domain_subset (B : ℝ) :
    LearnedRecurrent.domain O State B ⊆ domain O B := by
  apply convexHull_mono
  rintro r ⟨f, hf, hr⟩
  exact ⟨Sum.inr f, hf, hr⟩

/-- CM107: a complete finite mixture may contain either learned branch. -/
theorem finite_feasible {J : Type*} [Fintype J] (B : ℝ) (w : J → ℝ)
    (f : J → Component O) (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j = 1)
    (hf : ∀ j, ComponentUnit (f j)) :
    (∑ j, w j • (B • componentScore (f j))) ∈ domain O B := by
  exact mem_convexHull_of_exists_fintype w (fun j => B • componentScore (f j)) hw hm
    (fun j => ⟨f j, hf j, rfl⟩) rfl

/-- CM107: every joint point has finite actual learned branch representations. -/
theorem finite_representation (B : ℝ) (r : List O → O → ℝ) (hr : r ∈ domain O B) :
    ∃ (J : Type) (_ : Fintype J) (w : J → ℝ) (f : J → Component O),
      (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, ComponentUnit (f j)) ∧
      r = ∑ j, w j • (B • componentScore (f j)) := by
  obtain ⟨J, hJ, w, z, hw, hm, hz, he⟩ := mem_convexHull_iff_exists_fintype.mp hr
  choose f hf heq using hz
  refine ⟨J, hJ, w, f, hw, hm, hf, ?_⟩
  rw [← he]
  apply Finset.sum_congr rfl
  intro j hj
  rw [heq j]

/-- CM107: the common budget survives all complete finite learned mixtures. -/
theorem domain_bound (B : ℝ) (hB : 0 ≤ B) (r : List O → O → ℝ)
    (hr : r ∈ domain O B) (xs : List O) (c : O) : |r xs c| ≤ B := by
  obtain ⟨J, hJ, w, f, hw, hm, hf, rfl⟩ := finite_representation B r hr
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  calc
    |∑ j, w j * (B * componentScore (f j) xs c)| ≤
        ∑ j, |w j * (B * componentScore (f j) xs c)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, w j * B := by
      apply Finset.sum_le_sum
      intro j hj
      rw [abs_mul, abs_mul, abs_of_nonneg (hw j), abs_of_nonneg hB]
      have h := mul_le_mul_of_nonneg_left (component_bound (f j) (hf j) xs c)
        (mul_nonneg (hw j) hB)
      simpa only [mul_assoc, mul_one] using h
    _ = B := by rw [← Finset.sum_mul, hm, one_mul]

/-- CM107: ordinary complete-class CE is convex in these joint scores. -/
theorem loss_convex [Fintype O] (B : ℝ) (xs : List O) (target : O) :
    ConvexOn ℝ (domain O B) (fun r => Landscape.Neural.crossEntropy (r xs) target) := by
  have h := (ConvexRecall.convex_crossEntropy target).comp_linearMap
    (LearnedRecurrent.evaluation xs)
  have hu : ConvexOn ℝ Set.univ
      (fun r : List O → O → ℝ => Landscape.Neural.crossEntropy (r xs) target) := by
    simpa only [Set.preimage_univ, Function.comp_def, LearnedRecurrent.evaluation,
      LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (domain_convex B)

example : |componentScore (.inr (witness (3 / 5) (4 / 5))) [true, false] true| ≤ 1 :=
  component_bound (.inr (witness (3 / 5) (4 / 5))) witness_unit _ _

example : (1 : ℝ) • componentScore (.inr (witness (3 / 5) (4 / 5))) ∈ domain Bool 1 :=
  generator_mem _ (.inr (witness (3 / 5) (4 / 5))) witness_unit

example : (∑ _ : Fin 1, (1 : ℝ) • ((1 : ℝ) •
    componentScore (.inr (witness (3 / 5) (4 / 5))))) ∈ domain Bool 1 :=
  finite_feasible 1 (fun _ => 1) (fun _ : Fin 1 => .inr (witness (3 / 5) (4 / 5)))
    (by norm_num) (by simp) (fun _ => witness_unit)

example : ∃ (J : Type) (_ : Fintype J) (w : J → ℝ) (f : J → Component Bool),
    (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, ComponentUnit (f j)) ∧
    (1 : ℝ) • componentScore (.inr (witness (3 / 5) (4 / 5))) =
      ∑ j, w j • ((1 : ℝ) • componentScore (f j)) :=
  finite_representation 1 _
    (generator_mem 1 (.inr (witness (3 / 5) (4 / 5))) witness_unit)

example : |((1 : ℝ) • componentScore (.inr (witness (3 / 5) (4 / 5))))
    [true, false] true| ≤ 1 :=
  domain_bound 1 (by norm_num) _
    (generator_mem 1 (.inr (witness (3 / 5) (4 / 5))) witness_unit) _ _

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert
