import Transformer.Memorization.Section2_Shannon

/-!
# Dataset entropy

arXiv:2505.24832v3, Section 2.1 and Appendix A.6: finite-vector entropy
is subadditive, and is additive for independent coordinates.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory
open scoped BigOperators

variable {Ω A : Type*} [MeasurableSpace Ω] [MeasurableSpace A]
  [Fintype A] [MeasurableSingletonClass A]

/-- Section 2.1, proof of Proposition 1: subadditivity for a complete finite
dataset vector; no independence is needed for this direction. -/
theorem entropy_vector_le_sum (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (X : Fin n → Ω → A) (hX : ∀ i, Measurable (X i)) :
    entropy (fun ω i => X i ω) μ ≤ ∑ i, entropy (X i) μ := by
  induction n with
  | zero =>
    have he : (fun ω (i : Fin 0) => X i ω) = fun _ => Fin.elim0 := by
      funext ω i
      exact Fin.elim0 i
    rw [he]
    simp [entropy_const]
  | succ n ih =>
    let f : (Fin (n + 1) → A) → (Fin n → A) × A :=
      fun x => (fun i => x i.castSucc, x (Fin.last n))
    have hf : Function.Injective f := by
      intro x y he
      simp only [Prod.mk.injEq, f] at he
      funext i
      rcases Fin.eq_castSucc_or_eq_last i with ⟨j, rfl⟩ | rfl
      · exact congrFun he.1 j
      · exact he.2
    have hm : Measurable (fun ω i => X i ω) := by fun_prop
    calc
      entropy (fun ω i => X i ω) μ =
          entropy (fun ω => ((fun i : Fin n => X i.castSucc ω), X (Fin.last n) ω)) μ :=
        (entropy_comp_of_injective μ hm f hf).symm
      _ ≤ entropy (fun ω (i : Fin n) => X i.castSucc ω) μ +
          entropy (X (Fin.last n)) μ := entropy_pair_le_add (by fun_prop) (hX _) μ
      _ ≤ (∑ i : Fin n, entropy (X i.castSucc) μ) + entropy (X (Fin.last n)) μ :=
        add_le_add (ih _ (fun i => hX i.castSucc)) le_rfl
      _ = ∑ i, entropy (X i) μ :=
        (Fin.sum_univ_castSucc (fun i => entropy (X i) μ)).symm

/-- Section 2.1: a two-coordinate constant dataset on a Dirac probability
space satisfies all subadditivity side conditions. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    ∀ i : Fin 2, Measurable (fun _ : Unit => i) :=
  ⟨inferInstance, fun _ => measurable_const⟩

/-- Section 2.1, proof of Proposition 1: independent-coordinate additivity
under an explicitly specified probability measure. -/
theorem entropy_vector_eq_sum (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (X : Fin n → Ω → A) (hX : ∀ i, Measurable (X i))
    (hind : iIndepFun X μ) :
    entropy (fun ω i => X i ω) μ = ∑ i, entropy (X i) μ := by
  let : MeasureSpace Ω := ⟨μ⟩
  exact iIndepFun.entropy_eq_add hX hind

/-- Section 2.1: constant coordinates are jointly independent. -/
example : (∀ i : Fin 1, Measurable (fun _ : Unit => i)) ∧
    iIndepFun (fun i : Fin 1 => fun _ : Unit => i) (Measure.dirac ()) := by
  exact ⟨fun _ => measurable_const, iIndepFun.of_subsingleton⟩

end Transformer.Memorization
