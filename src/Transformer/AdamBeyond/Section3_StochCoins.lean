import Transformer.AdamBeyond.Section3_StochBound
import Mathlib.Logic.Function.DependsOn

/-
# Adam and beyond — §3: functions of finitely many coins

What the expectations in the lemma of Theorem 3 rest on: a function of the
coins that depends on finitely many of them is a bounded random variable
(`coin_integrable`), independent of every coin outside them (`coin_indep`),
so that `E[[b_t] Y] = p E[Y]` (`coin_integral_ind_mul`); and
`E[Σ_j c_j [b_j]] = p Σ_j c_j` (`coin_integral_sum`).

Source: arXiv:1904.09237, Appendix, proof of Theorem 3: the independence of
`g_t` from `m_{t-1}` and `v_{t-1}`.
-/

open MeasureTheory ProbabilityTheory Finset Function

namespace Transformer
namespace AdamBeyond

/-- The coins of `S`, extended by `false` outside `S`. -/
def coinExt (S : Finset ℕ) (v : S → Bool) : ℕ → Bool :=
  fun j => if h : j ∈ S then v ⟨j, h⟩ else false

theorem coin_factor {S : Finset ℕ} {F : (ℕ → Bool) → ℝ} (hF : DependsOn F (S : Set ℕ))
    (u : ℕ → Bool) : F u = F (coinExt S fun j => u j) :=
  hF fun i hi => by simp [coinExt, Finset.mem_coe.1 hi]

theorem dependsOn_sum (n : ℕ) (f : ℕ → Bool → ℝ) :
    DependsOn (fun u : ℕ → Bool => ∑ j ∈ Icc 1 n, f j (u j)) (Icc 1 n : Set ℕ) :=
  fun _ _ h => sum_congr rfl fun j hj => by rw [h j (mem_coe.2 hj)]

theorem dependsOn_comp₂ {s : Set ℕ} {M V : (ℕ → Bool) → ℝ} (hM : DependsOn M s)
    (hV : DependsOn V s) (g : ℝ → ℝ → ℝ) : DependsOn (fun u => g (M u) (V u)) s :=
  fun _ _ h => by simp only [hM h, hV h]

theorem dependsOn_coin_mul {S : Finset ℕ} {F : (ℕ → Bool) → ℝ} (hF : DependsOn F (S : Set ℕ))
    (t : ℕ) (ψ : Bool → ℝ) :
    DependsOn (fun u => ψ (u t) * F u) (insert t S : Finset ℕ) := fun x y h => by
  dsimp only
  rw [h t (by simp), hF fun i hi => h i (by simp [Finset.mem_coe.1 hi])]

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {b : ℕ → Ω → Bool} {p : ℝ}

/-- A function of the coins of a finite `S` is a random variable. -/
theorem coin_measurable (hb : IsBernoulliSeq μ b p) {S : Finset ℕ} {F : (ℕ → Bool) → ℝ}
    (hF : DependsOn F (S : Set ℕ)) : Measurable fun ω => F fun j => b j ω := by
  have e : (fun ω => F fun j => b j ω) =
      (fun v : S → Bool => F (coinExt S v)) ∘ fun ω (j : S) => b j ω :=
    funext fun ω => coin_factor hF _
  rw [e]
  exact (measurable_of_finite _).comp (measurable_pi_iff.2 fun j => hb.measurable j)

/-- A function of the coins of a finite `S` is integrable. -/
theorem coin_integrable [IsProbabilityMeasure μ] (hb : IsBernoulliSeq μ b p) {S : Finset ℕ}
    {F : (ℕ → Bool) → ℝ} (hF : DependsOn F (S : Set ℕ)) :
    Integrable (fun ω => F fun j => b j ω) μ := by
  refine Integrable.of_bound (coin_measurable hb hF).aestronglyMeasurable
    (∑ v : S → Bool, ‖F (coinExt S v)‖) (ae_of_all _ fun ω => ?_)
  rw [coin_factor hF]
  exact single_le_sum (f := fun v : S → Bool => ‖F (coinExt S v)‖) (fun _ _ => norm_nonneg _)
    (mem_univ _)

/-- A function of the coins of a finite `S` is independent of every coin outside `S`. -/
theorem coin_indep (hb : IsBernoulliSeq μ b p) {S : Finset ℕ} {F : (ℕ → Bool) → ℝ}
    (hF : DependsOn F (S : Set ℕ)) {t : ℕ} (ht : t ∉ S) (ψ : Bool → ℝ) :
    IndepFun (fun ω => F fun j => b j ω) (fun ω => ψ (b t ω)) μ := by
  have h := hb.indep.indepFun_finset S {t} (disjoint_singleton_right.2 ht) hb.measurable
  have e : (fun ω => F fun j => b j ω) =
      (fun v : S → Bool => F (coinExt S v)) ∘ fun ω (j : S) => b j ω :=
    funext fun ω => coin_factor hF _
  rw [e]
  exact h.comp (measurable_of_finite _)
    (measurable_of_finite (fun v : ({t} : Finset ℕ) → Bool => ψ (v ⟨t, mem_singleton_self t⟩)))

/-- `E[[b_t]] = p`. -/
theorem coin_integral_ind [IsProbabilityMeasure μ] (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p)
    (t : ℕ) :
    ∫ ω, coinInd (b t ω) ∂μ = p := by
  have e : (fun ω => coinInd (b t ω)) = (b t ⁻¹' {true}).indicator 1 := by
    funext ω
    by_cases h : b t ω <;> simp [coinInd, h]
  rw [e, integral_indicator_one ((hb.measurable t) (measurableSet_singleton true)),
    measureReal_def, show b t ⁻¹' {true} = {ω | b t ω = true} from rfl, hb.prob,
    ENNReal.toReal_ofReal hp]

/-- `E[[b_t] Y] = p E[Y]` for `Y` a function of coins other than `b_t`. -/
theorem coin_integral_ind_mul [IsProbabilityMeasure μ] (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p)
    {S : Finset ℕ}
    {F : (ℕ → Bool) → ℝ} (hF : DependsOn F (S : Set ℕ)) {t : ℕ} (ht : t ∉ S) :
    ∫ ω, coinInd (b t ω) * F (fun j => b j ω) ∂μ = p * ∫ ω, F (fun j => b j ω) ∂μ := by
  have h := (coin_indep hb hF ht coinInd).symm.integral_mul_eq_mul_integral
    (measurable_of_finite coinInd |>.comp (hb.measurable t)).aestronglyMeasurable
    (coin_measurable hb hF).aestronglyMeasurable
  rw [← coin_integral_ind hb hp t]
  exact h

/-- `E[(1 - [b_t]) Y] = (1 - p) E[Y]` for `Y` a function of coins other than `b_t`. -/
theorem coin_integral_not_mul [IsProbabilityMeasure μ] (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p)
    {S : Finset ℕ}
    {F : (ℕ → Bool) → ℝ} (hF : DependsOn F (S : Set ℕ)) {t : ℕ} (ht : t ∉ S) :
    ∫ ω, (1 - coinInd (b t ω)) * F (fun j => b j ω) ∂μ =
      (1 - p) * ∫ ω, F (fun j => b j ω) ∂μ := by
  have hI := coin_integrable hb hF
  have hJ : Integrable (fun ω => coinInd (b t ω) * F (fun j => b j ω)) μ :=
    coin_integrable hb (dependsOn_coin_mul hF t coinInd)
  simp only [sub_mul, one_mul]
  rw [integral_sub hI hJ, coin_integral_ind_mul hb hp hF ht]

/-- `E[Σ_{j∈S} c_j [b_j]] = p Σ_{j∈S} c_j`. -/
theorem coin_integral_sum [IsProbabilityMeasure μ] (hb : IsBernoulliSeq μ b p) (hp : 0 ≤ p)
    (S : Finset ℕ) (c : ℕ → ℝ) : ∫ ω, ∑ j ∈ S, c j * coinInd (b j ω) ∂μ = p * ∑ j ∈ S, c j := by
  rw [integral_finsetSum]
  · rw [mul_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [integral_const_mul, coin_integral_ind hb hp]
    ring
  · intro j _
    exact coin_integrable hb (F := fun u => c j * coinInd (u j)) (S := {j})
      fun x y h => by dsimp only; rw [h j (by simp)]

/-- The hypotheses are satisfiable: fair coins exist, and `p = 1/2 ≥ 0`. -/
example : ∃ μ : Measure (ℕ → Bool), IsProbabilityMeasure μ ∧
    IsBernoulliSeq μ (fun t ω => ω t) (1 / 2) :=
  exists_isBernoulliSeq (by norm_num) (by norm_num)

end AdamBeyond
end Transformer
