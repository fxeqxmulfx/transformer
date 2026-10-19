module

public import Transformer.Homogenized.Section7_SimplexScalarBarriers
public import Transformer.Homogenized.Section7_SimplexRealization

/-!
# The original scalar ODE remains in its realizable simplex domain

ArXiv:2604.01978v1, §7, eq:f_g_def_selfcontained,
eq:gamma_ODE_selfcontained and thm:clustering_random_init.
At every feasible overlap, construct the actual reference simplex
from the initial tokens. Its bounded normalized attention rows
give f,g in [0,1] and f+(n-1)g≥1, hence 0≤b≤2(1-γ).

The proved scalar barriers then give invariant strict admissibility
and monotonicity of the original ODE. Neither a feasible trajectory,
a reference simplex at later times, nor moment bounds are added as
hypotheses. The only moment inputs are the source's original
identities for f and g on actual simplex configurations.
The witness uses uniform attention and a nonconstant scalar solution.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory Set

namespace Transformer.Homogenized

section Ode

variable {d n : ℕ} (hn : 2 ≤ n + 1) (β : ℝ)
  (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (f g : ℝ → ℝ)
  (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
    (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
  (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y →
    ∀ i j, i ≠ j → (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
  {γ₀ : ℝ} (hγ₀ : γ₀ ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1)
  {X₀ : Idx (n + 1) → EucSpace d} (hX₀ : IsSimplexConfig γ₀ X₀)

include hn β ρ hf hg hγ₀ hX₀

/-- At every feasible overlap, the drift bounds follow from the source's
actual attention moments on the explicitly realized simplex.
Source: arXiv:2604.01978v1, §7, eq:f_g_def_selfcontained
and eq:Phi_def_selfcontained. -/
theorem simplexDrift_mem_Icc_of_initial (c : ℝ)
    (hc : c ∈ Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) :
    simplexDrift f g c ∈ Icc (0 : ℝ) (2 * (1 - c)) := by
  let Y := simplexRecenter X₀ γ₀ c
  have hY : IsSimplexConfig c Y := isSimplexConfig_simplexRecenter hn hγ₀ hX₀ hc
  let i : Idx (n + 1) := ⟨0, by omega⟩
  let j : Idx (n + 1) := ⟨1, by omega⟩
  have hij : i ≠ j := by
    intro h
    have hval := congrArg Fin.val h
    norm_num [i, j] at hval
  have hfBound : f c ∈ Icc (0 : ℝ) 1 := by
    have h := integral_sum_attnProb_product_mem_Icc β ρ Y i i
    simp only [← pow_two] at h
    rw [hf c Y hY i] at h
    exact h
  have hgBound : g c ∈ Icc (0 : ℝ) 1 := by
    rw [← hg c Y hY i j hij]
    exact integral_sum_attnProb_product_mem_Icc β ρ Y i j
  have hcolumn := attention_moments_column_bound β ρ Y (f c) (g c) (hf c Y hY) (hg c Y hY)
  apply simplexDrift_mem_Icc hn f g hc hfBound hgBound
  simpa only [Nat.cast_add, Nat.cast_one, add_sub_cancel_right] using hcolumn

/-- The original ODE remains strictly admissible and is nondecreasing.
Source: arXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained
in thm:clustering_random_init. Only the actual original f,g moment
identities and actual initial simplex are used to obtain the drift bounds. -/
theorem simplex_ode_domain_and_monotone (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) :
    (∀ t ∈ Ici (0 : ℝ), γ t ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) ∧
      MonotoneOn γ (Ici (0 : ℝ)) := by
  have hb := simplexDrift_mem_Icc_of_initial hn β ρ f g hf hg hγ₀ hX₀
  exact ⟨fun t ht => scalar_ode_mem_Ioo _ _ hγ₀ _ hb γ hinit hd t ht,
    scalar_ode_monotoneOn _ _ hγ₀ _ hb γ hinit hd⟩

/-- Every ODE value is realized by the explicit simplex in the original
space. Source: arXiv:2604.01978v1, §7, the use of R(γ(t)) in
thm:clustering_random_init. No additional realization premise is required. -/
theorem isSimplexConfig_ode_reference (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : IsSimplexConfig (γ t) (simplexRecenter X₀ γ₀ (γ t)) := by
  have hc := (simplex_ode_domain_and_monotone hn β ρ f g hf hg hγ₀ hX₀ γ hinit hd).1 t ht
  exact isSimplexConfig_simplexRecenter hn hγ₀ hX₀ ⟨hc.1.le, hc.2.le⟩

/-- The original initial overlap is a lower bound at every later time.
Source: arXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained. -/
theorem simplex_ode_initial_le (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : γ₀ ≤ γ t := by
  rw [← hinit]
  exact (simplex_ode_domain_and_monotone hn β ρ f g hf hg hγ₀ hX₀ γ hinit hd).2
    (by simp) ht ht

end Ode

/-- The actual reference path is continuous on the whole nonnegative
time axis. Source: arXiv:2604.01978v1, §7, R(γ(t)) in
thm:clustering_random_init. -/
theorem continuousOn_ode_simplex_reference {d n : ℕ} (X₀ : Idx n → EucSpace d)
    (γ₀ : ℝ) (f g γ : ℝ → ℝ)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) :
    ContinuousOn (fun t => simplexRecenter X₀ γ₀ (γ t)) (Ici (0 : ℝ)) := by
  exact (continuous_simplexRecenter X₀ γ₀).comp_continuousOn
    (fun t ht => (hd t ht).continuousWithinAt)

/-- The nonconstant curve (exp(t)-1)/(exp(t)+1) solves the original
scalar ODE for the actual uniform-attention moments f=g=1/2.
Source: arXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained,
specialized to two tokens and zero attention variance. -/
theorem hasDerivAt_simplex_uniform_attention (t : ℝ) :
    HasDerivAt (fun s : ℝ => (Real.exp s - 1) / (Real.exp s + 1))
      (simplexDrift (fun _ => 1 / 2) (fun _ => 1 / 2)
        ((Real.exp t - 1) / (Real.exp t + 1))) t := by
  have hden : Real.exp t + 1 ≠ 0 := by positivity
  have hd := ((Real.hasDerivAt_exp t).sub_const 1).div
    ((Real.hasDerivAt_exp t).add_const 1) hden
  convert hd using 1
  dsimp [simplexDrift]
  field_simp
  ring

/-- All original moment, simplex and ODE hypotheses hold simultaneously
for the orthonormal pair, the actual zero head, and a nonconstant γ.
In particular this witness supplies the full ODE, not only its initial data. -/
example : ∃ (X : Idx 2 → EucSpace 2) (γ : ℝ → ℝ),
    2 ≤ (2 : ℕ) ∧ IsProbabilityMeasure (Measure.dirac (0 : HeadParam 2)) ∧
    (0 : ℝ) ∈ Ioo (-(1 / ((2 : ℝ) - 1))) 1 ∧ IsSimplexConfig (0 : ℝ) X ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k ^ 2 ∂(Measure.dirac (0 : HeadParam 2))) = (1 / 2 : ℝ)) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k * attnProb 1 θ.2 Y j k
        ∂(Measure.dirac (0 : HeadParam 2))) = (1 / 2 : ℝ)) ∧
    γ 0 = 0 ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ
      (simplexDrift (fun _ => 1 / 2) (fun _ => 1 / 2) (γ t)) (Ici 0) t) ∧
    (0 : ℝ) < γ 1 ∧ (0 : ℝ) ≤ 1 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  let γ : ℝ → ℝ := fun t => (Real.exp t - 1) / (Real.exp t + 1)
  refine ⟨X, γ, le_rfl, inferInstance, by norm_num, ?_, ?_, ?_, ?_, ?_, ?_, by norm_num⟩
  · constructor
    · intro k; simp [X]
    · intro i j hij; simp [X, EuclideanSpace.inner_single_left, hij.symm]
  · intro c Y hY i
    norm_num [attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro c Y hY i j hij
    norm_num [attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · simp [γ]
  · intro t ht
    exact (hasDerivAt_simplex_uniform_attention t).hasDerivWithinAt
  · exact div_pos (sub_pos.mpr (Real.one_lt_exp_iff.mpr (by norm_num))) (by positivity)

end Transformer.Homogenized
