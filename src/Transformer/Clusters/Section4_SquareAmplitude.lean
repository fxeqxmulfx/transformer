import Transformer.Clusters.Section4_SquareScalar
import Mathlib.Topology.Order.MonotoneConvergence

/-!
# A positive limiting amplitude for the rescaled square

Source: arXiv:2305.05465v6, §4 `c:codim` and the rescaling
preceding `e:Rres`. For `V = I₂`, the rescaled square amplitude is
`r(t) = exp(-t) u(t)`, where the actual scalar equation is
`u' = u tanh(u²)`. Starting at two gives `r(t) ≥ 1` for `t ≥ 0`.
The upper drift bound makes `r` nonincreasing on that half-line,
so it has a limit `q ≥ 1`.

The proof extends the amplitude constantly to negative times solely
to apply the library's order convergence theorem. The genuine scalar
solution and its rescaling remain defined by the original equations.
No convergence property is included in their definitions.
-/

open Real Filter Topology
namespace Transformer.Clusters

/-- The literal rescaling of a four-corner scalar solution for `V = I₂`.
Source: arXiv:2305.05465v6, the change of variables before `e:Rres`. -/
noncomputable def squareAmplitude (u : ℝ → ℝ) (t : ℝ) : ℝ := Real.exp (-t) * u t

/-- Differentiating the actual rescaled amplitude.
Source: arXiv:2305.05465v6, `e:Rres`, with the square scalar equation. -/
theorem hasDerivAt_squareAmplitude {u : ℝ → ℝ} {t : ℝ}
    (hu : HasDerivAt u (squareScalarDrift (u t)) t) :
    HasDerivAt (squareAmplitude u)
      (Real.exp (-t) * (squareScalarDrift (u t) - u t)) t := by
  have he : HasDerivAt (fun s : ℝ => Real.exp (-s)) (-Real.exp (-t)) t := by
    simpa only [Pi.neg_apply, mul_neg, mul_one] using ((hasDerivAt_id' (x := t)).neg.exp)
  unfold squareAmplitude
  convert he.fun_mul hu using 1
  ring

example : ∃ u : ℝ → ℝ, HasDerivAt u (squareScalarDrift (u 0)) 0 := by
  obtain ⟨u, _, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu 0⟩

/-- Rescaling an actual scalar solution gives a continuous amplitude on all real times.
Source: arXiv:2305.05465v6, `p:wellposednessrescaledparticles`,
the regularity of the genuine four-corner specialization. -/
theorem continuous_squareAmplitude (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) :
    Continuous (squareAmplitude u) :=
  continuous_iff_continuousAt.mpr (fun t => (hasDerivAt_squareAmplitude (hu t)).continuousAt)

example : ∃ u : ℝ → ℝ, ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := by
  obtain ⟨u, _, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu⟩

/-- The unrescaled amplitude of the future square is strictly positive.
Source: arXiv:2305.05465v6, §4 `c:codim`, the lower affine
comparison for the actual scalar solution started at two. -/
theorem squareScalar_solution_pos (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) : 0 < u t := by
  exact lt_of_lt_of_le (by positivity) (one_add_exp_le_squareScalar_solution u hu h0 ht)

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, h0, hu, zero_le_one⟩

/-- A square started at amplitude two stays away from zero after rescaling.
Source: arXiv:2305.05465v6, §4 `c:codim`, the lower bound for its
four-corner test obtained from the actual scalar drift. -/
theorem one_le_squareAmplitude (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) : 1 ≤ squareAmplitude u t := by
  have hbound := mul_le_mul_of_nonneg_left
    (one_add_exp_le_squareScalar_solution u hu h0 ht) (Real.exp_pos (-t)).le
  have hexp : Real.exp (-t) * Real.exp t = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  rw [mul_add, mul_one, hexp] at hbound
  dsimp only [squareAmplitude]
  linarith [Real.exp_pos (-t)]

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, h0, hu, zero_le_one⟩

/-- The derivative of the rescaled square amplitude is nonpositive for future times.
Source: arXiv:2305.05465v6, `e:Rres`, the square specialization
of the convex attention average. -/
theorem squareAmplitude_derivative_nonpos (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) :
    Real.exp (-t) * (squareScalarDrift (u t) - u t) ≤ 0 := by
  have hut : 0 ≤ u t := by
    have hb := one_add_exp_le_squareScalar_solution u hu h0 ht
    linarith [Real.exp_pos t]
  exact mul_nonpos_of_nonneg_of_nonpos (Real.exp_pos _).le
    (sub_nonpos.mpr (squareScalarDrift_le_self hut))

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, h0, hu, zero_le_one⟩

/-- The genuine rescaled amplitude is nonincreasing on the nonnegative half-line.
Source: arXiv:2305.05465v6, §4 `c:codim`, integrating the
nonpositive derivative of the actual four-corner solution. -/
theorem antitoneOn_squareAmplitude (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2) :
    AntitoneOn (squareAmplitude u) (Set.Ici 0) := by
  apply antitoneOn_of_hasDerivWithinAt_nonpos (convex_Ici (0 : ℝ))
    (HasDerivAt.continuousOn (fun t _ => hasDerivAt_squareAmplitude (hu t)))
    (fun t _ => (hasDerivAt_squareAmplitude (hu t)).hasDerivWithinAt)
  intro t ht
  exact squareAmplitude_derivative_nonpos u hu h0 (interior_subset ht)

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := exists_squareScalar_solution 2

/-- The future rescaled amplitude never exceeds its initial value two.
Source: arXiv:2305.05465v6, §4 `c:codim`, the upper square
amplitude bound from the actual rescaled equation. -/
theorem squareAmplitude_le_two (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) : squareAmplitude u t ≤ 2 := by
  have hb := antitoneOn_squareAmplitude u hu h0
    (show (0 : ℝ) ∈ Set.Ici 0 from by simp) ht ht
  simpa only [squareAmplitude, neg_zero, Real.exp_zero, one_mul, h0] using hb

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, h0, hu, zero_le_one⟩

/-- The actual rescaled square converges to a strictly positive amplitude.
Source: arXiv:2305.05465v6, §4 `c:codim`. This is a property
proved for its four-corner test, not an assumption on a token curve. -/
theorem exists_squareAmplitude_limit (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2) :
    ∃ q : ℝ, 1 ≤ q ∧ Tendsto (squareAmplitude u) atTop (nhds q) := by
  let f : ℝ → ℝ := fun t => squareAmplitude u (max 0 t)
  have hlower (t : ℝ) : 1 ≤ f t :=
    one_le_squareAmplitude u hu h0 (le_max_left 0 t)
  have hbdd : BddBelow (Set.range f) := by
    refine ⟨1, ?_⟩
    rintro _ ⟨t, rfl⟩
    exact hlower t
  have hanti : Antitone f := by
    intro a b hab
    exact antitoneOn_squareAmplitude u hu h0 (le_max_left 0 a) (le_max_left 0 b)
      (max_le_max le_rfl hab)
  refine ⟨⨅ t, f t, le_ciInf hlower, ?_⟩
  apply (tendsto_atTop_ciInf hanti hbdd).congr'
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
  exact congrArg (squareAmplitude u) (max_eq_right ht)

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := exists_squareScalar_solution 2

end Transformer.Clusters
