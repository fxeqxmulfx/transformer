module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.KernelBasis.Basic

/-!
# A checked basis covers every feasible forward-query direction

Source: `review/convex/moments.md`, CM81. All universally forced rows are
retained in the equality matrix. A finite basis certificate then represents
exactly the directions admitted by CM80, including degenerate patterns.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.KernelBasis

variable {D R J : Type*} [Fintype D] [Fintype R] [Fintype J]

/-- Keep every original row index; non-forced rows contribute zero to the
equality matrix only. They remain constraints in all feasibility checks, CM81. -/
noncomputable def equalityRows (normal : R → D → ℝ) (middle : D → ℝ) : R → D → ℝ := by
  classical
  exact fun r d => if dot (normal r) middle = 0 then normal r d else 0

omit [Fintype R] in
/-- The equality matrix tests precisely its retained forced row, CM81. -/
theorem equality_row_dot (normal : R → D → ℝ) (middle v : D → ℝ) (r : R) :
    dot (equalityRows normal middle r) v =
      if dot (normal r) middle = 0 then dot (normal r) v else 0 := by
  classical
  by_cases h : dot (normal r) middle = 0
  · rw [ite_eq_left h]
    unfold dot
    simp only [equalityRows, h, ↓reduceIte]
  · rw [ite_eq_right h]
    unfold dot
    simp only [equalityRows, h, ↓reduceIte, zero_mul, Finset.sum_const_zero]

/-- CM80 certificates turn the finite equality matrix into the exact global
forced-row conditions, CM81. No geometric completeness premise is assumed. -/
theorem equality_kernel_iff {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (h : ∀ r, Projection.Certificate normal (normal r) (projected r) (dual r))
    (v : D → ℝ) :
    (∀ r, dot (equalityRows normal (centre projected) r) v = 0) ↔
      ∀ r, (∀ z, Feasible normal z → dot (normal r) z = 0) → dot (normal r) v = 0 := by
  constructor
  · intro hv r hz
    have hc := hz _ (centre_feasible (fun s => (h s).2.1))
    have hr := hv r
    rw [equality_row_dot, ite_eq_left hc] at hr
    exact hr
  · intro hv r
    rw [equality_row_dot]
    split_ifs with hc
    · exact hv r ((centre_zero_iff h r).mp hc)
    · rfl

/-- The finite checked basis represents exactly every direction admitting
two-sided feasible queries at the certified centre, CM81. This is full kernel
coverage, not coverage of a sampled collection of successful queries. -/
theorem query_directions_iff [DecidableEq D] [DecidableEq J]
    {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (hp : ∀ r, Projection.Certificate normal (normal r) (projected r) (dual r))
    {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (hb : Certificate (equalityRows normal (centre projected)) basis free recovery) (v : D → ℝ) :
    (∃ step : ℝ, 0 < step ∧ Feasible normal (shift (centre projected) v step) ∧
      Feasible normal (shift (centre projected) v (-step))) ↔
      ∃ coefficients, v = combine basis coefficients := by
  rw [← centre_query_iff hp v, ← kernel_iff hb v]
  exact (equality_kernel_iff hp v).symm

/-- Every full basis combination supplies feasible centre queries, CM81. -/
theorem basis_queries [DecidableEq D] [DecidableEq J]
    {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (hp : ∀ r, Projection.Certificate normal (normal r) (projected r) (dual r))
    {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (hb : Certificate (equalityRows normal (centre projected)) basis free recovery)
    (coefficients : J → ℝ) :
    ∃ step : ℝ, 0 < step ∧
      Feasible normal (shift (centre projected) (combine basis coefficients) step) ∧
      Feasible normal (shift (centre projected) (combine basis coefficients) (-step)) := by
  exact (query_directions_iff hp hb _).mpr ⟨coefficients, rfl⟩

/-- The same complete basis works after any positive centre mixture of a
feasible iterate, CM81. All original inequalities still bound the query radius. -/
theorem mixture_basis_queries [DecidableEq D] [DecidableEq J]
    {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (hp : ∀ r, Projection.Certificate normal (normal r) (projected r) (dual r))
    {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (hb : Certificate (equalityRows normal (centre projected)) basis free recovery)
    {point : D → ℝ} (hc : Feasible normal point) (coefficients : J → ℝ)
    {amount : ℝ} (ha : 0 < amount) (hd : amount ≤ 1) :
    ∃ step : ℝ, 0 < step ∧
      Feasible normal (shift (mixture point (centre projected) amount)
        (combine basis coefficients) step) ∧
      Feasible normal (shift (mixture point (centre projected) amount)
        (combine basis coefficients) (-step)) := by
  exact mixture_queries hp hc _
    ((equality_kernel_iff hp _).mp (combine_kernel hb coefficients)) ha hd

/-- The checked diagonal basis also satisfies the actual centre-derived
equality matrix, CM81. Both original row indices remain present. -/
theorem diagonal_equality_certificate :
    Certificate (equalityRows diagonalNormal (centre (fun _ : Bool => 0)))
      (fun _ : Unit => fun _ => 1) (fun _ => true) diagonalRecovery := by
  have he : equalityRows diagonalNormal (centre (fun _ : Bool => 0)) = diagonalNormal := by
    funext r d
    have hz : dot (diagonalNormal r) (centre (fun _ : Bool => 0)) = 0 := by
      simp [centre, dot]
    simp only [equalityRows, hz, ↓reduceIte]
  rw [he]
  exact diagonal_certificate

example := equality_kernel_iff diagonal_normal_certificates (fun _ => 3)
example := query_directions_iff diagonal_normal_certificates diagonal_equality_certificate (fun _ => 3)
example := basis_queries diagonal_normal_certificates diagonal_equality_certificate (fun _ => 3)
example := mixture_basis_queries diagonal_normal_certificates diagonal_equality_certificate
  (point := 0) (amount := 1 / 2) (by norm_num [Feasible, dot]) (fun _ => 3)
  (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.KernelBasis
