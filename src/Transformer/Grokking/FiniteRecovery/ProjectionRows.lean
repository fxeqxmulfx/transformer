module

public import Transformer.Grokking.FiniteRecovery.SoftsignAlignment

/-! Euclidean projection of disjoint real-coordinate rows. For a complex row
the coordinate set contains every real and imaginary component. This is the
row clip of `experiments/argand/src/engine/row_clip.rs` in real arithmetic. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.ProjectionRows

open Finset SoftsignMomentum
open scoped Classical

variable {ι G : Type*} [Fintype ι]

noncomputable def row (tags : ι → Option G) (key : Option G) : Finset ι := univ.filter (tags · = key)

noncomputable def normOn (indices : Finset ι) (w : ι → ℝ) : ℝ := Real.sqrt (∑ i ∈ indices, (w i)^2)

noncomputable def scale (R n : ℝ) : ℝ := if n ≤ R then 1 else R/n

theorem scale_bounds {R : ℝ} (hR : 0 ≤ R) (n : ℝ) : 0 ≤ scale R n ∧ scale R n ≤ 1 := by
  unfold scale
  split_ifs with h
  · norm_num
  · have hn := lt_of_le_of_lt hR (lt_of_not_ge h)
    exact ⟨div_nonneg hR hn.le, (div_le_one hn).2 (lt_of_not_ge h).le⟩

noncomputable def project (tags : ι → Option G) (R : ℝ) (w : ι → ℝ) (i : ι) : ℝ :=
  match tags i with
  | none => w i
  | some b => scale R (normOn (row tags (some b)) w)*w i

theorem abs_project_le (tags : ι → Option G) {R : ℝ} (hR : 0 ≤ R) (w : ι → ℝ) (i : ι) :
    |project tags R w i| ≤ |w i| := by
  unfold project
  split
  · exact le_rfl
  · rw [abs_mul,abs_of_nonneg (scale_bounds hR _).1]
    exact mul_le_of_le_one_left (abs_nonneg _) (scale_bounds hR _).2

/-- Each selected output row has Euclidean norm at most the prescribed radius
immediately after projection, including for a zero row or zero radius. -/
theorem norm_project_le (tags : ι → Option G) {R : ℝ} (hR : 0 ≤ R) (w : ι → ℝ) (b : G) :
    normOn (row tags (some b)) (project tags R w) ≤ R := by
  let I := row tags (some b)
  let n := normOn I w
  let a := scale R n
  have ha := (scale_bounds hR n).1
  have he : (∑ i ∈ I, (project tags R w i)^2) = a^2 * ∑ i ∈ I, (w i)^2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    have htag := (Finset.mem_filter.1 hi).2
    simp only [project,htag,mul_pow]
    rfl
  have hnorm : normOn I (project tags R w) = a*n := by
    change Real.sqrt _ = _
    rw [he,Real.sqrt_mul (sq_nonneg _),Real.sqrt_sq ha]
    rfl
  change normOn I (project tags R w) ≤ _
  rw [hnorm]
  by_cases h : n ≤ R
  · simp [a,scale,h]
  · have hpos := hR.trans_lt (lt_of_not_ge h)
    simp [a,scale,h,hpos.ne']

omit [Fintype ι] in
/-- Expanding the two squared distances and using Cauchy–Schwarz shows that a
radial projection cannot increase distance to any feasible representative. -/
theorem row_error_le (I : Finset ι) {R : ℝ} (hR : 0 ≤ R) (u w : ι → ℝ) (hu : normOn I u ≤ R) :
    ∑ i ∈ I, (scale R (normOn I w)*w i-u i)^2 ≤ ∑ i ∈ I, (w i-u i)^2 := by
  by_cases hin : normOn I w ≤ R
  · simp [scale,hin]
  let W := ∑ i ∈ I, (w i)^2
  let U := ∑ i ∈ I, (u i)^2
  let D := ∑ i ∈ I, w i*u i
  let n := normOn I w
  let a := scale R n
  have hW : 0 ≤ W := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hU : 0 ≤ U := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hn : 0 < n := hR.trans_lt (lt_of_not_ge hin)
  have ha : a = R/n := ite_eq_right hin
  have hn2 : n^2 = W := Real.sq_sqrt hW
  have hu2 : U ≤ R^2 := (Real.sq_sqrt hU).symm.le.trans
    ((sq_le_sq₀ (Real.sqrt_nonneg _) hR).2 hu)
  have hD : D ≤ n*R := by
    by_cases h : 0 ≤ D
    · apply (sq_le_sq₀ h (mul_nonneg hn.le hR)).1
      rw [mul_pow,hn2]
      exact (Finset.sum_mul_sq_le_sq_mul_sq I w u).trans (mul_le_mul_of_nonneg_left hu2 hW)
    · exact (lt_of_not_ge h).le.trans (mul_nonneg hn.le hR)
  have haW : a*W=n*R := by rw [ha,← hn2]; field_simp
  have hdot : 0 ≤ a*W-D := sub_nonneg.mpr (haW.symm ▸ hD)
  have hone : 0 ≤ 1-a := sub_nonneg.mpr (scale_bounds hR n).2
  have expand (z : ℝ) : ∑ i ∈ I, (z*w i-u i)^2 = z^2*W-2*z*D+U := by
    dsimp [W,D,U]
    simp only [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun _ _ => by ring
  change (∑ i ∈ I, (a*w i-u i)^2) ≤ _
  rw [expand,show (∑ i ∈ I, (w i-u i)^2) = 1^2*W-2*1*D+U from by simpa using expand 1]
  nlinarith [mul_nonneg hone hdot, mul_nonneg (sq_nonneg (1-a)) hW]

/-- Whole-vector squared error inherits the per-row inequality. Coordinates
outside every selected row, including a tied readout, contribute exactly their
original error. -/
theorem projected_error_le [Fintype G] (tags : ι → Option G) {R : ℝ} (hR : 0 ≤ R) (u w : ι → ℝ)
    (hu : ∀ b, normOn (row tags (some b)) u ≤ R) :
    squaredError u (project tags R w) ≤ squaredError u w := by
  have partition (f : ι → ℝ) : (∑ b : Option G, ∑ i ∈ row tags b, f i) = ∑ i, f i := by
    simp_rw [row,Finset.sum_filter]
    rw [Finset.sum_comm]
    simp
  unfold squaredError
  rw [← partition, ← partition (fun i => (w i-u i)^2)]
  apply Finset.sum_le_sum
  intro b _
  cases b with
  | none =>
    exact le_of_eq (Finset.sum_congr rfl (fun i hi => by
      have he := (Finset.mem_filter.1 hi).2
      simp [project,he]))
  | some b =>
    apply le_trans _ (row_error_le _ hR u w (hu b))
    exact le_of_eq (Finset.sum_congr rfl (fun i hi => by
      have he := (Finset.mem_filter.1 hi).2
      simp [project,he]))

example := row_error_le (Finset.univ : Finset (Fin 2)) (R := 2) (by norm_num)
  (fun _ => 0) (fun _ => 3) (by norm_num [normOn])

example := projected_error_le (fun i : Fin 2 => if i=0 then some true else none)
  (R := 2) (by norm_num) (fun _ => 0) (fun _ => 3) (by intro; simp [normOn])

end Transformer.Grokking.FiniteRecovery.ProjectionRows
