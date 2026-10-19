module

public import Transformer.ScheduleFree.Section2_GradientDescent

/-!
# Projection onto the comparator ball

arXiv:2405.15682v4, Appendices D.1 and E. The ball projection is the
explicit Euclidean radial projection, including its zero-radius case.
No projection or regret property is postulated as a structure field.
-/

/-!
# Weighted strongly convex surrogate regret

arXiv:2405.15682v4, Appendix E, proof of Theorem 6. We use the
paper's actual projected update and prove its weighted regret bound.
Zero-based `t` corresponds to source weight `t+1` and step `2/(μ(t+2))`.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendices D.1 and E: Euclidean projection onto `‖z‖≤D`. -/
def projectBall (D : ℝ) (v : E) : E :=
  if ‖v‖ ≤ D then v else (D / ‖v‖) • v

theorem norm_projectBall {D : ℝ} (hD : 0 ≤ D) (v : E) : ‖projectBall D v‖ ≤ D := by
  unfold projectBall
  split_ifs with h
  · exact h
  · have hv : 0 < ‖v‖ := lt_of_le_of_lt hD (lt_of_not_ge h)
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hD hv.le),
      div_mul_cancel₀ _ hv.ne']

example (v : E) : ‖projectBall 0 v‖ ≤ 0 := norm_projectBall le_rfl v

/-- Projecting a proposal cannot increase squared distance to an in-ball comparator. -/
theorem projectBall_distance {D : ℝ} (hD : 0 ≤ D) (v u : E) (hu : ‖u‖ ≤ D) :
    ‖projectBall D v - u‖ ^ 2 ≤ ‖v - u‖ ^ 2 := by
  unfold projectBall
  split_ifs with h
  · exact le_rfl
  · have hv : 0 < ‖v‖ := lt_of_le_of_lt hD (lt_of_not_ge h)
    have ha : 0 ≤ D / ‖v‖ := div_nonneg hD hv.le
    have ha1 : D / ‖v‖ ≤ 1 := (div_le_one hv).mpr (le_of_not_ge h)
    have he : D / ‖v‖ * ‖v‖ = D := div_mul_cancel₀ _ hv.ne'
    have hi : inner ℝ v u ≤ ‖v‖ * D :=
      (real_inner_le_norm v u).trans (mul_le_mul_of_nonneg_left hu hv.le)
    have hm := mul_le_mul_of_nonneg_left hi (show 0 ≤ 2 * (1 - D / ‖v‖) by positivity)
    have he2 : D / ‖v‖ * (‖v‖ * D) = D ^ 2 := by rw [← mul_assoc, he]; ring
    rw [norm_sub_sq_real, norm_sub_sq_real, real_inner_smul_left, norm_smul,
      Real.norm_eq_abs, abs_of_nonneg ha, he]
    nlinarith [sq_nonneg (‖v‖ - D)]

example (v : E) : ‖projectBall 1 v - 0‖ ^ 2 ≤ ‖v - 0‖ ^ 2 :=
  projectBall_distance zero_le_one v 0 (by simp)

/-- The projected counterpart of the one-step OGD identity. -/
theorem projected_energy {D : ℝ} (hD : 0 ≤ D) (η : ℝ) (z g u : E) (hu : ‖u‖ ≤ D) :
    ‖projectBall D (z - η • g) - u‖ ^ 2 ≤
      ‖z - u‖ ^ 2 - 2 * η * inner ℝ g (z - u) + η ^ 2 * ‖g‖ ^ 2 := by
  exact (projectBall_distance hD (z - η • g) u hu).trans_eq (descent_energy η z g u)

example (η : ℝ) (z g : E) :
    ‖projectBall 1 (z - η • g) - 0‖ ^ 2 ≤
      ‖z - 0‖ ^ 2 - 2 * η * inner ℝ g (z - 0) + η ^ 2 * ‖g‖ ^ 2 :=
  projected_energy zero_le_one η z g 0 (by simp)

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix E: the strongly convex surrogate loss `ℓ_t`. -/
def surrogate (μ : ℝ) (y g v : E) : ℝ := inner ℝ g v + μ / 2 * ‖v - y‖ ^ 2

/-- Appendix E: the exact first-order model of the surrogate. -/
theorem surrogate_difference (μ : ℝ) (y g z u : E) :
    surrogate μ y g z - surrogate μ y g u =
      inner ℝ (g + μ • (z - y)) (z - u) - μ / 2 * ‖z - u‖ ^ 2 := by
  simp only [surrogate, norm_sub_sq_real, inner_add_left, real_inner_smul_left,
    inner_sub_left, inner_sub_right, real_inner_self_eq_norm_sq]
  rw [real_inner_comm y z, real_inner_comm y u]
  ring

/-- Appendix E: a single weighted projected step, with its telescoping potential. -/
theorem strong_regret_step {μ D C s : ℝ} (hμ : 0 < μ) (hD : 0 ≤ D)
    (hC : 0 ≤ C) (hs : 0 ≤ s) (y g z u : E) (hu : ‖u‖ ≤ D)
    (hbound : ‖g + μ • (z - y)‖ ≤ C) :
    s * (surrogate μ y g z - surrogate μ y g u) ≤
      μ / 4 * (s * (s - 1) * ‖z - u‖ ^ 2 - s * (s + 1) *
        ‖projectBall D (z - (2 / (μ * (s + 1))) • (g + μ • (z - y))) - u‖ ^ 2) +
        C ^ 2 / μ := by
  have he := projected_energy hD (2 / (μ * (s + 1))) z (g + μ • (z - y)) u hu
  have hm := mul_le_mul_of_nonneg_left he
    (by positivity : 0 ≤ μ / 4 * s * (s + 1))
  have hrewrite : μ / 4 * s * (s + 1) *
      (‖z - u‖ ^ 2 - 2 * (2 / (μ * (s + 1))) * inner ℝ (g + μ • (z - y)) (z - u) +
        (2 / (μ * (s + 1))) ^ 2 * ‖g + μ • (z - y)‖ ^ 2) =
      μ / 4 * s * (s + 1) * ‖z - u‖ ^ 2 - s * inner ℝ (g + μ • (z - y)) (z - u) +
        s / (μ * (s + 1)) * ‖g + μ • (z - y)‖ ^ 2 := by
    field_simp
    ring
  rw [hrewrite] at hm
  have hratio : s / (μ * (s + 1)) ≤ 1 / μ := by
    apply (div_le_div_iff₀ (by positivity : 0 < μ * (s + 1)) hμ).mpr
    nlinarith
  have hc : s / (μ * (s + 1)) * ‖g + μ • (z - y)‖ ^ 2 ≤ C ^ 2 / μ := by
    calc
      _ ≤ s / (μ * (s + 1)) * C ^ 2 :=
        mul_le_mul_of_nonneg_left ((sq_le_sq₀ (norm_nonneg _) hC).mpr hbound) (by positivity)
      _ ≤ (1 / μ) * C ^ 2 := mul_le_mul_of_nonneg_right hratio (sq_nonneg C)
      _ = _ := by ring
  rw [surrogate_difference]
  nlinarith

example (z g : E) :
    (0 : ℝ) * (surrogate 1 z g z - surrogate 1 z g 0) ≤
      1 / 4 * (0 * (0 - 1) * ‖z - 0‖ ^ 2 - 0 * (0 + 1) *
        ‖projectBall 1 (z - ((2 : ℝ) / (1 * (0 + 1))) • (g + (1 : ℝ) • (z - z))) - 0‖ ^ 2) +
        ‖g‖ ^ 2 / 1 :=
  strong_regret_step zero_lt_one zero_le_one (norm_nonneg g) le_rfl z g z 0
    (by simp) (by simp)

/-- Appendix E: the actual projected algorithm has regret at most `TC²/μ`. -/
theorem strong_projected_regret {μ D C : ℝ} (hμ : 0 < μ) (hD : 0 ≤ D) (hC : 0 ≤ C)
    (z y g : ℕ → E) (u : E) (hu : ‖u‖ ≤ D)
    (hstep : ∀ t, z (t + 1) = projectBall D
      (z t - (2 / (μ * ((t : ℝ) + 2))) • (g t + μ • (z t - y t))))
    (hbound : ∀ t, ‖g t + μ • (z t - y t)‖ ≤ C) (T : ℕ) :
    (∑ t ∈ Finset.range T, ((t : ℝ) + 1) *
      (surrogate μ (y t) (g t) (z t) - surrogate μ (y t) (g t) u)) ≤ (T : ℝ) * C ^ 2 / μ := by
  have hall (n : ℕ) :
      (∑ t ∈ Finset.range n, ((t : ℝ) + 1) *
        (surrogate μ (y t) (g t) (z t) - surrogate μ (y t) (g t) u)) ≤
      -(μ / 4 * (n : ℝ) * ((n : ℝ) + 1) * ‖z n - u‖ ^ 2) + (n : ℝ) * C ^ 2 / μ := by
    induction n with
    | zero => simp
    | succ n ih =>
      have hs := strong_regret_step hμ hD hC (by positivity : 0 ≤ (n : ℝ) + 1)
        (y n) (g n) (z n) u hu (hbound n)
      have hc : (n : ℝ) + 1 + 1 = (n : ℝ) + 2 := by ring
      rw [hc, ← hstep n] at hs
      simp only [Finset.sum_range_succ, Nat.cast_add, Nat.cast_one]
      simp only [div_eq_mul_inv] at ih hs ⊢
      nlinarith
  have h := hall T
  have hp : 0 ≤ μ / 4 * (T : ℝ) * ((T : ℝ) + 1) * ‖z T - u‖ ^ 2 := by positivity
  linarith

example (T : ℕ) :
    (∑ t ∈ Finset.range T, ((t : ℝ) + 1) *
      (surrogate 1 (0 : ℝ) 0 0 - surrogate 1 (0 : ℝ) 0 0)) ≤ (T : ℝ) * 0 ^ 2 / 1 :=
  strong_projected_regret (E := ℝ) zero_lt_one zero_le_one le_rfl (fun _ => 0) (fun _ => 0)
    (fun _ => 0) 0 (by simp) (by intro t; simp [projectBall]) (by intro t; simp) T

end Transformer.ScheduleFree

end
