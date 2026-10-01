/-
# A common period for rational rotation angles

arXiv:2506.16055v3, Appendix F, `sec:sinusoidal_pes` and `sec:rope_pes`.
Only the finitely many angles used by a model's coordinates need a common
period. Twice the product of their rational denominators suffices.
-/

import Transformer.CRASP.PositionalTransformers

namespace Transformer.CRASP

/-- Clearing a rational angle's denominator gives a full-turn multiple (F). -/
theorem rational_full_turn (q : ℚ) (D : ℕ) (hD : q.den ∣ D) :
    ∃ z : ℤ, ((2 * D : ℕ) : ℝ) * (Real.pi * q) = (z : ℝ) * (2 * Real.pi) := by
  obtain ⟨n, rfl⟩ := hD
  refine ⟨(n : ℤ) * q.num, ?_⟩
  have h : (q : ℝ) * (q.den : ℝ) = (q.num : ℝ) := by
    exact_mod_cast q.mul_den_eq_num
  push_cast
  calc
    2 * ((q.den : ℝ) * n) * (Real.pi * q) =
        (q * q.den) * n * (2 * Real.pi) := by ring
    _ = (q.num : ℝ) * n * (2 * Real.pi) := by rw [h]
    _ = (n : ℝ) * q.num * (2 * Real.pi) := by ring

/-- Trigonometric coordinates depend only on the residue of the position (F). -/
theorem trig_mod_period (θ : ℝ) (M : ℕ) (z : ℤ)
    (hM : (M : ℝ) * θ = (z : ℝ) * (2 * Real.pi)) (i : ℕ) :
    Real.cos ((i : ℝ) * θ) = Real.cos ((i % M : ℕ) * θ) ∧
      Real.sin ((i : ℝ) * θ) = Real.sin ((i % M : ℕ) * θ) := by
  have hi : (i : ℝ) = (i % M : ℕ) + (M : ℝ) * (i / M : ℕ) := by
    exact_mod_cast (Nat.mod_add_div i M).symm
  have he : (i : ℝ) * θ = (i % M : ℕ) * θ +
      (((i / M : ℕ) : ℤ) * z : ℤ) * (2 * Real.pi) := by
    simp only [Int.cast_mul, Int.cast_natCast]
    rw [hi]
    calc
      ((i % M : ℕ) + (M : ℝ) * (i / M : ℕ)) * θ =
          (i % M : ℕ) * θ + (i / M : ℕ) * ((M : ℝ) * θ) := by ring
      _ = _ := by rw [hM]; ring
  rw [he, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
  exact ⟨rfl, rfl⟩

/-- Every finite-dimensional rational rotation has a positive common period.
Source: arXiv:2506.16055v3, Appendix F, periodicity assumption preceding
`thm:rtfr_eq_tlclmod` and `thm:rtfr_to_TLClmod`. -/
theorem exists_rotation_period (d : ℕ) (θ : ℕ → ℝ)
    (hθ : ∃ q : ℕ → ℚ, ∀ c, θ c = Real.pi * q c) :
    ∃ M : ℕ, 0 < M ∧ ∀ (i : ℕ) (v : Fin d → ℝ),
      rotate d θ i v = rotate d θ (i % M) v := by
  classical
  obtain ⟨q, hq⟩ := hθ
  let D := ∏ c : Fin d, (q (c.val / 2)).den
  have hD : 0 < D := Finset.prod_pos (fun c _ => (q (c.val / 2)).den_pos)
  refine ⟨2 * D, by omega, fun i v => ?_⟩
  funext c
  obtain ⟨z, hz⟩ := rational_full_turn (q (c.val / 2)) D
    (Finset.dvd_prod_of_mem (fun c : Fin d => (q (c.val / 2)).den) (Finset.mem_univ c))
  rw [← hq] at hz
  obtain ⟨hc, hs⟩ := trig_mod_period (θ (c.val / 2)) (2 * D) z hz i
  simp only [rotate, hc, hs]

/-- The rationality and common-turn hypotheses have zero-angle witnesses (F). -/
example : (∃ q : ℕ → ℚ, ∀ c : ℕ, (0 : ℝ) = Real.pi * q c) ∧
    (2 : ℝ) * 0 = (0 : ℤ) * (2 * Real.pi) := by
  exact ⟨⟨fun _ => 0, fun _ => by simp⟩, by simp⟩

end Transformer.CRASP
