/-
# Original key Gram contractions of actual query score derivatives

ArXiv:2604.01978v1, §7, the query-block trace computation
preceding `eq:estimate_bound_h`.
The actual bilinear Hessian and linear gradient products are expanded
on the original finite score coordinates. Summing over original key
feature coordinates gives their exact original key Gram contraction.

The actual original key Gram entries are bounded using the proved
coordinate Cauchy-Schwarz inequality and the sum of their squared norms.
Applying the proved summed softmax derivative estimates then preserves
the constants six and four, and keeps the original softmax factors.
Every estimate is pointwise, before taking Gaussian expectations.
-/

import Transformer.Homogenized.Section7_GaussianAttentionKeyBound

open scoped BigOperators

namespace Transformer.Homogenized

/-- An actual continuous bilinear form expands on the original finite
score axes. Source: arXiv:2604.01978v1, §7, the query Hessian
coordinate sums preceding `eq:estimate_bound_h`. -/
theorem bilinear_pi_coordinate_expansion {n : ℕ}
    (T : ContinuousMultilinearMap ℝ (fun _ : Fin 2 => Idx n → ℝ) ℝ) (u v : Idx n → ℝ) :
    T ![u, v] = ∑ r, ∑ s, (u r * v s) * T ![Pi.single r 1, Pi.single s 1] := by
  have hu0 (x y z : Idx n → ℝ) : Function.update ![x, y] (0 : Fin 2) z = ![z, y] := by
    ext t
    fin_cases t <;> simp
  have hu1 (x y z : Idx n → ℝ) : Function.update ![x, y] (1 : Fin 2) z = ![x, z] := by
    ext t
    fin_cases t <;> simp
  have hs0 (g : Idx n → Idx n → ℝ) (v : Idx n → ℝ) :
      T ![∑ r, g r, v] = ∑ r, T ![g r, v] := by
    change T.toMultilinearMap ![∑ r, g r, v] = ∑ r, T.toMultilinearMap ![g r, v]
    simpa only [hu0] using T.toMultilinearMap.map_update_sum Finset.univ (0 : Fin 2) g ![0, v]
  have hs1 (u : Idx n → ℝ) (g : Idx n → Idx n → ℝ) :
      T ![u, ∑ s, g s] = ∑ s, T ![u, g s] := by
    change T.toMultilinearMap ![u, ∑ s, g s] = ∑ s, T.toMultilinearMap ![u, g s]
    simpa only [hu1] using T.toMultilinearMap.map_update_sum Finset.univ (1 : Fin 2) g ![u, 0]
  have hv (x : Idx n → ℝ) : (∑ r : Idx n, x r • Pi.single r 1) = x := by
    ext r
    simp [Pi.single_apply]
  calc
    T ![u, v] = T ![∑ r, u r • Pi.single r 1, ∑ s, v s • Pi.single s 1] := by simp only [hv]
    _ = ∑ r, T ![u r • Pi.single r 1, ∑ s, v s • Pi.single s 1] := hs0 _ _
    _ = _ := by
      apply Finset.sum_congr rfl
      intro r hr
      rw [hs1]
      exact Finset.sum_congr rfl fun s hs => by rw [multilinear_two_smul_apply]; rfl

/-- Summing the actual bilinear forms over original feature coordinates
contracts their actual score axes with the original key Gram entries.
Source: arXiv:2604.01978v1, §7, the trace before `eq:estimate_bound_h`. -/
theorem sum_bilinear_key_gram {d n : ℕ}
    (T : ContinuousMultilinearMap ℝ (fun _ : Fin 2 => Idx n → ℝ) ℝ) (G : Idx n → Fin d → ℝ) :
    (∑ q : Fin d, T ![(fun r => G r q), (fun s => G s q)]) =
      ∑ r, ∑ s, (∑ q : Fin d, G r q * G s q) * T ![Pi.single r 1, Pi.single s 1] := by
  have he : (∑ q : Fin d, T ![(fun r => G r q), (fun s => G s q)]) =
      ∑ q : Fin d, ∑ r, ∑ s, (G r q * G s q) * T ![Pi.single r 1, Pi.single s 1] :=
    Finset.sum_congr rfl fun q hq => bilinear_pi_coordinate_expansion T _ _
  rw [he]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r hr
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro s hs
  rw [Finset.sum_mul]

/-- The original finite coordinate expansion of an actual linear form.
Source: arXiv:2604.01978v1, §7, the query first derivatives
used before `eq:estimate_bound_h`. -/
theorem linear_pi_coordinate_expansion {n : ℕ} (L : (Idx n → ℝ) →L[ℝ] ℝ) (v : Idx n → ℝ) :
    L v = ∑ r, v r * L (Pi.single r 1) := by
  have hv : (∑ r : Idx n, v r • Pi.single r 1) = v := by
    ext r
    simp [Pi.single_apply]
  rw [← hv, map_sum]
  simp only [map_smul, smul_eq_mul]
  simp [Pi.single_apply]

/-- The actual pair of key-weighted linear forms contracts with the
original finite key Gram matrix after summing feature coordinates.
Source: arXiv:2604.01978v1, §7, the mixed trace in `eq:estimate_bound_h`. -/
theorem sum_linear_key_gram {d n : ℕ} (L M : (Idx n → ℝ) →L[ℝ] ℝ) (G : Idx n → Fin d → ℝ) :
    (∑ q : Fin d, L (fun r => G r q) * M (fun s => G s q)) =
      ∑ r, ∑ s, (∑ q : Fin d, G r q * G s q) * L (Pi.single r 1) * M (Pi.single s 1) := by
  have he : (∑ q : Fin d, L (fun r => G r q) * M (fun s => G s q)) =
      ∑ q : Fin d, ∑ r, ∑ s, (G r q * G s q) * L (Pi.single r 1) * M (Pi.single s 1) := by
    apply Finset.sum_congr rfl
    intro q hq
    rw [linear_pi_coordinate_expansion, linear_pi_coordinate_expansion, Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun r hr => Finset.sum_congr rfl fun s hs => by ring
  rw [he, Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r hr
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro s hs
  simp only [Finset.sum_mul]

/-- Every actual original key Gram entry is bounded by the sum of
all original key squared norms, by the actual Cauchy-Schwarz estimate.
Source: arXiv:2604.01978v1, §7, the estimate in `eq:estimate_bound_h`. -/
theorem abs_key_gram_entry_le_total_squares {d n : ℕ} (G : Idx n → Fin d → ℝ) (r s : Idx n) :
    |∑ q : Fin d, G r q * G s q| ≤ ∑ m : Idx n, ∑ q : Fin d, G m q ^ 2 := by
  have h0 (m : Idx n) : 0 ≤ ∑ q : Fin d, G m q ^ 2 :=
    Finset.sum_nonneg fun q hq => sq_nonneg _
  have hr := Finset.single_le_sum (fun m hm => h0 m) (Finset.mem_univ r)
  have hs := Finset.single_le_sum (fun m hm => h0 m) (Finset.mem_univ s)
  have hc := two_mul_abs_sum_coordinate_products_le (G r) (G s)
  linarith

/-- The actual query Hessian coordinate sum has constant six and
retains its original softmax factor and the actual key squared norms.
Source: arXiv:2604.01978v1, §7, the Hessian terms in `eq:estimate_bound_h`. -/
theorem abs_sum_query_softmax_hessians_le {d n : ℕ} (hn : 0 < n) (u : Idx n → ℝ) (k : Idx n)
    (G : Idx n → Fin d → ℝ) :
    |∑ q : Fin d, iteratedFDeriv ℝ 2 (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u
      ![(fun r => G r q), (fun s => G s q)]| ≤
      6 * (∑ m : Idx n, ∑ q : Fin d, G m q ^ 2) * Perspective.softmaxWeight u k := by
  rw [sum_bilinear_key_gram]
  exact abs_sum_matrix_softmax_hessian_le hn u k _ _ (abs_key_gram_entry_le_total_squares G)

/-- The actual mixed query derivative sum has constant four and
retains both original softmax factors and actual key squared norms.
Source: arXiv:2604.01978v1, §7, the mixed terms in `eq:estimate_bound_h`. -/
theorem abs_sum_query_softmax_gradient_products_le {d n : ℕ} (hn : 0 < n) (u v : Idx n → ℝ) (k l : Idx n)
    (G : Idx n → Fin d → ℝ) :
    |∑ q : Fin d,
      fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w k) u (fun r => G r q) *
        fderiv ℝ (fun w : Idx n → ℝ => Perspective.softmaxWeight w l) v (fun s => G s q)| ≤
      4 * (∑ m : Idx n, ∑ q : Fin d, G m q ^ 2) *
        Perspective.softmaxWeight u k * Perspective.softmaxWeight v l := by
  rw [sum_linear_key_gram]
  exact abs_sum_matrix_softmax_gradients_le hn hn u v k l _ _ (abs_key_gram_entry_le_total_squares G)

/-- Two nonzero original keys have total squared norm five and
produce a genuine biased-softmax Hessian contraction of `-15/32`,
while satisfying the positive score count and original Gram bounds. -/
example : ∃ (G : Idx 2 → Fin 2 → ℝ) (u : Idx 2 → ℝ), (0 : ℕ) < 2 ∧
    (∑ m : Idx 2, ∑ q : Fin 2, G m q ^ 2) = 5 ∧
    (∀ r s, |∑ q : Fin 2, G r q * G s q| ≤ 5) ∧
    (∑ q : Fin 2, iteratedFDeriv ℝ 2 (fun w : Idx 2 → ℝ => Perspective.softmaxWeight w 0) u
      ![(fun r => G r q), (fun s => G s q)]) = -(15 / 32 : ℝ) := by
  let G : Idx 2 → Fin 2 → ℝ := ![![1, 0], ![0, 2]]
  let u : Idx 2 → ℝ := ![Real.log 3, 0]
  have hG : (∑ m : Idx 2, ∑ q : Fin 2, G m q ^ 2) = 5 := by
    norm_num [G, Fin.sum_univ_two]
  have hw0 : Perspective.softmaxWeight u (0 : Idx 2) = (3 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, u, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  have hw1 : Perspective.softmaxWeight u (1 : Idx 2) = (1 / 4 : ℝ) := by
    norm_num [Perspective.softmaxWeight, u, Real.exp_log (by norm_num : (0 : ℝ) < 3), Fin.sum_univ_two]
  refine ⟨G, u, by norm_num, hG, ?_, ?_⟩
  · intro r s
    have h := abs_key_gram_entry_le_total_squares G r s
    rw [hG] at h
    exact h
  · rw [sum_bilinear_key_gram]
    simp only [Fin.sum_univ_two, iteratedFDeriv_two_softmaxWeight (m := 2) (by norm_num)]
    norm_num [G, hw0, hw1]

end Transformer.Homogenized

