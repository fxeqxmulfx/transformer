/-
# The emergence of clusters in self-attention dynamics — two bounded tokens beside an escaping one

The case of `l:onlyone` (§7 of arXiv:2305.05465v6) where the largest token
runs off to `+∞`: two bounded tokens cannot keep a gap `g > 0`.  The upper
one ends up below `g/2` (`eventually_le_of_bounded`), so the lower one stays
below `-g/2`, which `frequently_lt_of_bounded` excludes.  This is the source's
"if `i ∈ ℬ`, necessarily `x_i(t) → 0`", used for two tokens at once.

The witness for the hypotheses is the symmetric triple `(-u, 0, u)` of
`Section7_Symmetric`, started at `u(0) > 2`, beyond the constant `A` of
`l:auxiliary` for three tokens (`auxiliary_constant_three_lt_two`).

Source: arXiv:2305.05465v6, proof of `l:onlyone`.
-/

import Transformer.Clusters.Section7_OnlyOneEscape
import Transformer.Clusters.Section7_Symmetric

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **Two bounded tokens beside an escaping one.**  If the largest token runs
off to `+∞` and every token is bounded or runs off to `±∞`, no two bounded
tokens keep a gap `g > 0`.

Source: arXiv:2305.05465v6, proof of `l:onlyone`. -/
theorem not_gap_of_bounded (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (L M : Idx (m + 1)) (hmin : ∀ t, 0 ≤ t → ∀ j, x t L ≤ x t j)
    (hmax : ∀ t, 0 ≤ t → ∀ j, x t j ≤ x t M) (hM : Tendsto (fun t => x t M) atTop atTop)
    (htok : ∀ j, (∃ R, ∀ t, 0 ≤ t → |x t j| ≤ R) ∨ Tendsto (fun t => x t j) atTop atTop ∨
      Tendsto (fun t => x t j) atTop atBot)
    (i j : Idx (m + 1)) (hi : ∃ R, ∀ t, 0 ≤ t → |x t i| ≤ R)
    (hj : ∃ R, ∀ t, 0 ≤ t → |x t j| ≤ R) {g : ℝ} (hg : 0 < g) :
    ¬ ∀ t, 0 ≤ t → x t i + g ≤ x t j := by
  intro hgap
  obtain ⟨Ri, hRi⟩ := hi
  obtain ⟨Rj, hRj⟩ := hj
  have hjle := eventually_le_of_bounded x hder M hmax hM j
    (fun t ht => (le_abs_self _).trans (hRj t ht)) (half_pos hg)
  have hder' : ∀ t k, HasDerivAt (fun s => -x s k)
      (∑ l, Perspective.softmaxWeight (fun l' => -x t k * -x t l') l * -x t l) t := by
    intro t k
    convert (hder t k).fun_neg using 1
    simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]
  have htok' : ∀ l, (∀ η, 0 < η → ∀ᶠ t in atTop, -η ≤ -x t l) ∨
      Tendsto (fun t => -x t l) atTop atBot := by
    intro l
    rcases htok l with ⟨R, hR⟩ | h | h
    · exact Or.inl fun η hη => (eventually_le_of_bounded x hder M hmax hM l
        (fun t ht => (le_abs_self _).trans (hR t ht)) hη).mono fun t h => neg_le_neg h
    · exact Or.inr (tendsto_neg_atTop_atBot.comp h)
    · exact Or.inl fun η _ => (h.eventually (eventually_le_atBot η)).mono fun t h => neg_le_neg h
  have hfreq := frequently_lt_of_bounded (fun t k => -x t k) hder' L
    (fun t ht l => neg_le_neg (hmin t ht l)) i (half_pos hg)
    (fun t ht => (neg_le_abs _).trans (hRi t ht)) htok'
  obtain ⟨t, h1, h2, h3⟩ := (hfreq.and_eventually (hjle.and (eventually_ge_atTop 0))).exists
  have := hgap t h3
  linarith

/-- The constant `A` of `l:auxiliary` for three tokens is below `2`:
`A² = 9 e^{-A²}` and `9 e^{-4} < 4`, since `e^4 ≥ 13`. -/
theorem auxiliary_constant_three_lt_two {A : ℝ} (hA : 0 < A)
    (hAeq : A ^ 2 = ((2 + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2)) : A < 2 := by
  by_contra h
  push Not at h
  have h4 : 4 ≤ A ^ 2 := by nlinarith
  have he : Real.exp (-A ^ 2) ≤ Real.exp (-4) := Real.exp_le_exp.2 (by linarith)
  have h13 : 13 ≤ Real.exp 4 := by
    have := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 4 by norm_num)
    linarith
  have hinv : Real.exp (-4) * Real.exp 4 = 1 := by rw [← Real.exp_add]; norm_num
  push_cast at hAeq
  nlinarith [Real.exp_pos (-4)]

/-- **The symmetric triple, started beyond `A`, escapes.**  With `u(0) > 2`,
the scalar curves `(-u, 0, u)` solve `e:Idnonresca`, stay ordered, and the
last one runs off to `+∞`.

Source: arXiv:2305.05465v6, `l:auxiliary`, on the triple of
`Section7_Symmetric`. -/
theorem symTriple_escape (u : ℝ → ℝ) (hu : ∀ t : ℝ, HasDerivAt u (symDrift (u t)) t)
    (hu0 : 2 < u 0) :
    (∀ t k, HasDerivAt (fun s => symTriple (u s) k 0)
      (∑ j, Perspective.softmaxWeight
        (fun l => symTriple (u t) k 0 * symTriple (u t) l 0) j * symTriple (u t) j 0) t) ∧
    (∀ t, 0 ≤ t → IsOrderedConfig (symTriple (u t))) ∧
    Tendsto (fun t => symTriple (u t) 2 0) atTop atTop := by
  have hX := idNonrescaledDynamics_symTriple u hu
  have hord : IsOrderedConfig (symTriple (u 0)) := isOrderedConfig_symTriple (by linarith)
  refine ⟨hasDerivAt_coord _ hX, fun t ht => isOrderedConfig_of_nonneg _ hX hord t ht, ?_⟩
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant (2 + 1) (Nat.succ_pos 2)
  have hA2 := auxiliary_constant_three_lt_two hA hAeq
  obtain ⟨c, hc, hev⟩ := exists_exp_lower_bound A hA hAeq _ hX 2 0
    (by simp only [symTriple, coord_smul_unit1, symSign_two, one_mul]; linarith)
  exact tendsto_atTop_mono' atTop hev (tendsto_exp_atTop.const_mul_atTop hc)

/-- The hypothesis of `auxiliary_constant_three_lt_two` is satisfiable: it is
the constant of `exists_auxiliary_constant` for three tokens. -/
example : ∃ A : ℝ, 0 < A ∧ A ^ 2 = ((2 + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2) :=
  exists_auxiliary_constant _ (Nat.succ_pos 2)

/-- The symmetric triple, as the scalar curves `(-u, 0, u)`: the smallest and
the largest token, the escape of the largest, and the bounded middle token. -/
theorem symTriple_escape_extremes (u : ℝ → ℝ) (hu : ∀ t : ℝ, HasDerivAt u (symDrift (u t)) t)
    (hu0 : 2 < u 0) :
    (∀ t, 0 ≤ t → ∀ j, symTriple (u t) 0 0 ≤ symTriple (u t) j 0) ∧
    (∀ t, 0 ≤ t → ∀ j, symTriple (u t) j 0 ≤ symTriple (u t) (Fin.last 2) 0) ∧
    Tendsto (fun t => symTriple (u t) 0 0) atTop atBot ∧
    ∀ t, |symTriple (u t) 1 0| ≤ 0 := by
  obtain ⟨-, hord, hlim⟩ := symTriple_escape u hu hu0
  refine ⟨fun t ht j => (Fin.zero_le j).lt_or_eq.elim (fun hj => (hord t ht 0 j hj).le)
      (fun hj => by rw [← hj]),
    fun t ht j => (Fin.le_last j).lt_or_eq.elim (fun hj => (hord t ht j _ hj).le)
      (fun hj => by rw [hj]), ?_, fun t => by simp [symTriple, symSign]⟩
  have h : (fun t => symTriple (u t) 0 0) = fun t => -symTriple (u t) 2 0 := by
    funext t; simp [symTriple, symSign]
  rw [h]
  exact tendsto_neg_atTop_atBot.comp hlim

/-- The hypotheses of `eventually_le_of_bounded` and `not_gap_of_bounded` are
satisfiable on the symmetric triple, for a solution `u` of the scalar equation
with `u(0) > 2`: the largest token `u` escapes, the smallest `-u` runs off to
`-∞`, and the middle token is bounded.  The solution `u` is the hypothesis, as
in `Section7_Bounded`: no solution with three distinct tokens is available in
closed form. -/
example (u : ℝ → ℝ) (hu : ∀ t : ℝ, HasDerivAt u (symDrift (u t)) t) (hu0 : 2 < u 0) :
    let x : ℝ → Idx 3 → ℝ := fun t j => symTriple (u t) j 0
    (∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t) ∧
    (∀ t, 0 ≤ t → ∀ j, x t 0 ≤ x t j) ∧ (∀ t, 0 ≤ t → ∀ j, x t j ≤ x t (Fin.last 2)) ∧
    Tendsto (fun t => x t (Fin.last 2)) atTop atTop ∧
    (∀ j, (∃ R, ∀ t, 0 ≤ t → |x t j| ≤ R) ∨ Tendsto (fun t => x t j) atTop atTop ∨
      Tendsto (fun t => x t j) atTop atBot) ∧
    ∃ R, ∀ t, 0 ≤ t → |x t 1| ≤ R := by
  obtain ⟨hder, -, hlim⟩ := symTriple_escape u hu hu0
  obtain ⟨hmin, hmax, hbot, hmid⟩ := symTriple_escape_extremes u hu hu0
  refine ⟨hder, hmin, hmax, hlim, fun j => ?_, ⟨0, fun t _ => hmid t⟩⟩
  fin_cases j
  · exact Or.inr (Or.inr hbot)
  · exact Or.inl ⟨0, fun t _ => hmid t⟩
  · exact Or.inr (Or.inl hlim)

/-- The hypotheses of `frequently_lt_of_bounded` are satisfiable on the
mirrored symmetric triple `(u, 0, -u)`, for a solution `u` of the scalar
equation with `u(0) > 2`: its largest token is the first one. -/
example (u : ℝ → ℝ) (hu : ∀ t : ℝ, HasDerivAt u (symDrift (u t)) t) (hu0 : 2 < u 0) :
    let y : ℝ → Idx 3 → ℝ := fun t j => -symTriple (u t) j 0
    (∀ t k, HasDerivAt (fun s => y s k)
      (∑ j, Perspective.softmaxWeight (fun l => y t k * y t l) j * y t j) t) ∧
    (∀ t, 0 ≤ t → ∀ j, y t j ≤ y t 0) ∧ (∀ t, 0 ≤ t → y t 1 ≤ 0) ∧
    ∀ j, (∀ η, 0 < η → ∀ᶠ t in atTop, -η ≤ y t j) ∨ Tendsto (fun t => y t j) atTop atBot := by
  obtain ⟨hder, -, hlim⟩ := symTriple_escape u hu hu0
  obtain ⟨hmin, -, hbot, hmid⟩ := symTriple_escape_extremes u hu hu0
  refine ⟨fun t k => ?_, fun t ht j => neg_le_neg (hmin t ht j),
    fun t _ => by simp [symTriple, symSign], fun j => ?_⟩
  · convert (hder t k).fun_neg using 1
    simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]
  fin_cases j
  · exact Or.inl fun η _ => ((tendsto_neg_atBot_atTop.comp hbot).eventually
      (eventually_ge_atTop (-η))).mono fun t h => h
  · exact Or.inl fun η hη => Eventually.of_forall fun t => by
      have := hmid t; simp only [abs_nonpos_iff] at this; simp [this, hη.le]
  · exact Or.inr (tendsto_neg_atTop_atBot.comp hlim)

end Clusters
end Transformer
