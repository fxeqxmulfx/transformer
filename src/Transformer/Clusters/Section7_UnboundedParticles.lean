/-
# The emergence of clusters in self-attention dynamics — `l:unboundedparticles`

§7 of arXiv:2305.05465v6: in `d = 1`, a token that is not uniformly bounded
runs off to `±∞` at an exponential rate, and its row of the self-attention
matrix converges to a standard basis row with doubly exponential rate.

The proof: an unbounded token is once beyond `±A`, and `l:auxiliary` — in the
form proved here, for every token — gives `±x_i(t) ≥ c e^t`.  The gaps of
`e:infdist` persist for `t ≥ 0` (`exists_gap`), so for `j ≠ n`,
`P_ij ≤ e^{x_i(x_j - x_n)} ≤ e^{-g x_i}`, and `1 - P_in` is the sum of those.

Source: arXiv:2305.05465v6, `e:infdist`, `l:unboundedparticles`.
-/

import Transformer.Clusters.Section7_Auxiliary

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {n m : ℕ}

/-- In `d = 1`, the attention weight of `e:Idnonresca` is the softmax of the
products `x_i x_l`. -/
theorem attentionMatrix_one_eq (Y : Idx n → EucSpace 1) (i j : Idx n) :
    attentionMatrix (1 : ParamMatrix 1) 1 Y i j =
      Perspective.softmaxWeight (fun l => Y i 0 * Y l 0) j := by
  simp only [attentionMatrix, one_apply_eq_self]
  simp [PiLp.inner_apply, mul_comm]

/-- **`e:infdist`.**  The tokens of an ordered configuration keep a uniform
gap: there is `g > 0` with `x_i(t) + g ≤ x_j(t)` for `i < j` and `t ≥ 0`.  The
source takes `g = min_i |x_{i+1}(0) - x_i(0)|` and invokes `l:distnondec`.

Source: arXiv:2305.05465v6, `e:infdist`. -/
theorem exists_gap (X : ℝ → Idx n → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (hord : IsOrderedConfig (X 0)) :
    ∃ g : ℝ, 0 < g ∧ ∀ t, 0 ≤ t → ∀ i j : Idx n, i < j → X t i 0 + g ≤ X t j 0 := by
  set S := (Finset.univ : Finset (Idx n × Idx n)).filter fun p => p.1 < p.2
  have key : ∀ g : ℝ, (∀ p ∈ S, g ≤ X 0 p.2 0 - X 0 p.1 0) →
      ∀ t, 0 ≤ t → ∀ i j : Idx n, i < j → X t i 0 + g ≤ X t j 0 := by
    intro g hg t ht i j hij
    have h1 := norm_sub_monotone X hX j i ht
    simp only [norm_eq_abs_coord, PiLp.sub_apply] at h1
    rw [abs_of_pos (sub_pos.2 (hord i j hij)),
      abs_of_pos (sub_pos.2 (isOrderedConfig_of_nonneg X hX hord t ht i j hij))] at h1
    have := hg (i, j) (by simp [S, hij])
    linarith
  rcases S.eq_empty_or_nonempty with hS | hS
  · exact ⟨1, one_pos, key 1 (by simp [hS])⟩
  · obtain ⟨p, hp, hmin⟩ := S.exists_min_image (fun p => X 0 p.2 0 - X 0 p.1 0) hS
    have hp' : p.1 < p.2 := (Finset.mem_filter.1 hp).2
    exact ⟨_, sub_pos.2 (hord _ _ hp'), key _ hmin⟩

/-- The hypotheses of `exists_gap` are satisfiable: the one-token solution. -/
example (z : EucSpace 1) :
    IdNonrescaledDynamics (n := 1) (fun t _ => Real.exp t • z) ∧
      IsOrderedConfig (n := 1) (fun _ => Real.exp 0 • z) :=
  ⟨idNonrescaledDynamics_single z, isOrderedConfig_subsingleton _⟩

/-- A softmax weight is at most `e^{u_j - u_N}`, for any reference score. -/
theorem softmaxWeight_le_exp_sub (u : Idx (m + 1) → ℝ) (N j : Idx (m + 1)) :
    Perspective.softmaxWeight u j ≤ Real.exp (u j - u N) := by
  unfold Perspective.softmaxWeight
  have hZ : Real.exp (u N) ≤ ∑ k : Idx (m + 1), Real.exp (u k) :=
    Finset.single_le_sum (f := fun k => Real.exp (u k))
      (fun k _ => (Real.exp_pos _).le) (Finset.mem_univ N)
  rw [Real.exp_sub]
  exact div_le_div_of_nonneg_left (Real.exp_pos _).le (Real.exp_pos _) hZ

/-- **The attention row of a far token.**  If `y_i > 0` and every `y_j`,
`j ≠ N`, lies `g` below `y_N`, then the row `P_i·` is within
`n e^{-g y_i}` of the basis row `δ_N`.

Source: arXiv:2305.05465v6, proof of `l:unboundedparticles`. -/
theorem abs_softmaxWeight_sub_le (y : Idx (m + 1) → ℝ) (i N : Idx (m + 1)) (g : ℝ)
    (hy : 0 < y i) (hgap : ∀ j, j ≠ N → y j + g ≤ y N) (j : Idx (m + 1)) :
    |Perspective.softmaxWeight (fun l => y i * y l) j - (if j = N then 1 else 0)| ≤
      (m + 1) * Real.exp (-(g * y i)) := by
  set p := Perspective.softmaxWeight (fun l => y i * y l)
  have hoff : ∀ k, k ≠ N → p k ≤ Real.exp (-(g * y i)) := fun k hk =>
    (softmaxWeight_le_exp_sub _ N k).trans
      (Real.exp_le_exp.2 (by nlinarith [hgap k hk]))
  have he : 0 ≤ Real.exp (-(g * y i)) := (Real.exp_pos _).le
  split_ifs with hj
  · subst hj
    have hs := Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => y i * y l)
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j)] at hs
    have hle : ∑ k ∈ Finset.univ.erase j, p k ≤
        ∑ _k ∈ Finset.univ.erase j, Real.exp (-(g * y i)) :=
      Finset.sum_le_sum fun k hk => hoff k (Finset.ne_of_mem_erase hk)
    have hnn : 0 ≤ ∑ k ∈ Finset.univ.erase j, p k :=
      Finset.sum_nonneg fun k _ => Perspective.softmaxWeight_nonneg _ _
    rw [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ j), Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, Nat.add_sub_cancel] at hle
    rw [abs_of_nonpos (by linarith), neg_sub]
    have : (m : ℝ) * Real.exp (-(g * y i)) ≤ (m + 1) * Real.exp (-(g * y i)) := by nlinarith
    linarith
  · rw [sub_zero, abs_of_nonneg (Perspective.softmaxWeight_nonneg _ _)]
    exact (hoff j hj).trans (le_mul_of_one_le_left he (by linarith [(m.cast_nonneg : (0 : ℝ) ≤ m)]))

/-- The hypothesis of `abs_softmaxWeight_sub_le` is satisfiable: one positive
score, with no other token to keep below it. -/
example : ∃ (y : Idx 1 → ℝ), 0 < y 0 ∧ ∀ j : Idx 1, j ≠ 0 → y j + 1 ≤ y 0 :=
  ⟨fun _ => 1, one_pos, fun j hj => absurd (Subsingleton.elim j 0) hj⟩

/-- **A row that converges with doubly exponential rate.**  If `x_i ≥ c e^t`
eventually and every `x_j`, `j ≠ N`, stays `g` below `x_N` for `t ≥ 0`, then
`|P_ij(t) - δ_{Nj}| ≤ exp(-c' e^t)` eventually, with `c' = gc/2`: the factor
`n` of `abs_softmaxWeight_sub_le` is absorbed by half the exponent.

Source: arXiv:2305.05465v6, proof of `l:unboundedparticles`. -/
theorem row_tendsto (x : ℝ → Idx (m + 1) → ℝ) (i N : Idx (m + 1)) {g c : ℝ}
    (hg : 0 < g) (hc : 0 < c) (hgap : ∀ t, 0 ≤ t → ∀ j, j ≠ N → x t j + g ≤ x t N)
    (hgrow : ∀ᶠ t in atTop, c * Real.exp t ≤ x t i) :
    ∃ c' : ℝ, 0 < c' ∧ ∀ᶠ t in atTop, ∀ j : Idx (m + 1),
      |Perspective.softmaxWeight (fun l => x t i * x t l) j - (if j = N then 1 else 0)| ≤
        Real.exp (-(c' * Real.exp t)) := by
  refine ⟨g * c / 2, by positivity, ?_⟩
  have hbig := (tendsto_exp_atTop.const_mul_atTop (by positivity : 0 < g * c / 2)).eventually
    (eventually_ge_atTop (m : ℝ))
  filter_upwards [hgrow, hbig, eventually_ge_atTop 0] with t hgt hK ht j
  set K := g * c / 2 * Real.exp t
  have hxi : 0 < x t i := (by positivity : 0 < c * Real.exp t).trans_le hgt
  refine (abs_softmaxWeight_sub_le (x t) i N g hxi (hgap t ht) j).trans ?_
  have h1 : Real.exp (-(g * x t i)) ≤ Real.exp (-(2 * K)) :=
    Real.exp_le_exp.2 (by simp only [K]; nlinarith)
  have h2 : (m : ℝ) + 1 ≤ Real.exp K := by linarith [Real.add_one_le_exp K]
  have h3 : Real.exp K * Real.exp (-(2 * K)) = Real.exp (-K) := by
    rw [← Real.exp_add]; ring_nf
  calc ((m : ℝ) + 1) * Real.exp (-(g * x t i)) ≤ Real.exp K * Real.exp (-(2 * K)) :=
        mul_le_mul h2 h1 (Real.exp_pos _).le (Real.exp_pos _).le
    _ = Real.exp (-(K)) := h3

/-- The hypotheses of `row_tendsto` are satisfiable: the one-token curve
`x(t) = e^t`, with `g = c = 1` and no other token to keep below it. -/
example : ∃ x : ℝ → Idx 1 → ℝ, (∀ t, 0 ≤ t → ∀ j, j ≠ 0 → x t j + 1 ≤ x t 0) ∧
    ∀ᶠ t in atTop, 1 * Real.exp t ≤ x t 0 :=
  ⟨fun t _ => Real.exp t, fun _ _ j hj => absurd (Subsingleton.elim j 0) hj,
    Eventually.of_forall fun _ => by simp⟩

/-- **Lemma (l:unboundedparticles).**  If `x_i(t)` is not uniformly bounded,
then it converges to `+∞` or to `-∞`; in the first case `P_ij(t) → δ_{nj}`, in
the second `P_ij(t) → δ_{1j}`, both with doubly exponential rate.

Source: arXiv:2305.05465v6, `l:unboundedparticles`. -/
theorem unbounded_tendsto_atTop_or_atBot (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) (i : Idx (m + 1))
    (hub : ¬ IsBoundedToken X i) :
    (Tendsto (fun t => X t i 0) atTop atTop ∧
        ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, ∀ j : Idx (m + 1),
          |attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j
              - (if j = Fin.last m then 1 else 0)| ≤ Real.exp (-(c * Real.exp t))) ∨
      (Tendsto (fun t => X t i 0) atTop atBot ∧
        ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, ∀ j : Idx (m + 1),
          |attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j
              - (if j = 0 then 1 else 0)| ≤ Real.exp (-(c * Real.exp t))) := by
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant (m + 1) (Nat.succ_pos m)
  obtain ⟨g, hg, hgap⟩ := exists_gap X hX hord
  simp only [IsBoundedToken, not_exists, not_forall, not_le] at hub
  obtain ⟨t₁, -, ht₁⟩ := hub A
  simp only [attentionMatrix_one_eq]
  rcases lt_abs.1 ht₁ with h | h
  · obtain ⟨c, hc, hev⟩ := exists_exp_lower_bound A hA hAeq X hX i t₁ h
    refine Or.inl ⟨tendsto_atTop_mono' atTop hev (tendsto_exp_atTop.const_mul_atTop hc), ?_⟩
    exact row_tendsto (fun t j => X t j 0) i (Fin.last m) hg hc
      (fun t ht j hj => hgap t ht j _ (Fin.lt_last_iff_ne_last.2 hj)) hev
  · obtain ⟨c, hc, hev⟩ := exists_exp_upper_bound A hA hAeq X hX i t₁ (by linarith)
    refine Or.inr ⟨tendsto_atBot_mono' atTop hev
      (tendsto_neg_atTop_atBot.comp (tendsto_exp_atTop.const_mul_atTop hc)), ?_⟩
    have := row_tendsto (fun t j => -X t j 0) i 0 hg hc
      (fun t ht j hj => by linarith [hgap t ht 0 j (Fin.pos_iff_ne_zero.2 hj)])
      (hev.mono fun t ht => by linarith)
    simpa only [neg_mul_neg] using this

/-- The hypotheses of `unbounded_tendsto_atTop_or_atBot` are satisfiable at
`m = 0`: the one-token solution `x(t) = e^t` is not uniformly bounded. -/
example :
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      IsOrderedConfig (n := 1)
        (fun _ => Real.exp 0 • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      ¬ IsBoundedToken
        (fun t (_ : Idx 1) => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) 0 :=
  ⟨idNonrescaledDynamics_single _, isOrderedConfig_subsingleton _, not_isBoundedToken_exp⟩

end Clusters
end Transformer
