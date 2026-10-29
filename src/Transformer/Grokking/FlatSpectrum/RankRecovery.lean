module

public import Transformer.Grokking.FlatSpectrum.PatternGraph
public import Transformer.Grokking.FlatSpectrum.RankCertificate

/-!
# When M recovers a table of any rank

Source: `review/flat_spectrum.md`, *Rank above one* (added), items 1 and 2. Sufficient: with a
certificate `Y = U Vᴴ + W` as in `le_nuclearNorm_add_of_certificate` for some `c > 1`, and no
nonzero `U Aᴴ + B Vᴴ` vanishing on `Ω`, `K` is the unique minimum of the nuclear norm among the
matrices equal to it on `Ω` (`isMinOn_iff_of_certificate`): a minimum `K + H` has
`(1 − U Uᴴ) H (1 − V Vᴴ) = 0`, so `H = U (Uᴴ H) + ((1 − U Uᴴ) H V) Vᴴ`, and it vanishes on `Ω`.
Necessary: let `r i + s j = 0` on the cells of `Ω` where `K` is nonzero, and `w = r i₀ + s j₀ ≠ 0`
at a nonzero entry `K i₀ j₀`. The phases `u i = exp(π i Re(r i / w))`, `v j = exp(π i Re(s j / w))`
give `diag(u) K diag(v)`, of the same nuclear norm, equal to `K` on `Ω` and to `−K i₀ j₀` at
`(i₀, j₀)`, so `K` is not the unique minimum (`not_forall_isMinOn_of_not_rigid_support`). With
`r = 1` and `s = −1` on the component of row `i₀`, this says that the cells of `Ω` where `K` is
nonzero join the row and the column of every nonzero entry (`reachable_of_forall_isMinOn`). For
`K = J` the sufficient condition is Theorem D's item 2; for `K = a bᵀ` the necessary one is the
rigidity half of Theorem G's item 3.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset SimpleGraph

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

/-- **Exact recovery at every rank**, the rank-`r` form of Theorem D, item 2: with a
certificate `Y = U Vᴴ + W` as in `le_nuclearNorm_add_of_certificate` for some `c > 1`, and no
nonzero `U Aᴴ + B Vᴴ` vanishing on `Ω`, `K` is the unique minimum of the nuclear norm among the
matrices equal to `K` on `Ω`. -/
theorem isMinOn_iff_of_certificate {K : Matrix m n ℂ} {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K) {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0) (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    (hT : ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → U * Aᴴ + B * Vᴴ = 0)
    {X : Matrix m n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} X ↔ X = K := by
  have hlow (X' : Matrix m n ℂ) (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2) :
      nuclearNorm K + (1 - c⁻¹) * nuclearNorm ((1 - U * Uᴴ) * (X' - K) * (1 - V * Vᴴ)) ≤
        nuclearNorm X' := by
    have h := le_nuclearNorm_add_of_certificate hU hV hKU hK hYΩ hWU hWV
      (zero_lt_one.trans hc) hW (H := X' - K) fun ij hij => by
        rw [Matrix.sub_apply, hX' ij hij, sub_self]
    rwa [add_sub_cancel] at h
  have hc1 : 0 < 1 - c⁻¹ := sub_pos.mpr (inv_lt_one_of_one_lt₀ hc)
  rw [isMinOn_iff]
  constructor
  · intro hmin
    have h1 := (hlow X hX).trans (hmin K fun _ _ => rfl)
    set H := X - K with hHdef
    have h0 := nuclearNorm_nonneg ((1 - U * Uᴴ) * H * (1 - V * Vᴴ))
    have hA := eq_zero_of_nuclearNorm_eq_zero (le_antisymm (by nlinarith) h0)
    have hdec : H = U * (Hᴴ * U)ᴴ + (1 - U * Uᴴ) * H * V * Vᴴ := by
      have e : H = U * Uᴴ * H + (1 - U * Uᴴ) * H * (V * Vᴴ) +
          (1 - U * Uᴴ) * H * (1 - V * Vᴴ) := by
        simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one]
        abel
      rw [hA, add_zero] at e
      rw [conjTranspose_mul, conjTranspose_conjTranspose]
      conv_lhs => rw [e]
      simp only [Matrix.mul_assoc]
    have hzero := hT (Hᴴ * U) ((1 - U * Uᴴ) * H * V) fun ij hij => by
      rw [← hdec, hHdef, Matrix.sub_apply, hX ij hij, sub_self]
    rwa [← hdec, hHdef, sub_eq_zero] at hzero
  · intro hXK X' hX'
    rw [hXK]
    have := nuclearNorm_nonneg ((1 - U * Uᴴ) * (X' - K) * (1 - V * Vᴴ))
    nlinarith [hlow X' hX']

example :=
  have h := svd_hypotheses (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
    (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
    fun k => by fin_cases k <;> norm_num
  (isMinOn_iff_of_certificate (by simp) (by simp) h.1 h.2.1 (Ω := Set.univ) (W := 0)
    (fun _ _ hij => absurd (Set.mem_univ _) hij) (by simp) (by simp) one_lt_two
    (by rw [smul_zero]; exact isContraction_zero)
    (fun _ _ hT => by ext i j; exact hT (i, j) (Set.mem_univ _)) fun _ _ => rfl).mpr rfl

omit [Fintype r] [DecidableEq r] in
/-- **Phase rigidity on the support is necessary at every rank**: if `r i + s j = 0` on the
cells of `Ω` where `K` is nonzero, and `r i₀ + s j₀ ≠ 0` at a nonzero entry `K i₀ j₀`, then `K`
is not the unique minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem not_forall_isMinOn_of_not_rigid_support {K : Matrix m n ℂ} {Ω : Set (m × n)}
    {r : m → ℂ} {s : n → ℂ} (hrs : ∀ ij ∈ Ω, K ij.1 ij.2 ≠ 0 → r ij.1 + s ij.2 = 0)
    {i₀ : m} {j₀ : n} (hK₀ : K i₀ j₀ ≠ 0) (h₀ : r i₀ + s j₀ ≠ 0) :
    ¬ ∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} X ↔ X = K) := by
  intro hall
  set w := r i₀ + s j₀
  set u : m → ℂ := fun i => Complex.exp (↑(Real.pi * (r i / w).re) * Complex.I) with hu
  set v : n → ℂ := fun j => Complex.exp (↑(Real.pi * (s j / w).re) * Complex.I) with hv
  have hu1 (i : m) : ‖u i‖ = 1 := by simp only [hu, Complex.norm_exp_ofReal_mul_I]
  have hv1 (j : n) : ‖v j‖ = 1 := by simp only [hv, Complex.norm_exp_ofReal_mul_I]
  have hK' (i : m) (j : n) : (diagonal u * K * diagonal v) i j =
      Complex.exp (↑(Real.pi * ((r i + s j) / w).re) * Complex.I) * K i j := by
    rw [mul_diagonal, diagonal_mul]
    simp only [hu, hv, add_div, Complex.add_re, mul_add, Complex.ofReal_add, add_mul,
      Complex.exp_add]
    ring
  have hnu : (diagonal u)ᴴ * diagonal u = 1 := by
    rw [diagonal_conjTranspose, diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext i
    rw [Pi.star_apply, Complex.star_def, Complex.conj_mul', hu1 i]
    norm_num
  have hnv : (diagonal (star v))ᴴ * diagonal (star v) = 1 := by
    rw [diagonal_conjTranspose, star_star, diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext j
    rw [Pi.star_apply, Complex.star_def, mul_comm, Complex.conj_mul', hv1 j]
    norm_num
  have hnorm : nuclearNorm (diagonal u * K * diagonal v) = nuclearNorm K := by
    have h := nuclearNorm_isometry hnu hnv K
    rwa [diagonal_conjTranspose, star_star] at h
  have hmem : ∀ ij ∈ Ω, (diagonal u * K * diagonal v) ij.1 ij.2 = K ij.1 ij.2 := by
    intro ij hij
    rw [hK']
    by_cases hz : K ij.1 ij.2 = 0
    · rw [hz, mul_zero]
    · rw [hrs ij hij hz, zero_div, Complex.zero_re, mul_zero, Complex.ofReal_zero, zero_mul,
        Complex.exp_zero, one_mul]
  have hKmin := isMinOn_iff.mp ((hall K fun _ _ => rfl).mpr rfl)
  have hne : diagonal u * K * diagonal v ≠ K := by
    intro h
    have h1 := congrFun (congrFun h i₀) j₀
    rw [hK', div_self h₀, Complex.one_re, mul_one, Complex.exp_pi_mul_I] at h1
    exact hK₀ (by linear_combination -h1 / 2)
  refine hne ((hall _ hmem).mp (isMinOn_iff.mpr fun X hX => ?_))
  rw [hnorm]
  exact hKmin X hX

/-- The rank-two `!![1, 1; 0, 1]` is not recovered from its diagonal. -/
example := not_forall_isMinOn_of_not_rigid_support (K := !![(1 : ℂ), 1; 0, 1])
  (Ω := {ij | ij.2 = ij.1}) (r := ![1, 0]) (s := ![-1, 0])
  (fun ij hij _ => by
    obtain ⟨i, j⟩ := ij
    have h : j = i := hij
    subst h
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp) (by simp)

omit [Fintype r] [DecidableEq r] in
/-- **The cells where `K` is nonzero join every nonzero entry**, the graph form of
`not_forall_isMinOn_of_not_rigid_support`: if M recovers `K` from `Ω`, then in the bipartite graph
of the cells of `Ω` where `K` is nonzero, the row and the column of every nonzero entry of `K` are
joined by a path. -/
theorem reachable_of_forall_isMinOn {K : Matrix m n ℂ} {Ω : Set (m × n)}
    (hK : ∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} X ↔ X = K))
    {i₀ : m} {j₀ : n} (hK₀ : K i₀ j₀ ≠ 0) :
    (patternGraph {ij | ij ∈ Ω ∧ K ij.1 ij.2 ≠ 0}).Reachable (.inl i₀) (.inr j₀) := by
  classical
  set G := patternGraph {ij | ij ∈ Ω ∧ K ij.1 ij.2 ≠ 0}
  by_contra hj
  refine not_forall_isMinOn_of_not_rigid_support
    (r := fun i => if G.Reachable (.inl i₀) (.inl i) then 1 else 0)
    (s := fun j => if G.Reachable (.inl i₀) (.inr j) then -1 else 0) (fun ij hij hz => ?_) hK₀
    (by rw [ite_eq_left (Reachable.refl _), ite_eq_right hj, add_zero]; exact one_ne_zero) hK
  have hadj : G.Adj (.inl ij.1) (.inr ij.2) := patternGraph_adj_inl_inr.mpr ⟨hij, hz⟩
  by_cases hi : G.Reachable (.inl i₀) (.inl ij.1)
  · rw [ite_eq_left hi, ite_eq_left (hi.trans hadj.reachable), add_neg_cancel]
  · rw [ite_eq_right hi, ite_eq_right fun h => hi (h.trans hadj.symm.reachable), add_zero]

/-- From all four cells M recovers `!![1, 1; 0, 1]`, so row `0` and column `1` are joined. -/
example := reachable_of_forall_isMinOn (K := !![(1 : ℂ), 1; 0, 1]) (Ω := Set.univ)
  (fun X hX => ⟨fun _ => Matrix.ext fun i j => hX (i, j) (Set.mem_univ _), fun h =>
    isMinOn_iff.mpr fun X' hX' => by
      rw [h, show X' = !![(1 : ℂ), 1; 0, 1] from
        Matrix.ext fun i j => hX' (i, j) (Set.mem_univ _)]⟩)
  (i₀ := 0) (j₀ := 1) (by simp)

end Transformer.Grokking.FlatSpectrum
