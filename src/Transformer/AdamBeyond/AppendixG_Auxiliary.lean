import Transformer.AMSGrad.Section2_Proj

/-
# Adam and beyond — the auxiliary lemmas

The last appendix of arXiv:1904.09237, "Auxiliary Lemma": the projection
lemma of McMahan and Streeter (lem:proj-lemma), Auer's bound
(lem:simple-grad-bound), and the one-dimensional projection property used in
the proof of Theorem 2 (lem:1d-proj-prop).

**What the source says and what is carried here.**

* lem:proj-lemma is stated for `Q` positive *semi*definite, where
  arXiv:1904.03590 (`Transformer.AMSGrad.mcm_str`) has positive definite.
  "`u = min_{x∈F} ‖Q^{1/2}(x - z)‖`" is taken as: `u ∈ F` minimizes
  `(x - z)ᵀQ(x - z)` over `F`, any minimizer, since for singular `Q` it need
  not be unique.  The source's proof starts from
  "`⟨z₁ - u₁, Q(z₂ - z₁)⟩ ≥ 0`", which is not the variational inequality and
  does not hold in general; the proof here is the variational inequality
  `(u - z)ᵀQ(x - u) ≥ 0`, `proj_variational_psd`, added twice.

* lem:simple-grad-bound: `y_i/√(Σ_{j≤i} y_j)` is read as `0` when the sum
  vanishes, which it does only if `y_i = 0`.

* lem:1d-proj-prop assumes `i ∈ [T]`; the bound holds for every `i`, and is
  stated so.  `Π_F` on `F = [a, b]` is `y ↦ max a (min b y)`.

Source: arXiv:1904.09237, appendix, "Auxiliary Lemma": lem:proj-lemma,
lem:simple-grad-bound, lem:1d-proj-prop.
-/

open Finset Matrix

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- A positive semidefinite real matrix gives a symmetric bilinear form. -/
theorem psd_comm {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosSemidef)
    (v w : Vec d) : v ⬝ᵥ Q *ᵥ w = w ⬝ᵥ Q *ᵥ v := by
  have hT : Qᵀ = Q := by
    have := hQ.isHermitian.eq
    rwa [conjTranspose_eq_transpose_of_trivial] at this
  rw [dotProduct_mulVec, dotProduct_comm, ← vecMul_transpose, hT]

/-- A positive semidefinite real matrix gives a non-negative quadratic form. -/
theorem psd_nonneg {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosSemidef)
    (v : Vec d) : 0 ≤ v ⬝ᵥ Q *ᵥ v := by
  simpa using hQ.dotProduct_mulVec_nonneg v

/-- **The variational inequality**, for `Q` positive semidefinite.  If `u ∈ F`
minimizes `(x - z)ᵀQ(x - z)` over a convex `F`, then `(u - z)ᵀQ(x - u) ≥ 0` for
every `x ∈ F`.  arXiv:1904.09237, appendix, the proof of lem:proj-lemma. -/
theorem proj_variational_psd {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosSemidef)
    {F : Set (Vec d)} (hF : Convex ℝ F) {z u : Vec d} (hu : u ∈ F)
    (hmin : ∀ x ∈ F, (u - z) ⬝ᵥ Q *ᵥ (u - z) ≤ (x - z) ⬝ᵥ Q *ᵥ (x - z))
    {x : Vec d} (hx : x ∈ F) : 0 ≤ (u - z) ⬝ᵥ Q *ᵥ (x - u) := by
  set a := (u - z) ⬝ᵥ Q *ᵥ (x - u)
  set b := (x - u) ⬝ᵥ Q *ᵥ (x - u)
  have hb : 0 ≤ b := psd_nonneg hQ _
  have hs : ∀ s : ℝ, 0 < s → s ≤ 1 → 0 ≤ 2 * a + s * b := by
    intro s hs0 hs1
    have hw : u + s • (x - u) ∈ F := by
      have := hF hu hx (by linarith : 0 ≤ 1 - s) hs0.le (by ring)
      convert this using 1
      rw [smul_sub, sub_smul, one_smul]; abel
    have h := hmin _ hw
    have he : u + s • (x - u) - z = (u - z) + s • (x - u) := by abel
    rw [he] at h
    simp only [mulVec_add, mulVec_smul, dotProduct_add, add_dotProduct, dotProduct_smul,
      smul_dotProduct, smul_eq_mul] at h
    have hc := psd_comm hQ (x - u) (u - z)
    have : s * 0 ≤ s * (2 * a + s * b) := by nlinarith
    exact le_of_mul_le_mul_left this hs0
  by_contra ha
  replace ha := not_le.mp ha
  set s := min 1 (-a / (b + 1))
  have hs0 : 0 < s := lt_min one_pos (div_pos (by linarith) (by linarith))
  have hsb : s * b ≤ -a / (b + 1) * b := mul_le_mul_of_nonneg_right (min_le_right _ _) hb
  have hlt : -a / (b + 1) * b ≤ -a := by
    rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]; nlinarith
  linarith [hs s hs0 (min_le_left _ _)]

/-- **lem:proj-lemma (McMahan and Streeter).**  For `Q` positive semidefinite
and `F` convex, if `u₁, u₂ ∈ F` minimize `‖Q^{1/2}(x - z₁)‖`, `‖Q^{1/2}(x - z₂)‖`
over `F`, then `‖Q^{1/2}(u₁ - u₂)‖ ≤ ‖Q^{1/2}(z₁ - z₂)‖`.

Source: arXiv:1904.09237, appendix, lem:proj-lemma. -/
theorem proj_lemma {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosSemidef) {F : Set (Vec d)}
    (hF : Convex ℝ F) {z₁ z₂ u₁ u₂ : Vec d} (hu₁ : u₁ ∈ F) (hu₂ : u₂ ∈ F)
    (hmin₁ : ∀ x ∈ F, (u₁ - z₁) ⬝ᵥ Q *ᵥ (u₁ - z₁) ≤ (x - z₁) ⬝ᵥ Q *ᵥ (x - z₁))
    (hmin₂ : ∀ x ∈ F, (u₂ - z₂) ⬝ᵥ Q *ᵥ (u₂ - z₂) ≤ (x - z₂) ⬝ᵥ Q *ᵥ (x - z₂)) :
    Real.sqrt ((u₁ - u₂) ⬝ᵥ Q *ᵥ (u₁ - u₂)) ≤ Real.sqrt ((z₁ - z₂) ⬝ᵥ Q *ᵥ (z₁ - z₂)) := by
  refine Real.sqrt_le_sqrt ?_
  have h₁ := proj_variational_psd hQ hF hu₁ hmin₁ hu₂
  have h₂ := proj_variational_psd hQ hF hu₂ hmin₂ hu₁
  have h₃ := psd_nonneg hQ ((z₁ - z₂) - (u₁ - u₂))
  have c := psd_comm hQ
  simp only [mulVec_sub, dotProduct_sub, sub_dotProduct] at h₁ h₂ h₃ ⊢
  linarith [c u₁ u₂, c u₁ z₁, c u₁ z₂, c u₂ z₁, c u₂ z₂, c z₁ z₂]

/-- The hypotheses of `proj_lemma` are satisfiable with a singular `Q`:
`Q = 0`, `F = ℝ^d`, any `uᵢ`. -/
example (z₁ z₂ u₁ u₂ : Vec d) :
    (0 : Matrix (Fin d) (Fin d) ℝ).PosSemidef ∧ Convex ℝ (Set.univ : Set (Vec d)) ∧
      (∀ x ∈ (Set.univ : Set (Vec d)),
        (u₁ - z₁) ⬝ᵥ (0 : Matrix (Fin d) (Fin d) ℝ) *ᵥ (u₁ - z₁) ≤
          (x - z₁) ⬝ᵥ (0 : Matrix (Fin d) (Fin d) ℝ) *ᵥ (x - z₁)) ∧
      ∀ x ∈ (Set.univ : Set (Vec d)),
        (u₂ - z₂) ⬝ᵥ (0 : Matrix (Fin d) (Fin d) ℝ) *ᵥ (u₂ - z₂) ≤
          (x - z₂) ⬝ᵥ (0 : Matrix (Fin d) (Fin d) ℝ) *ᵥ (x - z₂) :=
  ⟨PosSemidef.zero, convex_univ, fun _ _ => by simp, fun _ _ => by simp⟩

/-- **lem:simple-grad-bound (Auer et al.).**  For `y_1, …, y_t ≥ 0`,
`Σ_{i=1}^t y_i/√(Σ_{j=1}^i y_j) ≤ 2√(Σ_{i=1}^t y_i)`.

Source: arXiv:1904.09237, appendix, lem:simple-grad-bound. -/
theorem sum_div_sqrt_partial_le {y : ℕ → ℝ} (hy : ∀ i, 0 ≤ y i) (t : ℕ) :
    ∑ i ∈ Icc 1 t, y i / Real.sqrt (∑ j ∈ Icc 1 i, y j) ≤
      2 * Real.sqrt (∑ i ∈ Icc 1 t, y i) := by
  induction t with
  | zero => simp
  | succ n ih =>
    rw [sum_Icc_succ_top (by omega)]
    set s := ∑ j ∈ Icc 1 n, y j
    have hs : 0 ≤ s := sum_nonneg fun j _ => hy j
    rw [sum_Icc_succ_top (by omega : 1 ≤ n + 1)]
    set a := Real.sqrt s
    set b := Real.sqrt (s + y (n + 1))
    have ha : 0 ≤ a := Real.sqrt_nonneg _
    have hab : a ≤ b := Real.sqrt_le_sqrt (by linarith [hy (n + 1)])
    have ha2 : a ^ 2 = s := Real.sq_sqrt hs
    have hb2 : b ^ 2 = s + y (n + 1) := Real.sq_sqrt (by linarith [hy (n + 1)])
    have : y (n + 1) / b ≤ 2 * b - 2 * a := by
      have hb0 : 0 ≤ b := Real.sqrt_nonneg _
      rcases hb0.eq_or_lt with hb | hb
      · rw [← hb, div_zero]; linarith
      · rw [div_le_iff₀ hb]; nlinarith [sq_nonneg (b - a)]
    linarith

/-- **lem:1d-proj-prop.**  On `F = [a, b]`, if `y_{t+1} = Π_F(y_t + δ_t)` for
`t ≥ 1`, `y_1 ∈ F`, `δ_j ≤ 0` for `j ≤ i` and `δ_j > 0` for `i < j ≤ T`, then
`y_{T+1} ≥ min(b, y_1 + Σ_{j=1}^T δ_j)`.

Source: arXiv:1904.09237, appendix, lem:1d-proj-prop. -/
theorem proj_1d {a b : ℝ} {y δ : ℕ → ℝ}
    (hy : ∀ t, 1 ≤ t → y (t + 1) = max a (min b (y t + δ t))) (hy₁ : y 1 ∈ Set.Icc a b)
    {i T : ℕ} (hneg : ∀ j ∈ Icc 1 i, δ j ≤ 0) (hpos : ∀ j ∈ Ioc i T, 0 < δ j) :
    min b (y 1 + ∑ j ∈ Icc 1 T, δ j) ≤ y (T + 1) := by
  have hle : ∀ n, y (n + 1) ≤ b := by
    intro n
    cases n with
    | zero => exact hy₁.2
    | succ n => rw [hy _ (by omega)]; exact max_le (hy₁.1.trans hy₁.2) (min_le_left _ _)
  have h₁ : ∀ n, n ≤ i → y 1 + ∑ j ∈ Icc 1 n, δ j ≤ y (n + 1) := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      intro hn
      have hd := hneg (n + 1) (mem_Icc.mpr ⟨by omega, hn⟩)
      rw [sum_Icc_succ_top (by omega), hy (n + 1) (by omega),
        min_eq_right (by linarith [hle n])]
      exact le_max_of_le_right (by linarith [ih (by omega)])
  have h₂ : ∀ n, i ≤ n → n ≤ T → min b (y 1 + ∑ j ∈ Icc 1 n, δ j) ≤ y (n + 1) := by
    intro n hin
    induction n, hin using Nat.le_induction with
    | base => exact fun _ => (min_le_right _ _).trans (h₁ i le_rfl)
    | succ n hin ih =>
      intro hn
      have hd := hpos (n + 1) (mem_Ioc.mpr ⟨by omega, hn⟩)
      have ih := ih (by omega)
      rw [sum_Icc_succ_top (by omega), hy (n + 1) (by omega), ← add_assoc]
      refine le_max_of_le_right (le_min (min_le_left _ _) ?_)
      rcases le_total (y 1 + ∑ j ∈ Icc 1 n, δ j) b with h | h
      · rw [min_eq_right h] at ih; exact (min_le_right _ _).trans (by linarith)
      · rw [min_eq_left h] at ih; exact (min_le_left _ _).trans (by linarith)
  rcases le_total i T with h | h
  · exact h₂ T h le_rfl
  · exact (min_le_right _ _).trans (h₁ T h)

/-- The hypotheses of `proj_1d` are satisfiable: `F = [0, 1]`, `y ≡ 0`,
`δ ≡ 0`, `i = T = 1`. -/
example :
    (∀ t, 1 ≤ t → (fun _ : ℕ => (0 : ℝ)) (t + 1) =
      max 0 (min 1 ((fun _ : ℕ => (0 : ℝ)) t + (fun _ : ℕ => (0 : ℝ)) t))) ∧
      (0 : ℝ) ∈ Set.Icc 0 1 ∧ (∀ j ∈ Icc 1 1, (fun _ : ℕ => (0 : ℝ)) j ≤ 0) ∧
      ∀ j ∈ Ioc 1 1, 0 < (fun _ : ℕ => (0 : ℝ)) j :=
  ⟨fun _ _ => by simp, ⟨le_rfl, zero_le_one⟩, fun _ _ => le_rfl, fun _ hj => by simp at hj⟩

end AdamBeyond
end Transformer
