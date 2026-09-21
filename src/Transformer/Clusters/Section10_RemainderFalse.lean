/-
# The emergence of clusters in self-attention dynamics — `e:boundrj` is false

§10 of arXiv:2305.05465v6, the proof of Step 2'.  The bound `e:boundrj` on
the remainder `r_j(t)` (`scoreRemainder`) does not follow from its stated
inputs: `not_exists_bound_scoreRemainder` is a counterexample in `ℝ²`, built
on the coordinate helpers `mat2`, `coordKer` and the fact that `e^{tV}` fixes
`ker V`.

Source: arXiv:2305.05465v6, `e:boundrj`.
-/

import Transformer.Clusters.Section10_Remainder
import Mathlib.Topology.Algebra.InfiniteSum.ContinuousEval

open Real

namespace Transformer
namespace Clusters

/-- The `2 × 2` matrix `[[a, b], [c, e]]` acting on `ℝ²`. -/
noncomputable def mat2 (a b c e : ℝ) : ParamMatrix 2 :=
  LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin !![a, b; c, e])

/-- The first coordinate of `mat2 a b c e x`. -/
@[simp] theorem mat2_apply_zero (a b c e : ℝ) (x : EucSpace 2) :
    mat2 a b c e x 0 = a * x 0 + b * x 1 := by
  simp [mat2, Matrix.toEuclideanLin, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- The second coordinate of `mat2 a b c e x`. -/
@[simp] theorem mat2_apply_one (a b c e : ℝ) (x : EucSpace 2) :
    mat2 a b c e x 1 = c * x 0 + e * x 1 := by
  simp [mat2, Matrix.toEuclideanLin, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- The inner product of `ℝ²` in coordinates. -/
theorem inner2 (x y : EucSpace 2) : inner (𝕜 := ℝ) x y = x 0 * y 0 + x 1 * y 1 := by
  simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two]; ring

/-- Two vectors of `ℝ²` agreeing in both coordinates are equal. -/
theorem ext2 {x y : EucSpace 2} (h0 : x 0 = y 0) (h1 : x 1 = y 1) : x = y := by
  ext i; fin_cases i <;> simp [h0, h1]

/-- `e^{tV}` fixes the kernel of `V`. -/
theorem expTime_apply_of_map_eq_zero {d : ℕ} (V : ParamMatrix d) (t : ℝ) {w : EucSpace d}
    (hw : V w = 0) : expTime V t w = w := by
  rw [expTime, NormedSpace.exp_eq_tsum ℝ]
  show (∑' n : ℕ, ((n.factorial : ℝ)⁻¹) • (t • V) ^ n) w = w
  rw [
    tsum_apply (NormedSpace.expSeries_summable' (𝕂 := ℝ) (t • V)), tsum_eq_single 0]
  · simp
  · intro n hn
    obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn
    have h1 : (t • V) w = 0 := by
      show t • V w = 0
      rw [hw, smul_zero]
    have h2 : ((t • V) ^ (k + 1)) w = ((t • V) ^ k) ((t • V) w) := by
      rw [pow_succ]
      rfl
    show ((k + 1).factorial : ℝ)⁻¹ • (((t • V) ^ (k + 1)) w) = 0
    rw [h2, h1, map_zero, smul_zero]

/-- The coordinate hyperplane `{x | x_i = 0}` of `ℝ²`. -/
def coordKer (i : Fin 2) : Submodule ℝ (EucSpace 2) where
  carrier := {x | x i = 0}
  add_mem' := by intro a b ha hb; simp_all
  zero_mem' := by simp
  smul_mem' := by intro c x hx; simp_all

/-- Membership in `coordKer`. -/
@[simp] theorem mem_coordKer (i : Fin 2) (x : EucSpace 2) : x ∈ coordKer i ↔ x i = 0 :=
  Iff.rfl

/-- The squared norm of `ℝ²` in coordinates. -/
theorem norm2_sq (x : EucSpace 2) : ‖x‖ ^ 2 = x 0 ^ 2 + x 1 ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, Fin.sum_univ_two, Real.norm_eq_abs, Real.norm_eq_abs,
    sq_abs, sq_abs]

/-- A norm bound on `ℝ²` from the coordinates. -/
theorem norm2_le {x : EucSpace 2} {c : ℝ} (hc : 0 ≤ c) (h : x 0 ^ 2 + x 1 ^ 2 ≤ c ^ 2) :
    ‖x‖ ≤ c :=
  le_of_sq_le_sq (by rw [norm2_sq]; exact h) hc

/-- **Equation (e:boundrj) is false as written.**

The source's claim: for a good triple with multiplicity `(Q, K, V)`
(`d:goodmulti`), `Q^⊤K = A^⊤A`, a solution `z(t)` of the rescaled dynamics,
and the growth bounds `|π_ℱ(y_j(t))| ≤ Ce^{λ₁t}`, `|π_𝒢(y_j(t))| ≤ Ce^{t|λ₂|}`
of `y_j(t) = Ae^{tV}z_j(t)` (from `l:nottoofassst`), the remainder
`r_j(t) = ⟨y_i(t), y_j(t) - y_i(t)⟩ - a_j(t)e^{2λ₁t}` obeys
`|r_j(t)| ≤ Ce^{t(λ₁+|λ₂|)}`.  Every one of those inputs is a hypothesis
below, and the conclusion fails.

Counterexample: `d = n = 2`, `V = diag(1, 0)`, so `ℱ = ℝe₁`, `𝒢 = ℝe₂`,
`λ₁ = 1`, `|λ₂| = 0`; `A = Q = K = [[1, 1/2], [1/2, 1]]`; the stationary
configuration `z₁ = (0, 1)`, `z₂ = (0, 2)`, which `V` does not move.  Then
`y_j = Az_j` is bounded, while `a_2 = 1/4`, so `r_2(t) = 5/4 - e^{2t}/4`.

The source's derivation writes `r_j` as the three cross terms `P₁, P₂, P₃`,
which silently uses `π_ℱ(Ae^{tV}z) = e^{λ₁t}π_ℱ(Az)`; that holds when `A`
respects `ℱ ⊕ 𝒢`, and not in general.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`,
`e:boundrj`. -/
theorem not_exists_bound_scoreRemainder :
    ¬ ∀ (Q K V A proj projG : ParamMatrix 2) (F G : Submodule ℝ (EucSpace 2)) (lam mu : ℝ),
      IsGoodTripleMulti Q K V F G lam → IsAttentionRoot Q K A →
      IsProjOnto proj F G → IsProjOnto projG G F → 0 ≤ mu → mu < lam →
      ∀ Z : ℝ → Idx 2 → EucSpace 2, RescaledDynamics Q K V Z →
      (∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ j : Idx 2,
        ‖proj (A (expTime V t (Z t j)))‖ ≤ C * Real.exp (lam * t)) →
      (∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ j : Idx 2,
        ‖projG (A (expTime V t (Z t j)))‖ ≤ C * Real.exp (mu * t)) →
      ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx 2,
        |scoreRemainder A V proj lam Z t i j| ≤ C * Real.exp ((lam + mu) * t) := by
  intro h
  set A := mat2 1 (1/2) (1/2) 1 with hA
  set V := mat2 1 0 0 0 with hV
  set PG := mat2 0 0 0 1 with hPG
  set z : Idx 2 → EucSpace 2 := ![!₂[0, 1], !₂[0, 2]] with hz
  have hVz : ∀ j, V (z j) = 0 := by
    intro j
    fin_cases j <;> exact ext2 (by simp [hV, hz]) (by simp [hV, hz])
  have hVd : ∀ i j, V (z j - z i) = 0 := by
    intro i j
    rw [map_sub, hVz, hVz, sub_zero]
  have hexp : ∀ t j, expTime V t (z j) = z j := fun t j =>
    expTime_apply_of_map_eq_zero V t (hVz j)
  -- The hypotheses.
  have hApos : ∀ u : EucSpace 2, u ≠ 0 → 0 < u 0 ^ 2 + u 1 ^ 2 := by
    intro u hu
    by_contra hle
    push Not at hle
    have h0 : u 0 = 0 := by nlinarith [sq_nonneg (u 0), sq_nonneg (u 1)]
    have h1 : u 1 = 0 := by nlinarith [sq_nonneg (u 0), sq_nonneg (u 1)]
    exact hu (ext2 (by simp [h0]) (by simp [h1]))
  have hroot : IsAttentionRoot A A A := by
    refine ⟨fun u v => ?_, fun u hu => ?_, fun u v => rfl⟩
    · simp only [inner2, hA, mat2_apply_zero, mat2_apply_one]; ring
    · have := hApos u hu
      simp only [inner2, hA, mat2_apply_zero, mat2_apply_one]
      nlinarith [sq_nonneg (u 0 + u 1)]
  have hFG : IsCompl (coordKer 1) (coordKer 0) := by
    refine isCompl_iff.mpr ⟨Submodule.disjoint_def.mpr fun x hx hx' => ?_, ?_⟩
    · exact ext2 (by simpa using hx') (by simpa using hx)
    · refine codisjoint_iff.mpr (Submodule.eq_top_iff'.mpr fun x => Submodule.mem_sup.mpr
        ⟨!₂[x 0, 0], by simp, !₂[0, x 1], by simp, ext2 (by simp) (by simp)⟩)
  have hgood : IsGoodTripleMulti A A V (coordKer 1) (coordKer 0) 1 := by
    refine ⟨isPosDefQK_of_isAttentionRoot hroot, fun w _ => by simp [hV], fun w hw => ?_,
      hFG, one_pos, fun w hw => ?_, ⟨1, 0, one_pos, le_rfl, one_pos, fun k w hw => ?_⟩⟩
    · simp only [mem_coordKer] at hw ⊢; simp [hV, hw]
    · simp only [mem_coordKer] at hw
      exact ext2 (by simp [hV]) (by simp [hV, hw])
    · have hVw : V w = 0 := ext2 (by simpa [hV] using hw) (by simp [hV])
      cases k with
      | zero => simp
      | succ k =>
        have : (V ^ (k + 1)) w = (V ^ k) (V w) := by rw [pow_succ]; rfl
        rw [this, hVw, map_zero, norm_zero]
        simp
  have hP : IsProjOnto V (coordKer 1) (coordKer 0) := by
    refine ⟨fun x => by simp [hV], fun x hx => ?_, fun x hx => ?_⟩
    · simp only [mem_coordKer] at hx
      exact ext2 (by simp [hV]) (by simp [hV, hx])
    · simp only [mem_coordKer] at hx
      exact ext2 (by simp [hV, hx]) (by simp [hV])
  have hPG' : IsProjOnto PG (coordKer 0) (coordKer 1) := by
    refine ⟨fun x => by simp [hPG], fun x hx => ?_, fun x hx => ?_⟩
    · simp only [mem_coordKer] at hx
      exact ext2 (by simp [hPG, hx]) (by simp [hPG])
    · simp only [mem_coordKer] at hx
      exact ext2 (by simp [hPG]) (by simp [hPG, hx])
  have hZ : RescaledDynamics A A V (fun _ => z) := by
    intro t i
    simpa [hVd] using hasDerivAt_const t (z i)
  have hF : ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ j : Idx 2,
      ‖V (A (expTime V t ((fun _ => z) t j)))‖ ≤ C * Real.exp (1 * t) := by
    refine ⟨1, one_pos, fun t ht j => ?_⟩
    rw [hexp, one_mul, one_mul]
    refine (norm2_le zero_le_one ?_).trans (Real.one_le_exp ht)
    fin_cases j <;> norm_num [hV, hA, hz]
  have hG : ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ j : Idx 2,
      ‖PG (A (expTime V t ((fun _ => z) t j)))‖ ≤ C * Real.exp (0 * t) := by
    refine ⟨2, two_pos, fun t ht j => ?_⟩
    rw [hexp, zero_mul, Real.exp_zero, mul_one]
    refine norm2_le zero_le_two ?_
    fin_cases j <;> simp [hPG, hA, hz]
  obtain ⟨C, hC, hbound⟩ := h A A V A V PG (coordKer 1) (coordKer 0) 1 0 hgood hroot hP hPG'
    le_rfl one_pos (fun _ => z) hZ hF hG
  -- The remainder is `5/4 - e^{2t}/4`.
  have hr : ∀ t, scoreRemainder A V V 1 (fun _ => z) t 0 1 = 5 / 4 - Real.exp (2 * t) / 4 := by
    intro t
    have hd : z 1 - z 0 = !₂[0, 1] := ext2 (by simp [hz]) (by simp [hz]; norm_num)
    have hVd' : V (!₂[(0 : ℝ), 1]) = 0 := ext2 (by simp [hV]) (by simp [hV])
    simp only [scoreRemainder, projScore, hexp, hd, expTime_apply_of_map_eq_zero V t hVd']
    simp [norm2_sq, hA, hV, hz]
    ring_nf
  set s := 4 * C + 5 with hs
  have hs1 : 1 < s := by linarith
  have ht : 0 ≤ Real.log s := Real.log_nonneg hs1.le
  have := hbound (Real.log s) ht 0 1
  rw [hr, add_zero, one_mul, Real.exp_log (by linarith)] at this
  have he2 : Real.exp (2 * Real.log s) = s ^ 2 := by
    rw [show 2 * Real.log s = Real.log s + Real.log s by ring, Real.exp_add,
      Real.exp_log (by linarith)]
    ring
  rw [he2] at this
  obtain ⟨hlo, -⟩ := abs_le.mp this
  nlinarith

end Clusters
end Transformer
