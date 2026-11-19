module

public import Mathlib.Algebra.Order.Field.GeomSum
public import Mathlib.Analysis.Complex.ExponentialBounds
public import Transformer.KERPLE.Section5_ComplicatedKernels

/-!
# Appendix A.3: diagonal dominance and the row sums of Gaussian-like kernels on `ℕ`

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.3. The tools for
`SectionA_GaussianPositions`, which shows that on the integer positions `ℕ` the power of
`exp(−r|m − n|^p)` need not lie in `(0, 2]`.

* Diagonal dominance (`isPDKernel_of_sum_erase_le`): a symmetric kernel `k ≥ 0` with `k(x, x) = 1`
  and off-diagonal row sums at most `1` on every finite set is PD.
* On `ℕ` the off-diagonal row sums of `exp(−r|m − n|^p)` are at most
  `2 Σ_{d ≥ 1} e · e^{−2d} = 2e^{−1} / (1 − e^{−2}) < 1` for `r ≥ 1` and `p ≥ 2`, since
  `r d^p ≥ d² ≥ 2d − 1` for `d ≥ 1` (`sum_erase_gaussKernel_le`).
-/

@[expose] public section

namespace Transformer.KERPLE

variable {X : Type*}

/-- The off-diagonal part of a quadratic form is bounded by the row sums: for a symmetric `k ≥ 0`,
`Σ_x Σ_{y ≠ x} v_x k(x, y) v_y ≥ −Σ_x v_x² Σ_{y ≠ x} k(x, y)`. -/
theorem neg_le_sum_sum_erase [DecidableEq X] {k : X → X → ℝ}
    (hsymm : ∀ x y, k x y = k y x) (hnonneg : ∀ x y, 0 ≤ k x y) (s : Finset X) (v : X → ℝ) :
    -∑ x ∈ s, v x ^ 2 * ∑ y ∈ s.erase x, k x y ≤ ∑ x ∈ s, ∑ y ∈ s.erase x, v x * k x y * v y := by
  have hswap : ∑ x ∈ s, ∑ y ∈ s.erase x, v y ^ 2 * k x y =
      ∑ x ∈ s, v x ^ 2 * ∑ y ∈ s.erase x, k x y := by
    rw [Finset.sum_comm' (t' := s) (s' := fun y => s.erase y) fun x y => by
      simp only [Finset.mem_erase]
      exact ⟨fun h => ⟨⟨fun e => h.2.1 e.symm, h.1⟩, h.2.2⟩,
        fun h => ⟨h.1.2, fun e => h.1.1 e.symm, h.2⟩⟩]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun x _ => by rw [hsymm]
  have hpair : ∀ x ∈ s, ∀ y ∈ s.erase x,
      -(v x ^ 2 * k x y + v y ^ 2 * k x y) ≤ 2 * (v x * k x y * v y) := fun x _ y _ => by
    nlinarith [mul_nonneg (hnonneg x y) (sq_nonneg (v x + v y))]
  have hle := Finset.sum_le_sum fun x hx => Finset.sum_le_sum (hpair x hx)
  simp only [Finset.sum_neg_distrib, Finset.sum_add_distrib, ← Finset.mul_sum] at hle
  linarith

/-- The hypotheses of `neg_le_sum_sum_erase` hold for the kernel `(x − y)²`. -/
example (s : Finset ℕ) (v : ℕ → ℝ) :
    -∑ x ∈ s, v x ^ 2 * ∑ y ∈ s.erase x, ((x : ℝ) - y) ^ 2 ≤
      ∑ x ∈ s, ∑ y ∈ s.erase x, v x * ((x : ℝ) - y) ^ 2 * v y :=
  neg_le_sum_sum_erase (k := fun x y : ℕ => ((x : ℝ) - y) ^ 2) (fun x y => by ring)
    (fun _ _ => sq_nonneg _) s v

/-- **Diagonal dominance**: a symmetric kernel `k ≥ 0` with `k(x, x) = 1` whose off-diagonal row
sums `Σ_{y ∈ s, y ≠ x} k(x, y)` are at most `1` on every finite set `s` is PD. -/
theorem isPDKernel_of_sum_erase_le [DecidableEq X] {k : X → X → ℝ}
    (hsymm : ∀ x y, k x y = k y x) (hnonneg : ∀ x y, 0 ≤ k x y) (hdiag : ∀ x, k x x = 1)
    (hrow : ∀ (s : Finset X), ∀ x ∈ s, ∑ y ∈ s.erase x, k x y ≤ 1) : IsPDKernel k := by
  rw [isPDKernel_iff_posSemidef]
  unfold Matrix.PosSemidef
  refine ⟨Matrix.IsHermitian.ext fun i j => by simp [hsymm i j], fun v => ?_⟩
  simp only [Finsupp.sum, Matrix.of_apply, star_trivial]
  have hsplit : ∀ x ∈ v.support, ∑ y ∈ v.support, v x * k x y * v y =
      v x ^ 2 + ∑ y ∈ v.support.erase x, v x * k x y * v y := fun x hx => by
    rw [← Finset.add_sum_erase _ _ hx, hdiag]
    ring
  rw [Finset.sum_congr rfl hsplit, Finset.sum_add_distrib]
  have hoff := neg_le_sum_sum_erase hsymm hnonneg v.support v
  have hdom : ∑ x ∈ v.support, v x ^ 2 * ∑ y ∈ v.support.erase x, k x y ≤
      ∑ x ∈ v.support, v x ^ 2 :=
    Finset.sum_le_sum fun x hx => mul_le_of_le_one_right (sq_nonneg _) (hrow _ x hx)
  linarith

/-- The hypotheses of `isPDKernel_of_sum_erase_le` hold for the identity kernel. -/
example : IsPDKernel fun m n : ℕ => if m = n then (1 : ℝ) else 0 :=
  isPDKernel_of_sum_erase_le
    (fun m n => by
      by_cases h : m = n
      · simp [h]
      · simp [h, Ne.symm h])
    (fun m n => by positivity) (fun m => by simp) fun s m _ => by simp

/-- `exp(−r d^p) ≤ e · e^{−2d}` for a natural `d ≥ 1`, `r ≥ 1` and `p ≥ 2`, as
`r d^p ≥ d² ≥ 2d − 1`. -/
theorem exp_neg_mul_rpow_le_geom {r p : ℝ} (hr : 1 ≤ r) (hp : 2 ≤ p) {d : ℕ} (hd : 1 ≤ d) :
    Real.exp (-r * (d : ℝ) ^ p) ≤ Real.exp 1 * Real.exp (-2) ^ d := by
  have hd' : (1 : ℝ) ≤ d := by exact_mod_cast hd
  have hpow : (d : ℝ) ^ (2 : ℝ) ≤ (d : ℝ) ^ p := Real.rpow_le_rpow_of_exponent_le hd' hp
  rw [Real.rpow_two] at hpow
  have h1 : (d : ℝ) ^ p ≤ r * (d : ℝ) ^ p := le_mul_of_one_le_left (by positivity) hr
  rw [← Real.exp_nat_mul, ← Real.exp_add, Real.exp_le_exp]
  nlinarith [sq_nonneg ((d : ℝ) - 1)]

/-- The hypotheses of `exp_neg_mul_rpow_le_geom` hold at `d = 1` for the cube. -/
example : Real.exp (-1 * ((1 : ℕ) : ℝ) ^ (3 : ℝ)) ≤ Real.exp 1 * Real.exp (-2) ^ 1 :=
  exp_neg_mul_rpow_le_geom le_rfl (by norm_num) le_rfl

/-- A sum of `q^{g(b)}` over distinct exponents `g(b) ≥ 1` is at most `q / (1 − q)`. -/
theorem sum_pow_le_of_injOn {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1) (T : Finset ℕ) {g : ℕ → ℕ}
    (hg : Set.InjOn g T) (hg1 : ∀ b ∈ T, 1 ≤ g b) : ∑ b ∈ T, q ^ g b ≤ q / (1 - q) := by
  calc ∑ b ∈ T, q ^ g b = ∑ k ∈ T.image g, q ^ k := (Finset.sum_image hg).symm
    _ ≤ ∑ k ∈ Finset.Ico 1 ((T.image g).sup id + 1), q ^ k :=
        Finset.sum_le_sum_of_subset_of_nonneg (fun k hk => by
          obtain ⟨b, hb, rfl⟩ := Finset.mem_image.mp hk
          exact Finset.mem_Ico.mpr ⟨hg1 b hb, Nat.lt_succ_of_le (Finset.le_sup (f := id) hk)⟩)
          fun _ _ _ => pow_nonneg hq0 _
    _ ≤ q ^ 1 / (1 - q) := geom_sum_Ico_le_of_lt_one hq0 hq1
    _ = q / (1 - q) := by rw [pow_one]

/-- The hypotheses of `sum_pow_le_of_injOn` hold for the exponents `1, 2` at `q = 1/2`. -/
example : ∑ b ∈ ({1, 2} : Finset ℕ), (1 / 2 : ℝ) ^ id b ≤ 1 / 2 / (1 - 1 / 2) :=
  sum_pow_le_of_injOn (by norm_num) (by norm_num) _ (Set.injOn_id _) fun b hb => by
    simp only [Finset.mem_insert, Finset.mem_singleton] at hb
    rcases hb with rfl | rfl <;> simp

/-- **The off-diagonal row sums of `exp(−r|m − n|^p)` on `ℕ` are at most `1`** for `r ≥ 1` and
`p ≥ 2`: on either side of `m` they are at most `e Σ_{d ≥ 1} e^{−2d} = e^{−1} / (1 − e^{−2})`,
and `2e^{−1} / (1 − e^{−2}) < 1`. -/
theorem sum_erase_gaussKernel_le {r p : ℝ} (hr : 1 ≤ r) (hp : 2 ≤ p) (s : Finset ℕ) (a : ℕ) :
    ∑ b ∈ s.erase a, gaussKernel r p a b ≤ 1 := by
  set q := Real.exp (-2) with hq
  have hq0 : 0 ≤ q := (Real.exp_pos _).le
  have hq1 : q < 1 := Real.exp_lt_one_iff.mpr (by norm_num)
  have hside : ∀ (T : Finset ℕ) (g : ℕ → ℕ), Set.InjOn g T → (∀ b ∈ T, 1 ≤ g b) →
      (∀ b ∈ T, gaussKernel r p a b = Real.exp (-r * (g b : ℝ) ^ p)) →
      ∑ b ∈ T, gaussKernel r p a b ≤ Real.exp 1 * (q / (1 - q)) := fun T g hg hg1 heq =>
    calc ∑ b ∈ T, gaussKernel r p a b ≤ ∑ b ∈ T, Real.exp 1 * q ^ g b :=
          Finset.sum_le_sum fun b hb => by
            rw [heq b hb]
            exact exp_neg_mul_rpow_le_geom hr hp (hg1 b hb)
      _ = Real.exp 1 * ∑ b ∈ T, q ^ g b := (Finset.mul_sum _ _ _).symm
      _ ≤ Real.exp 1 * (q / (1 - q)) :=
          mul_le_mul_of_nonneg_left (sum_pow_le_of_injOn hq0 hq1 T hg hg1) (Real.exp_pos 1).le
  have hup := hside ((s.erase a).filter fun b => a < b) (fun b => b - a)
    (fun b hb c hc h => by
      simp only [Finset.mem_coe, Finset.mem_filter] at hb hc
      simp only at h
      omega)
    (fun b hb => by
      simp only [Finset.mem_filter] at hb
      omega)
    fun b hb => by
      simp only [Finset.mem_filter] at hb
      show gaussKernel r p a b = Real.exp (-r * ((b - a : ℕ) : ℝ) ^ p)
      rw [gaussKernel, abs_sub_comm, abs_of_nonneg (sub_nonneg.mpr (by exact_mod_cast hb.2.le)),
        Nat.cast_sub hb.2.le]
  have hdown := hside ((s.erase a).filter fun b => ¬a < b) (fun b => a - b)
    (fun b hb c hc h => by
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_erase] at hb hc
      simp only at h
      omega)
    (fun b hb => by
      simp only [Finset.mem_filter, Finset.mem_erase] at hb
      omega)
    fun b hb => by
      simp only [Finset.mem_filter, Finset.mem_erase] at hb
      have hba : b ≤ a := by omega
      show gaussKernel r p a b = Real.exp (-r * ((a - b : ℕ) : ℝ) ^ p)
      rw [gaussKernel, abs_of_nonneg (sub_nonneg.mpr (by exact_mod_cast hba)), Nat.cast_sub hba]
  have hnum : Real.exp 1 * (q / (1 - q)) ≤ 1 / 2 := by
    have he := Real.exp_one_gt_d9
    have ht0 := Real.exp_pos (-1)
    have h1 : Real.exp (-1) * Real.exp 1 = 1 := by rw [← Real.exp_add]; norm_num
    have h2 : q = Real.exp (-1) ^ 2 := by rw [hq, sq, ← Real.exp_add]; norm_num
    have h3 : Real.exp 1 * Real.exp (-1) ^ 2 = Real.exp (-1) := by
      rw [sq, ← mul_assoc, mul_comm (Real.exp 1), h1, one_mul]
    have ht : Real.exp (-1) < 0.37 := by nlinarith [mul_pos ht0 (sub_pos.mpr he)]
    rw [h2, mul_div_assoc', h3, div_le_iff₀ (by nlinarith)]
    nlinarith
  have hsplit := Finset.sum_filter_add_sum_filter_not (s.erase a) (fun b => a < b)
    fun b => gaussKernel r p a b
  linarith

/-- The hypotheses of `sum_erase_gaussKernel_le` hold for the cube at scale `1`. -/
example (s : Finset ℕ) : ∑ b ∈ s.erase 0, gaussKernel 1 3 (0 : ℕ) b ≤ 1 :=
  sum_erase_gaussKernel_le le_rfl (by norm_num) s 0

end Transformer.KERPLE
