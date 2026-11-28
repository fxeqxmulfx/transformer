module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Defs

/-! One AdamW step of two particles in the distance `gap`. The square root
of the second moment is Lipschitz with constants at most one. On the
moment cone the bias-corrected direction is Lipschitz with constants of
order 1/eps, so one step grows the distance by at most 1 + 12η/eps. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open Transformer.AMSGradW.Kernite

variable {P : Type*}

/-- `AdamW.moment_cone` for one step of any particle. -/
theorem cone_step (eps η : ℝ) (t : ℕ) (g : P → ℝ) {s : AdamW.State P} (h : Cone s) :
    Cone (step eps η t g s) := by
  intro p
  change 14 * ((9/10 : ℝ) * s.first p + 1/10 * g p)^2 ≤
    19 * ((95/100 : ℝ) * s.second p + 5/100 * (g p)^2)
  nlinarith [h p, sq_nonneg ((14/10 : ℝ) * s.first p - (9/10 : ℝ) * g p)]

theorem second_nonneg_of_cone {s : AdamW.State P} (h : Cone s) (p : P) : 0 ≤ s.second p := by
  nlinarith [h p, sq_nonneg (s.first p)]

/-- The root of the second moment moves by at most the old root gap and
a quarter of the gradient gap. -/
theorem sqrt_second_lipschitz {v v' g g' : ℝ} (hv : 0 ≤ v) (hv' : 0 ≤ v') :
    |Real.sqrt (95/100 * v + 5/100 * g^2) - Real.sqrt (95/100 * v' + 5/100 * g'^2)| ≤
      |Real.sqrt v - Real.sqrt v'| + |g - g'| / 4 := by
  have hA : 0 ≤ 95/100 * v + 5/100 * g^2 := by positivity
  have hB : 0 ≤ 95/100 * v' + 5/100 * g'^2 := by positivity
  have har : 95/100 * Real.sqrt v ≤ Real.sqrt (95/100 * v + 5/100 * g^2) :=
    (Real.le_sqrt (by positivity) hA).2 (by rw [mul_pow, Real.sq_sqrt hv]; nlinarith [sq_nonneg g])
  have hbr : 95/100 * Real.sqrt v' ≤ Real.sqrt (95/100 * v' + 5/100 * g'^2) :=
    (Real.le_sqrt (by positivity) hB).2 (by rw [mul_pow, Real.sq_sqrt hv']; nlinarith [sq_nonneg g'])
  have hag : |g| / 5 ≤ Real.sqrt (95/100 * v + 5/100 * g^2) :=
    (Real.le_sqrt (by positivity) hA).2 (by rw [div_pow, sq_abs]; nlinarith [sq_nonneg g])
  have hbg : |g'| / 5 ≤ Real.sqrt (95/100 * v' + 5/100 * g'^2) :=
    (Real.le_sqrt (by positivity) hB).2 (by rw [div_pow, sq_abs]; nlinarith [sq_nonneg g'])
  have ha2 := Real.sq_sqrt hA
  have hb2 := Real.sq_sqrt hB
  have hr2 := Real.sq_sqrt hv
  have hr2' := Real.sq_sqrt hv'
  set a := Real.sqrt (95/100 * v + 5/100 * g^2)
  set b := Real.sqrt (95/100 * v' + 5/100 * g'^2)
  set r := Real.sqrt v
  set r' := Real.sqrt v'
  obtain ⟨ha0, hb0, hr0, hr0'⟩ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ r ∧ 0 ≤ r' :=
    ⟨Real.sqrt_nonneg _, Real.sqrt_nonneg _, Real.sqrt_nonneg _, Real.sqrt_nonneg _⟩
  have e : (a - b) * (a + b) = 95/100 * ((r - r') * (r + r')) + 5/100 * ((g - g') * (g + g')) := by
    linear_combination ha2 - hb2 - 95/100 * hr2 + 95/100 * hr2'
  have key : |a - b| * (a + b) ≤ (|r - r'| + |g - g'| / 4) * (a + b) := by
    calc |a - b| * (a + b) = |(a - b) * (a + b)| := by
          rw [abs_mul, abs_of_nonneg (add_nonneg ha0 hb0)]
      _ = |95/100 * ((r - r') * (r + r')) + 5/100 * ((g - g') * (g + g'))| := by rw [e]
      _ ≤ 95/100 * (|r - r'| * (r + r')) + 5/100 * (|g - g'| * (|g| + |g'|)) := by
          refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
          · rw [abs_mul, abs_mul, abs_of_nonneg (add_nonneg hr0 hr0')]; norm_num
          · rw [abs_mul, abs_mul]
            exact mul_le_mul (by norm_num) (mul_le_mul_of_nonneg_left (abs_add_le g g') (abs_nonneg _))
              (by positivity) (by norm_num)
      _ ≤ (|r - r'| + |g - g'| / 4) * (a + b) := by
          nlinarith [mul_nonneg (abs_nonneg (r - r')) (by linarith : 0 ≤ a + b - 95/100 * (r + r')),
            mul_nonneg (abs_nonneg (g - g')) (by linarith : 0 ≤ a + b - (|g| + |g'|) / 5)]
  rcases (add_nonneg ha0 hb0).eq_or_lt with h | h
  · have ha : a = 0 := by linarith
    have hb : b = 0 := by linarith
    rw [ha, hb, sub_zero, abs_zero]
    positivity
  · exact le_of_mul_le_mul_right key h

/-- On the moment cone, the direction is Lipschitz with constants 10/eps
in the first moment and 12/eps in the root of the second moment. -/
theorem direction_lipschitz {eps : ℝ} (he : 0 < eps) (t : ℕ) {m v m' v' : ℝ}
    (hc : 14 * m^2 ≤ 19 * v) :
    |direction eps t m v - direction eps t m' v'| ≤
      (10 * |m - m'| + 12 * |Real.sqrt v - Real.sqrt v'|) / eps := by
  have hv : 0 ≤ v := by nlinarith [sq_nonneg m]
  have hc0 : 0 < Real.sqrt (1 - (95/100 : ℝ)^(t+1)) :=
    Real.sqrt_pos.2 (sub_pos.2 (pow_lt_one₀ (by norm_num) (by norm_num) (Nat.succ_ne_zero t)))
  have hb := firstBias_lower t
  have hm : |m| ≤ 6/5 * Real.sqrt v := by
    nlinarith [sq_abs m, abs_nonneg m, Real.sqrt_nonneg v, Real.sq_sqrt hv,
      sq_nonneg (|m| - (6/5 : ℝ) * Real.sqrt v)]
  set c := Real.sqrt (1 - (95/100 : ℝ)^(t+1))
  set b := firstBias t
  have hDe : denominator eps t v = b * (Real.sqrt v / c + eps) := rfl
  have hDe' : denominator eps t v' = b * (Real.sqrt v' / c + eps) := rfl
  have hx : 0 ≤ Real.sqrt v / c := div_nonneg (Real.sqrt_nonneg _) hc0.le
  have hx' : 0 ≤ Real.sqrt v' / c := div_nonneg (Real.sqrt_nonneg _) hc0.le
  set D := denominator eps t v
  set D' := denominator eps t v'
  have hD : 0 < D := by rw [hDe]; exact mul_pos (by linarith) (by linarith)
  have hD' : eps / 10 ≤ D' := by rw [hDe']; nlinarith
  have hD'0 : 0 < D' := by linarith
  have split : direction eps t m v - direction eps t m' v' =
      m * (D' - D) / (D * D') + (m - m') / D' := by
    show m / D - m' / D' = _
    field_simp
    ring
  have first : |m| * |D' - D| ≤ 6/5 * D * |Real.sqrt v - Real.sqrt v'| := by
    have hdiff : D' - D = b * (Real.sqrt v' - Real.sqrt v) / c := by
      rw [hDe, hDe']; field_simp; ring
    rw [hdiff, abs_div, abs_mul, abs_of_pos hc0, abs_of_pos (by linarith : (0:ℝ) < b),
      abs_sub_comm (Real.sqrt v')]
    have hmc : |m| * b / c ≤ 6/5 * D := by
      rw [div_le_iff₀ hc0, hDe]
      have he' : b * (Real.sqrt v / c + eps) * c = b * (Real.sqrt v + eps * c) := by
        field_simp
      rw [mul_assoc, he']
      nlinarith [mul_le_mul_of_nonneg_left hm (by linarith : (0:ℝ) ≤ b),
        mul_nonneg (mul_nonneg (by linarith : (0:ℝ) ≤ b) he.le) hc0.le]
    calc |m| * (b * |Real.sqrt v - Real.sqrt v'| / c)
        = (|m| * b / c) * |Real.sqrt v - Real.sqrt v'| := by ring
      _ ≤ 6/5 * D * |Real.sqrt v - Real.sqrt v'| :=
          mul_le_mul_of_nonneg_right hmc (abs_nonneg _)
  rw [split]
  calc |m * (D' - D) / (D * D') + (m - m') / D'|
      ≤ |m * (D' - D) / (D * D')| + |(m - m') / D'| := abs_add_le _ _
    _ ≤ 12 * |Real.sqrt v - Real.sqrt v'| / eps + 10 * |m - m'| / eps := by
        apply add_le_add
        · rw [abs_div, abs_mul, abs_of_pos (mul_pos hD hD'0), div_le_div_iff₀ (mul_pos hD hD'0) he]
          nlinarith [mul_le_mul_of_nonneg_right first he.le,
            mul_le_mul_of_nonneg_left hD' (mul_nonneg hD.le (abs_nonneg (Real.sqrt v - Real.sqrt v')))]
        · rw [abs_div, abs_of_pos hD'0, div_le_div_iff₀ hD'0 he]
          nlinarith [mul_le_mul_of_nonneg_left hD' (abs_nonneg (m - m'))]
    _ = _ := by ring

/-- One step of one coordinate, the first particle on the moment cone. -/
theorem coordinate_step {eps η : ℝ} (he : 0 < eps) (hη0 : 0 ≤ η) (hη : η ≤ 10) (t : ℕ)
    {w m v w' m' v' g g' : ℝ} (hc : 14 * m^2 ≤ 19 * v) (hv' : 0 ≤ v') :
    |((1 - η/10) * w - η * direction eps t (9/10*m + 1/10*g) (95/100*v + 5/100*g^2)) -
        ((1 - η/10) * w' - η * direction eps t (9/10*m' + 1/10*g') (95/100*v' + 5/100*g'^2))| +
      |(9/10*m + 1/10*g) - (9/10*m' + 1/10*g')| +
      |Real.sqrt (95/100*v + 5/100*g^2) - Real.sqrt (95/100*v' + 5/100*g'^2)| ≤
      (1 + 12*η/eps) * (|w - w'| + |m - m'| + |Real.sqrt v - Real.sqrt v'| + |g - g'|) := by
  have hv : 0 ≤ v := by nlinarith [sq_nonneg m]
  have hc1 : 14 * (9/10*m + 1/10*g)^2 ≤ 19 * (95/100*v + 5/100*g^2) := by
    nlinarith [sq_nonneg ((14/10 : ℝ) * m - (9/10 : ℝ) * g)]
  have hd := direction_lipschitz he t (m' := 9/10*m' + 1/10*g') (v' := 95/100*v' + 5/100*g'^2) hc1
  have hr := sqrt_second_lipschitz (g := g) (g' := g') hv hv'
  set d := direction eps t (9/10*m + 1/10*g) (95/100*v + 5/100*g^2)
  set d' := direction eps t (9/10*m' + 1/10*g') (95/100*v' + 5/100*g'^2)
  set Δm := |(9/10*m + 1/10*g) - (9/10*m' + 1/10*g')|
  set Δr := |Real.sqrt (95/100*v + 5/100*g^2) - Real.sqrt (95/100*v' + 5/100*g'^2)|
  have hm : Δm ≤ 9/10 * |m - m'| + 1/10 * |g - g'| := by
    have e : (9/10*m + 1/10*g) - (9/10*m' + 1/10*g') = 9/10 * (m - m') + 1/10 * (g - g') := by ring
    simp only [Δm, e]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
    norm_num
  have hw : |((1 - η/10) * w - η * d) - ((1 - η/10) * w' - η * d')| ≤ |w - w'| + η * |d - d'| := by
    have e : ((1 - η/10) * w - η * d) - ((1 - η/10) * w' - η * d') =
        (1 - η/10) * (w - w') - η * (d - d') := by ring
    rw [e]
    refine (abs_sub _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_nonneg (by linarith : (0:ℝ) ≤ 1 - η/10), abs_of_nonneg hη0]
    nlinarith [abs_nonneg (w - w')]
  set k := η / eps
  have hk : 0 ≤ k := div_nonneg hη0 he.le
  have hdk : η * |d - d'| ≤ k * (10 * Δm + 12 * Δr) := by
    have e : η * ((10 * Δm + 12 * Δr) / eps) = k * (10 * Δm + 12 * Δr) := by simp only [k]; ring
    rw [← e]
    exact mul_le_mul_of_nonneg_left hd hη0
  have hlam : 12 * η / eps = 12 * k := by simp only [k]; ring
  rw [hlam]
  have h0m : 0 ≤ Δm := abs_nonneg _
  have h0r : 0 ≤ Δr := abs_nonneg _
  nlinarith [mul_le_mul_of_nonneg_left hm hk, mul_le_mul_of_nonneg_left hr hk, abs_nonneg (w - w'),
    abs_nonneg (m - m'), abs_nonneg (g - g'), abs_nonneg (Real.sqrt v - Real.sqrt v'), mul_nonneg hk h0m,
    mul_nonneg hk (abs_nonneg (w - w')), mul_nonneg hk (abs_nonneg (g - g'))]

/-- One step of two particles: `gap` grows by at most 1 + 12η/eps, the
gradient gap entering in ℓ¹. Only the first particle needs the cone. -/
theorem gap_step [Fintype P] {eps η : ℝ} (he : 0 < eps) (hη0 : 0 ≤ η) (hη : η ≤ 10) (t : ℕ)
    {s s' : AdamW.State P} (hs : Cone s) (hs' : ∀ p, 0 ≤ s'.second p) (g g' : P → ℝ) :
    gap (step eps η t g s) (step eps η t g' s') ≤ (1 + 12*η/eps) * (gap s s' + l1 (g - g')) := by
  unfold gap l1
  rw [← Finset.sum_add_distrib, Finset.mul_sum]
  refine Finset.sum_le_sum fun p _ => ?_
  exact coordinate_step (w := s.weights p) (w' := s'.weights p) (m' := s'.first p) (g := g p)
    (g' := g' p) he hη0 hη t (hs p) (hs' p)

example := cone_step 1 1 0 (fun _ : Fin 1 => 1) (s := ⟨0, 0, 0⟩) fun _ => by simp
example := second_nonneg_of_cone (s := (⟨0, 0, 0⟩ : AdamW.State (Fin 1))) (fun _ => by simp) 0
example := sqrt_second_lipschitz (g := 1) (g' := 0) le_rfl zero_le_one
example := direction_lipschitz one_pos 0 (m := 1) (v := 1) (m' := 0) (v' := 0) (by norm_num)
example := coordinate_step one_pos zero_le_one (by norm_num) 0 (w := 0) (w' := 0) (m := 0) (v := 0)
  (m' := 0) (g := 1) (g' := 0) (by norm_num) le_rfl
example := gap_step one_pos zero_le_one (by norm_num) 0 (s := (⟨0, 0, 0⟩ : AdamW.State (Fin 1)))
  (s' := ⟨0, 0, 0⟩) (fun _ => by simp) (fun _ => le_rfl) 1 0

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
