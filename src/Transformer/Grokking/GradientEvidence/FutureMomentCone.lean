module

public import Transformer.Grokking.GradientEvidence.MainAdamMoments

/-!
# Future moment cones from a checkpoint

Source: `review/flat_spectrum.md`, *Future forcing from checkpoint moments*,
FF1–FF2. The betas are the main model's .9 and .95. The next gradient is arbitrary;
no bound measured from a future trajectory is a premise. Sharpness concerns
one scalar step with unrestricted gradients, not discovery by the tied model.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

/-- FF1: exact slack identity, including checkpoints with zero variance. -/
theorem future_cone_slack (m v g r : ℝ) :
    (((81 / 95) * r + 1 / 5) * mainSecond v g - (mainFirst m g) ^ 2) * v =
      (81 / 95) * (r * v - m ^ 2) * mainSecond v g +
        (19 * v - 9 * m * g) ^ 2 / 1900 := by
  unfold mainFirst mainSecond
  ring

/-- FF1: the sharp scalar cone update for every next gradient. -/
theorem future_cone_step (m v g r : ℝ) (hv : 0 ≤ v) (hr : 0 ≤ r)
    (h : m ^ 2 ≤ r * v) :
    (mainFirst m g) ^ 2 ≤ ((81 / 95) * r + 1 / 5) * mainSecond v g := by
  rcases eq_or_lt_of_le hv with hv0 | hvp
  · have hm : m = 0 := by nlinarith [sq_nonneg m]
    subst m
    have hg := mul_nonneg hr (sq_nonneg g)
    simp only [← hv0, mainFirst, mainSecond]
    nlinarith
  · have hs : 0 ≤ mainSecond v g := by
      unfold mainSecond
      positivity
    have hp := mul_nonneg (sub_nonneg.mpr h) hs
    have he := future_cone_slack m v g r
    apply le_of_mul_le_mul_right (a := v) _ hvp
    nlinarith [sq_nonneg (19 * v - 9 * m * g)]

/-- FF1: when both buffers are nonzero, the displayed gradient attains the
one-step ratio bound. It is not asserted to be the model's loss gradient. -/
theorem future_ratio_attained (m v : ℝ) (hv : 0 < v) (hm : m ≠ 0) :
    (mainFirst m (19 * v / (9 * m))) ^ 2 /
        mainSecond v (19 * v / (9 * m)) = (81 / 95) * (m ^ 2 / v) + 1 / 5 := by
  have hs : 0 < mainSecond v (19 * v / (9 * m)) := by
    unfold mainSecond
    positivity
  have hc : m ^ 2 / v * v = m ^ 2 := div_mul_cancel₀ _ (ne_of_gt hv)
  have hg : 19 * v - 9 * m * (19 * v / (9 * m)) = 0 := by
    field_simp
    ring
  have he := future_cone_slack m v (19 * v / (9 * m)) (m ^ 2 / v)
  rw [hc, hg] at he
  apply (div_eq_iff (ne_of_gt hs)).mpr
  nlinarith

/-- FF1: the cone after h future steps, using only the checkpoint ratio. -/
def futureCone (r : ℝ) (h : ℕ) : ℝ :=
  19 / 14 + (r - 19 / 14) * (81 / 95 : ℝ) ^ h

/-- FF1: closed-form solution of the scalar cone recurrence. -/
theorem future_cone_recurrence (r : ℝ) (h : ℕ) :
    futureCone r (h + 1) = (81 / 95) * futureCone r h + 1 / 5 := by
  unfold futureCone
  rw [pow_succ]
  ring

/-- FF1: every future cone is nonnegative from a valid checkpoint ratio. -/
theorem future_cone_nonneg (r : ℝ) (hr : 0 ≤ r) (h : ℕ) :
    0 ≤ futureCone r h := by
  induction h with
  | zero => simpa [futureCone] using hr
  | succ h ih => rw [future_cone_recurrence]; positivity

/-- FF1: dependence on the checkpoint contracts by exactly 81/95 per step.
This concerns a worst-case magnitude envelope, not the model's alignment. -/
theorem future_cone_memory (r s : ℝ) (h : ℕ) :
    futureCone r h - futureCone s h = (r - s) * (81 / 95 : ℝ) ^ h := by
  unfold futureCone
  ring

/-- FF2: continue the actual main-model moment recurrences from saved buffers. -/
def warmMoments (m v : ℝ) (g : ℕ → ℝ) : ℕ → ℝ × ℝ
  | 0 => (m, v)
  | h+1 => (mainFirst (warmMoments m v g h).1 (g h),
      mainSecond (warmMoments m v g h).2 (g h))

/-- FF2: resuming saved buffers agrees with the original moment history. -/
theorem main_moments_resume (g : ℕ → ℝ) (start h : ℕ) :
    mainMoments g (start + h) =
      warmMoments (mainMoments g start).1 (mainMoments g start).2
        (fun j => g (start+j)) h := by
  induction h with
  | zero => rfl
  | succ h ih => simp only [mainMoments, warmMoments, Nat.add_eq, ih]

/-- FF2: nonnegative variance persists for every future gradient sequence. -/
theorem warm_second_nonneg (m v : ℝ) (g : ℕ → ℝ) (hv : 0 ≤ v) (h : ℕ) :
    0 ≤ (warmMoments m v g h).2 := by
  induction h with
  | zero => exact hv
  | succ h ih =>
    simp only [warmMoments, mainSecond]
    positivity

/-- FF2: the closed-form cone controls all future prefixes, without reading
or restricting any future gradient. The initial ratio is the only cone premise. -/
theorem warm_moment_cone (m v r : ℝ) (g : ℕ → ℝ) (hv : 0 ≤ v) (hr : 0 ≤ r)
    (hcone : m ^ 2 ≤ r * v) (h : ℕ) :
    (warmMoments m v g h).1 ^ 2 ≤ futureCone r h * (warmMoments m v g h).2 := by
  induction h with
  | zero => simpa [warmMoments, futureCone] using hcone
  | succ h ih =>
    rw [future_cone_recurrence]
    exact future_cone_step _ _ _ _ (warm_second_nonneg m v g hv h)
      (future_cone_nonneg r hr h) ih

/-- FF2: simultaneous sign reversal preserves second moments and reverses
first moments. A cone bound cannot distinguish these opposite directions. -/
theorem warm_moments_sign_symmetry (m v : ℝ) (g : ℕ → ℝ) (h : ℕ) :
    warmMoments (-m) v (fun j => -g j) h =
      (-(warmMoments m v g h).1, (warmMoments m v g h).2) := by
  induction h with
  | zero => rfl
  | succ h ih =>
    apply Prod.ext <;> simp only [warmMoments, ih, mainFirst, mainSecond] <;> ring

/-- FF1: a nonnegative checkpoint variance and consistent zero buffers give
a computable initial cone. This is the native constructor's validity check. -/
theorem checkpoint_ratio_cone (m v : ℝ) (hv : 0 ≤ v) (hz : v = 0 → m = 0) :
    let r := if v = 0 then 0 else m^2/v
    0 ≤ r ∧ m^2 ≤ r*v := by
  dsimp only
  split_ifs with hv0
  · rw [hv0, hz hv0]
    norm_num
  · exact ⟨div_nonneg (sq_nonneg m) hv, by rw [div_mul_cancel₀ _ hv0]⟩

example := checkpoint_ratio_cone (1/10) (1/100) (by norm_num) (by norm_num)

example (g : ℕ → ℝ) (h : ℕ) := warm_second_nonneg (1/10) (1/100) g (by norm_num) h
example (g : ℕ → ℝ) (h : ℕ) := warm_moment_cone (1/10) (1/100) 1 g
  (by norm_num) (by norm_num) (by norm_num) h

example (g : ℝ) := future_cone_step (1/10) (1/100) g 1
  (by norm_num) (by norm_num) (by norm_num)
example := future_ratio_attained (1/10) (1/100) (by norm_num) (by norm_num)
example (h : ℕ) := future_cone_nonneg 1 (by norm_num) h

end Transformer.Grokking.GradientEvidence
