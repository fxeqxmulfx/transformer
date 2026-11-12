module

public import Transformer.Grokking.Protocol.AdamLimitWitness

/-!
# Under weight decay a coordinate that is idle again and again converges to zero only

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*; arXiv:2201.02177v1 (Power et al.),
appendix *Additional experimental details*: "AdamW optimizer with learning rate `10⁻³`, weight
decay `1`". The paper states no such fact; `review/grok_division_convergence.md`, section 5.

`AdamLimit` describes the limit of a run that sees one continuous field, and `AdamStationary`
a coordinate that converges without weight decay. A run of the protocol sees another batch at
every update, and here the weight decay is not zero.

Update `t` subtracts the learning rate times the *drive* of the coordinate (`adamDrive`): the
corrected first moment over the corrected root of the second moment plus `ε`
(`adamRun_succ_params_drive`). If the coordinate converges to `a`, the drive converges to
`− wd · a` (`tendsto_adamDrive`). Call an update *idle* in a coordinate when the gradient it
sees there is `0`. An idle update multiplies the first moment by `β₁` and the second by `β₂`,
so it shrinks the drive by the factor `β₁ / √β₂` up to the bias corrections
(`abs_adamDrive_succ_le`); the factor is below `1` when `β₁² < β₂`. A drive that converges
and is shrunk by a fixed factor again and again converges to `0`. Hence, with a weight decay
that is not zero:

    a coordinate that converges and is idle at infinitely many updates converges to `0`

(`eq_zero_of_tendsto_adamRun_of_frequently_idle`), whatever fields the updates see and
whatever the other coordinates do. Without weight decay this fails: a coordinate that is
always idle keeps its start. The source applies the decay only when it is positive, and
`adamStep` writes one formula for every decay: at a negative decay the statement is about
that formula, not about the source. Nothing here says that a run converges.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {ι : Type*} {h : Hyper} {grad : ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : ι → ℝ} {i : ι} {a : ℝ}

/-- **The drive of update `t`** in a coordinate: the corrected first moment over the corrected
root of the second moment plus `ε`. The update subtracts the learning rate times it. -/
noncomputable def adamDrive (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) (t : ℕ)
    (i : ι) : ℝ :=
  (adamRun h grad θ₀ t).first i /
    (√((adamRun h grad θ₀ t).second i) / √(1 - h.beta2 ^ t) + h.eps) / (1 - h.beta1 ^ t)

/-- The parameters after update `t + 1`, in the drive of that update. -/
lemma adamRun_succ_params_drive (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) (t : ℕ)
    (i : ι) :
    (adamRun h grad θ₀ (t + 1)).params i =
      (1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i -
        h.rateAt (t + 1) * adamDrive h grad θ₀ (t + 1) i := by
  rw [adamRun_succ_params, adamDrive]
  ring

/-- **The drive of a coordinate that converges** to `a` converges to `− wd · a`. -/
theorem tendsto_adamDrive (hr : h.rate ≠ 0)
    (hlim : Tendsto (fun t => (adamRun h grad θ₀ t).params i) atTop (𝓝 a)) :
    Tendsto (fun t => adamDrive h grad θ₀ (t + 1) i) atTop (𝓝 (-(h.decay * a))) := by
  have hnext := (tendsto_add_atTop_iff_nat 1).2 hlim
  have h0 : Tendsto (fun t => ((1 - h.rate * h.decay) * (adamRun h grad θ₀ t).params i -
      (adamRun h grad θ₀ (t + 1)).params i) / h.rate) atTop
      (𝓝 (((1 - h.rate * h.decay) * a - a) / h.rate)) :=
    ((hlim.const_mul _).sub hnext).div_const _
  have ha : ((1 - h.rate * h.decay) * a - a) / h.rate = -(h.decay * a) := by
    field_simp
    ring
  rw [ha] at h0
  refine h0.congr' ((eventually_gt_atTop h.warmup).mono fun t ht => ?_)
  have hrate : h.rateAt (t + 1) = h.rate := Hyper.rateAt_of_lt (by simpa using ht)
  rw [adamRun_succ_params_drive, hrate]
  field_simp
  ring

/-- The arithmetic of an idle update: the first moment is multiplied by `β₁` and the root of
the second by `q = √β₂`, while the bias corrections grow from `A`, `B` to `A'`, `B'`. -/
lemma abs_idle_drive_le {β₁ q m s B B' A A' ε : ℝ} (h1 : 0 ≤ β₁) (hq : 0 < q) (hq' : q ≤ 1)
    (hs : 0 ≤ s) (hB : 0 < B) (hBB : B ≤ B') (hA : 0 < A) (hAA : A ≤ A') (he : 0 < ε) :
    |β₁ * m / (q * s / B' + ε) / A'| ≤ β₁ / q * (B' / B) * |m / (s / B + ε) / A| := by
  have hB' : 0 < B' := hB.trans_le hBB
  have hA' : 0 < A' := hA.trans_le hAA
  have hD : 0 < s / B + ε := by positivity
  have hD' : 0 < q * s / B' + ε := by positivity
  have key : q * B * ((s / B + ε) * A) ≤ B' * ((q * s / B' + ε) * A') := by
    have e1 : q * B * (s / B + ε) = q * s + q * B * ε := by field_simp
    have e2 : B' * (q * s / B' + ε) = q * s + B' * ε := by field_simp
    have h3 : q * B * ε ≤ B' * ε := by
      have : q * B ≤ 1 * B' := mul_le_mul hq' hBB hB.le zero_le_one
      nlinarith
    calc q * B * ((s / B + ε) * A) = (q * s + q * B * ε) * A := by rw [← e1]; ring
      _ ≤ (q * s + B' * ε) * A' := mul_le_mul (by linarith) hAA hA.le (by positivity)
      _ = B' * ((q * s / B' + ε) * A') := by rw [← e2]; ring
  rw [abs_div, abs_div, abs_div, abs_div, abs_mul, abs_of_nonneg h1, abs_of_pos hD',
    abs_of_pos hA', abs_of_pos hD, abs_of_pos hA, div_div, div_div]
  have e : β₁ / q * (B' / B) * (|m| / ((s / B + ε) * A)) =
      β₁ * |m| * B' / (q * B * ((s / B + ε) * A)) := by
    field_simp
  rw [e, div_le_div_iff₀ (by positivity) (by positivity)]
  have := mul_le_mul_of_nonneg_left key (mul_nonneg h1 (abs_nonneg m))
  linarith

/-- **An idle update shrinks the drive**: if update `t + 1` sees the gradient `0` in a
coordinate, the drive there is at most `β₁ / √β₂` times the drive of update `t`, up to the
ratio of the bias corrections of the second moment. -/
theorem abs_adamDrive_succ_le (h1 : 0 ≤ h.beta1) (h1' : h.beta1 < 1) (h2 : 0 < h.beta2)
    (h2' : h.beta2 < 1) (he : 0 < h.eps) {t : ℕ} (ht : 1 ≤ t)
    (hg : grad (t + 1) (adamRun h grad θ₀ t).params i = 0) :
    |adamDrive h grad θ₀ (t + 1) i| ≤
      h.beta1 / √h.beta2 * (√(1 - h.beta2 ^ (t + 1)) / √(1 - h.beta2 ^ t)) *
        |adamDrive h grad θ₀ t i| := by
  have hAA : 1 - h.beta1 ^ t ≤ 1 - h.beta1 ^ (t + 1) := by
    have := pow_le_pow_of_le_one h1 h1'.le (Nat.le_add_right t 1)
    linarith
  have hBB : √(1 - h.beta2 ^ t) ≤ √(1 - h.beta2 ^ (t + 1)) := Real.sqrt_le_sqrt (by
    have := pow_le_pow_of_le_one h2.le h2'.le (Nat.le_add_right t 1)
    linarith)
  have hfirst : (adamRun h grad θ₀ (t + 1)).first i =
      h.beta1 * (adamRun h grad θ₀ t).first i := by
    rw [adamRun_succ_first, hg]
    ring
  have hsecond : (adamRun h grad θ₀ (t + 1)).second i =
      h.beta2 * (adamRun h grad θ₀ t).second i := by
    rw [adamRun_succ_second, hg]
    ring
  rw [adamDrive, adamDrive, hfirst, hsecond, Real.sqrt_mul h2.le]
  exact abs_idle_drive_le h1 (Real.sqrt_pos.2 h2) (Real.sqrt_le_one.2 h2'.le)
    (Real.sqrt_nonneg _) (Real.sqrt_pos.2 (one_sub_pow_pos h2.le h2' ht)) hBB
    (one_sub_pow_pos h1 h1' ht) hAA he

/-- **Under weight decay a coordinate that converges and is idle at infinitely many updates
converges to zero**: if the weight decay is not `0`, `β₁² < β₂`, the coordinate `i` of the run
converges to `a`, and infinitely many updates see the gradient `0` in it, then `a = 0`. -/
theorem eq_zero_of_tendsto_adamRun_of_frequently_idle (hh : h.Regular) (h2 : 0 < h.beta2)
    (h12 : h.beta1 ^ 2 < h.beta2) (hd : h.decay ≠ 0)
    (hlim : Tendsto (fun t => (adamRun h grad θ₀ t).params i) atTop (𝓝 a))
    (hidle : ∃ᶠ t in atTop, grad (t + 1) (adamRun h grad θ₀ t).params i = 0) : a = 0 := by
  obtain ⟨hr, h1, h1', -, h2', he⟩ := hh
  have hc1 := tendsto_adamDrive hr hlim
  have hc0 : Tendsto (fun t => adamDrive h grad θ₀ t i) atTop (𝓝 (-(h.decay * a))) :=
    (tendsto_add_atTop_iff_nat 1).1 hc1
  have hq : 0 < √h.beta2 := Real.sqrt_pos.2 h2
  have hp : Tendsto (fun t : ℕ => √(1 - h.beta2 ^ t)) atTop (𝓝 1) := by
    have h3 : Tendsto (fun t : ℕ => 1 - h.beta2 ^ t) atTop (𝓝 (1 - 0)) :=
      tendsto_const_nhds.sub (tendsto_pow_atTop_nhds_zero_of_lt_one h2.le h2')
    have h4 := (Real.continuous_sqrt.tendsto _).comp h3
    rwa [sub_zero, Real.sqrt_one] at h4
  have hfac : Tendsto (fun t : ℕ => √(1 - h.beta2 ^ (t + 1)) / √(1 - h.beta2 ^ t)) atTop
      (𝓝 (1 / 1)) := ((tendsto_add_atTop_iff_nat 1).2 hp).div hp one_ne_zero
  have hf := hc1.abs.sub (((hfac.const_mul (h.beta1 / √h.beta2))).mul hc0.abs)
  have hle := le_of_tendsto_of_frequently hf
    ((hidle.and_eventually (eventually_ge_atTop 1)).mono fun t ht =>
      sub_nonpos.2 (abs_adamDrive_succ_le h1 h1' h2 h2' he ht.2 ht.1))
  have hlt : h.beta1 / √h.beta2 < 1 := (div_lt_one hq).2 ((Real.lt_sqrt h1).2 h12)
  have habs : |-(h.decay * a)| ≤ 0 := by
    have h5 := abs_nonneg (-(h.decay * a))
    rw [div_one, mul_one] at hle
    nlinarith
  have h0 : h.decay * a = 0 := by simpa using abs_nonpos_iff.1 habs
  exact (mul_eq_zero.1 h0).resolve_left hd

/-! The witnesses: the source's constants with weight decay `1`, and the run that stays at
zero on the zero field, where every update is idle. The first lemma is the arithmetic at the
source's `β₁ = 0.9` and `q = 0.9`. -/

example : |(9 / 10 : ℝ) * 1 / (9 / 10 * 1 / 1 + 1) / 1| ≤
    (9 / 10 : ℝ) / (9 / 10) * (1 / 1) * |1 / (1 / 1 + 1) / 1| :=
  abs_idle_drive_le (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    le_rfl (by norm_num) le_rfl (by norm_num)

example := adamRun_succ_params_drive (Hyper.source 1) (fun _ _ (_ : Unit) => (0 : ℝ)) 0 3 ()

example : Tendsto (fun t => adamDrive (Hyper.source 1) (fun _ _ => 0) (0 : Unit → ℝ) (t + 1) ())
    atTop (𝓝 (-((Hyper.source 1).decay * 0))) :=
  tendsto_adamDrive (by norm_num [Hyper.source]) (by simp [adamRun_zero])

example := abs_adamDrive_succ_le (h := Hyper.source 1) (grad := fun _ _ => 0)
  (θ₀ := (0 : Unit → ℝ)) (i := ()) (t := 1) (by norm_num [Hyper.source])
  (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
  (by norm_num [Hyper.source]) le_rfl rfl

example : (0 : ℝ) = 0 :=
  eq_zero_of_tendsto_adamRun_of_frequently_idle (h := Hyper.source 1) (grad := fun _ _ => 0)
    (θ₀ := (0 : Unit → ℝ)) (i := ()) (Hyper.source_regular 1) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by simp [adamRun_zero])
    (Frequently.of_forall fun _ => rfl)

end Transformer.Grokking.Protocol
