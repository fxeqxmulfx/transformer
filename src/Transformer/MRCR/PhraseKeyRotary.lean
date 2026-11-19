module

public import Transformer.MRCR.PhraseKeySandwich
public import Transformer.RoPE.YaRN.Section3_Algorithm
public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies

/-!
# RoPE, YaRN and LeRoPE do not read the token before

In no paper. A head with rotary embeddings (arXiv:2309.00071v3, §2.1, Eq. (5)) whose query and
key carry, beside the content, a constant pair of coordinates in each rotated plane scores the
key at the distance `i − j` by the dot product `x` of the content plus a RoPE score of the
distance, `Transformer.MRCR.ropeScore` with the coefficients of the constant pairs
(`re_sum_rotary_eq_ropeScore`). `ropeLogit` divides `x` by `c` and multiplies both by a
temperature `g`. The frequencies are free: RoPE's geometric grid, YaRN's rescaled grid with its
temperature (arXiv:2309.00071v3, Definition 2 and §3.4, `yarnLogit`), and LeRoPE's learned
frequencies `e^{α_m} θ_m` (arXiv:2607.10134v1, §3.1, Eq. (5)).

A RoPE score of the distance does not converge unless it is constant (`ropeScore_eq_of_tendsto`):
a return `d` as far as wanted (`exists_return_ge`) brings the score at `t + d` as near as wanted
to the score at `t`. Rescaling the frequencies (YaRN) or learning them (LeRoPE) cannot change
that, since every set of frequencies has such returns. In particular the key at the distance
`1 + d` scores within `1 / (2g)` of the key at the distance `1`, so the logit cannot be `0` at
the token before and at most `−1` at every key two or more tokens back. `ReadsPrev` fails for
every frequency, coefficient, divisor, temperature `g > 0` and dot product
(`not_readsPrev_ropeLogit`, `not_readsPrev_yarn`, `not_readsPrev_lerope`), and the five layers
of `Transformer.MRCR.PhraseKeyLogits` have no RoPE, YaRN or LeRoPE version. Whether RoPE reads
it by other means, such as a head whose rounded weights vanish without a margin, is not settled
here.
-/

@[expose] public section

namespace Transformer.MRCR

open Filter Topology Finset Transformer.CRASP Transformer.CRASP.Grams

variable {K : ℕ}

theorem ropeNorm_nonneg (a b : Fin K → ℝ) : 0 ≤ ropeNorm a b :=
  sum_nonneg fun _ _ => by positivity

/-- **A RoPE score that converges is constant**: if the score at the distances `0, 1, 2, …`
tends to `L`, it is `L` at every distance. -/
theorem ropeScore_eq_of_tendsto (θ a b : Fin K → ℝ) {L : ℝ}
    (h : Tendsto (fun t : ℕ => ropeScore θ a b t) atTop (𝓝 L)) (t : ℕ) :
    ropeScore θ a b t = L := by
  refine eq_of_forall_dist_le fun ε hε => ?_
  obtain ⟨N, hN⟩ := Metric.tendsto_atTop.1 h (ε / 2) (half_pos hε)
  have hn := ropeNorm_nonneg a b
  obtain ⟨d, hdN, hd⟩ :=
    exists_return_ge θ (div_pos (half_pos hε) (by linarith : (0 : ℝ) < ropeNorm a b + 1)) N
  have h1 := abs_ropeScore_add_sub_le (a := a) (b := b) hd t
  have h2 := hN (t + d) (by omega)
  have h3 : ε / 2 / (ropeNorm a b + 1) * ropeNorm a b ≤ ε / 2 := by
    rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]
    nlinarith
  rw [Real.dist_eq] at h2 ⊢
  push_cast at h2
  have h4 := abs_le.1 (h1.trans h3)
  have h5 := abs_lt.1 h2
  exact abs_le.2 ⟨by linarith, by linarith⟩

/-- The hypothesis of `ropeScore_eq_of_tendsto` holds for the constant score of the frequency
`0`. -/
example := ropeScore_eq_of_tendsto (fun _ : Fin 1 => 0) (fun _ => 1) (fun _ => 0) (L := 1)
  (by simp [ropeScore]) 3

/-- **A RoPE head's logit**: the temperature `g` times the dot product `x` divided by `c` plus the
RoPE score of the distance `i − j`. -/
noncomputable def ropeLogit (g c : ℝ) (θ a b : Fin K → ℝ) (i j : ℕ) (x : ℝ) : ℝ :=
  g * (x / c + ropeScore θ a b ((i : ℝ) - j))

/-- **No RoPE logit meets `ReadsPrev`**, whatever the frequencies, coefficients, divisor `c`,
temperature `g > 0` and dot product `b₀`: at a return `d ≥ 1` the key at the distance `1 + d`
scores within `1 / (2g)` of the key at the distance `1`. -/
theorem not_readsPrev_ropeLogit {g : ℝ} (hg : 0 < g) (c : ℝ) (θ a b : Fin K → ℝ) (b₀ : ℕ) :
    ¬ ReadsPrev (ropeLogit g c θ a b) b₀ := fun hL => by
  have hn := ropeNorm_nonneg a b
  have hη : 0 < 1 / (2 * g * (ropeNorm a b + 1)) := by positivity
  obtain ⟨d, hd1, hd⟩ := exists_return_ge θ hη 1
  have h1 := abs_ropeScore_add_sub_le (a := a) (b := b) hd 1
  have hprev := hL.prev (1 + d) (by omega)
  have hfar := (hL.far (1 + d) 0 (by omega)).2
  simp only [ropeLogit] at hprev hfar
  rw [show ((1 + d : ℕ) : ℝ) - ((1 + d - 1 : ℕ) : ℝ) = 1 by
    rw [Nat.add_sub_cancel_left]; push_cast; ring] at hprev
  rw [show ((1 + d : ℕ) : ℝ) - ((0 : ℕ) : ℝ) = 1 + d by push_cast; ring] at hfar
  rw [mul_add] at hprev hfar
  have h3 : 1 / (2 * g * (ropeNorm a b + 1)) * ropeNorm a b ≤ 1 / (2 * g) := by
    rw [div_mul_eq_mul_div, one_mul, div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith
  have h4 := mul_le_mul_of_nonneg_left (abs_le.1 (h1.trans h3)).1 hg.le
  have e : g * -(1 / (2 * g)) = -1 / 2 := by field_simp
  rw [e, mul_sub] at h4
  linarith

/-- The hypothesis of `not_readsPrev_ropeLogit` holds for RoPE's grid of `64` frequencies with the
base `10000`, at the temperature `1`. -/
example := not_readsPrev_ropeLogit one_pos 1 (fun m : Fin 64 => RoPE.frequency 10000 128 m)
  0 0 1

/-- **YaRN's logit** (arXiv:2309.00071v3, Definition 2 and §3.4): its temperature
`logitGain (contextScale original target)` scales the dot product `x` divided by `c` as it scales
`fullScore`, the score of the coefficients `a, b` at YaRN's frequencies of `omega`. -/
noncomputable def yarnLogit (c : ℝ) (a b omega : Fin K → ℝ) (original target : ℝ) (i j : ℕ)
    (x : ℝ) : ℝ :=
  RoPE.YaRN.logitGain (RoPE.YaRN.contextScale original target) * (x / c) +
    RoPE.YaRN.fullScore a b omega original target ((i : ℝ) - j)

/-- YaRN's logit is a RoPE logit at YaRN's frequencies and temperature. -/
theorem yarnLogit_eq (c : ℝ) (a b omega : Fin K → ℝ) (original target : ℝ) :
    yarnLogit c a b omega original target =
      ropeLogit (RoPE.YaRN.logitGain (RoPE.YaRN.contextScale original target)) c
        (fun m => RoPE.YaRN.yarnFrequency (omega m) (RoPE.YaRN.contextScale original target)
          original) a b := by
  funext i j x
  simp only [yarnLogit, ropeLogit, RoPE.YaRN.fullScore, RoPE.Diagnostics.headScore, RoPE.score,
    ropeScore, mul_add]

/-- **YaRN does not meet `ReadsPrev`**, whatever the frequencies, coefficients, divisor, the
original and target lengths, and the dot product: its temperature is at least `1`. -/
theorem not_readsPrev_yarn (c : ℝ) (a b omega : Fin K → ℝ) (original target : ℝ) (b₀ : ℕ) :
    ¬ ReadsPrev (yarnLogit c a b omega original target) b₀ := by
  rw [yarnLogit_eq]
  have := RoPE.YaRN.logit_gain_ge_one (RoPE.YaRN.context_scale_ge_one original target)
  exact not_readsPrev_ropeLogit (by linarith) c _ a b b₀

/-- **Nor do LeRoPE's learned frequencies** `e^{α_m} θ_m` (arXiv:2607.10134v1, §3.1, Eq. (5)),
whatever they learn. -/
theorem not_readsPrev_lerope (c : ℝ) (a b omega alpha : Fin K → ℝ) (b₀ : ℕ) :
    ¬ ReadsPrev
      (ropeLogit 1 c (fun m => RoPE.LeRoPE.learnedFrequency (omega m) (alpha m)) a b) b₀ :=
  not_readsPrev_ropeLogit one_pos c _ a b b₀

end Transformer.MRCR
