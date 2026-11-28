module

public import Transformer.Precision.Binary32Error

/-!
# The binary32 AdamW kernel at small learning rates

Source: the `adam` kernel of `research/main_model/phase_score_native/src/optimizer_kernels.rs`
in the GPT checkout pinned by `experiments/argand/build.rs` (revision `5a37c515`). For one
coordinate it executes fifteen PTX instructions, each rounding its exact result to nearest
binary32 once (`fma.rn` included):

`m ← fma(0.1, g - m, m)`, `v ← fma(0.05, g·g, v·0.95)`,
`z ← √v / bc2 + 1e-8`, `t ← (lr / bc1)·m / z`, `p ← p·(1 - lr·decay) - t`.

`adamKernel` transcribes them with `Q` any nearest rounding to `grid 8 23`; the tie rule
stays free. The PTX embedded in the registered persistence binary (`experiments/argand`,
sha256 `27388112…8903`) has this sequence with the immediates `0f3DCCCCCD`, `0f3F733333`,
`0f3D4CCCCD`, `0f322BCC77`, `0f3F800000`; the Argand runner passes `decay = 0.1f`.
Overflow, NaN and flush-to-zero are not modeled.

At every rate `0 ≤ lr ≤ 2^{-22}` the decoupled decay factor `1 - lr·0.1` rounds to exactly
`1` (`kernel_decay_eq_one`): the registered WSD floor `0x3300D959 ≈ 3·10⁻⁸` qualifies
(`floorRate32_decay_eq_one`). There the kernel applies no weight decay
(`adamKernel_weight_of_small_rate`), unlike the real update `(1 - η/10)·p - η·d`, and every
coordinate of magnitude at least `1` whose Adam step is below `2^{-25}` stays fixed
(`adamKernel_frozen`): the spacing of binary32 numbers near such a coordinate is at least
`2^{-24}` (`binary32_spacing`).
-/

@[expose] public section
noncomputable section

namespace Transformer.Precision

/-- binary32 `0.1f`, bits `0x3DCCCCCD`. -/
def tenth32 : ℝ := binary32Value false 123 5033165

/-- binary32 `0.95f`, bits `0x3F733333`. -/
def ninetyFive32 : ℝ := binary32Value false 126 7549747

/-- binary32 `0.05f`, bits `0x3D4CCCCD`. -/
def twentieth32 : ℝ := binary32Value false 122 5033165

/-- binary32 `1e-8f`, bits `0x322BCC77`. -/
def eps32 : ℝ := binary32Value false 100 2870391

/-- The rate the runner passes on the WSD floor of the registered division protocol:
`(f64::from(1e-8f32) * 3) as f32`, bits `0x3300D959`. -/
def floorRate32 : ℝ := binary32Value false 102 55641

/-- A binary32 number of magnitude at least `1/2` is an integer multiple of `2^{-24}`. -/
theorem grid_int_multiple {w : ℝ} (hw : w ∈ grid 8 23) (half : 1/2 ≤ |w|) :
    ∃ a : ℤ, w * 2^24 = a := by
  obtain ⟨b,-,hb⟩ := hw
  obtain ⟨n,hn,k,hk⟩ := ieee_abs hb
  have hn' : (n:ℝ) < 2^24 := by exact_mod_cast hn
  have hk24 : -24 ≤ k := by
    by_contra hlt
    have hle : (2:ℝ)^k ≤ 2^(-25:ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h25 : (2:ℝ)^(-25:ℤ) = 1/2^25 := by norm_num
    have : |w| < 1/2 := by
      rw [hk]
      calc (n:ℝ) * 2^k ≤ n * (1/2^25) := by rw [← h25]; gcongr
        _ < 2^24 * (1/2^25) := by gcongr
        _ = 1/2 := by norm_num
    linarith
  obtain ⟨j,hj⟩ : ∃ j : ℕ, k = j - 24 := ⟨(k+24).toNat, by omega⟩
  have habs : |w| * 2^24 = ((n * 2^j : ℕ) : ℝ) := by
    rw [hk, hj, zpow_sub₀ (by norm_num), zpow_natCast]
    push_cast
    field_simp
  rcases abs_choice w with h | h
  · exact ⟨(n * 2^j : ℕ), by rw [Int.cast_natCast, ← habs, h]⟩
  · exact ⟨-((n * 2^j : ℕ) : ℤ), by rw [Int.cast_neg, Int.cast_natCast, ← habs, h]; ring⟩

/-- **binary32 spacing near large numbers.** Another binary32 number is at least
`2^{-24}` away from one of magnitude at least `1`. -/
theorem binary32_spacing {p z : ℝ} (hp : p ∈ grid 8 23) (hz : z ∈ grid 8 23) (h1 : 1 ≤ |p|)
    (hne : z ≠ p) : 1/2^24 ≤ |z - p| := by
  by_cases hzs : |z| < 1/2
  · have h := abs_sub_abs_le_abs_sub p z
    rw [abs_sub_comm] at h
    have : (1:ℝ)/2^24 ≤ 1/2 := by norm_num
    linarith
  · obtain ⟨a,ha⟩ := grid_int_multiple hz (le_of_not_gt hzs)
    obtain ⟨b,hb⟩ := grid_int_multiple hp (by linarith)
    have hab : a ≠ b := by
      rintro rfl
      exact hne (mul_right_cancel₀ (by positivity) (ha.trans hb.symm))
    have h1ab : (1:ℝ) ≤ |(a:ℝ) - b| := by exact_mod_cast Int.one_le_abs (sub_ne_zero.mpr hab)
    have he : |z - p| * 2^24 = |(a:ℝ) - b| := by
      rw [← ha, ← hb, ← sub_mul, abs_mul, abs_of_pos (by positivity : (0:ℝ) < 2^24)]
    rw [div_le_iff₀ (by positivity)]
    linarith

theorem one_mem_grid : (1:ℝ) ∈ grid 8 23 := by
  have h := binary32_value_mem false 127 0 (by norm_num) (by norm_num)
  norm_num [binary32Value] at h
  exact h

example : ∃ a : ℤ, (1:ℝ) * 2^24 = a := grid_int_multiple one_mem_grid (by norm_num)

example : (1:ℝ)/2^24 ≤ |0 - 1| :=
  binary32_spacing one_mem_grid (zero_mem_grid (by norm_num) 23) (by norm_num) (by norm_num)

/-- **The kernel's decay factor is exactly one at small rates.** For `0 ≤ lr ≤ 2^{-22}`,
`fl(1 - fl(lr·0.1f)) = 1` under every nearest rounding. -/
theorem kernel_decay_eq_one {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q) {lr : ℝ}
    (h0 : 0 ≤ lr) (h : lr ≤ 1/2^22) : Q (1 - Q (lr * tenth32)) = 1 := by
  have ht : tenth32 = 13421773/2^27 := by norm_num [tenth32, binary32Value]
  have hc : 1/2^22 * tenth32 ∈ grid 8 23 := by
    have h := binary32_value_mem false 101 5033165 (by norm_num) (by norm_num)
    convert h using 1
    rw [ht]; norm_num [binary32Value]
  have hx0 : 0 ≤ lr * tenth32 := mul_nonneg h0 (by rw [ht]; positivity)
  have hxc : lr * tenth32 ≤ 1/2^22 * tenth32 :=
    mul_le_mul_of_nonneg_right h (by rw [ht]; positivity)
  have up := (hQ (lr * tenth32)).2 _ hc
  have low := (hQ (lr * tenth32)).2 0 (zero_mem_grid (by norm_num) 23)
  rw [abs_of_nonneg (sub_nonneg.mpr hxc)] at up
  rw [zero_sub,abs_neg,abs_of_nonneg hx0] at low
  have dlow : 0 ≤ Q (lr * tenth32) := by linarith [(abs_le.mp low).1]
  have dup : Q (lr * tenth32) ≤ 1/2^22 * tenth32 := by linarith [(abs_le.mp up).2]
  have small : 2 * |-Q (lr * tenth32)| < 1/2^24 := by
    rw [abs_neg,abs_of_nonneg dlow]
    calc 2 * Q (lr * tenth32) ≤ 2 * (1/2^22 * tenth32) := by linarith
      _ < 1/2^24 := by rw [ht]; norm_num
  have := hQ.add_eq_of_abs one_mem_grid
    (fun z hz hne => binary32_spacing one_mem_grid hz (by norm_num) hne) small
  rwa [← sub_eq_add_neg] at this

/-- The registered WSD floor lies in the range of `kernel_decay_eq_one`. -/
theorem floorRate32_decay_eq_one {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q) :
    Q (1 - Q (floorRate32 * tenth32)) = 1 :=
  kernel_decay_eq_one hQ (by norm_num [floorRate32, binary32Value])
    (by norm_num [floorRate32, binary32Value])

example : ∃ Q, IsNearest (grid 8 23) Q ∧ Q (1 - Q (1/2^22 * tenth32)) = 1 := by
  obtain ⟨Q,hQ⟩ := exists_isNearest_grid (E := 8) (by norm_num) 23
  exact ⟨Q,hQ,kernel_decay_eq_one hQ (by norm_num) le_rfl⟩

example : ∃ Q, IsNearest (grid 8 23) Q ∧ Q (1 - Q (floorRate32 * tenth32)) = 1 := by
  obtain ⟨Q,hQ⟩ := exists_isNearest_grid (E := 8) (by norm_num) 23
  exact ⟨Q,hQ,floorRate32_decay_eq_one hQ⟩

/-- The step `fl(fl(fl(lr/bc1)·m')/z)` the kernel subtracts, from the updated moments. -/
def kernelStep (Q : ℝ → ℝ) (m v g lr bc1 bc2 : ℝ) : ℝ :=
  let m' := Q (tenth32 * Q (g - m) + m)
  let v' := Q (twentieth32 * Q (g * g) + Q (v * ninetyFive32))
  Q (Q (Q (lr / bc1) * m') / Q (Q (Q (Real.sqrt v') / bc2) + eps32))

/-- One coordinate of the kernel: new weight, first and second moment. -/
def adamKernel (Q : ℝ → ℝ) (p m v g lr decay bc1 bc2 : ℝ) : ℝ × ℝ × ℝ :=
  (Q (Q (p * Q (1 - Q (lr * decay))) - kernelStep Q m v g lr bc1 bc2),
    Q (tenth32 * Q (g - m) + m), Q (twentieth32 * Q (g * g) + Q (v * ninetyFive32)))

/-- **No weight decay at small rates.** With the runner's `decay = 0.1f` and
`0 ≤ lr ≤ 2^{-22}`, the kernel's weight update is `fl(p - step)`. -/
theorem adamKernel_weight_of_small_rate {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q)
    {p m v g lr bc1 bc2 : ℝ} (hp : p ∈ grid 8 23) (h0 : 0 ≤ lr) (h : lr ≤ 1/2^22) :
    (adamKernel Q p m v g lr tenth32 bc1 bc2).1 = Q (p - kernelStep Q m v g lr bc1 bc2) := by
  simp only [adamKernel, kernel_decay_eq_one hQ h0 h, mul_one, hQ.eq_self hp]

example : ∃ Q, IsNearest (grid 8 23) Q ∧ (adamKernel Q 1 0 0 0 floorRate32 tenth32 1 1).1 =
    Q (1 - kernelStep Q 0 0 0 floorRate32 1 1) := by
  obtain ⟨Q,hQ⟩ := exists_isNearest_grid (E := 8) (by norm_num) 23
  exact ⟨Q,hQ,adamKernel_weight_of_small_rate hQ one_mem_grid
    (by norm_num [floorRate32, binary32Value]) (by norm_num [floorRate32, binary32Value])⟩

/-- **Large coordinates are frozen.** A coordinate of magnitude at least `1` does not
move when the kernel's Adam step is below `2^{-25}`. -/
theorem adamKernel_frozen {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q)
    {p m v g lr bc1 bc2 : ℝ} (hp : p ∈ grid 8 23) (h1 : 1 ≤ |p|) (h0 : 0 ≤ lr)
    (h : lr ≤ 1/2^22) (hs : 2 * |kernelStep Q m v g lr bc1 bc2| < 1/2^24) :
    (adamKernel Q p m v g lr tenth32 bc1 bc2).1 = p := by
  rw [adamKernel_weight_of_small_rate hQ hp h0 h, sub_eq_add_neg]
  exact hQ.add_eq_of_abs hp (fun z hz hne => binary32_spacing hp hz h1 hne) (by rwa [abs_neg])

/-- With zero moments and gradient the step vanishes, so the kernel keeps `p = 1` at the
floor rate, where the real update would give `1 - η/10`. -/
example : ∃ Q, IsNearest (grid 8 23) Q ∧ (adamKernel Q 1 0 0 0 floorRate32 tenth32 1 1).1 = 1 := by
  obtain ⟨Q,hQ⟩ := exists_isNearest_grid (E := 8) (by norm_num) 23
  have z : Q 0 = 0 := hQ.eq_self (zero_mem_grid (by norm_num) 23)
  refine ⟨Q,hQ,adamKernel_frozen hQ one_mem_grid (by norm_num)
    (by norm_num [floorRate32, binary32Value]) (by norm_num [floorRate32, binary32Value]) ?_⟩
  simp [kernelStep, z]

end Transformer.Precision
