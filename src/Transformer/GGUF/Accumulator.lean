/-
# A float accumulator stops counting

`ggml_compute_forward_flash_attn_ext_f16_one_chunk` in
`ggml/src/ggml-cpu/ops.cpp` (llama.cpp commit `335b21f`) is the CPU flash
attention of one query row.  It runs the online softmax over the keys `k` of a
range `[ic_start, ic_end)`: with `M` the running maximum score and
`p_k = exp(s_k - M) ≤ 1`, it keeps

* `VKQ += v_k · p_k`, the unnormalized output, and
* `S = S · ms + p_k`, the denominator (`float`),

and divides once at the end.  When the value cache is `F16`, the llama.cpp
default, `VKQ` is the `ggml_fp16_t` array `VKQ16`, and `ggml_vec_mad_f16`
(`ggml/src/ggml-cpu/vec.h`) stores it back to binary16 after every key:
`y[i] = FP32_TO_FP16(FP16_TO_FP32(y[i]) + FP16_TO_FP32(x[i]) * v)`.

That is `a ← r(a + t)` with `r` a rounding to binary16.  Once `a ≥ 2^{10} · 2^s`,
the binary16 numbers above `a` are at least `2^s` apart, and an increment
`0 ≤ t < 2^{s-1}` rounds back to `a` (`Precision.accum_stall`, through
`grid_hasSignificand`): **an accumulator stops
counting once it is `2^{11}` times its increments**, whatever their scale.  The
binary32 `S` keeps growing, until it is `2^{24}` times its own.  The output
`VKQ / S` then decays like `1 / n` instead of converging: for `v ≡ 1` it should
be exactly `1`, and after some `2^{11}` keys of comparable weight it is
`a / S → 0` (`fa_f16_output_le`).  `accum_stall` at `M = 23` is the stall of
`S` itself.

Which keys go through this loop: decoding (one query) with `F16` `K` and `V`
and at least 512 keys takes `use_split_kv_path`, splitting the keys into one
chunk per thread, each with its own `VKQ16`, merged in `float` by
`ggml_flash_attn_ext_reduce_partials`; so the stall is per chunk of `n / nth`
keys.  Batches of fewer than `Q_TILE_SZ` queries, decoding under 512 keys, and
a quantized `K` over an `F16` `V` (`-ctk q8_0 -ctv f16`) run the whole range
through one `VKQ16`.  Prompt processing in tiles
(`ggml_compute_forward_flash_attn_ext_tiled`) and a quantized `V` accumulate in
`float` instead.

The model here is one rounding per key, `r(a + t)` with `t` the exact product:
exactly the SVE and `zvfh` paths of `ggml_vec_mad_f16` (where `v` itself is first
rounded to binary16).  The x86 and scalar paths round `a + t` to binary32
first; that double rounding is not covered.  A new maximum rescales `VKQ16` by
`ms < 1`; the theorems are about runs of keys below the maximum, where `ms = 1`.
-/

import Transformer.GGUF.Nearest
import Transformer.Precision.Accumulate

namespace Transformer
namespace GGUF

open Precision

/-- Every number of the format is `± n · 2^k` with `n < 2^{M+1}`. -/
theorem ieee_abs {E M b : ℕ} {z : ℝ} (h : ieee E M b = some z) :
    ∃ n : ℕ, n < 2 ^ (M + 1) ∧ ∃ k : ℤ, |z| = n * (2 : ℝ) ^ k := by
  have hsgn : |(if b / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1)| = 1 := by
    split_ifs <;> simp
  have hm : b % 2 ^ M < 2 ^ M := Nat.mod_lt _ (by positivity)
  unfold ieee at h
  simp only at h
  generalize (if b / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1) = sgn at h hsgn
  split_ifs at h <;> obtain rfl := Option.some.inj h
  · refine ⟨b % 2 ^ M, by rw [pow_succ]; omega, 1 - bias E - M, ?_⟩
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _)]
  · refine ⟨2 ^ M + b % 2 ^ M, by rw [pow_succ]; omega, ((b / 2 ^ M % 2 ^ E : ℕ) : ℤ) - bias E - M, ?_⟩
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _)]
    push_cast; rfl

/-- **An IEEE format with `M` mantissa bits has `M + 1` significant bits**, so
`Precision.accum_stall` and `Precision.accum_le` apply to it. -/
theorem grid_hasSignificand (E M : ℕ) : HasSignificand (M + 1) (grid E M) :=
  fun _ ⟨_, _, h⟩ => ieee_abs h

/-- **Flash attention in binary16 loses the signal.**  Values `v ≡ 1`, so the
exact output is `1`; weights `p_k ∈ [c, 2^{s-1})` past key `m`, and a binary16
accumulator `a` that has reached `2^{10} · 2^s` there.  With the denominator
`S` kept exactly, the computed output after `n` keys is at most
`a_m / ((n - m) c)`, and goes to `0`. -/
theorem fa_f16_output_le {Q : ℝ → ℝ} (hQ : IsNearest (grid 5 10) Q) {a p S : ℕ → ℝ}
    (ha : ∀ k, a (k + 1) = Q (a k + p k)) (hS : ∀ k, S (k + 1) = S k + p k) {m : ℕ} {s : ℤ} {c : ℝ}
    (hm : a m ∈ grid 5 10) (hge : (2 : ℝ) ^ 10 * 2 ^ s ≤ a m) (hS0 : 0 ≤ S m) (hc : 0 < c)
    (hp : ∀ k, m ≤ k → c ≤ p k ∧ 2 * p k < 2 ^ s) {n : ℕ} (hn : m < n) :
    a n / S n ≤ a m / ((n - m) * c) := by
  have hstall := accum_stall (grid_hasSignificand 5 10) hQ ha hm hge
    (fun k hk => ⟨by linarith [(hp k hk).1], (hp k hk).2⟩) n hn.le
  have hSn : ∀ j, S m + j * c ≤ S (m + j) := by
    intro j
    induction j with
    | zero => simp
    | succ j ih =>
      rw [← add_assoc, hS]; push_cast
      linarith [(hp (m + j) (by omega)).1]
  have := hSn (n - m)
  rw [Nat.add_sub_cancel' hn.le, Nat.cast_sub hn.le] at this
  have hpos : 0 < ((n : ℝ) - m) * c := mul_pos (by
    have : (m : ℝ) < n := by exact_mod_cast hn
    linarith) hc
  have ha0 : 0 ≤ a m := le_trans (by positivity) hge
  rw [hstall]
  exact div_le_div_of_nonneg_left ha0 hpos (by linarith)

/-- The hypotheses of `fa_f16_output_le` are satisfiable: from `a_0 = 2048`
(`0x6800`) and `S_0 = 0`, the weights `p_k = 1/2`. -/
example : ∃ (Q : ℝ → ℝ) (a p S : ℕ → ℝ), IsNearest (grid 5 10) Q ∧
    (∀ k, a (k + 1) = Q (a k + p k)) ∧ (∀ k, S (k + 1) = S k + p k) ∧ a 0 ∈ grid 5 10 ∧
    (2 : ℝ) ^ 10 * 2 ^ (1 : ℤ) ≤ a 0 ∧ 0 ≤ S 0 ∧ (0 : ℝ) < 1 / 2 ∧
    ∀ k, 0 ≤ k → 1 / 2 ≤ p k ∧ 2 * p k < 2 ^ (1 : ℤ) := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 5) (by norm_num) 10
  refine ⟨Q, fun k => Nat.rec 2048 (fun _ ak => Q (ak + 1 / 2)) k, fun _ => 1 / 2,
    fun k => (k : ℝ) / 2, hQ, fun _ => rfl, fun k => by push_cast; ring,
    ⟨0x6800, by norm_num, by norm_num [ieee, bias]⟩, by norm_num, by norm_num, by norm_num,
    fun _ _ => by norm_num⟩

end GGUF
end Transformer
