module

public import Transformer.LeRoPEBlock.Defs
public import Transformer.LeRoPEBlock.Basic
public import Transformer.LeRoPEBlock.PiHead
public import Transformer.LeRoPEBlock.PiSwap
public import Transformer.LeRoPEBlock.NoPE
public import Transformer.LeRoPEBlock.Limit
public import Transformer.LeRoPEBlock.TwoPi
public import Transformer.LeRoPEBlock.Bound
public import Transformer.LeRoPEBlock.SlotFejer
public import Transformer.LeRoPEBlock.SlotWeight
public import Transformer.LeRoPEBlock.SlotDFT

/-!
# A block of DiffGPT's main model with LeRoPE, against other positional schemes

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`), `research/main_model/README.txt`. Review
note: `review/lerope_block.md`.

The question is whether the main model's block with LeRoPE approximates better than the same
block with another positional scheme and nothing else changed. Everything but the attention acts
on one position at a time, so the block at the query `i` is a map `Φ` of the query's token and of
`H` attentions whose scores read the distance `i − j` of each key (`block`). The statements leave
`Φ`, the projections, the values and the number of heads arbitrary, so they cover the main
model's own.

* **No positional encoding.** A block whose scores ignore the distance does not see the order of
  its keys: permuting the keys `j < i` leaves it unchanged (`block_noPos_perm`), so it misses
  every target that sees that order by half of what the permutation changes
  (`norm_sub_le_of_noPos`). LeRoPE's π head, one head of the main model with one channel of
  frequency `π`, sees the order of the first two keys: swapping them changes its output by at
  least `(1 − N^{−2κ}) / N` times `‖u_a − u_b‖` (`norm_piHead_sub_swap`), so every block with no
  positional encoding misses it by at least half of that (`gap_noPos_piHead`). Conversely LeRoPE
  loses nothing: its frequencies are positive, never zero, but the frequency `2π` leaves a plane
  unrotated at natural distances, so every rotary block with frequencies `θ ≥ 0` (RoPE's grid,
  partial RoPE, no rotation at all) is a LeRoPE block with the same `Φ`, projections and values
  on every input (`exists_pos_block_eq`, `exists_lerope_eq_noPos`), the positive frequencies
  being LeRoPE's own (`exists_alpha_block_eq`). Small frequencies give it up to `ε` as well
  (`exists_lerope_close`).
* **Logits.** The score of every head of the main model, SSMax's gain `ln(i + 1)/d` times
  `Re Z + β (Im Z)² / d`, is at most `(5/4) ln (i + 1)` for every distance, every frequency and
  every `|β| ≤ 1`, once `Q x` and `K y` have squared length at most `d`, as after the per-head
  Q/K RMS (`leropeLogit_abs_le`, `plainLogit_abs_le`).
* **Reading a slot.** A head of the main model keeps at most `1 / (1 + R N^{−(1 + |β|)(1 + 1/R)})`
  of its weight on a key whose `R` neighbours on each side carry its token, `N = i + 1`, whatever
  the frequencies (`depthWeight_leropeLogit_le`). Over a window of distances `c + s − s'` the bands
  of the score sum to their value at the slot times a Fejér factor in `[0, n²]`, so the margins sum
  to at most `n²` times the amplitude (`sum_sum_phaseLogit_sub_le`), and Jensen's inequality turns
  that into the weight (`depthWeight_le_of_sum_sum_le`). At `β = 0` the middle of three keys keeps
  at most `9/10`, and the bound tends to `2/3`. The cap is nearly reached: `R` bands at the
  frequencies of the discrete Fourier transform on `2R + 1` points keep
  `1 / (1 + 2R N^{−1−1/(2R)})` on the middle key, more than half (`depthWeight_dft`,
  `depthWeight_dft_mem`).
-/
