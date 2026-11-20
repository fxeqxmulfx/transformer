module

public import Transformer.ReRoPEBlock.Defs
public import Transformer.ReRoPEBlock.Basic
public import Transformer.ReRoPEBlock.PiHead
public import Transformer.ReRoPEBlock.FarOrder
public import Transformer.ReRoPEBlock.Leaky
public import Transformer.ReRoPEBlock.Peak
public import Transformer.ReRoPEBlock.Softmax
public import Transformer.ReRoPEBlock.Returns
public import Transformer.ReRoPEBlock.Clamp

/-!
# A block of DiffGPT's main model: LeRoPE against ReRoPE

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`), `research/main_model/README.txt`; ReRoPE of
Su, kexue.fm/archives/9708 (`papers/kexue-9708-rerope`). Review note: `review/rerope_block.md`.

The question was whether the main model's block approximates better or worse when ReRoPE
replaces LeRoPE and nothing else changes. Everything but the attention acts on one position at a
time, so the block at the query `i` is a map `Φ` of the query's token and of `H` attentions whose
scores read the position `ψ(i − j)` of each key (`block`): `i − j` for LeRoPE, `min(i − j, w)`
for ReRoPE.

* Inside the window the two blocks are one function (`block_rePosition_of_lt`).
* Beyond it every ReRoPE block is unchanged when the keys beyond the window are permuted among
  themselves (`block_rePosition_perm`), so it misses every target that sees their order by half
  of what the permutation changes (`norm_sub_le_of_rePosition`).
* LeRoPE's π head, one head of the main model with every frequency `π`, sees the order of the
  first two keys: swapping them changes its output by at least `(1 − N^{−2κ}) / N` times
  `‖u_a − u_b‖`, `N` keys, `κ = qᵀq/d`, just below `1` after the head RMS
  (`norm_piHead_sub_lePosition`). So at every query `w + 1` every ReRoPE block, whatever its
  `Φ`, heads and frequencies, misses it by at least half of that (`gap_rePosition_piHead`).
* With scores free functions of the position the ReRoPE blocks are LeRoPE blocks
  (`rePosition_class_subset`), strictly by that gap.
* Leaky ReRoPE (`Leaky`) is LeRoPE inside the window (`block_lkPosition_of_lt`). Its position is
  invertible, so with free scores its blocks are exactly LeRoPE's (`lkPosition_class_eq`). With
  rotary scores, beyond the window its heads are LeRoPE heads at `k` times lower frequencies
  with a constant turn of the key (`headScore_lkPosition_of_le`). For an integer `k` every ReRoPE
  block misses its `kπ` head as it misses LeRoPE's π head (`gap_rePosition_lkHead`).
* ReRoPE's peak head (`Peak`), one head with every frequency `π/(W − 1)` and the key turned by
  `1`, notices at every query `D ≥ W` a swap of the keys at the distances `1` and `D`: for
  `κ ≥ 1/2` its output moves by at least `‖u_a − u_b‖ / (2(W + 1))` (`le_norm_peakHead_sub`).
* A block with `Φ` continuous whose scores at the distance `D` come within any `ε` of its scores
  at the distance `1`, at queries as far as wanted, misses the peak head by at least
  `‖u_a − u_b‖ / (4(W + 1))` (`le_of_approx_peakHead`). The main model's scores come back so at
  the near returns of all frequencies, despite SSMax's growing gain (`abs_headScore_add_sub_le`,
  `exists_return_log`).

The instances of `le_of_approx_peakHead` for LeRoPE and Leaky ReRoPE blocks with the main
model's rotary scores are not stated here, so whether those blocks approximate ReRoPE blocks
beyond the window is not settled here.
-/
