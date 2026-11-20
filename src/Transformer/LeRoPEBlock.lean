module

public import Transformer.LeRoPEBlock.Defs
public import Transformer.LeRoPEBlock.Basic

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
  (`norm_sub_le_of_noPos`).
-/
