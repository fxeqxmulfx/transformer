/-
# Tuple distances versus raw-token convolution offsets

Arora et al., arXiv:2312.04927v1, Appendix `sec: data-dep-ar`, Setup,
equations `eq: data-dependent-kernels-t1` and `eq: y-data-ind-t1`.
The Setup calls `i-j` the distance from query `q_i` to key `k_j`, where
`i,j` index key-value-query triples. A convolution over the raw token
sequence sees key `k_j` at `3j` and query `q_i` at `3i+2`; its offset is
`3(i-j)+2`. The value-to-query offset is one less. The displayed kernel
`X^(i-j)` therefore needs an index conversion in this raw-token model.
-/

import Transformer.Zoology.Appendix_Shift

namespace Transformer.Zoology

/-- Query and key raw positions in a flattened key-value-query sequence.
Source: Appendix equation `eq: kqv-indices`. -/
theorem raw_key_query_gap (i j : ℕ) (hj : j ≤ i) :
    (3 * i + 2) - 3 * j = 3 * (i - j) + 2 := by
  omega

/-- The corresponding value-to-query offset is one less because the value
follows its key immediately. Source: Appendix equations
`eq: data-dependent-kernels-t1` and `eq: z-data-ind-t1`. -/
theorem raw_value_query_gap (i j : ℕ) (hj : j ≤ i) :
    (3 * i + 2) - (3 * j + 1) = 3 * (i - j) + 1 := by
  omega

/-- On two triples, a first key occupies raw position `0` and its later
matching query occupies raw position `5`. This key channel isolates that
first position. Source: Appendix equation `eq: kqv-indices`. -/
def firstRawKey : RealSequence 6 1 :=
  fun p _ => if p = 0 then 1 else 0

/-- Literal use of the tuple distance `1` fails to move the earlier key
from raw position `0` to query position `5`; the correct raw offset `5`
does. This refutes the unconverted `s=i-j` reading of the displayed raw
convolution kernel, without refuting the result after index conversion.
Source: Appendix Setup and equation `eq: y-data-ind-t1`. -/
theorem tuple_gap_is_not_raw_shift :
    shiftDown firstRawKey (1 : Fin 6) 5 0 = 0 ∧
      shiftDown firstRawKey (5 : Fin 6) 5 0 = 1 := by
  constructor <;> norm_num [shiftDown, firstRawKey]

end Transformer.Zoology
