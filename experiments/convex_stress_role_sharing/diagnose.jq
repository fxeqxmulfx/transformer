# CM40 training-only diagnostic on a reviewed, complete two-dimensional code.
# This does not verify the source certificate or evaluate any query labels.
def require($ok; $message):
  if $ok then . else error($message) end;
def natural: type == "number" and . >= 0 and . == floor;
def modp($p): ((. % $p) + $p) % $p;
def project($v; $d; $p): ($v[0] * $d[0] + $v[1] * $d[1]) | modp($p);

.discovery.grouped.code
| require(type == "object"; "missing grouped training code")
| require(.dimension == 2; "diagnostic requires a complete dimension-two code")
| .prime as $p
| require(($p | natural) and $p >= 2 and $p <= 997; "field size outside diagnostic cap")
| require(all(range(2; $p); ($p % .) != 0); "field size is not prime")
| require(all([.names.rows, .names.columns, .names.classes][];
    type == "array" and length > 0 and length <= 10000 and
    (length == (unique | length)) and all(.[]; natural and . <= 4294967295));
    "invalid or repeated categorical names")
| require((.rows | length) == (.names.rows | length) and
    (.columns | length) == (.names.columns | length) and
    (.classes | length) == (.names.classes | length); "code/name cardinality mismatch")
| require(all([.rows, .columns, .classes][];
    type == "array" and all(.[]; type == "array" and length == 2 and
      all(.[]; natural and . < $p))); "invalid code coordinates")
| . as $code
| (reduce range(0; ($code.names.columns | length)) as $j
    ({}; .[($code.names.columns[$j] | tostring)] = $j)) as $column_index
| [range(0; ($code.names.rows | length)) as $i
    | ($code.names.rows[$i] | tostring) as $token
    | select($column_index | has($token))
    | [$i, $column_index[$token]]] as $alignments
| require(($alignments | length) >= 2; "need at least two shared categorical identities")
| [[range(0; $p) as $b | [1, $b]][], [0, 1]] as $directions
| [$directions[] as $d
    | {direction: $d,
       distinct_classes: ([$code.classes[] | project(.; $d; $p)] | unique | length),
       role_differences: ([$alignments[] as $ij
         | (project($code.rows[$ij[0]]; $d; $p) -
            project($code.columns[$ij[1]]; $d; $p)) | modp($p)] | unique)}] as $tests
| {prime: $p, dimension: $code.dimension,
   observed_groups: ($code.classes | length),
   aligned_tokens: ($alignments | length),
   enumerated_projective_directions: ($directions | length),
   observed_separating_directions: ([$tests[]
     | select(.distinct_classes == ($code.classes | length))] | length),
   basis_role_difference_counts: [range(0; 2) as $k
     | {coordinate: $k, distinct: ([$alignments[] as $ij
       | ($code.rows[$ij[0]][$k] - $code.columns[$ij[1]][$k]) | modp($p)] | unique | length)}],
   role_shared_separating: [$tests[]
     | select(.distinct_classes == ($code.classes | length) and (.role_differences | length) == 1)
     | {direction, role_difference: .role_differences[0]}]}
