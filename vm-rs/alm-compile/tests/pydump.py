"""The Python side of the graph differential.

Run inside the vendored `transformer-vm/` checkout and diff against
`cargo run -p alm-compile --bin alm-graph-dump`; the two dumps are
expected to be identical line for line, coefficients included, which is
what says the port reproduces the graph rather than something like it.
"""

import struct
from transformer_vm.graph.core import (
    Expression, InputDimension, PersistDimension, ReGLUDimension,
    LookUpDimension, CumSumDimension, reset_graph, _all_dims, _all_lookups,
)
from transformer_vm.wasm.interpreter import build

def terms(e):
    parts = []
    for d, c in e.terms.items():
        bits = struct.unpack("<Q", struct.pack("<d", float(c)))[0]
        parts.append(f"{d.name}:{bits:016x}")
    return "[" + " ".join(parts) + "]"

reset_graph()
input_tokens, output_tokens = build()
out = []
for dim in _all_dims:
    if isinstance(dim, InputDimension):
        out.append(f"dim {dim.id} input {dim.name}")
    elif isinstance(dim, PersistDimension):
        out.append(f"dim {dim.id} persist {dim.name} {terms(dim.expr)}")
    elif isinstance(dim, ReGLUDimension):
        out.append(f"dim {dim.id} reglu {dim.name} a={terms(dim.a_expr)} b={terms(dim.b_expr)}")
    elif isinstance(dim, LookUpDimension):
        out.append(f"dim {dim.id} lookup {dim.name} lu={dim.lookup.id} v={dim.value_index}")
    elif isinstance(dim, CumSumDimension):
        out.append(f"dim {dim.id} cumsum {dim.name} {terms(dim.value_expr)}")
    else:
        out.append(f"dim {dim.id} ??? {dim.name}")
for lu in _all_lookups:
    out.append(
        f"lookup {lu.id} {lu.name or '-'} tie={lu.tie_break} "
        f"qx={terms(lu.query_exprs_2d[0])} qy={terms(lu.query_exprs_2d[1])} "
        f"kx={terms(lu.key_exprs_2d[0])} ky={terms(lu.key_exprs_2d[1])}"
    )
    for i, v in enumerate(lu.value_exprs):
        out.append(f"lookup {lu.id} value {i} {terms(v)}")
for name in sorted(input_tokens):
    out.append(f"in {name} {terms(input_tokens[name])}")
for name in sorted(output_tokens):
    out.append(f"out {name} {terms(output_tokens[name])}")
print("\n".join(out))
