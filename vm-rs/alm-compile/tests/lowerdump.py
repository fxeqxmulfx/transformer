"""The Python half of the lowering check: `compilation/lower.py` applied to
every function of a module, printed in the decoder's canonical form.

The one normalization is the constant immediates.  A decoded `i32.const` is
already truncated to 32 bits on both sides, but the expansions write Python
literals — `-1`, `-2147483648` — which stay signed here and are `u32` in the
port.  Masking makes the two comparable, and the machine reads them as 32-bit
words in any case.
"""

import os
import sys

# The vendored release, three levels up from this file.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "transformer-vm"))

from transformer_vm.compilation import decoder as D  # noqa: E402
from transformer_vm.compilation.lower import check_basic_only, lower_hard_ops  # noqa: E402


def imm_text(op, imm):
    if not imm:
        return ""
    if op in (D.OP_BLOCK, D.OP_LOOP, D.OP_IF):
        return "block=0x%02x" % imm[0]
    if op == D.OP_BR_TABLE:
        return "targets=[%s] default=%d" % (", ".join(str(t) for t in imm[0]), imm[1])
    if op == D.OP_CALL_INDIRECT:
        return "type=%d table=%d" % imm
    if op in (D.OP_MEMORY_SIZE, D.OP_MEMORY_GROW):
        return "reserved=%d" % imm[0]
    if len(imm) == 2:
        return "align=%d offset=%d" % imm
    if op == D.OP_I32_CONST:
        return str(imm[0] & 0xFFFFFFFF)
    return str(imm[0])


def main():
    mod = D.decode(open(sys.argv[1], "rb").read())
    for fi, func in enumerate(mod.functions):
        num_params = len(mod.types[mod.func_type_indices[fi]].params)
        mod.functions[fi] = lower_hard_ops(func, num_params)

    out = []
    for t in mod.types:
        out.append("type [%s] -> [%s]" % (", ".join("%02x" % p for p in t.params),
                                          ", ".join("%02x" % r for r in t.results)))
    for i in mod.imports:
        out.append("import %s %s kind=%d index=%d" % (i.module, i.name, i.kind, i.index))
    out.append("functypes [%s]" % ", ".join(str(i) for i in mod.func_type_indices))
    for e in mod.exports:
        out.append("export %s kind=%d index=%d" % (e.name, e.kind, e.index))
    for g in mod.globals:
        out.append("global type=%02x mut=%d init=%d" % (g["valtype"], g["mutable"], g["init"]))
    for d in mod.data_segments:
        out.append("data offset=%d bytes=%s" % (d.offset, d.data.hex()))
    for fi, f in enumerate(mod.functions):
        locs = ", ".join("(%d, %d)" % (c, t) for c, t in f.locals)
        out.append("func %d locals=[%s] n=%d" % (fi, locs, f.num_locals))
        for i, ins in enumerate(f.instructions):
            name = D.WASM_OP_NAMES.get(ins.opcode, "?")
            out.append("  %d %s %s" % (i, name, imm_text(ins.opcode, ins.immediates)))
    for fi, f in enumerate(mod.functions):
        for name, count in check_basic_only(f, fi).items():
            out.append("unsupported func %d %s %d" % (fi, name, count))
    print("\n".join(out))


main()
