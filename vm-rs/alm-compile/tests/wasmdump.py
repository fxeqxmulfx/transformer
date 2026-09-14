"""The Python half of the decoder check: the same canonical form, from
`compilation/decoder.py`, so the two can be diffed on the same binary."""

import os
import sys

# The vendored release, three levels up from this file.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "transformer-vm"))

from transformer_vm.compilation import decoder as D  # noqa: E402


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
    return str(imm[0])


def main():
    mod = D.decode(open(sys.argv[1], "rb").read())
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
    print("\n".join(out))


main()
