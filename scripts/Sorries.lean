import Transformer

/-!
# Sorry positions

Implementation imports are intentional: the legacy frontend loads transitive
proof bodies, which module imports omit. Reading public interfaces alone would
incorrectly erase the direct-`sorry` count.

`INDEX.md` marks a declaration as using `sorry` when it contains `sorry`
itself, which is what `lake build` warns about. A module that Lake restores
from its artifact cache comes with an empty build log, though, so its warnings
are never printed. This script reads the same fact from the `.olean` files:

    lake env lean scripts/Sorries.lean

It prints one line `<source>\t<line>\t<declaration>` for each declaration of a
`Transformer` module whose type or value contains `sorry`. The line is that
of the declaration's name, where `lake build` puts its warning. An auxiliary
declaration (`foo.proof_1`, `foo._unary`) has no position of its own and is
reported at the nearest declaration it belongs to; line `0` means none was
found. `scripts/index.py` reads the output.
-/

open Lean Elab Command

/-- The line of the name of `n`, or else of the nearest prefix of `n` that has a position. -/
private partial def nameLine (n : Name) : CommandElabM (Option Nat) := do
  if n.isAnonymous then return none
  match ← findDeclarationRanges? n with
  | some r => return some r.selectionRange.pos.line
  | none => nameLine n.getPrefix

run_cmd do
  let env ← getEnv
  for (mod, data) in env.header.moduleNames.zip env.header.moduleData do
    unless (`Transformer).isPrefixOf mod do continue
    let path := "src/" ++ "/".intercalate (mod.components.map (·.toString (escape := false)))
      ++ ".lean"
    for ci in data.constants do
      let valueHasSorry := ((ci.value? (allowOpaque := true)).map Expr.hasSorry).getD false
      unless ci.type.hasSorry || valueHasSorry do continue
      IO.println s!"{path}\t{(← nameLine ci.name).getD 0}\t{ci.name}"
