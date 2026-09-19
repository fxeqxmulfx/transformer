/-
# Axiom audit

`lake build` warns only about a declaration that contains `sorry` **itself**; a
theorem that *uses* a sorried theorem elaborates without a word, and the sorry
count in `INDEX.md` never sees it.  `#print axioms` does see it, and this
script runs `#print axioms` over every `Transformer.*` declaration at once:

    lake env lean scripts/Axioms.lean

It prints one line per declaration whose axioms are not contained in
`{propext, Classical.choice, Quot.sound}` —

  `sorry   Transformer.Foo.bar`     the declaration is sorried itself,
  `rests   Transformer.Foo.baz`     it is proved, but on top of a sorried one,
  `axiom   Transformer.Foo.qux      Lean.ofReduceBool`   any other axiom,

then a summary line.  `rests` and `axiom` must both be zero; `sorry` is the
debt of `INDEX.md` and must only ever go down.
-/
import Transformer

open Lean Elab Command

private def standard : List Name := [``propext, ``Classical.choice, ``Quot.sound]

run_cmd do
  let env ← getEnv
  let mut selfSorry : Array Name := #[]
  let mut rests : Array Name := #[]
  let mut extra : Array (Name × Name) := #[]
  for (n, _) in env.constants.toList do
    unless (`Transformer).isPrefixOf n && !n.isInternal do continue
    let ax ← liftCoreM (collectAxioms n)
    let others := ax.filter fun a => !(standard.contains a) && a != ``sorryAx
    for a in others do
      extra := extra.push (n, a)
    if ax.contains ``sorryAx then
      let ci ← liftCoreM (getConstInfo n)
      if ((ci.value? (allowOpaque := true)).map Expr.hasSorry).getD false then
        selfSorry := selfSorry.push n
      else
        rests := rests.push n
  for n in rests.qsort (·.toString < ·.toString) do
    logInfo m!"rests   {n}"
  for (n, a) in extra.qsort (fun x y => x.1.toString < y.1.toString) do
    logInfo m!"axiom   {n}   {a}"
  logInfo m!"{selfSorry.size} sorry · {rests.size} resting on a sorry · {extra.size} extra axioms"
