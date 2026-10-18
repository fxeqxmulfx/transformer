#!/usr/bin/env python3
"""Regenerate INDEX.md from the Lean sources.

Navigation aid and audit surface: every declaration, where it lives, and
whether the build can actually vouch for it.  Run after any change under
`src/` and commit the result in the same change (see AGENTS.md).

Three things are reported per declaration, in order of precedence:

  sorry        the declaration contains `sorry` itself, as `lake build`
               reports it.
  vacuous      the statement concludes `True` (`: True`, `: ∃ x, True`, …);
               such a theorem is provable by `trivial` and says nothing,
               yet is invisible both to the sorry count and to
               `#print axioms`.
  placeholder  a definition whose body is a constant (`True`, `0`, `∅`,
               `⊥`, `∀ _, True`) while its docstring promises content.

The sorry status is read from the build, never guessed: `lake build` brings
the `.olean` files up to date and `scripts/Sorries.lean` finds the sorries in
them.  The build's warnings would say the same, but a module restored from
Lake's artifact cache replays none.
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

SRC = "src"
ROOT = "Transformer"
OUT = "INDEX.md"

# A declaration's body runs to the next declaration, but must not swallow the
# file-structure lines that may sit between them (`end`, `section`, a docstring).
STOP = re.compile(r"^(?:end|namespace|section|open|variable|universe|attribute|"
                  r"set_option|/-|#)\b", re.M)

# What `code` looks for: outside comments and strings, inside a block comment,
# inside a string literal.
OUTSIDE = re.compile(r'--|/-|"')
IN_BLOCK = re.compile(r"/-|-/")
IN_STRING = re.compile(r'\\.|"', re.S)
LINE_END = re.compile(r"$", re.M)

DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(theorem|lemma|def|abbrev|structure|class|instance)\s+([^\s({\[:]+)", re.M)
THEOREMS = ("theorem", "lemma")
TRIVIAL_BODY = re.compile(
    r"^(?:by\s+)?(?:exact\s+)?(?:(?:∀|∃)[^,]*,\s*)*(?:True|0|∅|⊥)$")


def code(text):
    """`text` with the inside of every comment and string literal blanked and
    its newlines kept, so offsets and line numbers still hold.  The comments
    are Lean's: `--` to the end of the line, and `/- … -/`, docstrings
    included, which nest; inside a string `--` and `/-` are text.  A block
    comment keeps its opening `/-`, at which `STOP` ends a declaration."""
    out, i = [], 0
    while m := OUTSIDE.search(text, i):
        # Keep text[i:a] and text[b:j], blank text[a:b]; unclosed runs to the end.
        a = b = j = len(text)
        if m.group() == "--":
            a = m.start()
            b = j = LINE_END.search(text, a).start()
        elif m.group() == "/-":
            a = k = m.end()
            depth = 1
            while depth and (n := IN_BLOCK.search(text, k)):
                depth, k = depth + (1 if n.group() == "/-" else -1), n.end()
            if depth == 0:
                b = j = k
        else:
            a = k = m.end()
            while (n := IN_STRING.search(text, k)) and n.group() != '"':
                k = n.end()
            if n:
                b, j = n.start(), n.end()
        out += [text[i:a], re.sub(r"[^\n]", " ", text[a:b]), text[b:j]]
        i = j
    out.append(text[i:])
    return "".join(out)


def normalize(text):
    """Collapse whitespace, so multi-line and one-line declarations can be
    matched by the same patterns; `text` comes from `code`, so its comments
    are blank but for their opening `/-`."""
    return re.sub(r"\s+", " ", text.replace("/-", " ")).strip()


def modules():
    for dirpath, _, filenames in os.walk(SRC):
        for name in sorted(filenames):
            if name.endswith(".lean"):
                yield os.path.join(dirpath, name)


def module_name(path):
    return os.path.relpath(path, SRC)[:-5].replace(os.sep, ".")


def namespace(path):
    """The namespace the file declares into: `Transformer.` + its directory."""
    parts = os.path.relpath(path, SRC)[:-5].split(os.sep)
    if len(parts) == 1:
        return ROOT if parts[0] == ROOT else f"{ROOT}.{parts[0]}"
    if len(parts) == 2:
        return ROOT if parts[1] == "Basic" else f"{ROOT}.{parts[1]}"
    return ".".join(parts[:-1])


def incomplete_declarations():
    """{(path, line): name} for every declaration containing sorry, at the
    line of its name; `scan` removes each one it places."""
    build = subprocess.run(["lake", "build"], capture_output=True, text=True)
    if build.returncode != 0:
        sys.exit("lake build failed; index not regenerated")
    found = subprocess.run(["lake", "env", "lean", "scripts/Sorries.lean"],
                           capture_output=True, text=True)
    if found.returncode != 0:
        sys.exit("scripts/Sorries.lean failed; index not regenerated\n" + found.stdout)
    hits = {}
    for row in found.stdout.splitlines():
        path, line, name = row.split("\t")
        hits[(path, int(line))] = name
    return hits


def scan(path, incomplete):
    text = code(open(path).read())
    marks = list(DECL.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        stop = STOP.search(text, m.end(), end)
        whole = normalize(text[m.start():stop.start() if stop else end])
        line = text[:m.start()].count("\n") + 1
        # Lean places a declaration at its name, below any attribute line.
        name_line = text[:m.start(2)].count("\n") + 1
        kind, name = m.group(1), m.group(2)
        signature, _, body = whole.rpartition(":=")
        if incomplete.pop((path, name_line), None) is not None:
            status = "sorry"
        elif kind in THEOREMS:
            # A conclusion of `True` is the last atom of the signature,
            # whether written `: True` or `: ∃ x, True`.
            status = "vacuous" if signature.rstrip().endswith("True") else "proved"
        elif kind in ("def", "abbrev") and TRIVIAL_BODY.match(body.strip()):
            status = "placeholder"
        else:
            status = ""
        out.append((name, kind, status, line))
    return out


def arxiv_id(path):
    head = "\n".join(open(path).read().split("\n")[:12])
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", head)
    return m.group(1) if m else None


def main():
    incomplete = incomplete_declarations()
    by_namespace = defaultdict(list)
    papers = {}
    for path in modules():
        ns = namespace(path)
        source = open(path).read()
        by_namespace[ns].append(
            (module_name(path), path, source.count("\n"), scan(path, incomplete)))
        paper = arxiv_id(path)
        if paper and ns not in papers:
            papers[ns] = paper
    if incomplete:
        sys.exit("sorry outside every indexed declaration: "
                 + ", ".join(sorted(incomplete.values())))

    everything = [d for ms in by_namespace.values() for _, _, _, ds in ms for d in ds]
    theorems = [d for d in everything if d[1] in THEOREMS]
    counts = {s: sum(1 for d in everything if d[2] == s)
              for s in ("sorry", "vacuous", "placeholder")}
    n_modules = sum(len(ms) for ms in by_namespace.values())

    w = ["# Index", "",
         f"Generated by `scripts/index.py` from `src/` and `lake build`; "
         f"do not edit by hand.", "",
         f"{n_modules} modules · {len(theorems)} theorems · "
         f"{counts['sorry']} using `sorry` · {counts['vacuous']} concluding `True` · "
         f"{counts['placeholder']} placeholder definitions.", "",
         "`vacuous` and `placeholder` are the ones that do not show up in the sorry",
         "count or in `#print axioms`; see [Gaps](#gaps).", "",
         "## Subjects", "",
         "| namespace | paper | modules | theorems | sorry | vacuous | placeholder |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for ns in sorted(by_namespace):
        ds = [d for _, _, _, dd in by_namespace[ns] for d in dd]
        n_t = sum(1 for d in ds if d[1] in THEOREMS)
        cells = [sum(1 for d in ds if d[2] == s)
                 for s in ("sorry", "vacuous", "placeholder")]
        paper = f"arXiv:{papers[ns]}" if ns in papers else "—"
        w.append(f"| [`{ns}`](#{ns.lower().replace('.', '')}) | {paper} "
                 f"| {len(by_namespace[ns])} | {n_t} | " + " | ".join(map(str, cells)) + " |")

    w += ["", "## Declarations", ""]
    for ns in sorted(by_namespace):
        w += [f"### `{ns}`", ""]
        for mod, path, nlines, ds in sorted(by_namespace[ns]):
            if not ds:
                w += [f"**[{mod}]({path})** — {nlines} lines, aggregator", ""]
                continue
            w += [f"**[{mod}]({path})** — {nlines} lines", "",
                  "| declaration | kind | status |", "| --- | --- | --- |"]
            for name, kind, status, line in ds:
                w.append(f"| [`{name}`]({path}#L{line}) | {kind} | {status} |")
            w.append("")

    w += ["## Gaps", "",
          "Everything the build cannot vouch for, in one place.", ""]
    for status, title in (("sorry", "Incomplete proofs (`sorry`)"),
                          ("vacuous", "Vacuous statements (conclude `True`)"),
                          ("placeholder", "Placeholder definitions")):
        rows = [(n, k, mod, path, ln)
                for ns in sorted(by_namespace)
                for mod, path, _, ds in sorted(by_namespace[ns])
                for n, k, st, ln in ds if st == status]
        w += [f"### {title} — {len(rows)}", "",
              "| declaration | kind | module |", "| --- | --- | --- |"]
        for n, k, mod, path, ln in rows:
            w.append(f"| [`{n}`]({path}#L{ln}) | {k} | `{mod}` |")
        w.append("")

    open(OUT, "w").write("\n".join(w).rstrip() + "\n")
    print(f"{OUT}: {n_modules} modules, {len(theorems)} theorems, "
          f"{counts['sorry']} sorry, {counts['vacuous']} vacuous, "
          f"{counts['placeholder']} placeholder")


if __name__ == "__main__":
    main()
