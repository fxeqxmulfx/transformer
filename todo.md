# todo

The paper audit: for every directory under `papers/`, check that each of its
theorem-like statements is written out in `src/`, with `sorry` in proof
position wherever the proof is not carried over.  Debt counts may rise in
`sorry` only; `vacuous` and `placeholder` must not.

## 0. New papers

- [x] `arXiv-2106.06981v2` — Weiss, Goldberg, Yahav, "Thinking Like
      Transformers" (ICML 2021).  No theorem environments; the content is the
      RASP semantics (§3), the `selector_width` implementation (Figure 8), the
      worked programs (§3, Figure 12) and the compilation rule (§3.1, §4).
      Formalized as `Transformer.RASP`.
- [ ] `arXiv-2310.16028v1` — "What Algorithms can Transformers Learn? A Study
      in Length Generalization" (ICLR 2024).  RASP-L, the RASP-Generalization
      Conjecture, 4 lemmas (one `[Constructability]`).  Builds on
      `Transformer.RASP`; the conjecture is a conjecture and must be stated as
      a hypothesis, never as a theorem.
- [ ] `arXiv-2506.16055v3` — "Knee-Deep in C-RASP: A Transformer Depth
      Hierarchy" (COLM 2025).  The largest of the three: 11 definitions,
      ~10 theorems (`thm:TLC_depth`, `thm:TLCl_depth`,
      `thm:tlclpos_depth_hierarchy`, `thm:rtfr_depth_hierarchy`,
      `thm:rtfr_pes_depth_hierarchy`, `thm:rtfr_eq_tlclmod`,
      `thm:tlc_to_majtwo`, `thm:majtwo_to_tlc`, `thm:logical_inclusions`),
      5 propositions, ~8 lemmas.  Depth hierarchies are hard; expect `sorry`
      for most proofs and state them honestly.

## 1. Statements present in a formalized paper but absent from `src/`

Found by diffing every `\label` of a theorem-like environment against the
labels cited anywhere under `src/`.  Twenty-two of them.

- [ ] `arXiv-2312.10794v5` → `Transformer.Perspective`
      - `lem: wgf`
      - the unlabelled theorem: any global minimum of `H_β` is either a sharp
        configuration or the vertices of a 600-cell
- [ ] `arXiv-2410.06833v1` → `Transformer.Metastability`
      - the unlabelled claim on i.i.d. uniform points of `S¹`
      - `lem: appen_time_bound_collapse`
      - `lem: collaps_time_app`
      - `lem_app: exact time scale of clustering`
- [ ] `arXiv-2411.04551v3` → `Transformer.Interpolation`
      - `cl: balls`, `cl: W.to.ball`
      - `prop:generic-discrete`
      - `lem: mass.concentration.Q1`
      - `lem: tubular.mass.movement`
      - `lem: two.balls`
- [ ] `arXiv-2411.04990v2` → `Transformer.Causal`
      - `lemma:convergence`
      - `lemma:interaction`, `lem:interaction`
      - `lemma:scalar`
      - the unlabelled lemma on strong Rényi parking
- [ ] `arXiv-2510.22026v2` → `Transformer.Normalization`
      - the unlabelled corollary: for pre, peri with `n ≤ e^β`,
        unconditional synchronization
      - `lem:loj`
      - `sprop:time_change`
      - `lem:matrix`
- [ ] `arXiv-2512.01868v4` → `Transformer.MeanField`
      - `thm:agazzi_merge`

`arXiv-2603.09078v1` → `Transformer.XSA` has no theorem environments at all;
its two claims are empirical and deliberately stay prose.

## 2. Deliberately not formalized

- §4 of arXiv:2106.06981 concludes from the `Ω(n log n)` comparison lower
  bound for sorting that attention variants restricted to `o(n log n)` pairs
  lose expressive power.  The argument is informal and the restricted
  architectures are not modelled; `RASP.Sort` formalizes the sorting program
  and its correctness, which is the half that is a theorem.
- Dyck-`k`-PTF (Figures 3 and 9 of arXiv:2106.06981).  The programs are
  written out in the paper but no property of them is claimed beyond "solves
  the task in a fixed number of heads and layers for any `k`"; worth adding
  once `RASP.Compilation` is used to count the heads of a written-out program.

## 3. Typos noticed while reading

- arXiv:2106.06981 §2 states the selection matrix as `S[i][j] = p(k[i], q[j])`,
  but the worked example directly below it and the definition of `aggregate`
  both require the transpose — rows are queries, columns are keys.
  `RASP.Defs` takes the convention of the example and records the
  discrepancy.
