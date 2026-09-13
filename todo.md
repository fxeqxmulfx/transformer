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
- [x] `arXiv-2310.16028v1` — "What Algorithms can Transformers Learn? A Study
      in Length Generalization" (ICLR 2024).  Formalized as
      `Transformer.RASPL`: the RASP-L core (Listing 3), the Constructability
      Lemma and the max-aggregation construction (Appendix C), Lemma 4.1 and
      Lemma E.1 with the Fourier analysis they need, and the §4 separation.
      The conjecture is stated as a hypothesis, not a theorem.
- [ ] `arXiv-2506.16055v3` — "Knee-Deep in C-RASP: A Transformer Depth
      Hierarchy" (COLM 2025).  Formalized as `Transformer.CRASP`, 14 modules,
      27 `sorry`.  Written out: the syntax, semantics and depth of
      `TL[◁#]`/`TL[◁#, ▷#]` (`CRASP.Defs`, `CRASP.Basic`), the §4 vocabulary of
      Parikh vectors, intervals and affix restrictions (`CRASP.Parikh`),
      piecewise testability and `A_k` (`CRASP.PiecewiseTestable`), the Cropping
      and Reduction Lemmas with `thm:TLCl_depth` and `thm:TLC_depth`
      (`CRASP.Depth`, `CRASP.TLCDepth`), the Appendix A.3 sugar and its
      elimination (`CRASP.Extensions`), fixed-precision arithmetic
      (`CRASP.Fixed`, sorry-free), future-masked rounded transformers and
      `thm:rtfr_depth_hierarchy` (`CRASP.Transformers`), `MAJ²` with its
      quantifier elimination and the translations (`CRASP.MajTwo`,
      `CRASP.MajTwoEquiv`), and `TL[◁#]^pos` with `thm:ynf`,
      `lem:tlclpos_reduction` and `thm:tlclpos_depth_hierarchy`
      (`CRASP.Positional`, `CRASP.PositionalDepth`).

      **Where I stopped.**  What remains is Appendix E on the transformer
      side: transformers with sinusoidal position encodings
      (`thm:rtfr_eq_tlclmod` and the unlabelled depth theorem below it,
      appendix.tex 1148-1157), with RoPE (`thm:rtfr_to_TLClmod` and the
      unlabelled theorem, 1272-1288), and with ALiBi (`lem:alibi_window`,
      `thm:rtfr_to_TLCly` and the unlabelled theorem, 1300-1332).  Each needs
      `RTfr` of `CRASP.Transformers` re-scored: sinusoidal adds
      `R(θ)^{i-1}(0,1)` to the layer-0 embedding, RoPE replaces the score by
      `R(θ)^i q_i · R(θ)^j k_j`, ALiBi subtracts `a(i-j)` from it.  The plan
      was one file, `CRASP.PositionalTransformers`, with a single scored layer
      and the three instantiations, then the six statements with `sorry`.
      Also still missing: the step from `MAJ²` to `FO[<]`-uniform `LTC⁰`
      circuits in `thm:ltc0_hierarchy`, which needs a circuit model this
      development does not have (`CRASP.MajTwoEquiv` states the logical half).

      Deliberately omitted, both inside `\iffalse` blocks in the source and so
      not part of the paper: `lem:bb`, and
      `lem:piecewise_testable_depth_majtwo` together with `thm:mnf`,
      `thm:tlmod_to_rtfr` and `thm:TLCmod_to_rtfr` (appendix.tex 1171-1271).

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
- The footnote to §3 of arXiv:2310.16028: a next-token function requiring
  `Ω(n³)` time is representable by no Transformer, since Transformers are
  simulable by Turing machines in `O(n²)` time and such tasks exist by the
  Time Hierarchy Theorem.  Both halves are statements about Turing machines,
  which this development does not model.
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
- arXiv:2506.16055 Appendix A.3 defines the strict left-counting operator as
  `◁#_<[φ]^{w,i} = |{j ∈ [i, |w|-1] : w,j ⊨ φ}|`.  The range should be
  `[1, i-1]`: as printed it contradicts both the operator's name and its own
  rewriting rule `◁#_<[φ] ≡ ◁#[φ] - (φ ? 1 : 0)` three lines below.
  Recorded in `CRASP.Extensions`.
- arXiv:2506.16055 `eq:altsingle`, even case, sets `A_k = Σ*(aΣ*bΣ*)^k`; it
  should be `^{k/2}`, as the companion `B_k` correctly has.  As printed it
  fixes `2k` symbols and so is not `k`-piecewise testable, which is what the
  lemma below it needs.  Recorded in `CRASP.PiecewiseTestable`.
- arXiv:2506.16055 `lem:reduction` concludes with "a formula `φ'` of depth
  `(k-1)` of `TL[◁#]^P_{k-1}` (or `TL[◁#,▷#]^P_k`, resp.)".  The parenthetical
  should read `TL[◁#,▷#]^P_{k-1}`, as the sentence's own "of depth `(k-1)`"
  says and as the proof of `thm:TLC_depth` uses it.  Recorded in
  `CRASP.Depth`.
- arXiv:2506.16055 appendix.tex 1154, the proof of the sinusoidal depth
  theorem, reads "every language definable by a rtfr **with RoPE** is
  definable in `TL[◁#,MOD]_k`"; it should say "with sinusoidal positional
  encoding".  The sentence is copied verbatim from the RoPE subsection
  (appendix.tex 1287), which cites `thm:rtfr_to_TLClmod` where this one
  correctly cites `thm:rtfr_eq_tlclmod`.
