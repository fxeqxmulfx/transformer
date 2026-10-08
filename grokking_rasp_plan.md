# Grokking inside a fixed RASP task family

Updated 2026-10-08. Status: active. Work solo.

The user restricted the next research cycle to the RASP family. Every new
primary task must have an explicit RASP program and a checked task semantics.
The earlier fixed-table Gen/Mem laws are auxiliary results, not an explanation
of how a trained GPTMini discovers an available algorithm. Preserved mod-97
runs and internal measurements remain calibration evidence at one fixed model
size; they do not establish length generalization or a size law.

## Question

For a specified RASP task and a model that can implement its program, when
does unchanged AdamW find a solution which continues to compute the same
algorithm on held-out inputs and longer sequences? Which measured internal
changes distinguish this from fitting only the training examples?

The task, program, student architecture, data support, precision and tested
length domain must be explicit. Realizability is a first gate, not a supplied
successful Gen component. Width, depth, heads and parameter count are separate
resource quantities. The alphabet size and number of output classes are not
model size. A program's compilation cost is a sufficient resource budget;
it is not automatically a lower bound for every implementation of the task.

## Local sources and the existing Lean boundary

- Weiss et al., `papers/arXiv-2106.06981v2/03_RASP.tex`, section 3.1,
  and `05_experiments.tex`: program semantics and compilation resource counts.
- Zhou et al., `papers/arXiv-2310.16028v1/conjecture.tex`, section 3:
  realizability, simplicity and data diversity. The paper states a
  phenomenological length-generalization conjecture, not a proved universal
  impossibility outside RASP-L. Its transformer setting even permits infinite
  weights for saturating softmax; that premise must not silently enter a
  finite-weight lab model.
- `papers/arXiv-2310.16028v1/rasp.tex`, section 4: autoregressive use reads
  the last position of a causal program. Encoder RASP and this decoder
  interpretation require different serialization bridges.
- `Transformer.RASP.Programs` proves reversal at every sequence length.
  `Transformer.RASP.Sort` proves the sorting program is a permutation and
  orders its keys. `Transformer.RASP.Compilation` provides expression syntax,
  evaluation and head/layer counts. It does not yet implement a checked
  numerical compiler to the lab's GPTMini parameter tensors.
- `Transformer.RASPL.Conjecture` names realizability, simplicity and diversity
  as predicates on abstract represented programs. It does not prove that a
  concrete GPTMini represents a given task or that AdamW learns it.

## Initial task ledger

| Task | Existing semantic certificate | Next required link |
| --- | --- | --- |
| Copy | Token transport / identity semantics | Explicit causal next-token program for the lab prompt, separator and EOS protocol |
| Reverse | `RASP.reverse_apply`, for every length | Prove the explicit `Expr` computes reverse and record its head/layer budget; then the decoder serialization bridge |
| Histogram | `selectorWidth` counts selected equal tokens | Explicit syntax and number-token output contract over the declared length range |
| Sort | `RASP.sortProg_apply`, `RASP.sortProg_keys_monotone` | Explicit syntax/resource budget, duplicate handling and decoder serialization |

These are candidate tasks with known algorithms. They are not all marked as
having a proved numerical GPTMini implementation. Begin with the missing
syntax-to-semantics link for reverse instead of adding more scalar Gen/Mem
limit variants. Other tasks enter training only with their own recorded
realizability boundary, never on the strength of an unrelated teacher table.

## Cycle

1. Fix a task and its train / held-out / longer-length domain. Prove in Lean
   that one explicit RASP program computes its task on that entire domain.
2. Derive the program's heads, dependency depth, finite-value ranges and
   required width. Check a concrete numerical implementation with fixed
   weights. State finite-context, finite-weight and precision assumptions;
   prove the forward/decoding bridge instead of assuming task correctness.
3. Record which student sizes have that construction. Use smaller sizes as
   controls; call them impossible only when an applicable lower bound is
   proved. Do not infer a necessary size from a compiler's upper bound.
4. Train from ordinary random initialization with unchanged AdamW and
   answer CE. Compiled weights serve as a realizability reference, not an
   extra training target or replacement optimizer. Register every recipe
   with the lab DSL before running it.
5. Vary width, depth and heads separately. Report both equal-training-token
   and equal-training-FLOP comparisons, with several independent seeds.
   Retain the original full budgets and do not relabel runs stopped before
   a transition as proof of absence of grokking.
6. Compare train fit, held-out correctness and length extension separately.
   Read internal routes, functional contributions, margins, representation
   geometry and gradient/optimizer histories against the known program.
   A learned circuit need not equal the compiler's particular weights.
7. Formalize only the observed mechanism's supported consequences. Find
   counterexamples to proposed universal detectors. Keep all operational,
   confidence, geometry, spectral, compositional, precision and phase routes
   active within this RASP task family.

## Completion boundary

A useful result identifies a realizable task/program/model-size combination,
a repeatable delayed transition or an explicit failure within budget, and a
Lean statement explaining an observable part of the dynamics with its true
premises. A proof about a prepared teacher alone does not prove student
learning. Correct predictions on a finite held-out set do not prove a single
finite-weight model works at all unbounded lengths.
