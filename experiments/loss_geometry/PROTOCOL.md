# Loss-controlled geometry on the original cubic cohort

Fixed 2026-11-08, task #181, before measurements. No training or optimizer
intervention. Retain all 24 states: four arms, seeds 7/11, steps 0/5000/10000.
Reuse the FP64 forward, frozen scalar oracle and immutable split reconstruction
from flat_approx 9328bc83. Verify checkpoint/state/history/split hashes, sampled
scalar logits and stored train/validation correctness counts. Four OpenBLAS
CPU threads; five-minute analysis cap excluding compilation.

Question: which geometry follows from small observed cross-entropy? Compare
raw state scatter, final-normalized state scatter, raw class-contrast scatter,
normalized-logit contrast scatter, and softmax probability scatter. Group by
exact sum or the three-element cubic answer orbit. Compute means independently
on training and validation rows; never use a validation mean for training.
All 97 output classes compete, including the labels absent from the cubic image.

For each group and split retain count, mean CE, source accuracy, missing target
probability, squared one-hot probability error, probability scatter about the
computed group mean, and that mean's squared one-hot error. Record raw and
normalized state scatter per input, and contrast scatter per input per wrong
class. Verify the exact centering decomposition within + center error = total
one-hot squared error. Verify row missing probability <= CE, row squared
probability error <= 2*CE, and within-group probability scatter <= 2*mean CE.
Use a 1e-10 absolute tolerance for these numerical identity/inequality audits.
Report the maximum observed discrepancies, not only whether checks pass.

Retain weighted split aggregates and every group, including failures of
informal monotonicity expectations. For each adjacent saved interval record
changes in CE and each scatter measure. Count intervals where CE decreases
while raw contrast scatter increases, separately for each grouping and split.
These intervals are retrospective endpoints of actual training, not one-step
gradient tests or proofs about the optimizer's intermediate path.

Lean also checks an actual scalar gradient-descent counterexample: two same-label
binary examples with logits (theta,0) and (2*theta,0). A positive step from
theta>=0 lowers their mean CE and increases centered raw contrast energy.
Native tests verify this behavior and the probability identities on known rows.
This simplified shared-parameter model is not identified with the full transformer.

Small training CE controls a ceiling on training probability scatter; it does
not force monotone scatter, small hidden-state scatter, or low held-out CE.
Validation inequalities use validation loss and are not generalization bounds.
Report all arms and seeds. FP64 checks are not certified arithmetic. Earlier
reports, checkpoints, architecture and training authorization remain unchanged.
