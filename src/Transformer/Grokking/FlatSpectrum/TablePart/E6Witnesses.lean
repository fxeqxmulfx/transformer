module

public import Transformer.Grokking.FlatSpectrum.TablePart.Rectangles

/-!
# PT6: rectangles on the training cells of the tasks outside the class

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. For each task of the
arithmetic data that no function of the class fits, two rectangles of its training cells with its
labels. The test `lean_witnesses_are_training_cells` of `experiments/table_part` reads this file and
checks every cell and label against the real split, and that these are exactly the tasks the solve
puts outside the class.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

/-- The tasks with the full table at `p = 97` that lie outside the class. -/
def outside97 : List (String × Rect × Rect) :=
  [("arithmetic-abs-diff", (0, 1, 2, 3, 2, 1, 2, 3), (0, 3, 2, 3, 2, 1, 0, 3)),
   ("arithmetic-abs-diff-blind", (0, 1, 2, 3, 1, 0, 1, 1), (0, 1, 2, 4, 1, 0, 1, 2)),
   ("arithmetic-cubic-mixed", (0, 1, 2, 3, 2, 7, 13, 3), (0, 1, 2, 4, 2, 7, 21, 4)),
   ("arithmetic-floor-product", (0, 1, 2, 3, 0, 0, 0, 0), (0, 2, 2, 55, 0, 0, 1, 0)),
   ("arithmetic-max", (0, 1, 2, 3, 2, 2, 3, 3), (0, 3, 2, 3, 2, 3, 3, 3)),
   ("arithmetic-mixed", (0, 1, 2, 3, 95, 96, 65, 0), (0, 1, 2, 4, 95, 96, 94, 93)),
   ("arithmetic-parity-cubic", (0, 1, 2, 3, 0, 3, 4, 1), (0, 1, 2, 4, 0, 3, 5, 0)),
   ("arithmetic-parity-cubic-blind", (0, 1, 2, 3, 0, 1, 2, 0), (0, 1, 2, 4, 0, 1, 2, 0)),
   ("arithmetic-perm-add", (0, 1, 2, 3, 21, 52, 90, 52), (0, 1, 2, 4, 21, 52, 93, 90)),
   ("arithmetic-square-times", (0, 1, 2, 3, 0, 2, 3, 0), (0, 2, 2, 3, 0, 8, 12, 0)),
   ("arithmetic-sum-cube", (0, 1, 2, 3, 8, 27, 64, 27), (0, 1, 2, 4, 8, 27, 28, 64)),
   ("arithmetic-sum-interval", (0, 1, 2, 3, 1, 1, 1, 1), (0, 3, 2, 96, 1, 1, 1, 0)),
   ("arithmetic-sum-pow12", (0, 1, 2, 3, 22, 75, 96, 75), (0, 1, 2, 4, 22, 75, 64, 96)),
   ("arithmetic-sum-pow16", (0, 1, 2, 3, 61, 61, 35, 61), (0, 1, 2, 4, 61, 61, 36, 35)),
   ("arithmetic-sum-pow24", (0, 1, 2, 3, 96, 96, 1, 96), (0, 1, 2, 4, 96, 96, 22, 1)),
   ("arithmetic-sum-pow32", (0, 1, 2, 3, 35, 35, 61, 35), (0, 1, 2, 4, 35, 35, 35, 61)),
   ("arithmetic-sum-pow4", (0, 1, 2, 3, 16, 81, 62, 81), (0, 1, 2, 4, 16, 81, 43, 62)),
   ("arithmetic-sum-pow48", (0, 1, 2, 3, 1, 1, 1, 1), (0, 1, 2, 4, 1, 1, 96, 1)),
   ("arithmetic-sum-pow6", (0, 1, 2, 3, 64, 50, 22, 50), (0, 1, 2, 4, 64, 50, 8, 22)),
   ("arithmetic-sum-pow8", (0, 1, 2, 3, 62, 62, 61, 62), (0, 1, 2, 4, 62, 62, 6, 61)),
   ("arithmetic-sum-squares", (0, 1, 2, 3, 1, 1, 1, 1), (0, 1, 2, 4, 1, 1, 0, 1))]

end Transformer.Grokking.FlatSpectrum.TablePart.E6
