# Interrupted implementation pilot

This directory preserves 69 completed learning-rate screening runs from
the first launch. It was stopped before any final comparison runs because
a new float32 translation test exposed cancellation in the unshifted Sparsemax
projection. These measurements are excluded from the final ranking.

The corrected implementation subtracts each row maximum before the simplex
projection. The complete experiment is restarted under `../rtx3050/`; no
measurements from different source versions are mixed.

The original attention source is saved in `attention_source.txt`. Source hashes
and the original protocol remain in `metadata.json`.
