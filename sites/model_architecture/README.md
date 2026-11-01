# Local architecture diagram

From the repository root:

```sh
./run.py site
```

Open <http://127.0.0.1:8765>. Use `./run.py site --port 8766` for another port.
Stop the server with Ctrl-C. The command rebuilds the static files in the ignored
`dist/` directory before serving, binds to loopback, and needs only Python's
standard library. There is no network asset dependency, package install or
training step.

The default diagram shows the current fixed-scale GPT. Compare historical
ordinary addition and learned gates, choose prime 97 or 193, and select components
for their equations. Historical views update counts and evidence; the copyable
constructor always builds the current single-architecture GPT and has no legacy
architecture flags. Replay older models at their historical Git revisions.
On narrow screens, scroll the block diagram horizontally. SVG components support
keyboard focus and activation with Enter or Space.

The page presents the [architecture specification](../../MODEL_ARCHITECTURE.md).
`build.py` extracts actual validation observations from
`experiments/residual_generalization/results.json` and recovery counts from
`experiments/residual_gradient_paths/results.json`. It also includes all 48
RMS backward cases from `experiments/rms_backward/results.json`. Downloadable `evidence.json`
records the source paths and SHA-256 hashes. Both complete reports and the
architecture document are also included. Lines connect recorded observations;
there are no extrapolated curves or invented learned-gate results.

When `experiments/residual_scaled_long/results.json` is present, the builder
extends the two scaled curves with those complete-state continuations and adds
their downloadable report. The original task145 report stays available. The
first-99%-validation endpoint is distinguished from test accuracy and the older
three-observation 99.9% stability target.

The site describes the completed local comparison. It is not a training dashboard
and does not claim a universal optimal architecture or convergence theorem.
