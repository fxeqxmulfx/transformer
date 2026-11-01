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

Choose ordinary addition, fixed scaling or learned gates, choose prime 97 or 193,
and select a diagram component for its formula and source conventions. Controls
update parameter counts, the actual diff_gpt constructor and the evidence status.
On narrow screens, scroll the block diagram horizontally. SVG components support
keyboard focus and activation with Enter or Space.

The page presents the [architecture specification](../../MODEL_ARCHITECTURE.md).
`build.py` extracts actual validation observations from
`experiments/residual_generalization/results.json` and recovery counts from
`experiments/residual_gradient_paths/results.json`. Downloadable `evidence.json`
records the source paths and SHA-256 hashes. Both complete reports and the
architecture document are also included. Lines connect recorded observations;
there are no extrapolated curves or invented learned-gate results.

The site describes the completed local comparison. It is not a training dashboard
and does not claim a universal optimal architecture or convergence theorem.
