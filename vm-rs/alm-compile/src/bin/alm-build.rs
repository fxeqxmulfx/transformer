//! Build `model.bin` from the graph and a schedule, the way
//! `python -m transformer_vm.build --plan plan.yaml --save-weights` does.
//!
//! `--grid` drops the hard-attention query scale (`patches/on-the-grid.patch`,
//! todo3.md section 0) and `--mask` masks dying slots instead of erasing them.

use alm_compile::weights::Options;
use alm_compile::{interpreter, plan::Plan, weights};

fn main() {
    let mut opts = Options::default();
    let mut rest = Vec::new();
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--grid" => opts.query_scale = false,
            "--mask" => opts.use_erase = false,
            _ => rest.push(a),
        }
    }
    let mut args = rest.into_iter();
    let plan_path = args.next().unwrap_or_else(|| "plan.yaml".into());
    let out = args.next().unwrap_or_else(|| "model.bin".into());

    let mg = interpreter::build();
    let text = std::fs::read_to_string(&plan_path).expect("read the plan");
    let plan = Plan::load(&text, &mg.graph).expect("load the plan");
    let model = weights::build(&mg, &plan, opts);
    model.save(&out).expect("write the model");

    eprintln!(
        "{out}: vocab={} d_model={} layers={} heads={} d_ffn={} ({} bytes)",
        model.tokens.len(),
        model.d_model,
        model.layers.len(),
        model.n_heads,
        model.d_ffn,
        model.to_bytes().len()
    );
}
