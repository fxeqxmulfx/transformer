//! Build `model.bin` from the graph and a schedule, the way
//! `python -m transformer_vm.build --plan plan.yaml --save-weights` does.

use alm_compile::{interpreter, plan::Plan, weights};

fn main() {
    let mut args = std::env::args().skip(1);
    let plan_path = args
        .next()
        .unwrap_or_else(|| "../transformer-vm/plan.yaml".into());
    let out = args.next().unwrap_or_else(|| "model.bin".into());

    let mg = interpreter::build();
    let text = std::fs::read_to_string(&plan_path).expect("read the plan");
    let plan = Plan::load(&text, &mg.graph).expect("load the plan");
    let model = weights::build(&mg, &plan, true);
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
