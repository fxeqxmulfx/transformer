//! Expose each row's final decision from the pinned GPT scorer to paired comparisons.
use std::{env, fs, path::PathBuf};

fn replace_once(text: String, from: &str, to: &str) -> String {
    assert_eq!(text.matches(from).count(), 1, "pinned scoring site changed");
    text.replacen(from, to, 1)
}

fn main() {
    let source =
        "../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/metrics.rs";
    println!("cargo:rerun-if-changed={source}");
    let original = fs::read_to_string(source).unwrap();
    let entry = "pub fn evaluate(engine: &mut Engine, examples: &[Example], generate: bool) -> Result<Value> {";
    let body = replace_once(
        original,
        entry,
        &format!(
            "{entry}
    evaluate_with_rows(engine, examples, generate, |_, _| {{}})
}}

pub(crate) fn evaluate_with_rows(engine: &mut Engine, examples: &[Example], generate: bool,
    mut row: impl FnMut(usize, bool)) -> Result<Value> {{"
        ),
    );
    let dense = "teacher.exact += usize::from(exact);";
    let generated = "generated.exact += usize::from(exact);";
    let body = replace_once(
        body,
        dense,
        &format!("{dense}\n            if !generative {{ row(start + local, exact); }}"),
    );
    let body = replace_once(
        body,
        generated,
        &format!("{generated}\n            row(start + i, exact);"),
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(out.join("gpt_metrics.rs"), body).unwrap();
    fs::write(
        out.join("score_module.rs"),
        "#[path = \"gpt_metrics.rs\"] mod metrics;\n",
    )
    .unwrap();
}
