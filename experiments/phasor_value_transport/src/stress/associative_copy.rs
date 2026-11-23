//! Data-only port of DiffGPT 5a37c515's mqar_div97.copy_queries.
//! Both copy controls draw identical fixed memories and ordered query pairs.
//! AssociativeCopyScore.swap_target gives an independent check of every target.
use basis::data::Task;
use serde_json::{Value, json};

pub(super) const CONTROL: &str = "associative-divisor-n2-l1-w0";

pub(super) fn build(case: &Value, reference: &Value) -> basis::Result<(Task, Value)> {
    if reference["name"] != CONTROL
        || case["kind"] != "associative"
        || case["data_seed"] != reference["case"]["data_seed"]
        || case["data"] != json!({"compute": 1, "labels": "dividend", "loads": 2, "overwrites": 0})
    {
        return Err("expected the original two-register dividend-copy study".into());
    }
    let mut control = case["data"].clone();
    control["labels"] = "divisor".into();
    if control != reference["case"]["data"] {
        return Err("copy controls do not share the same memory and query distribution".into());
    }
    // Rebuilding the canonical control authenticates all prompts, their order,
    // the source split of each memory and the original dense target masks.
    let mut task = basis::data::build(reference)?;
    let mut fingerprints = serde_json::Map::new();
    for (name, expected) in [
        (
            "train",
            "78bc95b68ba5000e6de04400ed6005c27d5efdf9edbc4f3dae84590aaf87f854",
        ),
        (
            "validation",
            "36052aaefa9da9dda04122fa7c244247dd7578b81eb5143fa399cbc001c92690",
        ),
    ] {
        let rows = task
            .splits
            .get_mut(name)
            .ok_or("missing original copy split")?;
        for example in rows.iter_mut() {
            let [first, _, first_value, second, _, second_value, query, _] = example.tokens[..]
            else {
                return Err("invalid two-register frame".into());
            };
            let value = if query == first {
                first_value
            } else if query == second {
                second_value
            } else {
                return Err("query missing from the displayed memory".into());
            };
            *example
                .targets
                .last_mut()
                .ok_or("missing dense copy target")? = value;
        }
        let actual = basis::wire::tensor_hash(rows);
        if actual != expected {
            return Err(format!("dividend-copy target or mask mismatch on {name}").into());
        }
        fingerprints.insert(name.to_owned(), actual.into());
    }
    Ok((task, fingerprints.into()))
}
