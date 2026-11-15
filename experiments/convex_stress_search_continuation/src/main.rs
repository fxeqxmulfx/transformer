//! Data preparation selects the named source training pool. Search receives
//! only raw causal observations and vocabulary, never case IDs or heldout data.
use convex_stress_search_continuation::*;
use serde_json::json;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [case, output] = args.as_slice() else {
        return Err("usage: convex-stress-search-continuation CASE FRESH_JSON".into());
    };
    let input = prior::adapter::training_only(case)?;
    let report = search::run(
        &input.observations,
        input.vocabulary,
        search::Policy::default(),
    )?;
    let compatible = original::discovery::run(
        &input.observations,
        original::observations::coordinates(input.vocabulary)?,
        report.programs.clone(),
    );
    let (discovery, compatibility_error) = match compatible {
        Ok(d) => (Some(d), None),
        Err(e) => (None, Some(e.to_string())),
    };
    // Discovery's independent decoder check is additional to the search's
    // reserved Boolean-label audit; retain its explicit forecast and refusal.
    original::artifacts::write(
        Path::new(output),
        &json!({
            "input":input,"search":report,"discovery":discovery,
            "compatibility_error":compatibility_error,"heldout_pools_read":0,
            "new_accuracy_measurements":0,"target_realizability_certified":false
        }),
    )
}
