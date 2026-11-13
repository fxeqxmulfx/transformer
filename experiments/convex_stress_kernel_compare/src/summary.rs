//! Complete35-task evidence, including failed imports/evaluations and all
//! immutable construction costs. Primary success is the benchmark decision.
use super::*;
fn exit(root: &Path, file: &str) -> Result<i32> {
    Ok(fs::read_to_string(root.join(file))?.trim().parse()?)
}
pub fn summarize(root: &Path) -> Result<Value> {
    let (inv, parent, stage) = execute::inventory(root, true)?;
    let suite = suite()?;
    let planned: Vec<String> = serde_json::from_value(inv["planned"].clone())?;
    summarize_with(root, &inv, parent, &stage, |name| {
        inputs::build(&suite, name, &planned)
    })
}
pub(crate) fn summarize_with(
    root: &Path,
    inv: &Value,
    parent: Value,
    stage: &Value,
    build: impl Fn(&str) -> Result<inputs::Inputs>,
) -> Result<Value> {
    let all: Vec<String> = serde_json::from_value(inv["tasks"].clone())?;
    let (mut records, mut offers, mut evaluated) = (vec![], vec![], BTreeMap::new());
    let mut failures = 0;
    for task in &all {
        name(task)?;
        let x = build(task)?;
        inputs::check(inv, task, &x)?;
        let run_exit = exit(root, &format!("{task}.run.exit"))?;
        let replay_exit = if run_exit == 0 {
            Some(exit(root, &format!("{task}.replay.exit"))?)
        } else {
            None
        };
        let mut artifacts = vec![];
        for suffix in ["run.exit", "run.resources", "log"] {
            artifacts.push(parents::evidence(&root.join(format!("{task}.{suffix}")))?);
        }
        if replay_exit.is_some() {
            for suffix in [
                "replay.exit",
                "replay.resources",
                "replay.json",
                "replay.log",
            ] {
                artifacts.push(parents::evidence(&root.join(format!("{task}.{suffix}")))?);
            }
        }
        if root.join(task).exists() {
            parents::tree(&root.join(task), &mut artifacts)?;
        }
        if run_exit != 0 || replay_exit != Some(0) {
            failures += 1;
            records.push(json!({"task":task,"status":"process_failure","run_exit":run_exit,"replay_exit":replay_exit,"artifacts":artifacts,"prior_choices_retained":6}));
            continue;
        }
        let (result, current) = execute::retained(root, task, inv, &parent, stage, &x, false)?;
        let replay: Value = read(&root.join(format!("{task}.replay.json")))?;
        if replay != execute::seal(root, task, inv)? {
            return Err("independent comparison replay seal changed".into());
        }
        for row in result["evaluation"]
            .as_array()
            .ok_or("missing original evaluated outcomes")?
        {
            if row["status"] == "evaluated" {
                let key = row["reference"]
                    .as_str()
                    .ok_or("missing actual model reference")?;
                if evaluated.insert(key.to_string(), row.clone()).is_some() {
                    return Err("duplicate actual endpoint evaluation".into());
                }
            }
        }
        offers.extend(current);
        records.push(json!({"task":task,"status":"complete","result":result,"replay":replay,"artifacts":artifacts}));
    }
    let selected = comparison::select(&parent, &offers, &all)?;
    let arms = comparison::render(&parent, &selected, &evaluated)?;
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],"protocol":protocol()?,
        "parent_sha256":inv["request"]["parent"]["summary_sha256"],"staging_sha256":inv["request"]["staging"]["summary_sha256"],
        "complete_parent":parent,"staged_construction":stage,"parent_snapshots":inv["parent_snapshots"],"staging_snapshots":inv["staging_snapshots"],
        "parent_artifacts":inv["parent_artifacts"],"staging_artifacts":inv["staging_artifacts"],"preparation_resources":parents::evidence(&root.join("prepare.resources"))?,"records":records,"tasks":35,"process_failures":failures,
        "selection":selected,"arms":arms,"choose_between_initializers":false,"choices_per_arm":105,"all35_transformer_endpoints_retained":true,
        "unconditional_identification":false,"success_rule":"original_declared_primary_metric","auxiliary_splits":"retained_diagnostics",
        "constructor_accounting":{"new_construction_work":0,"new_discovery_work":0,"staged_costs":stage["constructor_accounting"],
            "charged_once_across_both_arms":true,"all_staged_costs_retained_including_failed_imports":true,"all_prior_costs_retained_in_complete_parent":true}}),
    )
}
