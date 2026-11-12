//! Allocation-only preflight. No head fitting or held-out scoring is performed.
use convex_stress_depth::{
    Result, data,
    features::{Bank, Transport},
};
use serde_json::{Value, json};
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const TASKS: [&str; 4] = [
    "basis-depth-easy",
    "basis-depth-hard",
    "basis-recall-easy",
    "gpt-probe-key-value-binding",
];

fn prepare() -> Result<Value> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let mut output = vec![];
    for name in TASKS {
        let row = suite["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("missing frozen task")?;
        let task = data::build(row)?;
        let train = &task.splits["train"];
        let observations = train
            .iter()
            .map(|e| e.targets.iter().filter(|t| **t >= 0).count())
            .sum::<usize>();
        let mut candidates = vec![];
        for depth in [0, 1, 2, 4] {
            let bank = Bank::discover(task.vocab, depth, 4096, Transport::Fourier, train)?;
            let bytes = bank.allocation_bytes(observations)?;
            candidates.push(
                json!({"depth":depth,"paths":bank.paths(),"features":bank.dimension(),
                "mask_count":bank.masks.len(),"masks":bank.masks,"explicit_buffer_bytes":bytes,
                "within_two_gib_buffer_budget":bytes<=2*1024*1024*1024usize}),
            );
        }
        output.push(json!({"task":name,"vocab":task.vocab,"original_train_examples":train.len(),
            "original_train_prefixes":observations,"data_fingerprints":row["data_fingerprints"],"candidates":candidates}));
    }
    Ok(
        json!({"status":"allocation_preflight_only","learned_head_or_stress_metrics":false,
        "explicit_buffer_budget_bytes":2*1024*1024*1024usize,"planned_process_limit_gib":8,"tasks":output}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["prepare"] {
        println!("{}", serde_json::to_string_pretty(&prepare()?)?);
        Ok(())
    } else {
        Err("usage: convex-stress-depth prepare".into())
    }
}
