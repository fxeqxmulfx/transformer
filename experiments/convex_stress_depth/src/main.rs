//! Allocation preflight and training-only optimizer checks. No held-out scoring.
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

fn prepare(retention: bool) -> Result<Value> {
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
            let mut bank = Bank::discover(task.vocab, depth, 4096, Transport::Fourier, train)?;
            if retention {
                bank.execution = convex_stress_depth::features::Execution::Cumulative;
                let plan = convex_stress_depth::stream_cache::allocation(
                    &bank,
                    observations,
                    train.len(),
                )?;
                candidates.push(json!({"depth":depth,"paths":bank.paths(),"features":bank.dimension(),
                    "parameters":bank.dimension()*bank.vocab,"masks":bank.masks,
                    "dense_cache_buffer_bytes":bank.allocation_bytes(observations)?,"streamed":plan}));
                continue;
            }
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
        "retained_prefixes":retention,"source_fingerprint":convex_stress_depth::training::fingerprint(),
        "explicit_buffer_budget_bytes":2*1024*1024*1024usize,"planned_process_limit_gib":8,"tasks":output}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["prepare"] || args == ["prepare-retention"] {
        println!(
            "{}",
            serde_json::to_string_pretty(&prepare(args[0] == "prepare-retention")?)?
        );
        Ok(())
    } else if args == ["fingerprint"] {
        println!("{}", convex_stress_depth::training::fingerprint());
        Ok(())
    } else if args == ["cohort-fingerprint"] {
        println!("{}", convex_stress_depth::cohort::fingerprint());
        Ok(())
    } else if args.len() == 2 && args[0] == "prepare-cohort" {
        convex_stress_depth::cohort::prepare(std::path::Path::new(&args[1]))
    } else if args.len() == 3 && args[0] == "run-cell" {
        convex_stress_depth::cohort::run_cell(std::path::Path::new(&args[1]), &args[2])
    } else if args.len() == 3 && args[0] == "replay-cell" {
        println!(
            "{}",
            serde_json::to_string_pretty(&convex_stress_depth::cohort::replay_cell(
                std::path::Path::new(&args[1]),
                &args[2]
            )?)?
        );
        Ok(())
    } else if args.len() == 2 && args[0] == "summarize-cohort" {
        println!(
            "{}",
            serde_json::to_string_pretty(&convex_stress_depth::cohort::summarize(
                std::path::Path::new(&args[1])
            )?)?
        );
        Ok(())
    } else if args.len() == 5 && args[0] == "train-check" {
        convex_stress_depth::training::run(
            &args[1],
            args[2].parse()?,
            &args[3],
            std::path::Path::new(&args[4]),
        )
    } else if args.len() == 2 && args[0] == "replay-train" {
        println!(
            "{}",
            serde_json::to_string_pretty(&convex_stress_depth::training::replay(
                std::path::Path::new(&args[1])
            )?)?
        );
        Ok(())
    } else {
        Err("usage: convex-stress-depth prepare | train-check TASK DEPTH MODE FRESH_DIRECTORY | replay-train DIRECTORY".into())
    }
}
