//! Compare the native adapter against all twelve archived source-generator
//! runs at seed 11 and inventory the complete original stress data at seed 1.
use baseline::hash;
use composed_retrieval::{
    Result,
    chain_data::{self, PRIMES, STEPS},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

fn archived(history: &Value, prime: usize, steps: usize) -> Result<&Value> {
    let runs = history["runs"].as_array().ok_or("historical runs")?;
    let mut matches = runs.iter().map(|r| &r["task"]).filter(|s| {
        s["operation"] == "chain"
            && s["data_seed"] == 11
            && s["prime"] == prime
            && s["steps"] == steps
    });
    let first = matches
        .next()
        .ok_or("missing historical generator evidence")?;
    if matches.any(|r| r != first) {
        return Err("inconsistent retained hashes across runs of one source dataset".into());
    }
    Ok(first)
}

fn inventory(path: &Path) -> Result<Value> {
    let sources: Value = serde_json::from_str(chain_data::SOURCES)?;
    for (file, source) in sources["files"].as_object().ok_or("source manifest")? {
        if source[2] != hash(&fs::read(path.join(file))?) {
            return Err(format!("changed original source {file}").into());
        }
    }
    let plan: Value = serde_json::from_slice(&fs::read(path.join("plan.json"))?)?;
    let history: Value = serde_json::from_slice(&fs::read(path.join("results.json"))?)?;
    let cases = plan["cases"].as_array().ok_or("165-case plan")?;
    let mut entries = BTreeMap::new();
    let mut historical_hashes = 0;
    for case in cases.iter().filter(|s| s["kind"] == "chain") {
        let spec = &case["data"];
        let prime = usize::try_from(spec["prime"].as_u64().ok_or("original modulus")?)?;
        let steps = usize::try_from(spec["steps"].as_u64().ok_or("composition length")?)?;
        let id = format!("modular-chain-p{prime}-l{steps}");
        if case["name"] != id
            || case["batch"] != 256
            || case["data_seed"] != 1
            || case["metric"] != "sequence_accuracy"
            || case["target"] != 0.99
        {
            return Err("changed original modular-chain case".into());
        }
        let previous = archived(&history, prime, steps)?;
        let historical = chain_data::build(prime, steps, 11)?;
        let mut source = BTreeMap::new();
        let original = chain_data::build(prime, steps, 1)?;
        for (name, historical_name, count, width) in [
            ("train", "train", 2048, 1 + 2 * steps),
            ("validation", "heldout", 1024, 1 + 2 * steps),
            ("test/length-double", "longer", 1024, 1 + 4 * steps),
        ] {
            let old = chain_data::fingerprints(&historical.splits[name], prime)?;
            let pins = &previous[historical_name];
            if pins["fingerprint"] != old.fingerprint
                || pins["input_fingerprint"] != old.input_fingerprint
                || pins["examples"] != old.examples
                || pins["input_length"] != old.input_length
            {
                return Err(
                    format!("{id} {name}: retained source-generator fingerprint differs").into(),
                );
            }
            historical_hashes += 2;
            let current = chain_data::fingerprints(&original.splits[name], prime)?;
            if current.examples != count || current.input_length != width {
                return Err("original chain split shape differs".into());
            }
            source.insert(name, current);
        }
        if entries.insert(id, source).is_some() {
            return Err("duplicate original chain case".into());
        }
    }
    if cases.len() != 165 || entries.len() != PRIMES.len() * STEPS.len() || historical_hashes != 72
    {
        return Err("incomplete original source coverage".into());
    }
    let examples: usize = entries
        .values()
        .flat_map(|s| s.values())
        .map(|s| s.examples)
        .sum();
    Ok(
        json!({"source_manifest_sha256":hash(chain_data::SOURCES.as_bytes()),"data_seed":1,
        "historical_fingerprint_seed":11,"verified_historical_hashes":historical_hashes,"cases":entries,
        "original_splits":entries.values().map(|s|s.len()).sum::<usize>(),
        "examples":examples,"supervised_objects":examples,"scored_models":0}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [dir] = args.as_slice() else {
        return Err("expected ORIGINAL_CHAIN_SOURCE_DIRECTORY".into());
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &inventory(Path::new(dir))?)?;
    println!();
    Ok(())
}
