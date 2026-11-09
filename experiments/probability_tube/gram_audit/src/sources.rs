//! Construction access is limited to sealed old forecasts, identities and anchors.
use crate::{
    Result,
    application::State,
    checkpoints,
    domain::{
        data::{Data, Plan},
        model::{Arm, initialize},
    },
    evidence::{self, Ledger, require},
    infrastructure::storage,
};
use serde_json::{Value, json};
use std::path::Path;
pub const COHORTS: [&str; 3] = ["lr001", "lr01", "settle"];
pub const HASHES: [&str; 2] = [
    "b237a39dddb8fc1f905849c390c8cbc7f4f5cb3cdd1d962a2c174a6e08fa1181",
    "e1a3a4adab7f179aa5e45b552f8b4fefc9f00cc8c3cef05ab81ead0b702d9d68",
];
pub fn array(v: &Value) -> Result<Vec<f64>> {
    Ok(serde_json::from_value(v.clone())?)
}
pub fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or("finite number".into())
}
pub fn support(data: &Data) -> Vec<bool> {
    let mut active = vec![false; 97];
    for row in data.train.as_chunks::<3>().0 {
        active[row[2] as usize] = true;
    }
    active
}
pub fn inventory() -> Vec<Value> {
    COHORTS
        .into_iter()
        .flat_map(|cohort| {
            checkpoints::SEEDS.into_iter().flat_map(move |seed| {
                checkpoints::ANCHORS.into_iter().map(move |anchor| {
                    json!({"cohort":cohort,"seed":seed,
            "anchor":anchor,"status":"pending","horizons":checkpoints::HORIZONS.into_iter()
                .filter(|h|anchor+h<=10000).collect::<Vec<_>>()})
                })
            })
        })
        .collect()
}
pub fn sealed(root: &Path, ledger: &mut Ledger) -> Result<Value> {
    let bytes = ledger.read(&root.join("construction.json"))?;
    let seal = ledger.read(&root.join("construction.sha256"))?;
    require(
        evidence::hash(&bytes).as_bytes() == seal,
        "construction seal",
    )?;
    let f: Value = serde_json::from_slice(&bytes)?;
    require(
        f["status"] == "frozen" && f["future_state_reads"] == 0,
        "old-only construction status",
    )?;
    let cases = f["cases"].as_array().ok_or("source cases")?;
    require(cases.len() == 144, "source case count")?;
    for (case, want) in cases.iter().zip(inventory()) {
        for key in ["cohort", "seed", "anchor", "horizons"] {
            require(case[key] == want[key], "source case inventory")?;
        }
        require(case["status"] == "frozen", "source case unavailable")?;
    }
    Ok(f)
}
pub fn payload(root: &Path, spec: &Value, ledger: &mut Ledger) -> Result<Value> {
    let bytes =
        ledger.read(&root.join(spec["forecast"]["file"].as_str().ok_or("forecast file")?))?;
    require(
        evidence::hash(&bytes) == spec["forecast"]["sha256"],
        "forecast hash",
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}
pub fn old(
    root: &Path,
    seed: u64,
    anchor: usize,
    ledger: &mut Ledger,
) -> Result<(State, Data, Plan)> {
    let path = root.join(format!("seed{seed}"));
    let identity = ledger.json(&path.join("identity.json"))?;
    let data = Data::new(seed);
    let plan = Plan::new(seed, 10000);
    require(
        identity["seed"] == seed
            && identity["initial_sha256"]
                == storage::hash(&storage::floats(&initialize(Arm::Plain, seed)))
            && identity["train_sha256"] == storage::hash(&storage::words(&data.train))
            && identity["validation_sha256"] == storage::hash(&storage::words(&data.validation))
            && identity["order_sha256"] == storage::hash(&storage::words(&plan.order)),
        "original source identity",
    )?;
    require(
        data.train.len() == 4704 * 3 && support(&data).iter().filter(|v| **v).count() == 33,
        "training support",
    )?;
    let bytes = ledger.read(&path.join(format!("checkpoints/step-{anchor:06}.bin")))?;
    let (step, state) = storage::decode(&bytes, Arm::Plain.count())?;
    require(step == anchor, "anchor time")?;
    Ok((state, data, plan))
}
pub fn pair(dirs: &[std::path::PathBuf; 2], ledger: &mut Ledger) -> Result<[Value; 2]> {
    let values = [sealed(&dirs[0], ledger)?, sealed(&dirs[1], ledger)?];
    for i in 0..2 {
        require(
            values[i]["source_sha256"] == HASHES[i],
            "frozen source fingerprint",
        )?;
    }
    require(
        values[0]["roots"] == values[1]["roots"],
        "source cohort roots",
    )?;
    Ok(values)
}
