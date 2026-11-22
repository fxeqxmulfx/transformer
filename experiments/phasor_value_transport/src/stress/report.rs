//! Audit the first scheduled 99% crossing of each fixed task and seed.
use super::*;
use std::io::{BufRead, BufReader};

pub(super) fn validate(row: &Value, seed: u32, dir: &Path, result: &Value) -> Result<()> {
    let train = row["train_examples"]
        .as_u64()
        .ok_or("missing training size")?;
    let held = row["heldout_examples"]
        .as_u64()
        .ok_or("missing held-out size")?;
    let case = &row["case"];
    let number = |k: &str| case[k].as_u64().ok_or("missing case budget");
    let (batch, observe, budget) = (number("batch")?, number("observe")?, number("max_updates")?);
    let epoch = train.div_ceil(batch);
    let invalid = || {
        format!(
            "inconsistent checkpoint or observation history in {}",
            dir.display()
        )
    };
    if result["model"] != MODEL
        || result["initial_gain"] != json!(pair_memory::INITIAL_GAIN)
        || result["fp32_affines"] != precision::FP32_AFFINE_REFERENCE
        || result["suite_sha256"] != suite_hash(row)
        || result["selection"] != row["selection"]
        || result["train_total"] != train
        || !result["train_correct"].as_u64().is_some_and(|n| n <= train)
        || !result["pair_parameters"].as_u64().is_some_and(|n| n > 1)
        || !result["gain"].as_f64().is_some_and(f64::is_finite)
        || (pair_memory::LEARN_BONUS
            && (result["initial_bonus"] != json!(pair_memory::INITIAL_BONUS)
                || !result["successor_bonus"]
                    .as_f64()
                    .is_some_and(f64::is_finite)))
    {
        return Err(invalid().into());
    }
    let history = BufReader::new(fs::File::open(dir.join("observations.jsonl"))?);
    let (mut step, mut previous, mut stopped) = (0, 0, false);
    for line in history.lines() {
        let point: Value = serde_json::from_str(&line?)?;
        let correct = point["correct"].as_u64().ok_or_else(invalid)?;
        if stopped || correct > held {
            return Err(invalid().into());
        }
        let pass = correct * 100 >= held * 99;
        let seen = step / epoch * train + step % epoch * batch;
        if point
            != json!({"case":row["name"],"tier":row["tier"],"seed":seed,"step":step,
                "examples_seen":seen,"correct":correct,"total":held,"pass":pass})
        {
            return Err(invalid().into());
        }
        stopped = pass || step == budget;
        if stopped && (point != result["observation"] || result["previous_step"] != previous) {
            return Err(invalid().into());
        }
        previous = step;
        step = (step + observe).min(budget);
    }
    if !stopped {
        return Err(invalid().into());
    }
    Ok(())
}

/// Read the three fixed seeds. A screen ending before budget is still pending.
fn observations(out: &Path, row: &Value) -> Result<(Vec<Value>, [usize; 3])> {
    let name = row["name"].as_str().ok_or("missing case name")?;
    let mut count = [0; 3];
    let mut seeds = Vec::new();
    for seed in 0..3 {
        let dir = out.join(name).join(format!("seed{seed}"));
        let file = dir.join("result.json");
        if !file.try_exists()? {
            count[2] += 1;
            seeds.push(json!({"seed":seed,"status":"pending"}));
            continue;
        }
        let result: Value = serde_json::from_slice(&fs::read(file)?)?;
        validate(row, seed, &dir, &result)?;
        let point = &result["observation"];
        let pass = point["pass"] == true;
        count[usize::from(!pass)] += 1;
        seeds.push(
            json!({"seed":seed,"status":if pass { "pass" } else { "budget miss" },
            "step":point["step"],"correct":point["correct"],"total":point["total"],
            "examples_seen":point["examples_seen"],"elapsed_seconds":result["elapsed_seconds"]}),
        );
    }
    Ok((seeds, count))
}

pub fn catalog(out: &Path, names: &[String]) -> Result<()> {
    let mut tasks = BTreeMap::new();
    for name in names {
        let (row, _) = load(name)?;
        if row["tier"] != "catalog" || tasks.contains_key(name) {
            return Err("expected distinct catalog additions outside the original cohort".into());
        }
        let (seeds, count) = observations(out, &row)?;
        let tier = if count[0] == 3 {
            let mut slowest = 0.0_f64;
            for run in &seeds {
                let seconds = run["elapsed_seconds"]
                    .as_f64()
                    .filter(|s| s.is_finite() && *s > 0.)
                    .ok_or("missing fresh cell time")?;
                slowest = slowest.max(seconds);
            }
            if slowest <= 60. {
                "short"
            } else if slowest <= 300. {
                "long"
            } else {
                "over five minutes"
            }
        } else if count[1] > 0 {
            "budget miss"
        } else {
            "pending"
        };
        tasks.insert(
            name,
            json!({"tier":tier,"seeds":seeds,"source_target":row["case"]["target"],
            "selection":row["selection"],"catalog_sha256":row["catalog_sha256"]}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"model":MODEL,"tasks":tasks}))?
    );
    Ok(())
}

pub fn run(out: &Path, extended: bool) -> Result<()> {
    if !out.is_dir() {
        return Err(format!("missing cohort directory {}", out.display()).into());
    }
    let mut totals = BTreeMap::<String, [usize; 3]>::new();
    let mut tasks = Vec::new();
    let mut rows = cases()?;
    rows.push(nv11_case());
    if extended {
        for (name, tier) in super::catalog::registered()? {
            let (mut row, _) = super::catalog::load(name)?;
            row["report_tier"] = json!(format!("{tier}-extension"));
            rows.push(row);
        }
    }
    for row in rows {
        let name = row["name"].as_str().ok_or("missing case name")?;
        let tier = row["report_tier"]
            .as_str()
            .or(row["tier"].as_str())
            .ok_or("missing tier")?;
        let count = totals.entry(tier.into()).or_default();
        let (seeds, measured) = observations(out, &row)?;
        for (total, increment) in count.iter_mut().zip(measured) {
            *total += increment;
        }
        tasks.push(json!({"case":name,"tier":tier,"seeds":seeds}));
    }
    let tiers: BTreeMap<_, _> = totals
        .into_iter()
        .map(|(name, [passed, missed, pending])| {
            (
                name,
                json!({"passed":passed,"budget_misses":missed,"pending":pending}),
            )
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"model":MODEL,"tiers":tiers,"tasks":tasks}))?
    );
    Ok(())
}
