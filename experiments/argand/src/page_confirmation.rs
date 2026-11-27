//! A fresh fixed-size confirmation cohort, independent of the screening runs.
use crate::{
    Result,
    page_runs::{Job, Rule},
    suite::publish,
};
use serde_json::{Value, json};
use std::{fs, io::Read, path::Path};

const N: usize = 12;

fn binomial_tail(p: f64, k: usize) -> f64 {
    let mut choose = 1.0;
    let mut sum = 0.0;
    for i in 0..=N {
        if i >= k {
            sum += choose * p.powi(i as i32) * (1.0 - p).powi((N - i) as i32);
        }
        choose *= (N - i) as f64 / (i + 1) as f64;
    }
    sum
}

fn lower_bound(k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if binomial_tail(mid, k) < 0.05 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

pub fn run(name: &str, dir: &Path, cap: usize, rule: Rule, plan_only: bool) -> Result<()> {
    let job = Job::new(name, cap)?;
    fs::create_dir_all(dir)?;
    let lock = fs::File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("cohort.lock"))?;
    lock.lock()?;
    let audit_and = matches!(name, "suite-boolean-and-shift" | "suite-boolean-and-random");
    let plan = |pairs: [[u32; 2]; N]| {
        let mut plan = json!({"case":name,"n":N,"updates":job.until,"seeds":pairs,
        "correction":rule.label(),
        "success":"strict correctness of every supervised prefix and exact primary suite score at the fixed horizon",
        "exhaustive_boolean_and_lengths":if audit_and {Some([8,16])} else {None},
        "null":0.75,"one_sided_alpha":0.05});
        if matches!(rule, Rule::AdamW(_)) {
            plan["seed_usage"] = "First word: native initialization and shuffled epochs (order seed = model seed + 10000 modulo 2^32). AdamW uses no estimator sampler. The second word is unused.".into();
            plan["data_split"] = "Identical for all draws and for the discovery run.".into();
        }
        plan
    };
    let path = dir.join("protocol.json");
    let pairs = if path.try_exists()? {
        let stored: Value = serde_json::from_slice(&fs::read(path)?)?;
        let pairs = serde_json::from_value(stored["seeds"].clone())?;
        if stored != plan(pairs) {
            return Err("confirmation protocol mismatch".into());
        }
        pairs
    } else {
        let mut rng = fs::File::open("/dev/urandom")?;
        let mut pairs = [[0; 2]; N];
        for pair in &mut pairs {
            let mut raw = [0; 8];
            rng.read_exact(&mut raw)?;
            *pair = [
                u32::from_le_bytes(raw[..4].try_into()?),
                u32::from_le_bytes(raw[4..].try_into()?),
            ];
        }
        publish(
            dir,
            "protocol.json",
            &serde_json::to_vec_pretty(&plan(pairs))?,
        )?;
        pairs
    };
    if plan_only {
        return Ok(());
    }
    let (mut k, mut native, mut full) = (0, 0, 0);
    for (i, [model, sampler]) in pairs.into_iter().enumerate() {
        let cell = dir.join(format!("draw-{i}"));
        let r = job.run(&cell, model, sampler, rule, true)?;
        let success = r["strict"]["all_strict"] == true && r["correct"] == r["total"];
        k += usize::from(success);
        if audit_and {
            let audit = crate::and_audit::run(&cell)?;
            native += usize::from(audit["unseen_native_support"]["all_strict"] == true);
            full += usize::from(audit["all_hidden_strict"] == true);
        }
        println!(
            "draw {i}: {}/{} correct, minimum gap {}, strict success {success}",
            r["correct"], r["total"], r["strict"]["min_gap"]
        );
    }
    let report = json!({"protocol":plan(pairs),"successes":k,"lower_95":lower_bound(k),"p_value_at_3_4":binomial_tail(0.75,k),
        "native_support_successes":audit_and.then_some(native),"all_hidden_bit_strings_successes":audit_and.then_some(full)});
    publish(
        dir,
        "confirmation.json",
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_twelve_are_needed_to_exclude_three_quarters_at_five_percent() {
        assert!(binomial_tail(0.75, 11) > 0.05 && binomial_tail(0.75, 12) < 0.05);
        assert!((lower_bound(12) - 0.05_f64.powf(1.0 / 12.0)).abs() < 1e-12);
        assert_eq!(lower_bound(0), 0.0);
    }
}
